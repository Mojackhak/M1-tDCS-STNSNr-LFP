"""Pre-60-second bootstrap calibration for contact-paired Ipsi Phase EMMs."""

from pathlib import Path
import argparse
import importlib
import json
import pickle
import pickletools
import zlib
import numpy as np
import pandas as pd
from lfptensorpipe.tabular.grid import (
    AxisInfo,
    parse_band_definitions,
    mean_sufficient_statistics,
)
from lfptensorpipe.lfp.burst.timeline import (
    MappedFragment,
    reduce_burst_scalars,
    burst_event_ids,
    sample_cell_edges,
)
from lfptensorpipe.utils.transforms import get_transform_policy, apply_transform_array

NBOOT = 10000
BLOCK = 10
BANDS = dict(
    zip(
        ["delta", "theta", "alpha", "beta_low", "beta_high", "gamma_low", "gamma_high"],
        [[2, 4], [4, 8], [8, 13], [13, 20], [20, 35], [35, 60], [60, 90]],
    )
)
MODES = {
    "ciplv": "logit",
    "wpli": "logit",
    "imcoh_abs": "fisherz",
    "raw_power": "dB",
    "periodic": "dB",
    "aperiodic": "none",
    "psi": "none",
    "trgc": "none",
}

# The alignment object is never unpickled. Numeric/summary sources are limited
# to the exact NumPy/Pandas reconstruction symbols observed during static audit.
SYMBOLS = [
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("builtins", "slice"),
    ("pandas", "DataFrame"),
    ("pandas", "StringDtype"),
    ("pandas", "Index"),
    ("pandas", "RangeIndex"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),
    ("pandas.arrays", "StringArray"),
    ("pandas.core.indexes.base", "_new_Index"),
]
ALLOWED = {key: getattr(importlib.import_module(key[0]), key[1]) for key in SYMBOLS}


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        key = (module, name)
        if key not in ALLOWED:
            raise pickle.UnpicklingError(f"Non-data reconstruction symbol: {key}")
        return ALLOWED[key]


def read_data(path):
    with Path(path).open("rb") as stream:
        return DataOnlyUnpickler(stream).load()


def timing(path):
    fields = {}
    name = None
    for opcode, arg, _ in pickletools.genops(Path(path).read_bytes()):
        if opcode.name in ("SHORT_BINUNICODE", "BINUNICODE"):
            name = arg
        elif opcode.name == "BINFLOAT" and name in (
            "start_t",
            "end_t",
            "pad_left",
            "anno_left",
            "anno_right",
            "pad_right",
        ):
            if name in fields:
                raise ValueError("Multiple epochs require an explicit mapping.")
            fields[name] = arg
            name = None
    assert np.isclose(fields["start_t"] - fields["pad_left"], 60)
    return fields


def integral_at(x, y, positions):
    """Exact integrals of finite piecewise-linear segments at requested times."""
    good = np.isfinite(y[:-1]) & np.isfinite(y[1:])
    dt = np.diff(x)
    area = np.where(good, 0.5 * (y[:-1] + y[1:]) * dt, 0.0)
    cumulative = np.r_[0.0, np.cumsum(area)]
    p = np.asarray(positions, dtype=float)
    i = np.clip(np.searchsorted(x, p, side="right") - 1, 0, len(x) - 2)
    h = np.clip(p - x[i], 0, dt[i])
    tail = np.where(good[i], y[i] * h + 0.5 * (y[i + 1] - y[i]) / dt[i] * h * h, 0.0)
    return cumulative[i] + tail


def circ_integrals(x, y):
    marks = integral_at(x, y, np.arange(61, dtype=float))
    starts = np.arange(60)
    ends = starts + BLOCK
    return np.where(
        ends <= 60,
        marks[np.minimum(ends, 60)] - marks[starts],
        marks[60] - marks[starts] + marks[ends % 60],
    )


def scalar_from_numden(num, den, metric):
    value = np.divide(num, den, out=np.full(np.shape(num), np.nan), where=den > 0)
    if get_transform_policy(MODES[metric]).reduction_domain == "native":
        value = apply_transform_array(value, mode=MODES[metric])
    return value


def numeric_blocks(values, freqs, source_times, start, band, metric):
    rate = 1.0 / np.median(np.diff(source_times))
    a, b = int(np.round(start * rate)), int(np.round((start + 120) * rate))
    t = source_times[a:b]
    v = values[:, a:b]
    mode = MODES[metric]
    if get_transform_policy(mode).reduction_domain == "transformed":
        v = apply_transform_array(v, mode=mode)
    inside = (t > start) & (t < start + 60)
    nodes = np.r_[start, t[inside], start + 60]
    framed = np.vstack([np.interp(nodes, t, row) for row in v])
    labels = pd.Index(freqs)
    numeric = np.issubdtype(np.asarray(freqs).dtype, np.number)
    axis = AxisInfo(
        labels,
        (
            np.asarray(freqs, dtype=float)
            if numeric
            else np.arange(len(freqs), dtype=float)
        ),
        numeric,
    )
    definition = {band: [BANDS[band]]} if numeric else {band: [[band]]}
    selection = parse_band_definitions(
        definition, axis, inclusive=True, interval_mode="absolute"
    )[0][1]
    parts = np.asarray(
        [
            mean_sufficient_statistics(axis, framed[:, k], selection)
            for k in range(len(nodes))
        ]
    )
    parts[parts[:, 1] <= 0] = np.nan
    bn = circ_integrals(nodes - start, parts[:, 0])
    bd = circ_integrals(nodes - start, parts[:, 1])
    base_n = integral_at(nodes - start, parts[:, 0], [60])[0]
    base_d = integral_at(nodes - start, parts[:, 1], [60])[0]
    reconstructed = scalar_from_numden(bn[::BLOCK].sum(), bd[::BLOCK].sum(), metric)
    base = scalar_from_numden(base_n, base_d, metric)
    assert np.isclose(base, reconstructed, rtol=1e-10, atol=1e-10, equal_nan=True)
    return bn, bd, float(base), float(base_d)


def burst_blocks(values, times, start):
    edges = sample_cell_edges(times)
    ids = burst_event_ids(values)
    full = reduce_burst_scalars(
        values,
        times,
        [MappedFragment(0, 60, start, start + 60, 0)],
        cell_edges=edges,
        event_ids=ids,
    )
    records = []
    for k in range(60):
        end = min(k + BLOCK, 60)
        fragments = [MappedFragment(0, end - k, start + k, start + end, 0)]
        if k + BLOCK > 60:
            fragments.append(
                MappedFragment(end - k, BLOCK, start, start + k + BLOCK - 60, 1)
            )
        result = reduce_burst_scalars(
            values, times, fragments, cell_edges=edges, event_ids=ids
        )
        left_time = start + k
        right_time = start + ((k + BLOCK - 1e-9) % 60)
        li = np.searchsorted(times, left_time, side="right") - 1
        ri = np.searchsorted(times, right_time, side="right") - 1
        lognum = (
            np.log10(result.mean) * result.burst_duration
            if result.burst_duration > 0
            else 0.0
        )
        records.append(
            [
                result.valid_duration,
                result.burst_duration,
                result.event_count,
                lognum,
                float(values[li] > 0),
                float(values[ri] > 0),
            ]
        )
    blocks = np.asarray(records)
    ordered = blocks[::BLOCK]
    count = ordered[:, 2].sum() - np.sum(ordered[:-1, 5] * ordered[1:, 4])
    assert count == full.event_count
    assert np.isclose(ordered[:, 0].sum(), full.valid_duration, atol=1e-8)
    assert np.isclose(ordered[:, 1].sum(), full.burst_duration, atol=1e-8)
    return blocks, full


def burst_values(blocks, draws, reducer):
    sampled = blocks[draws]
    sums = sampled[:, :, :4].sum(axis=1)
    event_count = sums[:, 2] - np.sum(sampled[:, :-1, 5] * sampled[:, 1:, 4], axis=1)
    if reducer == "mean-scalar":
        return np.divide(
            sums[:, 3],
            sums[:, 1],
            out=np.full(len(draws), np.nan),
            where=sums[:, 1] > 0,
        )
    if reducer == "duration-scalar":
        ratio = np.divide(
            sums[:, 1],
            event_count,
            out=np.full(len(draws), np.nan),
            where=event_count > 0,
        )
        return apply_transform_array(ratio, mode="log10")
    if reducer == "rate-scalar":
        ratio = np.divide(
            event_count,
            sums[:, 0],
            out=np.full(len(draws), np.nan),
            where=sums[:, 0] > 0,
        )
        return np.arcsinh(ratio)
    return np.divide(
        100 * sums[:, 1],
        sums[:, 0],
        out=np.full(len(draws), np.nan),
        where=sums[:, 0] > 0,
    )


def run_bootstrap(data_root: Path, stats_root: Path, output_root: Path):
    """Reconstruct paired Pre values and save contact-level bootstrap draws."""
    output_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "pre_reconstruction.csv",
        "bootstrap_contact_values.npz",
        "bootstrap_contact_index.csv",
        "validation.json",
    ):
        if (output_root / name).exists():
            raise FileExistsError(f"Output exists: {output_root / name}")
    results = []
    boot = {}
    registry = pd.read_csv(stats_root / "statistics.csv")

    def tables(kind):
        return pd.concat(
            [
                pd.read_csv(stats_root / p)
                for p in registry.loc[registry.TableType == kind, "Path"]
            ],
            ignore_index=True,
        )

    models, emm, tukey, contacts = (
        tables("models"),
        tables("emmeans"),
        tables("tukey"),
        tables("contact_pairing"),
    )
    emm = emm[(emm.TestID == "Phase-Lat") & (emm.Lat == "Ipsi")]
    tukey = tukey[(tukey.TestID == "Phase-Lat") & (tukey.Lat == "Ipsi")]
    contacts = contacts[(contacts.Lat == "Ipsi") & contacts.Included]
    assert (emm.groupby("ModelID").Phase.nunique() == 4).all()
    assert (tukey.groupby("ModelID").size() == 6).all()
    source_tables = {p: read_data(p) for p in models.SourcePath.unique()}
    all_source = pd.concat(source_tables.values(), ignore_index=True)
    record_lookup = all_source[["ID", "Record", "Trial", "Polar"]].drop_duplicates()
    assert not record_lookup.duplicated(["ID", "Polar"]).any()
    record_lookup = record_lookup.set_index(["ID", "Polar"])
    for metric in models.Metric.unique():
        print("METRIC", metric, flush=True)
        jobs = contacts[contacts.Metric == metric].merge(
            record_lookup.reset_index(), on=["ID", "Polar"], validate="many_to_one"
        )
        for (subject, record, trial), group in jobs.groupby(
            ["ID", "Record", "Trial"], sort=True
        ):
            root = data_root / subject / record
            bounds = timing(root / "alignment" / trial / "warp_labels.pkl")
            start = bounds["pad_left"]
            payload = read_data(root / "tensor" / metric / "tensor.pkl")
            tensor = np.asarray(payload["tensor"])
            axes = payload["meta"]["axes"]
            assert tensor.shape[0] == 1
            values = tensor[0]
            freqs = list(axes["freq"])
            times = np.asarray(axes["time"], dtype=float)
            channel_keys = [
                (
                    json.dumps(list(c), separators=(",", ":"))
                    if isinstance(c, (list, tuple))
                    else str(c)
                )
                for c in axes["channel"]
            ]
            seed = zlib.crc32((subject + "|" + record).encode())
            rng = np.random.default_rng(seed)
            da = rng.integers(0, 60, size=(NBOOT, 6))
            db = rng.integers(0, 60, size=(NBOOT, 6))
            for (unit, band), unit_jobs in group.groupby(
                ["ContactUnitID", "Band"], sort=True
            ):
                if metric in ("ciplv", "imcoh_abs", "wpli"):
                    wanted = frozenset(json.loads(unit))
                    ci = next(
                        i
                        for i, c in enumerate(axes["channel"])
                        if frozenset(c) == wanted
                    )
                else:
                    ci = channel_keys.index(unit)
                if metric == "burst":
                    blocks, full = burst_blocks(
                        values[ci, freqs.index(band)], times, start
                    )
                else:
                    bn, bd, base, valid = numeric_blocks(
                        values[ci], freqs, times, start, band, metric
                    )
                    ba = scalar_from_numden(
                        bn[da].sum(axis=1), bd[da].sum(axis=1), metric
                    )
                    bb = scalar_from_numden(
                        bn[db].sum(axis=1), bd[db].sum(axis=1), metric
                    )
                for row in unit_jobs.itertuples():
                    if metric == "burst":
                        ba = burst_values(blocks, da, row.FeatureOutput)
                        bb = burst_values(blocks, db, row.FeatureOutput)
                        raw = getattr(full, row.FeatureOutput.split("-")[0])
                        transform = {
                            "mean-scalar": "log10",
                            "duration-scalar": "log10",
                            "rate-scalar": "asinh",
                            "occupancy-scalar": "none",
                        }[row.FeatureOutput]
                        base = float(
                            apply_transform_array(np.array(raw), mode=transform)
                        )
                        valid = full.valid_duration
                    boot[(row.ModelID, subject, unit)] = np.stack([ba, bb])
                    source = source_tables[
                        models.set_index("ModelID").loc[row.ModelID, "SourcePath"]
                    ]
                    uc = "Channel" if row.Domain == "local" else "ChannelPair"
                    sub = source[
                        (source.ID == subject)
                        & (source.Polar == row.Polar)
                        & (source.Band == band)
                        & (source.Phase == "Pre")
                        & (source.Lat == "Ipsi")
                    ]
                    keys = sub[uc].map(
                        lambda c: (
                            json.dumps(list(c), separators=(",", ":"))
                            if isinstance(c, (list, tuple))
                            else str(c)
                        )
                    )
                    expected = float(sub.loc[keys == unit, "Value"].iloc[0])
                    assert np.isclose(
                        base, expected, rtol=1e-8, atol=1e-8, equal_nan=True
                    ), (metric, subject, band, base, expected)
                    results.append(
                        dict(
                            ModelID=row.ModelID,
                            ID=subject,
                            ContactUnitID=unit,
                            Metric=metric,
                            Band=band,
                            PreStart=start,
                            PreEnd=start + 60,
                            PreNative=base,
                            PreSource=expected,
                            PreError=base - expected,
                            ValidSupport=valid,
                            BootstrapFinite=float(np.isfinite(ba - bb).mean()),
                            CompletePhase=row.CompletePhase,
                        )
                    )
            print("record_done", subject, record, flush=True)

    validation = pd.DataFrame(results)
    validation.to_csv(output_root / "pre_reconstruction.csv", index=False)
    np.savez_compressed(
        output_root / "bootstrap_contact_values.npz",
        **{str(i): v for i, v in enumerate(boot.values())},
    )
    pd.DataFrame(list(boot.keys()), columns=["ModelID", "ID", "ContactUnitID"]).to_csv(
        output_root / "bootstrap_contact_index.csv", index=False
    )
    report = {
        "replicates": NBOOT,
        "block_seconds": BLOCK,
        "max_pre_error": float(validation.PreError.abs().max()),
    }
    (output_root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_bootstrap(args.data_root, args.stats_root, args.output_root)))


if __name__ == "__main__":
    main()
