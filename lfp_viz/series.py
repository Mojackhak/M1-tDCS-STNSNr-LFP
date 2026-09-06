"""Render configured spectral and trace Series figures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory
from statsmodels.nonparametric.smoothers_lowess import lowess

from . import visualdf
from .config import VizConfig
from .data import marginalize_laterality
from .scalar import top_strip_label


def smooth_trace_table(source: pd.DataFrame, config: VizConfig) -> pd.DataFrame:
    """LOWESS-smooth trace values while preserving their axes and missing mask."""

    if config.kind != "trace" or source.attrs.get("trace_smoothed") is True:
        return source
    smoothing = config["style"]["smoothing"]
    smoothed = source.copy()
    values: list[pd.Series] = []
    for value in source["Value"]:
        x = value.index.to_numpy(dtype=float)
        y = value.to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        result = np.full(y.shape, np.nan, dtype=float)
        if finite.sum() == 1:
            result[finite] = y[finite]
        elif finite.sum() > 1:
            result[finite] = lowess(
                y[finite],
                x[finite],
                frac=float(smoothing["fraction"]),
                it=int(smoothing["iterations"]),
                delta=float(smoothing["delta_seconds"]),
                is_sorted=True,
                missing="raise",
                return_sorted=False,
            )
        values.append(pd.Series(result, index=value.index, name=value.name))
    smoothed["Value"] = values
    smoothed.attrs["trace_smoothed"] = True
    return smoothed


def series_value_label(model: Mapping[str, Any], config: VizConfig) -> str:
    """Resolve one spectral or trace y-axis label."""

    metric = str(model["Metric"])
    configured = config["labels"]["value_axis"][metric]
    if isinstance(configured, Mapping):
        template = str(configured[str(model["Band"])])
    else:
        template = str(configured)
    band = str(model["Band"])
    band_label = config["labels"].get("bands", {}).get(band, band)
    return template.replace("{Band}", str(band_label))


def _style_axes(fig, n_data_axes: int, default_width: float) -> None:
    for axis in fig.axes[:n_data_axes]:
        for spine in axis.spines.values():
            spine.set_linewidth(default_width)
        axis.tick_params(width=default_width)


def _style_legend(fig, config: VizConfig) -> None:
    legend_style = config["style"]["legend"]
    if not legend_style["show"] or not fig.legends:
        return
    legend = fig.legends[0]
    legend.set_frame_on(legend_style["frame"])
    frame = legend.get_frame()
    frame.set_facecolor(legend_style["face_color"])
    frame.set_edgecolor(legend_style["edge_color"])
    frame.set_alpha(legend_style["frame_alpha"])
    frame.set_linewidth(legend_style["edge_width"])
    if legend_style["fancybox"]:
        frame.set_boxstyle("round", pad=0.4)


def _interval_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _add_trace_intervals(
    fig,
    intervals: pd.DataFrame,
    palette: Mapping[str, str],
    config: VizConfig,
) -> None:
    significant = intervals.loc[
        intervals["p_value"] < config["inference"]["alpha"]
    ].sort_values(["Direction", "Start", "End"])
    if significant.empty:
        return
    axis = fig.axes[0]
    transform = blended_transform_factory(axis.transData, axis.transAxes)
    lanes = {"above_0": 0, "below_0": 0}
    for _, interval in significant.iterrows():
        color = palette[str(interval["LineValue"])]
        axis.axvspan(
            float(interval["Start"]),
            float(interval["End"]),
            facecolor=color,
            alpha=config["inference"]["interval_shadow_alpha"],
            edgecolor="none",
            linewidth=0,
            zorder=0,
        )
        direction = str(interval["Direction"])
        lane = lanes[direction]
        lanes[direction] += 1
        y = 0.98 - lane * 0.06 if direction == "above_0" else 0.02 + lane * 0.06
        axis.text(
            (float(interval["Start"]) + float(interval["End"])) / 2,
            y,
            _interval_stars(float(interval["p_value"])),
            color=color,
            fontsize=config["inference"]["interval_star_pt"],
            fontweight="bold",
            ha="center",
            va="top" if direction == "above_0" else "bottom",
            transform=transform,
            zorder=7,
            clip_on=True,
        )


def render_series_figure(
    source: pd.DataFrame,
    model: Mapping[str, Any],
    figure: Mapping[str, Any],
    config: VizConfig,
    intervals: pd.DataFrame | None = None,
):
    """Render one spectral or trace Phase layout."""

    style = config["style"]
    font = style["font"]
    geometry = style["geometry_mm"]
    strips = style["strips"]
    widths = style["line_width_pt"]
    top_label = top_strip_label(model, config)
    plotted = smooth_trace_table(source, config).copy()
    row_var = None
    row_levels = None
    singleton_lat = (
        str(model["Lat"]) if figure["layout"] == "laterality_singletons" else None
    )
    if singleton_lat is not None:
        plotted = plotted.loc[plotted["Lat"].eq(singleton_lat)].copy()
        plotted["Lat"] = plotted["Lat"].astype(str)
    if config.kind == "spectral":
        line_var = "Phase"
        line_levels = list(config["factor_levels"]["Phase"])
        if figure["layout"] == "marginal":
            plotted = marginalize_laterality(plotted, ["ID", "Phase"])
        elif singleton_lat is not None:
            row_var = "Lat"
            row_levels = [singleton_lat]
        else:
            row_var = "Lat"
            row_levels = list(config["factor_levels"]["Lat"])
        n_grid = len(plotted.iloc[0]["Value"])
        vertical_lines = list(style["frequency_boundaries"])
        vertical_shadows = {
            (float(item["start"]), float(item["stop"])): item["color"]
            for item in style["band_shadows"]
        }
        horizontal_lines = None
    else:
        n_grid = len(plotted.iloc[0]["Value"])
        vertical_lines = list(style["temporal_boundaries"])
        vertical_shadows = None
        horizontal_lines = list(style["horizontal_reference"]["values"])
        if figure["layout"] == "marginal":
            plotted = marginalize_laterality(plotted, ["ID"])
            line_var = "_Line"
            line_levels = [str(figure["line_label"])]
            plotted[line_var] = line_levels[0]
        elif singleton_lat is not None:
            row_var = "Lat"
            row_levels = [singleton_lat]
            line_var = "Lat"
            line_levels = [singleton_lat]
        elif figure["layout"] == "laterality_rows":
            row_var = "Lat"
            row_levels = list(config["factor_levels"]["Lat"])
            line_var = "Lat"
            line_levels = list(config["factor_levels"]["Lat"])
        else:
            line_var = "Lat"
            line_levels = list(config["factor_levels"]["Lat"])
    plotted["_Panel"] = top_label
    if config.kind == "trace" and singleton_lat is not None:
        lat_palette = dict(zip(config["factor_levels"]["Lat"], style["palette"]))
        palette = {singleton_lat: lat_palette[singleton_lat]}
    else:
        palette = dict(zip(line_levels, style["palette"]))
    boundary = style["boundary_line"]
    horizontal = style.get("horizontal_reference", boundary)
    kwargs = dict(
        df=plotted,
        value_col="Value",
        x_var=line_var,
        panel_var="_Panel",
        x_levels=line_levels,
        panel_levels=[top_label],
        x_label=style["axes"]["x_label"],
        single_x_label=True,
        y_label=series_value_label(model, config),
        single_y_label=True,
        x_limits=tuple(style["axes"]["x_limits"]),
        y_limits=None,
        x_log=bool(style["axes"]["x_log"]),
        y_log=bool(style["axes"]["y_log"]),
        title=None,
        font_family=font["family"],
        line_palette=palette,
        line_width=widths["mean"],
        line_alpha=1.0,
        show_ribbon=True,
        ribbon_alpha=style["ribbon"]["alpha"],
        ribbon=style["ribbon"]["kind"],
        grid=style["axes"]["grid"],
        grid_alpha=0.0,
        dpi=style["dpi"],
        show_top_right_axes=style["axes"]["show_top_right_spines"],
        vertical_lines=vertical_lines,
        vline_color=boundary["color"],
        vline_style=boundary["style"],
        vline_width=widths["boundary"],
        vline_alpha=boundary["alpha"],
        horizontal_lines=horizontal_lines,
        hline_color=horizontal["color"],
        hline_style=horizontal["style"],
        hline_width=widths.get("reference", widths["default"]),
        hline_alpha=horizontal["alpha"],
        hline_zorder=horizontal.get("zorder", 4),
        vertical_shadows=vertical_shadows,
        horizontal_shadows=None,
        label_fontsize=font["strip_pt"],
        label_top_bg_color=strips["top_background"],
        label_right_bg_color=strips["right_background"],
        label_text_color=strips["text_color"],
        label_fontweight=strips["font_weight"],
        strip_top_height_mm=geometry["strip_top_height"],
        strip_right_width_mm=geometry["strip_right_width"],
        strip_pad_mm=geometry["strip_pad"],
        title_fontsize=font["axis_pt"],
        axis_label_fontsize=font["axis_pt"],
        tick_label_fontsize=font["tick_pt"],
        legend_loc=(
            style["legend"]["location"]
            if style["legend"]["show"]
            and not (
                config.kind == "trace"
                and (singleton_lat is not None or figure["layout"] == "laterality_rows")
            )
            else "none"
        ),
        legend_ncol=1,
        legend_framealpha=style["legend"]["frame_alpha"],
        legend_fontsize=font["legend_pt"],
        boxsize=(geometry["panel_width"], geometry["panel_height"]),
        panel_gap=tuple(geometry["panel_gap"]),
        include_global_label_margins=True,
        x_label_offset_mm=geometry["x_label_offset"],
        y_label_offset_mm=geometry["y_label_offset"],
        transparent=style["transparent"],
        n_grid=n_grid,
    )
    if row_var is None:
        kwargs.pop("label_right_bg_color")
        kwargs.pop("strip_right_width_mm")
        result = visualdf.plot_double_interaction_series(**kwargs)
        n_data_axes = 1
    else:
        result = visualdf.plot_triple_interaction_series(
            **kwargs,
            facet_var=row_var,
            facet_levels=row_levels,
        )
        n_data_axes = len(row_levels)
    _style_axes(result, n_data_axes, widths["default"])
    _style_legend(result, config)
    if config.kind == "trace" and intervals is not None:
        _add_trace_intervals(result, intervals, palette, config)
    return result
