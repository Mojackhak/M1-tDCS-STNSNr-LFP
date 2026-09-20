"""Translate the study visualization contract into visualdf scalar artists."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from . import visualdf
from .config import VizConfig


def _feature_name(feature_output: str) -> str:
    suffix = "-scalar"
    if not feature_output.endswith(suffix):
        raise ValueError(
            f"Scalar FeatureOutput must end with {suffix!r}: {feature_output}"
        )
    return feature_output[: -len(suffix)]


def y_axis_label(model: Mapping[str, Any], config: VizConfig) -> str:
    """Render the configured transformed-value label for one model."""

    metric = str(model["Metric"])
    band = str(model["Band"])
    labels = config["labels"]
    metric_labels = labels["y_axis"][metric]
    if metric == "aperiodic":
        template = metric_labels[band]
        band_label = band
    elif metric == "burst":
        template = metric_labels[_feature_name(str(model["FeatureOutput"]))]
        band_label = labels["bands"][band]
    else:
        template = metric_labels["default"]
        band_label = labels["bands"][band]
    rendered = str(template).replace("{Band}", str(band_label))
    contrast = model.get("Contrast")
    prefixes = labels.get("contrast_y_axis_prefix", {})
    if pd.notna(contrast) and str(contrast) in prefixes:
        rendered = f"{prefixes[str(contrast)]}{rendered}"
    return rendered


def top_strip_label(model: Mapping[str, Any], config: VizConfig) -> str:
    """Render canonical region and polarity in the single top strip."""

    region = config["labels"]["region"][str(model["RegionValue"])]
    contrast = model.get("Contrast")
    if pd.notna(contrast) and str(contrast):
        condition = config["labels"]["contrast"][str(contrast)]
    else:
        condition = config["labels"]["polarity"][str(model["Polar"])]
    suffix = " | Pre" if config.stats_analysis_id == "pre-by-polar" else ""
    return f"{region} | {condition}{suffix}"


def render_scalar_figure(
    raw: pd.DataFrame,
    emm: pd.DataFrame,
    inference: pd.DataFrame,
    model: Mapping[str, Any],
    figure: Mapping[str, Any],
    config: VizConfig,
):
    """Render one configured pairwise or EMM-versus-null scalar figure."""

    style = config["style"]
    font = style["font"]
    widths = style["line_width_pt"]
    geometry = style["geometry_mm"]
    strips = style["strips"]
    significance = config["significance"]
    bracket = significance.get(
        "bracket",
        {
            "y_start": 0.50,
            "y_end": 0.90,
            "y_step": 0.08,
            "height_fraction": 0,
            "text_offset_fraction": -0.035,
        },
    )
    sample = config["sample_size"]
    test_kind = config["inputs"]["test_kind"]
    pairwise = test_kind == "emm_pairwise"
    reference = config.data.get("reference_line")
    x_var = str(figure["x_var"])
    x_levels = list(config["factor_levels"][x_var])
    scale_limits = None
    if config["style"]["y_limits"].get("share_across_lat_layouts"):
        bounds = config["style"]["y_limits"].get("bounds", {})
        output_group = str(model.get("OutputGroup", ""))
        if output_group in bounds:
            scale_limits = tuple(float(value) for value in bounds[output_group])
        else:
            scale_limits = visualdf._auto_y_limits_scalar(
                raw,
                emm,
                value_col="Value",
                y_limits=None,
                lower_padding_fraction=config["style"]["y_limits"][
                    "lower_padding_fraction"
                ],
                upper_padding_fraction=config["style"]["y_limits"][
                    "upper_padding_fraction"
                ],
                reference_values=[reference["value"]] if reference else None,
            )
    row_var = figure.get("row_var")
    if figure.get("layout") == "laterality_singletons":
        displayed_lat = str(model["Lat"])
        raw = raw.loc[raw["Lat"].astype(str).eq(displayed_lat)].copy()
        if "Lat" in emm.columns:
            emm = emm.loc[emm["Lat"].astype(str).eq(displayed_lat)].copy()
        if "Lat" in inference.columns:
            inference = inference.loc[
                inference["Lat"].astype(str).eq(displayed_lat)
            ].copy()
        for table in (raw, emm, inference):
            if "Lat" in table.columns:
                table["Lat"] = table["Lat"].astype(str)
        row_var = "Lat"
        row_levels = [displayed_lat]
    elif figure.get("layout") == "polarity_singletons":
        displayed_polar = str(model["Polar"])
        raw = raw.loc[raw["Polar"].astype(str).eq(displayed_polar)].copy()
        if "Polar" in emm.columns:
            emm = emm.loc[emm["Polar"].astype(str).eq(displayed_polar)].copy()
        if "Polar" in inference.columns:
            inference = inference.loc[
                inference["Polar"].astype(str).eq(displayed_polar)
            ].copy()
        for table in (raw, emm, inference):
            if "Polar" in table.columns:
                table["Polar"] = table["Polar"].astype(str)
        row_var = "Polar"
        row_levels = [displayed_polar]
    elif row_var and figure.get("row_levels_from_model"):
        row_levels = [str(model[str(row_var)])]
        raw = raw.copy()
        emm = emm.copy()
        inference = inference.copy()
        for table in (raw, emm, inference):
            table[str(row_var)] = table[str(row_var)].astype(str)
    else:
        row_levels = list(figure.get("row_levels", [])) if row_var else None
    row_strip_labels = row_levels
    if config.stats_analysis_id == "pre-by-polar-contact-paired":
        row_strip_labels = ["Pre"]
    if row_var == "Polar" and row_levels is not None:
        row_strip_labels = [config["labels"]["polarity"][level] for level in row_levels]

    visualdf.set_greek_symbol_font_enabled(False)
    label = y_axis_label(model, config)
    result = visualdf._plot_interaction_scalar_grid(
        df=raw,
        emm=emm,
        tuk=inference if pairwise else pd.DataFrame(),
        value_col="Value",
        x_var=x_var,
        col_var=None,
        row_var=row_var,
        col_levels=None,
        row_levels=row_levels,
        jitter_var="Phase",
        fill_var="Phase",
        outline_var="Phase",
        x_levels=x_levels,
        x_label=x_var,
        single_x_label=True,
        y_label=label,
        single_y_label=True,
        x_limits=None,
        y_limits=scale_limits,
        x_log=False,
        y_log=False,
        xtick_rotation=0,
        ytick_rotation=0,
        title=None,
        font_family=font["latin"],
        jitter_palette=style["phase_palette"],
        fill_palette=style["phase_palette"],
        outline_palette=style["phase_palette"],
        jitter_width=style["jitter"]["width"],
        jitter_alpha=style["jitter"]["alpha"],
        jitter_size=style["jitter"]["size"],
        grid=style["axes"]["grid"],
        grid_alpha=0,
        dpi=style["dpi"],
        seed=style["jitter"]["seed"],
        show_top_right_axes=style["axes"]["show_top_right_spines"],
        show_box=True,
        box_width=style["box"]["width"],
        fill_alpha=style["box"]["fill_alpha"],
        whiskers=style["box"]["whiskers"],
        box_edge_color="black",
        box_edge_width=widths["box_edge"],
        median_color="black",
        median_linewidth=widths["median"],
        whisker_color="black",
        whisker_linewidth=widths["whisker"],
        cap_linewidth=widths["cap"],
        outlier_marker="o",
        outlier_markersize=style["box"]["outlier_marker_size"],
        show_emm_line=True,
        show_emm_ci=True,
        show_raw_mean_line=True,
        emm_line_width=widths["emm_line"],
        emm_line_color=style["emm"]["color"],
        emm_line_style=style["emm"]["line_style"],
        raw_mean_line_width=widths["raw_mean"],
        raw_mean_line_color=style["raw_mean"]["color"],
        raw_mean_line_style=style["raw_mean"]["line_style"],
        error_bar_linewidth=widths["emm_ci"],
        error_bar_cap=style["emm"]["ci_cap_pt"],
        error_bar_color=style["emm"]["color"],
        vertical_lines=None,
        vline_color="black",
        vline_style="-",
        vline_width=widths["default"],
        vline_alpha=1,
        horizontal_lines=[reference["value"]] if reference else None,
        hline_color=reference["color"] if reference else "black",
        hline_style=reference["line_style"] if reference else "-",
        hline_width=reference["line_width_pt"] if reference else widths["default"],
        hline_alpha=reference["alpha"] if reference else 1,
        hline_zorder=reference["zorder"] if reference else 4,
        vertical_shadows=None,
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
        legend_loc="none",
        legend_ncol=None,
        legend_framealpha=0,
        legend_fontsize=font["tick_pt"],
        boxsize=(geometry["phase_width"] * len(x_levels), geometry["panel_height"]),
        panel_gap=tuple(geometry["panel_gap"]),
        include_global_label_margins=True,
        x_label_offset_mm=geometry["x_label_offset"],
        y_label_offset_mm=geometry["y_label_offset"],
        show_brackets=pairwise,
        hide_ns=significance["hide_non_significant"],
        y_start=bracket["y_start"],
        y_end=bracket["y_end"],
        y_step=bracket["y_step"],
        bracket_height_frac=bracket["height_fraction"],
        bracket_color="black",
        bracket_linewidth=widths["significance_bracket"],
        bracket_text_size=font["bracket_pt"],
        transparent=style["transparent"],
        top_strip_labels=[top_strip_label(model, config)],
        right_strip_labels=row_strip_labels,
        sample_size_id_col=sample["id_column"],
        sample_size_template=sample["template"],
        sample_size_color=sample["color"],
        sample_size_fontsize=font["sample_size_pt"],
        sample_size_y_axes=sample["y_axes"],
        sample_size_vertical_alignment=sample["vertical_alignment"],
        axis_linewidth=widths["default"],
        tick_linewidth=widths["default"],
        jitter_linewidth=widths["default"],
        p_value_column=significance["p_column"],
        star_thresholds=[
            (item["p_lt"], item["label"]) for item in significance["thresholds"]
        ],
        bracket_text_offset_fraction=bracket["text_offset_fraction"],
        lower_y_padding_fraction=style["y_limits"]["lower_padding_fraction"],
        upper_y_padding_fraction=style["y_limits"]["upper_padding_fraction"],
        error_bar_marker=style["emm"]["mean_marker"],
        error_bar_marker_size=style["emm"]["mean_marker_size_pt"],
        error_bar_marker_color=style["emm"]["color"],
        error_bar_marker_edge_width=style["emm"]["mean_marker_edge_width_pt"],
        null_tests=inference if not pairwise else None,
        show_null_stars=not pairwise,
        null_p_value_column=significance["p_column"] if not pairwise else None,
        null_star_offset_fraction=significance.get("marker_offset_fraction", 0.035),
    )
    for text_artist in result.texts:
        if text_artist.get_text() == label and text_artist.get_rotation() == 90:
            text_artist.set_horizontalalignment("center")
    return result
