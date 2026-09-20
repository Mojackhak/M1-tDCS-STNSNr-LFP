"""Render configured raw DataFrame heatmaps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from cmcrameri import cm
from matplotlib.collections import QuadMesh

from . import visualdf
from .config import VizConfig
from .data import marginalize_laterality
from .scalar import top_strip_label


def raw_colorbar_label(model: Mapping[str, Any], config: VizConfig) -> str:
    """Resolve one raw heatmap colorbar label."""

    return str(config["labels"]["colorbar"][str(model["Metric"])])


def _numeric_index(index: pd.Index) -> np.ndarray | None:
    values = np.asarray(pd.to_numeric(index, errors="coerce"), dtype=float)
    return values if np.isfinite(values).all() else None


def _categorical_rows(
    table: pd.DataFrame, config: VizConfig
) -> tuple[pd.DataFrame, list[str]]:
    labels = list(table.iloc[0]["Value"].index.astype(str))
    display = [config["labels"]["bands"].get(label, label) for label in labels]
    converted = table.copy()
    converted["Value"] = converted["Value"].map(
        lambda value: value.set_axis(np.arange(len(value), dtype=float), axis=0)
    )
    return converted, display


def render_raw_figure(
    source: pd.DataFrame,
    model: Mapping[str, Any],
    figure: Mapping[str, Any],
    config: VizConfig,
):
    """Render one complete 0--240 s raw heatmap layout."""

    style = config["style"]
    font = style["font"]
    geometry = style["geometry_mm"]
    strips = style["strips"]
    widths = style["line_width_pt"]
    plotted = source.copy()
    row_var = None
    row_levels = None
    if figure["layout"] == "marginal":
        plotted = marginalize_laterality(plotted, ["ID"])
    elif figure["layout"] == "laterality_singletons":
        lat = str(model["Lat"])
        plotted = plotted.loc[plotted["Lat"].eq(lat)].copy()
        row_var = "Lat"
        row_levels = [lat]
    else:
        row_var = "Lat"
        row_levels = list(config["factor_levels"]["Lat"])
    top_label = top_strip_label(model, config)
    plotted["_Panel"] = top_label
    numeric_y = _numeric_index(plotted.iloc[0]["Value"].index)
    categorical_labels = None
    if numeric_y is None:
        plotted, categorical_labels = _categorical_rows(plotted, config)
        y_limits = (-0.5, len(categorical_labels) - 0.5)
        y_log = False
        horizontal_lines = None
    else:
        y_limits = tuple(style["axes"]["numeric_frequency_limits"])
        y_log = bool(style["axes"]["numeric_frequency_log"])
        horizontal_lines = list(style["frequency_boundaries"])
    boundary = style["boundary_line"]
    kwargs = dict(
        df=plotted,
        panel_var="_Panel",
        panel_levels=[top_label],
        value_col="Value",
        mode="mean",
        x_label=style["axes"]["x_label"],
        y_label="Frequency (Hz)" if numeric_y is not None else "Band",
        single_x_label=True,
        single_y_label=True,
        x_limits=tuple(style["axes"]["x_limits"]),
        y_limits=y_limits,
        x_log=False,
        y_log=y_log,
        title=None,
        font_family=font["family"],
        cmap=cm.vik,
        vmin=None,
        vmax=None,
        vmode="sym",
        dpi=style["dpi"],
        grid=style["axes"]["grid"],
        grid_alpha=0.0,
        boxsize=(geometry["panel_width"], geometry["panel_height"]),
        panel_gap=tuple(geometry["panel_gap"]),
        include_global_label_margins=True,
        x_label_offset_mm=geometry["x_label_offset"],
        y_label_offset_mm=geometry["y_label_offset"],
        cbar_label_offset_mm=geometry["colorbar_label_offset"],
        vertical_lines=list(style["temporal_boundaries"]),
        vline_color=boundary["color"],
        vline_style=boundary["style"],
        vline_width=widths["boundary"],
        vline_alpha=boundary["alpha"],
        vertical_shadows=None,
        horizontal_lines=horizontal_lines,
        hline_color=boundary["color"],
        hline_style=boundary["style"],
        hline_width=widths["boundary"],
        hline_alpha=boundary["alpha"],
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
        colorbar_label=raw_colorbar_label(model, config),
        colorbar_width_mm=geometry["colorbar_width"],
        colorbar_pad_mm=geometry["colorbar_pad"],
        annotate_input_vars=False,
        input_vars_box_loc="upper left",
        input_vars_fontsize=font["tick_pt"],
        input_vars_facecolor="white",
        input_vars_text_color="black",
        transparent=style["transparent"],
    )
    if row_var is None:
        kwargs.pop("label_right_bg_color")
        kwargs.pop("strip_right_width_mm")
        result = visualdf.plot_double_interaction_df(**kwargs)
        n_data_axes = 1
    else:
        result = visualdf.plot_triple_interaction_df(
            **kwargs,
            facet_var=row_var,
            facet_levels=row_levels,
        )
        n_data_axes = len(row_levels)
    for axis in result.axes[:n_data_axes]:
        for collection in axis.collections:
            if isinstance(collection, QuadMesh):
                collection.set_rasterized(True)
        for spine in axis.spines.values():
            spine.set_linewidth(widths["default"])
        axis.tick_params(width=widths["default"])
        axis.spines["top"].set_visible(style["axes"]["show_top_right_spines"])
        axis.spines["right"].set_visible(style["axes"]["show_top_right_spines"])
        if categorical_labels is not None:
            axis.set_yticks(np.arange(len(categorical_labels), dtype=float))
            axis.set_yticklabels(categorical_labels)
    return result
