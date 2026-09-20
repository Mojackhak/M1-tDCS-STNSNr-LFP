# -*- coding: utf-8 -*-
"""
visualdf.py

Deterministic, research-grade plotting helpers for faceted DataFrame-based visualizations.

All layout-related sizes are specified in millimeters (mm) and computed from `boxsize`:
- `boxsize` controls the inner plotting box per panel, in mm.
- `cellsize`, where available, derives `boxsize` from each matrix cell size.
- Figure size is derived from panel grid + strips + labels + colorbars.
- `figsize` is intentionally not supported to keep geometry reproducible across exports.
"""

from __future__ import annotations

import heapq
import re
import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from cmcrameri import cm

try:
    import nibabel as nib  # type: ignore
except ImportError:  # pragma: no cover
    nib = None  # type: ignore
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from scipy.ndimage import map_coordinates

# ----------------------- global matplotlib defaults -----------------------

DEFAULT_TEXT_FONT_FAMILY = "Arial Unicode MS"
GREEK_TEXT_FONT_FAMILY = "Symbol"
_GREEK_CHAR_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
_MIXED_FONT_OVERLAY_ATTR = "_visualdf_mixed_font_overlay"
_MIXED_FONT_ORIGINAL_ALPHA_ATTR = "_visualdf_mixed_font_original_alpha"
_GREEK_SYMBOL_FONT_ENABLED = False
_ORIGINAL_FIGURE_SAVEFIG = None


def set_greek_symbol_font_enabled(enabled: bool = True) -> None:
    """Enable or disable automatic Greek-to-Symbol rendering on figure save."""
    global _GREEK_SYMBOL_FONT_ENABLED
    _GREEK_SYMBOL_FONT_ENABLED = bool(enabled)
    if _GREEK_SYMBOL_FONT_ENABLED:
        _patch_figure_savefig_once()


def get_greek_symbol_font_enabled() -> bool:
    """Return whether automatic Greek-to-Symbol rendering is enabled."""
    return _GREEK_SYMBOL_FONT_ENABLED


def _contains_greek_text(text: object) -> bool:
    """Return True when text contains a Unicode Greek code point."""
    return bool(_GREEK_CHAR_RE.search(str(text)))


def _split_greek_runs(text: str) -> List[Tuple[str, bool]]:
    """Split text into adjacent Greek and non-Greek runs."""
    runs: List[Tuple[str, bool]] = []
    current: List[str] = []
    current_is_greek: Optional[bool] = None
    for char in text:
        char_is_greek = bool(_GREEK_CHAR_RE.match(char))
        if current_is_greek is None:
            current_is_greek = char_is_greek
        if char_is_greek != current_is_greek:
            runs.append(("".join(current), current_is_greek))
            current = []
            current_is_greek = char_is_greek
        current.append(char)
    if current:
        runs.append(("".join(current), bool(current_is_greek)))
    return runs


def _font_prop_for_text(text_artist: Text, font_family: str) -> FontProperties:
    """Build font properties for measuring one text run."""
    return FontProperties(
        family=font_family,
        size=text_artist.get_fontsize(),
        weight=text_artist.get_fontweight(),
        style=text_artist.get_fontstyle(),
    )


def _run_uses_mathtext(text_artist: Text, text: str) -> bool | str:
    """Return the renderer math mode flag for a text run."""
    if text_artist.get_usetex():
        return "TeX"
    return text.count("$") >= 2


def _measure_text_run(
    renderer,
    text_artist: Text,
    text: str,
    font_family: str,
) -> Tuple[float, float]:
    """Measure a text run in display pixels."""
    if text == "":
        return 0.0, 0.0
    prop = _font_prop_for_text(text_artist, font_family)
    try:
        width, height, _ = renderer.get_text_width_height_descent(
            text,
            prop,
            ismath=_run_uses_mathtext(text_artist, text),
        )
        return float(width), float(height)
    except Exception:
        temp = Text(0, 0, text, fontproperties=prop)
        temp.set_figure(text_artist.figure)
        temp.set_usetex(text_artist.get_usetex())
        bbox = temp.get_window_extent(renderer=renderer)
        return float(bbox.width), float(bbox.height)


def _alignment_start_offset(width: float, alignment: str) -> float:
    """Return the local x offset for a left-drawn run from an aligned anchor."""
    alignment = str(alignment or "left").lower()
    if alignment == "center":
        return -0.5 * width
    if alignment == "right":
        return -width
    return 0.0


def _alignment_top_offset(height: float, alignment: str) -> float:
    """Return the local y coordinate of the top edge from a vertical anchor."""
    alignment = str(alignment or "center").lower()
    if alignment == "top":
        return 0.0
    if alignment == "bottom":
        return height
    if alignment in {"center", "center_baseline"}:
        return 0.5 * height
    return 0.5 * height


def _rotation_degrees(text_artist: Text) -> float:
    """Return numeric text rotation in degrees."""
    rotation = text_artist.get_rotation()
    if rotation == "vertical":
        return 90.0
    if rotation == "horizontal":
        return 0.0
    try:
        return float(rotation)
    except (TypeError, ValueError):
        return 0.0


def _display_offset_from_text_offset(
    x_offset: float, y_offset: float, rotation: float
) -> Tuple[float, float]:
    """Rotate a local text offset into display-pixel coordinates."""
    theta = np.deg2rad(rotation)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    return (
        x_offset * cos_t - y_offset * sin_t,
        x_offset * sin_t + y_offset * cos_t,
    )


def _copy_text_style_kwargs(
    text_artist: Text, font_family: str, alpha: Optional[float]
) -> Dict[str, object]:
    """Return common Text kwargs for a mixed-font overlay run."""
    kwargs: Dict[str, object] = {
        "fontsize": text_artist.get_fontsize(),
        "fontfamily": font_family,
        "fontstyle": text_artist.get_fontstyle(),
        "fontweight": text_artist.get_fontweight(),
        "color": text_artist.get_color(),
        "rotation": text_artist.get_rotation(),
        "rotation_mode": text_artist.get_rotation_mode(),
        "ha": "left",
        "va": "center",
        "zorder": text_artist.get_zorder() + 0.01,
        "clip_on": text_artist.get_clip_on(),
        "usetex": (
            False if font_family == GREEK_TEXT_FONT_FAMILY else text_artist.get_usetex()
        ),
    }
    if alpha is not None:
        kwargs["alpha"] = alpha
    return kwargs


def _add_mixed_font_run(
    text_artist: Text,
    text: str,
    display_x: float,
    display_y: float,
    font_family: str,
    alpha: Optional[float],
) -> Optional[Text]:
    """Add one overlay text run at a display-space position."""
    if text == "" or text_artist.figure is None:
        return None
    transform = text_artist.get_transform()
    try:
        local_x, local_y = transform.inverted().transform((display_x, display_y))
    except Exception:
        return None

    kwargs = _copy_text_style_kwargs(text_artist, font_family, alpha)
    axes = text_artist.axes
    if axes is not None:
        overlay = axes.text(local_x, local_y, text, transform=transform, **kwargs)
    else:
        overlay = text_artist.figure.text(
            local_x, local_y, text, transform=transform, **kwargs
        )

    setattr(overlay, _MIXED_FONT_OVERLAY_ATTR, True)
    try:
        overlay.set_clip_box(text_artist.get_clip_box())
        overlay.set_clip_path(text_artist.get_clip_path())
    except Exception:
        pass
    try:
        overlay.set_in_layout(text_artist.get_in_layout())
    except Exception:
        pass
    return overlay


def _remove_mixed_font_overlays(fig: Figure) -> None:
    """Remove overlay text artists from a previous mixed-font pass."""
    for artist in list(fig.findobj(match=lambda item: isinstance(item, Text))):
        if bool(getattr(artist, _MIXED_FONT_OVERLAY_ATTR, False)):
            try:
                artist.remove()
            except Exception:
                pass


def _restore_mixed_font_originals(fig: Figure) -> None:
    """Restore original text alpha before recomputing mixed-font overlays."""
    for artist in fig.findobj(match=lambda item: isinstance(item, Text)):
        if hasattr(artist, _MIXED_FONT_ORIGINAL_ALPHA_ATTR):
            artist.set_alpha(getattr(artist, _MIXED_FONT_ORIGINAL_ALPHA_ATTR))


def _candidate_text_artists(fig: Figure) -> List[Text]:
    """Return visible, non-overlay text artists for mixed-font processing."""
    artists: List[Text] = []
    for artist in fig.findobj(match=lambda item: isinstance(item, Text)):
        if bool(getattr(artist, _MIXED_FONT_OVERLAY_ATTR, False)):
            continue
        if not artist.get_visible():
            continue
        if artist.get_text() == "":
            continue
        artists.append(artist)
    return artists


def apply_mixed_font_text(
    fig: Figure,
    *,
    latin_font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    greek_font_family: str = GREEK_TEXT_FONT_FAMILY,
    enabled: bool = True,
) -> Figure:
    """Render Greek characters with Symbol and other text with the default font.

    The original text artists remain in place for layout and caller inspection,
    but Greek-containing artists are made transparent and overlaid with
    per-run text artists using the default font for non-Greek runs and Symbol for Greek
    runs. Underlying label strings are not changed.
    """
    if (
        not enabled
        or fig is None
        or bool(getattr(fig, "_visualdf_mixed_font_applying", False))
    ):
        return fig

    setattr(fig, "_visualdf_mixed_font_applying", True)
    try:
        _remove_mixed_font_overlays(fig)
        _restore_mixed_font_originals(fig)
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        except Exception:
            return fig

        for text_artist in _candidate_text_artists(fig):
            text_artist.set_fontfamily(latin_font_family)
            text = str(text_artist.get_text())
            if not _contains_greek_text(text):
                continue

            original_alpha = text_artist.get_alpha()
            setattr(text_artist, _MIXED_FONT_ORIGINAL_ALPHA_ATTR, original_alpha)
            try:
                anchor_x, anchor_y = text_artist.get_transform().transform(
                    text_artist.get_position()
                )
            except Exception:
                continue

            lines = text.split("\n")
            line_infos: List[
                Tuple[List[Tuple[str, bool, float, float]], float, float]
            ] = []
            base_line_height = max(
                float(text_artist.get_fontsize()) * fig.dpi / 72.0, 1.0
            )
            for line in lines:
                run_infos: List[Tuple[str, bool, float, float]] = []
                line_width = 0.0
                line_height = base_line_height
                for run_text, is_greek in _split_greek_runs(line):
                    run_font = greek_font_family if is_greek else latin_font_family
                    run_width, run_height = _measure_text_run(
                        renderer, text_artist, run_text, run_font
                    )
                    run_infos.append((run_text, is_greek, run_width, run_height))
                    line_width += run_width
                    line_height = max(line_height, run_height)
                line_infos.append((run_infos, line_width, line_height))

            line_height = max([info[2] for info in line_infos] or [base_line_height])
            line_spacing = float(getattr(text_artist, "_linespacing", 1.2))
            line_step = max(line_height * line_spacing, line_height)
            block_height = line_height + line_step * max(0, len(line_infos) - 1)
            top_offset = _alignment_top_offset(
                block_height, text_artist.get_verticalalignment()
            )
            rotation = _rotation_degrees(text_artist)
            if hasattr(text_artist, "get_multialignment"):
                multiline_alignment = text_artist.get_multialignment()
            elif hasattr(text_artist, "_get_multialignment"):
                multiline_alignment = text_artist._get_multialignment()
            else:
                multiline_alignment = None
            multiline_alignment = (
                multiline_alignment or text_artist.get_horizontalalignment()
            )

            for line_index, (run_infos, line_width, _) in enumerate(line_infos):
                cursor = _alignment_start_offset(line_width, multiline_alignment)
                line_center_y = (
                    top_offset - (0.5 * line_height) - (line_index * line_step)
                )
                for run_text, is_greek, run_width, _ in run_infos:
                    run_font = greek_font_family if is_greek else latin_font_family
                    offset_x, offset_y = _display_offset_from_text_offset(
                        cursor, line_center_y, rotation
                    )
                    _add_mixed_font_run(
                        text_artist,
                        run_text,
                        anchor_x + offset_x,
                        anchor_y + offset_y,
                        run_font,
                        original_alpha,
                    )
                    cursor += run_width

            text_artist.set_alpha(0.0)
    finally:
        setattr(fig, "_visualdf_mixed_font_applying", False)
    return fig


def plot_categorical_emm_trajectory(
    emm: pd.DataFrame,
    *,
    x_var: str,
    line_var: str,
    x_levels: List[str],
    line_levels: List[str],
    mean_col: str = "emmean",
    lower_col: str = "lower.CL",
    upper_col: str = "upper.CL",
    x_label: str = "Phase",
    y_label: str,
    top_strip_label: str,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    line_palette: Dict[str, str],
    line_styles: Dict[str, str],
    line_zorders: Dict[str, float],
    line_width: float = 0.75,
    marker: str = "o",
    marker_size: float = 2.0,
    marker_edge_width: float = 0.0,
    error_bar_linewidth: float = 1.0,
    error_bar_cap: float = 0.0,
    y_limits: Optional[Tuple[float, float]] = None,
    reference_value: Optional[float] = 0.0,
    reference_color: str = "#9A9A9A",
    reference_style: str = ":",
    reference_width: float = 0.60,
    reference_alpha: float = 0.80,
    reference_zorder: float = 2.0,
    grid: bool = False,
    show_top_right_axes: bool = False,
    label_fontsize: float = 7.0,
    label_top_bg_color: str = "#D7E3E0",
    label_text_color: str = "black",
    label_fontweight: str = "bold",
    strip_top_height_mm: float = 4.5,
    strip_pad_mm: float = 2.0,
    axis_label_fontsize: float = 7.0,
    tick_label_fontsize: float = 6.0,
    legend_loc: str = "outside_right",
    legend_fontsize: float = 6.0,
    legend_frame: bool = False,
    boxsize: Optional[Tuple[float, float]] = None,
    x_label_offset_mm: float = 5.0,
    y_label_offset_mm: float = 7.0,
    axis_linewidth: float = 0.60,
    dpi: int = 600,
    transparent: bool = True,
) -> plt.Figure:
    """Plot precomputed categorical EMM trajectories in one stripped panel."""

    required = {x_var, line_var, mean_col, lower_col, upper_col}
    missing = sorted(required.difference(emm.columns))
    if missing:
        raise KeyError(f"EMM trajectory table is missing columns: {missing}")
    if emm.duplicated([line_var, x_var], keep=False).any():
        raise ValueError("EMM trajectory table contains duplicate line/x cells.")

    plotted = emm.copy()
    for column in (mean_col, lower_col, upper_col):
        plotted[column] = pd.to_numeric(plotted[column], errors="coerce")
    numeric = plotted[[mean_col, lower_col, upper_col]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("EMM trajectory means and confidence limits must be finite.")
    if (plotted[lower_col] > plotted[mean_col]).any() or (
        plotted[upper_col] < plotted[mean_col]
    ).any():
        raise ValueError("EMM trajectory confidence limits do not contain the mean.")

    observed_x = set(plotted[x_var].astype(str))
    observed_lines = set(plotted[line_var].astype(str))
    if observed_x != set(x_levels):
        raise ValueError("EMM trajectory x levels do not match the configured levels.")
    if observed_lines != set(line_levels):
        raise ValueError(
            "EMM trajectory line levels do not match the configured levels."
        )

    if y_limits is None:
        lower = plotted[lower_col].to_numpy(dtype=float)
        upper = plotted[upper_col].to_numpy(dtype=float)
        if reference_value is not None:
            lower = np.append(lower, float(reference_value))
            upper = np.append(upper, float(reference_value))
        ymin = float(np.min(lower))
        ymax = float(np.max(upper))
        span = ymax - ymin
        if span <= 0:
            span = max(abs(ymin), 1.0)
        y_limits = (ymin - 0.15 * span, ymax + 0.30 * span)

    _configure_plot_fonts(font_family)
    fig, axes, layout = _init_box_figure(
        nrows=1,
        ncols=1,
        boxsize_mm=boxsize or DEFAULT_BOXSIZE_MM,
        panel_gap_mm=(0.0, 0.0),
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=0.0,
        strip_pad_mm=strip_pad_mm,
        single_x_label=True,
        single_y_label=True,
        axis_label_fontsize=axis_label_fontsize,
        include_global_label_margins=True,
        x_label_text=x_label,
        y_label_text=y_label,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        dpi=dpi,
        font_family=font_family,
        transparent=transparent,
    )
    ax = axes[0, 0]
    x_positions = np.arange(len(x_levels), dtype=float)

    if reference_value is not None:
        ax.axhline(
            float(reference_value),
            color=reference_color,
            linestyle=reference_style,
            linewidth=reference_width,
            alpha=reference_alpha,
            zorder=reference_zorder,
        )

    for line_level in line_levels:
        line = plotted.loc[plotted[line_var].astype(str).eq(line_level)].copy()
        line[x_var] = pd.Categorical(
            line[x_var].astype(str), categories=x_levels, ordered=True
        )
        line = line.sort_values(x_var)
        means = line[mean_col].to_numpy(dtype=float)
        lower = line[lower_col].to_numpy(dtype=float)
        upper = line[upper_col].to_numpy(dtype=float)
        color = line_palette[line_level]
        base_zorder = float(line_zorders[line_level])
        ax.errorbar(
            x_positions,
            means,
            yerr=np.vstack((means - lower, upper - means)),
            fmt="none",
            ecolor=color,
            elinewidth=error_bar_linewidth,
            capsize=error_bar_cap,
            zorder=base_zorder,
        )
        ax.plot(
            x_positions,
            means,
            color=color,
            linestyle=line_styles[line_level],
            linewidth=line_width,
            marker=marker,
            markersize=marker_size,
            markeredgewidth=marker_edge_width,
            markeredgecolor=color,
            markerfacecolor=color,
            label=line_level,
            zorder=base_zorder + 1.0,
        )

    ax.set_xlim(-0.5, len(x_levels) - 0.5)
    ax.set_ylim(*y_limits)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_levels)
    ax.tick_params(labelsize=tick_label_fontsize, width=axis_linewidth)
    if grid:
        ax.grid(True, axis="y", alpha=0.30, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(show_top_right_axes)
    ax.spines["right"].set_visible(show_top_right_axes)
    for spine in ax.spines.values():
        spine.set_linewidth(axis_linewidth)

    _add_strips_mm(
        fig,
        axes,
        col_labels=[top_strip_label],
        row_labels=None,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=0.0,
        strip_pad_mm=strip_pad_mm,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_top_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
    )
    if transparent:
        _ensure_strip_background_opaque(fig)

    _draw_global_labels_and_title(
        fig,
        layout,
        x_label=x_label,
        y_label=y_label,
        axis_label_fontsize=axis_label_fontsize,
        title=None,
        single_x_label=True,
        single_y_label=True,
        font_family=font_family,
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=line_palette[level],
            linestyle=line_styles[level],
            linewidth=line_width,
            marker=marker,
            markersize=marker_size,
            markeredgewidth=marker_edge_width,
            label=level,
        )
        for level in line_levels
    ]
    _place_legend(
        fig,
        ax,
        handles,
        line_levels,
        legend_loc=legend_loc,
        legend_fontsize=legend_fontsize,
        legend_ncol=1,
        legend_framealpha=0.0,
        inside_map=not _is_outside_legend_loc(legend_loc),
    )
    if fig.legends:
        fig.legends[0].set_frame_on(legend_frame)
    return fig


def apply_mixed_font_ticklabels(ax, *, axis: str = "both") -> None:
    """Apply mixed-font overlays to tick labels on an axes figure."""
    if ax is None or ax.figure is None:
        return
    if axis not in {"x", "y", "both"}:
        raise ValueError("axis must be one of {'x', 'y', 'both'}.")
    apply_mixed_font_text(ax.figure)


def maybe_apply_mixed_font_text(
    fig: Figure,
    *,
    enabled: Optional[bool] = None,
    latin_font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    greek_font_family: str = GREEK_TEXT_FONT_FAMILY,
) -> Figure:
    """Apply mixed-font rendering when a local or global switch is enabled."""
    if enabled is None:
        enabled = get_greek_symbol_font_enabled()
    return apply_mixed_font_text(
        fig,
        latin_font_family=latin_font_family,
        greek_font_family=greek_font_family,
        enabled=enabled,
    )


def _patch_figure_savefig_once() -> None:
    """Patch Matplotlib Figure.savefig so exports apply mixed-font text."""
    global _ORIGINAL_FIGURE_SAVEFIG
    if _ORIGINAL_FIGURE_SAVEFIG is not None:
        return

    _ORIGINAL_FIGURE_SAVEFIG = Figure.savefig

    def _savefig_with_mixed_font_text(self, *args, **kwargs):
        maybe_apply_mixed_font_text(self)
        return _ORIGINAL_FIGURE_SAVEFIG(self, *args, **kwargs)

    Figure.savefig = _savefig_with_mixed_font_text


mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.width"] = 1
mpl.rcParams["xtick.major.pad"] = 1  # default ~3.5
mpl.rcParams["ytick.major.pad"] = 1
mpl.rcParams["axes.labelpad"] = 1  # default ~4
mpl.rcParams["xtick.major.size"] = 2  # length of major ticks (x-axis)
mpl.rcParams["ytick.major.size"] = 2  # length of major ticks (y-axis)
mpl.rcParams["xtick.minor.size"] = 1
mpl.rcParams["ytick.minor.size"] = 1
mpl.rcParams["xtick.major.width"] = 1  # thickness
mpl.rcParams["ytick.major.width"] = 1


def _configure_plot_fonts(font_family: str = DEFAULT_TEXT_FONT_FAMILY) -> None:
    """Configure sans-serif text and mathtext fonts for figure labels."""
    family = font_family or DEFAULT_TEXT_FONT_FAMILY
    sans_serif = list(
        dict.fromkeys(
            [
                family,
                DEFAULT_TEXT_FONT_FAMILY,
                "Helvetica",
                "DejaVu Sans",
                "Liberation Sans",
            ]
        )
    )

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = sans_serif
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = family
    mpl.rcParams["mathtext.it"] = f"{family}:italic"
    mpl.rcParams["mathtext.bf"] = f"{family}:bold"
    mpl.rcParams["mathtext.sf"] = family
    mpl.rcParams["mathtext.default"] = "regular"
    try:
        mpl.rcParams["mathtext.fallback"] = "stixsans"
    except (KeyError, ValueError):
        pass


# ----------------------- unit helpers -----------------------

MM_PER_INCH = 25.4
DEFAULT_BOXSIZE_MM: Tuple[float, float] = (30.0, 25.0)


def _validate_boxsize(boxsize: Tuple[float, float]) -> Tuple[float, float]:
    """Validate `boxsize` is a (width_mm, height_mm) tuple of positive finite numbers."""
    if boxsize is None:
        raise ValueError("boxsize must be a (width_mm, height_mm) tuple; got None.")
    if not isinstance(boxsize, (tuple, list)) or len(boxsize) != 2:
        raise ValueError(
            f"boxsize must be a length-2 tuple/list (width_mm, height_mm); got {boxsize!r}."
        )
    w = float(boxsize[0])
    h = float(boxsize[1])
    if not np.isfinite(w) or not np.isfinite(h) or w <= 0 or h <= 0:
        raise ValueError(
            f"boxsize entries must be positive, finite numbers; got {boxsize!r}."
        )
    return (w, h)


def _validate_cellsize(cellsize: Tuple[float, float]) -> Tuple[float, float]:
    """Validate `cellsize` is a (cell_width_mm, cell_height_mm) tuple."""
    if cellsize is None:
        raise ValueError(
            "cellsize must be a (cell_width_mm, cell_height_mm) tuple; got None."
        )
    if not isinstance(cellsize, (tuple, list)) or len(cellsize) != 2:
        raise ValueError(
            f"cellsize must be a length-2 tuple/list (cell_width_mm, cell_height_mm); got {cellsize!r}."
        )
    w = float(cellsize[0])
    h = float(cellsize[1])
    if not np.isfinite(w) or not np.isfinite(h) or w <= 0 or h <= 0:
        raise ValueError(
            f"cellsize entries must be positive, finite numbers; got {cellsize!r}."
        )
    return (w, h)


def _mm_to_in(x_mm: float) -> float:
    """Convert millimeters to inches."""
    return float(x_mm) / MM_PER_INCH


def _tuple_mm_to_in(size_mm: Tuple[float, float]) -> Tuple[float, float]:
    """Convert a (w_mm, h_mm) tuple to inches."""
    return _mm_to_in(size_mm[0]), _mm_to_in(size_mm[1])


def _ordered_levels(series, user_levels):
    """Return ordered unique levels, preserving user-specified order if provided."""
    if user_levels is not None:
        return list(user_levels)
    if isinstance(series.dtype, pd.CategoricalDtype):
        return list(series.cat.categories)
    return list(pd.unique(series.dropna()))


def _rgba_color(c):
    """Convert a color specification to an RGBA tuple.

    Supported inputs:
      - Any Matplotlib color (e.g., "gray", (r, g, b), (r, g, b, a))
      - Hex "#RRGGBB"
      - Hex with alpha "#RRGGBBAA" (alpha last; matches Matplotlib/CSS)

    Notes:
      - "#AARRGGBB" (alpha first) is intentionally NOT supported.

    If parsing fails, the original value is returned unchanged.
    """
    if c is None:
        return c

    if isinstance(c, str) and c.startswith("#"):
        s = c.strip()
        try:
            # Matplotlib accepts #RRGGBB and #RRGGBBAA (alpha last).
            return mpl.colors.to_rgba(s)
        except ValueError:
            return c

    try:
        return mpl.colors.to_rgba(c)
    except Exception:
        return c


def p_to_stars(p):
    """Convert a p-value to significance stars."""
    try:
        p = float(p)
    except Exception:
        return "n.s."
    if np.isnan(p):
        return "n.s."
    # if p < 1e-4:
    #     return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


# ----------------------- p-value helpers -----------------------

_P_VALUE_PRIORITY: Tuple[str, ...] = ("p_tukey", "p_across", "p.value", "p_raw")

# Backwards-compatible aliases that show up in R / statsmodels exports.
_P_VALUE_FALLBACK: Tuple[str, ...] = (
    "pvalue",
    "p",
    "p_val",
    "p_value",
    "pval",
    "pval_adj",
    "Pr(>F)",
    "Pr(>|t|)",
    "q",
    "qval",
    "q_value",
)


def _coerce_float_or_nan(x: object) -> float:
    """Best-effort float conversion; return NaN for non-numeric or non-finite values."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _unique_preserve_order(items: List[str]) -> List[str]:
    """Deduplicate while preserving order."""
    seen: set[str] = set()
    out: List[str] = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _candidate_p_value_columns(*, preferred: Optional[str] = None) -> List[str]:
    """Return p-value candidates with p_tukey as the absolute first choice."""
    candidates: List[str] = ["p_tukey"]
    if preferred:
        candidates.append(str(preferred))
    candidates.extend([c for c in _P_VALUE_PRIORITY if c != "p_tukey"])
    candidates.extend(list(_P_VALUE_FALLBACK))
    return _unique_preserve_order([c for c in candidates if c])


def _resolve_p_value_from_row(
    row: pd.Series, *, preferred: Optional[str] = None
) -> float:
    """
    Resolve a single p-value from a row using a strict priority order.

    Priority (highest -> lowest):
        p_tukey
        preferred (if provided)
        p_across -> p.value -> p_raw
        then common legacy aliases (p, pvalue, q, ...)

    Any missing / empty / non-finite values are skipped.
    """
    candidates = _candidate_p_value_columns(preferred=preferred)

    for c in candidates:
        if c in row.index:
            v = _coerce_float_or_nan(row.get(c))
            if np.isfinite(v):
                return float(v)
    return float("nan")


def _resolve_p_value_series(
    df: pd.DataFrame, *, preferred: Optional[str] = None
) -> pd.Series:
    """
    Resolve a per-row p-value Series from a DataFrame using a strict priority order.

    This is the vectorized counterpart of `_resolve_p_value_from_row`.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    candidates = [
        c for c in _candidate_p_value_columns(preferred=preferred) if c in df.columns
    ]

    p = pd.Series(np.nan, index=df.index, dtype=float)
    p_arr = p.to_numpy()

    for c in candidates:
        v = pd.to_numeric(df[c], errors="coerce").astype(float)
        v_arr = v.to_numpy()
        fill_mask = (~np.isfinite(p_arr)) & np.isfinite(v_arr)
        if np.any(fill_mask):
            p_arr[fill_mask] = v_arr[fill_mask]

    # Final sanitize: keep only finite values
    p_arr = np.where(np.isfinite(p_arr), p_arr, np.nan)
    return pd.Series(p_arr, index=df.index, dtype=float)


def _is_ns_label(label: object) -> bool:
    """Return True if a star/label should be treated as non-significant."""
    s = str(label).strip().lower()
    return s in {"", "ns", "n.s.", "n.s", "na", "nan", "none"}


def _stars_sort_rank(label: object) -> int:
    """
    Turn a star label into an integer rank for sorting:
    smaller => more significant.
    """
    if _is_ns_label(label):
        return 10_000
    s = str(label)
    n = s.count("*")
    if n <= 0:
        return 9_999
    return 4 - min(n, 4)


def _build_color_map(levels, palette):
    """
    Build a dict mapping each level -> color.

    palette can be:
    - a matplotlib colormap name
    - a list of colors
    - a dict mapping levels -> colors
    """
    if isinstance(palette, dict):
        return dict(palette)

    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        n = max(1, len(levels))
        cols = [cmap(i / (n - 1 if n > 1 else 1)) for i in range(n)]
        return dict(zip(levels, cols))

    cols = list(palette)
    if len(cols) < len(levels):
        raise ValueError("Palette list is shorter than the number of levels.")
    return dict(zip(levels, cols))


def _series_grid_stats(
    series_list: List[pd.Series], n_grid: int = 400, ribbon: str = "sem"
):
    """
    Aggregate multiple pd.Series on a common x-grid.

    Returns:
        x, mean, low, high  (arrays)
    """
    if len(series_list) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    x_min = np.nanmin([s.index.min() for s in series_list if len(s) > 0])
    x_max = np.nanmax([s.index.max() for s in series_list if len(s) > 0])
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        return np.array([]), np.array([]), np.array([]), np.array([])

    valid = [s for s in series_list if isinstance(s, pd.Series) and not s.empty]
    preserve_native = (
        bool(valid)
        and int(n_grid) == len(valid[0])
        and all(s.index.equals(valid[0].index) for s in valid[1:])
    )
    if preserve_native:
        x_grid = valid[0].index.to_numpy(dtype=float)
        ys = [s.to_numpy(dtype=float) for s in valid]
    else:
        x_grid = np.linspace(x_min, x_max, int(n_grid))
        ys = [
            np.interp(x_grid, s.index.to_numpy(dtype=float), s.to_numpy(dtype=float))
            for s in valid
        ]

    if len(ys) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    Y = np.vstack(ys)
    mean = np.nanmean(Y, axis=0)
    finite_count = np.isfinite(Y).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sd = np.nanstd(Y, axis=0, ddof=1)
    sd = np.where(finite_count > 1, sd, 0.0)
    sem = sd / np.sqrt(np.maximum(1, finite_count))

    if ribbon == "sd":
        low, high = mean - sd, mean + sd
    elif ribbon == "95%ci":
        low, high = mean - 1.96 * sem, mean + 1.96 * sem
    else:  # "sem"
        low, high = mean - sem, mean + sem

    return x_grid, mean, low, high


def _finite(arr):
    """Return finite values from array-like."""
    a = np.asarray(arr)
    return a[np.isfinite(a)]


def _auto_limits(x_arrays, y_arrays, x_pad_frac, y_pad_frac, y_log: bool = False):
    """Compute padded limits from multiple x/y arrays."""
    if x_arrays:
        xs = _finite(
            np.concatenate([np.asarray(v).ravel() for v in x_arrays if v is not None])
        )
    else:
        xs = np.array([0.0, 1.0])

    if y_arrays:
        ys = _finite(
            np.concatenate([np.asarray(v).ravel() for v in y_arrays if v is not None])
        )
    else:
        ys = np.array([0.0, 1.0])

    if xs.size == 0:
        xs = np.array([0.0, 1.0])
    if ys.size == 0:
        ys = np.array([0.0, 1.0])

    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))

    xr = x1 - x0
    yr = y1 - y0
    x_pad = xr * float(x_pad_frac)
    y_pad = yr * float(y_pad_frac)

    if y_log:
        y0 = max(y0, 1e-12)

    return (x0 - x_pad, x1 + x_pad), (y0 - y_pad, y1 + y_pad)


def _safe_axvspan(ax, x0: float, x1: float, **kwargs):
    """axvspan that avoids invalid spans on log scales."""
    if ax.get_xscale() == "log":
        if x1 <= 0:
            return None
        x0 = max(x0, np.finfo(float).tiny)
    return ax.axvspan(x0, x1, **kwargs)


def _safe_axhspan(ax, y0: float, y1: float, **kwargs):
    """axhspan that avoids invalid spans on log scales."""
    if ax.get_yscale() == "log":
        if y1 <= 0:
            return None
        y0 = max(y0, np.finfo(float).tiny)
    return ax.axhspan(y0, y1, **kwargs)


def _to_numeric_index(idx):
    """
    Convert an index/array to a numeric numpy array when possible.

    Priority:
      1) Already-numeric dtype -> float
      2) pd.to_numeric (handles numeric strings) when fully convertible
      3) datetime-like -> seconds since epoch
      4) fallback -> 0..N-1
    """
    if isinstance(idx, pd.Index):
        idx = idx.to_numpy()
    arr = np.asarray(idx)

    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float)

    # Try numeric conversion (e.g., "10", "20.5")
    try:
        num = pd.to_numeric(arr, errors="coerce")
        if np.all(np.isfinite(num)):
            return num.to_numpy(dtype=float)
    except Exception:
        pass

    # Try datetime conversion
    try:
        dt = pd.to_datetime(arr, errors="raise")
        return dt.astype("int64").to_numpy(dtype=float) / 1e9
    except Exception:
        pass

    return np.arange(len(arr), dtype=float)


def _cell_aggregate(
    df_list: List[pd.DataFrame], agg: str = "mean"
) -> Optional[pd.DataFrame]:
    """Aggregate a list of aligned DataFrames into one DataFrame."""
    if len(df_list) == 0:
        return None
    dfs = [d for d in df_list if isinstance(d, pd.DataFrame) and not d.empty]
    if len(dfs) == 0:
        return None
    common_index = dfs[0].index
    common_cols = dfs[0].columns
    for d in dfs[1:]:
        common_index = common_index.intersection(d.index)
        common_cols = common_cols.intersection(d.columns)
    if len(common_index) == 0 or len(common_cols) == 0:
        return None
    stack = np.stack(
        [d.loc[common_index, common_cols].to_numpy(dtype=float) for d in dfs], axis=0
    )
    if agg == "median":
        out = np.nanmedian(stack, axis=0)
    else:
        out = np.nanmean(stack, axis=0)
    return pd.DataFrame(out, index=common_index, columns=common_cols)


def _measure_text_inches(
    text: str,
    fontsize: float = 12.0,
    rotation: float = 0.0,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    dpi: int = 100,
) -> Tuple[float, float]:
    """Measure rendered text size in inches using a tiny offscreen figure."""
    if not text:
        return (0.0, 0.0)

    _configure_plot_fonts(font_family)
    fig = plt.figure(figsize=(2, 2), dpi=dpi)
    fig.patch.set_alpha(0)

    t = fig.text(0, 0, text, fontsize=fontsize, rotation=rotation)
    fig.canvas.draw()
    bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
    plt.close(fig)

    return bbox.width / dpi, bbox.height / dpi


# ----------------------- layout engine (mm, boxsize-driven) -----------------------


def _compute_fig_layout_from_box_mm(
    *,
    nrows: int,
    ncols: int,
    boxsize_mm: Tuple[float, float],
    panel_gap_mm: Tuple[float, float] = (0.0, 0.0),
    strip_top_height_mm: float = 0.0,
    strip_right_width_mm: float = 0.0,
    strip_pad_mm: float = 0.0,
    colorbar_width_mm: float = 0.0,
    colorbar_pad_mm: float = 0.0,
    single_x_label: bool = True,
    single_y_label: bool = True,
    axis_label_fontsize: float = 16.0,
    include_global_label_margins: bool = True,
    x_label_text: str = "",
    y_label_text: str = "",
    colorbar_label_text: Optional[str] = None,
    x_label_offset_mm: float = 0.0,
    y_label_offset_mm: float = 0.0,
    cbar_label_offset_mm: float = 0.0,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    dpi: int = 100,
) -> Dict[str, float | Tuple[float, float] | List[float]]:
    """
    Compute deterministic figure size and grid rectangle from physical mm inputs.

    Returns a layout dict containing:
        - figsize_in: (w_in, h_in)
        - fig_w_in, fig_h_in
        - rect: [L, B, R, T] in figure fractions for the panel grid region
        - box_w_in, box_h_in, gap_x_in, gap_y_in
        - strip_top_h_in, strip_right_w_in, strip_pad_in
        - cb_w_in, cb_pad_in
        - x_off_in, y_off_in, cbar_off_in (label offsets in inches)
    """
    box_w_in, box_h_in = _tuple_mm_to_in(boxsize_mm)
    gap_x_in, gap_y_in = _tuple_mm_to_in(panel_gap_mm)

    strip_top_h_in = _mm_to_in(strip_top_height_mm)
    strip_right_w_in = _mm_to_in(strip_right_width_mm)
    strip_pad_in = _mm_to_in(strip_pad_mm)

    cb_w_in = _mm_to_in(colorbar_width_mm)
    cb_pad_in = _mm_to_in(colorbar_pad_mm)

    x_off_in = _mm_to_in(x_label_offset_mm) if single_x_label else 0.0
    y_off_in = _mm_to_in(y_label_offset_mm) if single_y_label else 0.0
    cbar_off_in = _mm_to_in(cbar_label_offset_mm) if colorbar_label_text else 0.0

    grid_w_in = ncols * box_w_in + (ncols - 1) * gap_x_in
    grid_h_in = nrows * box_h_in + (nrows - 1) * gap_y_in

    top_extras_in = (strip_top_h_in + strip_pad_in) if strip_top_h_in > 0 else 0.0

    right_extras_in = 0.0
    if strip_right_w_in > 0:
        right_extras_in += strip_pad_in + strip_right_w_in
    if cb_w_in > 0:
        right_extras_in += cb_pad_in + cb_w_in

    # Space for global axis labels
    pad_in = 0.03  # ~0.76 mm
    x_label_h_in = 0.0
    y_label_w_in = 0.0

    if include_global_label_margins:
        if single_x_label and x_label_text:
            _, x_label_h_in = _measure_text_inches(
                x_label_text,
                fontsize=axis_label_fontsize,
                font_family=font_family,
                dpi=dpi,
            )
        if single_y_label and y_label_text:
            y_label_w_in, _ = _measure_text_inches(
                y_label_text,
                fontsize=axis_label_fontsize,
                rotation=90,
                font_family=font_family,
                dpi=dpi,
            )

    bottom_margin_in = (
        (x_off_in + x_label_h_in + pad_in)
        if (include_global_label_margins and single_x_label)
        else 0.0
    )
    left_margin_in = (
        (y_off_in + y_label_w_in + pad_in)
        if (include_global_label_margins and single_y_label)
        else 0.0
    )

    # Space for colorbar label (outside to the right)
    if include_global_label_margins and colorbar_label_text:
        cbar_label_w_in, _ = _measure_text_inches(
            colorbar_label_text,
            fontsize=axis_label_fontsize,
            rotation=270,
            font_family=font_family,
            dpi=dpi,
        )
        right_extras_in += cbar_off_in + cbar_label_w_in + pad_in

    fig_w_in = left_margin_in + grid_w_in + right_extras_in
    fig_h_in = bottom_margin_in + grid_h_in + top_extras_in

    if fig_w_in <= 0 or fig_h_in <= 0:
        raise ValueError("Invalid figure size computed from box layout. Check inputs.")

    L = left_margin_in / fig_w_in
    B = bottom_margin_in / fig_h_in
    R = 1.0 - (right_extras_in / fig_w_in)
    T = 1.0 - (top_extras_in / fig_h_in)

    return {
        "figsize_in": (fig_w_in, fig_h_in),
        "fig_w_in": fig_w_in,
        "fig_h_in": fig_h_in,
        "rect": [L, B, R, T],
        "box_w_in": box_w_in,
        "box_h_in": box_h_in,
        "gap_x_in": gap_x_in,
        "gap_y_in": gap_y_in,
        "strip_top_h_in": strip_top_h_in,
        "strip_right_w_in": strip_right_w_in,
        "strip_pad_in": strip_pad_in,
        "cb_w_in": cb_w_in,
        "cb_pad_in": cb_pad_in,
        "x_off_in": x_off_in,
        "y_off_in": y_off_in,
        "cbar_off_in": cbar_off_in,
    }


def _compute_fig_layout_from_panel_sizes_mm(
    *,
    panel_widths_mm: List[float],
    panel_heights_mm: List[float],
    panel_gap_mm: Tuple[float, float] = (0.0, 0.0),
    strip_top_height_mm: float = 0.0,
    strip_right_width_mm: float = 0.0,
    strip_pad_mm: float = 0.0,
    colorbar_width_mm: float = 0.0,
    colorbar_pad_mm: float = 0.0,
    single_x_label: bool = True,
    single_y_label: bool = True,
    axis_label_fontsize: float = 16.0,
    include_global_label_margins: bool = True,
    x_label_text: str = "",
    y_label_text: str = "",
    colorbar_label_text: Optional[str] = None,
    x_label_offset_mm: float = 0.0,
    y_label_offset_mm: float = 0.0,
    cbar_label_offset_mm: float = 0.0,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    dpi: int = 100,
) -> Dict[str, float | Tuple[float, float] | List[float]]:
    """Compute deterministic figure size for a grid with variable panel sizes."""
    if not panel_widths_mm or not panel_heights_mm:
        raise ValueError("panel_widths_mm and panel_heights_mm must be non-empty.")
    if any(width <= 0 for width in panel_widths_mm) or any(
        height <= 0 for height in panel_heights_mm
    ):
        raise ValueError("Panel widths and heights must be positive.")

    panel_widths_in = [_mm_to_in(float(width)) for width in panel_widths_mm]
    panel_heights_in = [_mm_to_in(float(height)) for height in panel_heights_mm]
    gap_x_in, gap_y_in = _tuple_mm_to_in(panel_gap_mm)

    strip_top_h_in = _mm_to_in(strip_top_height_mm)
    strip_right_w_in = _mm_to_in(strip_right_width_mm)
    strip_pad_in = _mm_to_in(strip_pad_mm)

    cb_w_in = _mm_to_in(colorbar_width_mm)
    cb_pad_in = _mm_to_in(colorbar_pad_mm)

    x_off_in = _mm_to_in(x_label_offset_mm) if single_x_label else 0.0
    y_off_in = _mm_to_in(y_label_offset_mm) if single_y_label else 0.0
    cbar_off_in = _mm_to_in(cbar_label_offset_mm) if colorbar_label_text else 0.0

    ncols = len(panel_widths_in)
    nrows = len(panel_heights_in)
    grid_w_in = sum(panel_widths_in) + (ncols - 1) * gap_x_in
    grid_h_in = sum(panel_heights_in) + (nrows - 1) * gap_y_in

    top_extras_in = (strip_top_h_in + strip_pad_in) if strip_top_h_in > 0 else 0.0

    right_extras_in = 0.0
    if strip_right_w_in > 0:
        right_extras_in += strip_pad_in + strip_right_w_in
    if cb_w_in > 0:
        right_extras_in += cb_pad_in + cb_w_in

    pad_in = 0.03
    x_label_h_in = 0.0
    y_label_w_in = 0.0

    if include_global_label_margins:
        if single_x_label and x_label_text:
            _, x_label_h_in = _measure_text_inches(
                x_label_text,
                fontsize=axis_label_fontsize,
                font_family=font_family,
                dpi=dpi,
            )
        if single_y_label and y_label_text:
            y_label_w_in, _ = _measure_text_inches(
                y_label_text,
                fontsize=axis_label_fontsize,
                rotation=90,
                font_family=font_family,
                dpi=dpi,
            )

    bottom_margin_in = (
        (x_off_in + x_label_h_in + pad_in)
        if (include_global_label_margins and single_x_label)
        else 0.0
    )
    left_margin_in = (
        (y_off_in + y_label_w_in + pad_in)
        if (include_global_label_margins and single_y_label)
        else 0.0
    )

    if include_global_label_margins and colorbar_label_text:
        cbar_label_w_in, _ = _measure_text_inches(
            colorbar_label_text,
            fontsize=axis_label_fontsize,
            rotation=270,
            font_family=font_family,
            dpi=dpi,
        )
        right_extras_in += cbar_off_in + cbar_label_w_in + pad_in

    fig_w_in = left_margin_in + grid_w_in + right_extras_in
    fig_h_in = bottom_margin_in + grid_h_in + top_extras_in

    if fig_w_in <= 0 or fig_h_in <= 0:
        raise ValueError(
            "Invalid figure size computed from variable panel layout. Check inputs."
        )

    L = left_margin_in / fig_w_in
    B = bottom_margin_in / fig_h_in
    R = 1.0 - (right_extras_in / fig_w_in)
    T = 1.0 - (top_extras_in / fig_h_in)

    return {
        "figsize_in": (fig_w_in, fig_h_in),
        "fig_w_in": fig_w_in,
        "fig_h_in": fig_h_in,
        "rect": [L, B, R, T],
        "panel_widths_in": panel_widths_in,
        "panel_heights_in": panel_heights_in,
        "gap_x_in": gap_x_in,
        "gap_y_in": gap_y_in,
        "strip_top_h_in": strip_top_h_in,
        "strip_right_w_in": strip_right_w_in,
        "strip_pad_in": strip_pad_in,
        "cb_w_in": cb_w_in,
        "cb_pad_in": cb_pad_in,
        "x_off_in": x_off_in,
        "y_off_in": y_off_in,
        "cbar_off_in": cbar_off_in,
    }


def _place_panels_fixed(
    fig,
    axes,
    rect: List[float],
    box_w_in: float,
    box_h_in: float,
    gap_x_in: float,
    gap_y_in: float,
    align: str = "left",
):
    """
    Place a (nrows, ncols) axes grid with fixed per-panel sizes (inches) inside rect.

    rect is in figure fractions: [L, B, R, T] where R and T are absolute figure fractions.
    """
    L, B, R, T = rect
    fig_w_in, fig_h_in = fig.get_size_inches()

    total_w_frac = R - L
    avail_w_in = total_w_frac * fig_w_in
    nrows, ncols = axes.shape
    need_w_in = ncols * box_w_in + (ncols - 1) * gap_x_in

    if align == "center":
        x0_in = (avail_w_in - need_w_in) / 2.0
    elif align == "right":
        x0_in = avail_w_in - need_w_in
    else:
        x0_in = 0.0

    y0_in = 0.0

    base_x0 = L + (x0_in / fig_w_in)
    base_y0 = B + (y0_in / fig_h_in)

    box_w_frac = box_w_in / fig_w_in
    box_h_frac = box_h_in / fig_h_in
    gap_x_frac = gap_x_in / fig_w_in
    gap_y_frac = gap_y_in / fig_h_in

    for i in range(nrows):
        for j in range(ncols):
            x0 = base_x0 + j * (box_w_frac + gap_x_frac)
            y0 = base_y0 + (nrows - 1 - i) * (box_h_frac + gap_y_frac)
            axes[i, j].set_position([x0, y0, box_w_frac, box_h_frac])


def _place_panels_variable(
    fig,
    axes,
    rect: List[float],
    panel_widths_in: List[float],
    panel_heights_in: List[float],
    gap_x_in: float,
    gap_y_in: float,
    align: str = "left",
):
    """Place a grid whose column widths and row heights are fixed in inches."""
    L, B, R, T = rect
    fig_w_in, fig_h_in = fig.get_size_inches()

    total_w_frac = R - L
    avail_w_in = total_w_frac * fig_w_in
    nrows, ncols = axes.shape
    need_w_in = sum(panel_widths_in) + (ncols - 1) * gap_x_in

    if align == "center":
        x0_in = (avail_w_in - need_w_in) / 2.0
    elif align == "right":
        x0_in = avail_w_in - need_w_in
    else:
        x0_in = 0.0

    y0_in = 0.0
    base_x0 = L + (x0_in / fig_w_in)
    base_y0 = B + (y0_in / fig_h_in)

    gap_x_frac = gap_x_in / fig_w_in
    gap_y_frac = gap_y_in / fig_h_in
    width_fracs = [width / fig_w_in for width in panel_widths_in]
    height_fracs = [height / fig_h_in for height in panel_heights_in]

    x_offsets = [0.0]
    for width in width_fracs[:-1]:
        x_offsets.append(x_offsets[-1] + width + gap_x_frac)

    y_offsets = []
    for row_index in range(nrows):
        rows_below_height = sum(height_fracs[row_index + 1 :])
        gaps_below = (nrows - 1 - row_index) * gap_y_frac
        y_offsets.append(rows_below_height + gaps_below)

    for i in range(nrows):
        for j in range(ncols):
            x0 = base_x0 + x_offsets[j]
            y0 = base_y0 + y_offsets[i]
            axes[i, j].set_position([x0, y0, width_fracs[j], height_fracs[i]])


def _add_strips_mm(
    fig,
    axes,
    *,
    col_labels: Optional[List[str]] = None,
    row_labels: Optional[List[str]] = None,
    strip_top_height_mm: float = 0.0,
    strip_right_width_mm: float = 0.0,
    strip_pad_mm: float = 0.0,
    label_fontsize: float = 16.0,
    label_top_bg_color: str = "lightgray",
    label_right_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
):
    """
    Add facet strips using absolute mm units.

    Returns:
        right_edge (figure fraction): right-most edge after right strips (for colorbar placement).
    """
    nrows, ncols = axes.shape
    fig_w_in, fig_h_in = fig.get_size_inches()

    top_h_in = _mm_to_in(strip_top_height_mm)
    right_w_in = _mm_to_in(strip_right_width_mm)
    pad_in = _mm_to_in(strip_pad_mm)

    top_h_frac = top_h_in / fig_h_in if top_h_in > 0 else 0.0
    right_w_frac = right_w_in / fig_w_in if right_w_in > 0 else 0.0
    pad_y_frac = pad_in / fig_h_in if pad_in > 0 else 0.0
    pad_x_frac = pad_in / fig_w_in if pad_in > 0 else 0.0

    # Top strips
    if col_labels is not None and top_h_frac > 0:
        for j, lab in enumerate(col_labels):
            pos = axes[0, j].get_position()
            ax_strip = fig.add_axes(
                [pos.x0, pos.y1 + pad_y_frac, pos.width, top_h_frac]
            )
            ax_strip._visualdf_strip_axis = True
            ax_strip.set_facecolor(label_top_bg_color)
            ax_strip.text(
                0.5,
                0.5,
                str(lab),
                ha="center",
                va="center",
                fontsize=label_fontsize,
                color=label_text_color,
                fontweight=label_fontweight,
            )
            ax_strip.set_xticks([])
            ax_strip.set_yticks([])
            for spine in ax_strip.spines.values():
                spine.set_visible(False)

    # Right strips
    right_edge = 0.0
    if row_labels is not None and right_w_frac > 0:
        for i, lab in enumerate(row_labels):
            pos = axes[i, ncols - 1].get_position()
            ax_strip = fig.add_axes(
                [pos.x1 + pad_x_frac, pos.y0, right_w_frac, pos.height]
            )
            ax_strip._visualdf_strip_axis = True
            ax_strip.set_facecolor(label_right_bg_color)
            ax_strip.text(
                0.5,
                0.5,
                str(lab),
                ha="center",
                va="center",
                rotation=-90,
                fontsize=label_fontsize,
                color=label_text_color,
                fontweight=label_fontweight,
            )
            ax_strip.set_xticks([])
            ax_strip.set_yticks([])
            for spine in ax_strip.spines.values():
                spine.set_visible(False)
            right_edge = max(right_edge, pos.x1 + pad_x_frac + right_w_frac)
    else:
        # Right edge defaults to the rightmost panel edge
        right_edge = max(ax.get_position().x1 for ax in axes.ravel())

    return right_edge


def _init_box_figure(
    *,
    nrows: int,
    ncols: int,
    boxsize_mm: Tuple[float, float],
    panel_gap_mm: Tuple[float, float] = (0.0, 0.0),
    strip_top_height_mm: float = 0.0,
    strip_right_width_mm: float = 0.0,
    strip_pad_mm: float = 0.0,
    colorbar_width_mm: float = 0.0,
    colorbar_pad_mm: float = 0.0,
    single_x_label: bool = True,
    single_y_label: bool = True,
    axis_label_fontsize: float = 16.0,
    include_global_label_margins: bool = True,
    x_label_text: str = "",
    y_label_text: str = "",
    colorbar_label_text: Optional[str] = None,
    x_label_offset_mm: float = 0.0,
    y_label_offset_mm: float = 0.0,
    cbar_label_offset_mm: float = 0.0,
    dpi: int = 100,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    sharex: bool = False,
    sharey: bool = False,
    transparent: bool = False,
):
    """Create a fixed-geometry figure and axes grid from mm layout parameters."""
    _configure_plot_fonts(font_family)

    layout = _compute_fig_layout_from_box_mm(
        nrows=nrows,
        ncols=ncols,
        boxsize_mm=boxsize_mm,
        panel_gap_mm=panel_gap_mm,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=strip_right_width_mm,
        strip_pad_mm=strip_pad_mm,
        colorbar_width_mm=colorbar_width_mm,
        colorbar_pad_mm=colorbar_pad_mm,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        axis_label_fontsize=axis_label_fontsize,
        include_global_label_margins=include_global_label_margins,
        x_label_text=x_label_text,
        y_label_text=y_label_text,
        colorbar_label_text=colorbar_label_text,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        cbar_label_offset_mm=cbar_label_offset_mm,
        font_family=font_family,
        dpi=dpi,
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=layout["figsize_in"],  # inches
        dpi=dpi,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )

    if transparent:
        fig.patch.set_alpha(0)
        for ax in axes.ravel():
            ax.patch.set_alpha(0)

    _place_panels_fixed(
        fig,
        axes,
        rect=layout["rect"],
        box_w_in=layout["box_w_in"],
        box_h_in=layout["box_h_in"],
        gap_x_in=layout["gap_x_in"],
        gap_y_in=layout["gap_y_in"],
        align="left",
    )

    return fig, axes, layout


def _init_variable_box_figure(
    *,
    panel_widths_mm: List[float],
    panel_heights_mm: List[float],
    panel_gap_mm: Tuple[float, float] = (0.0, 0.0),
    strip_top_height_mm: float = 0.0,
    strip_right_width_mm: float = 0.0,
    strip_pad_mm: float = 0.0,
    colorbar_width_mm: float = 0.0,
    colorbar_pad_mm: float = 0.0,
    single_x_label: bool = True,
    single_y_label: bool = True,
    axis_label_fontsize: float = 16.0,
    include_global_label_margins: bool = True,
    x_label_text: str = "",
    y_label_text: str = "",
    colorbar_label_text: Optional[str] = None,
    x_label_offset_mm: float = 0.0,
    y_label_offset_mm: float = 0.0,
    cbar_label_offset_mm: float = 0.0,
    dpi: int = 100,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    sharex: bool = False,
    sharey: bool = False,
    transparent: bool = False,
):
    """Create a fixed-cell figure whose panel sizes vary by row and column."""
    _configure_plot_fonts(font_family)

    layout = _compute_fig_layout_from_panel_sizes_mm(
        panel_widths_mm=panel_widths_mm,
        panel_heights_mm=panel_heights_mm,
        panel_gap_mm=panel_gap_mm,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=strip_right_width_mm,
        strip_pad_mm=strip_pad_mm,
        colorbar_width_mm=colorbar_width_mm,
        colorbar_pad_mm=colorbar_pad_mm,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        axis_label_fontsize=axis_label_fontsize,
        include_global_label_margins=include_global_label_margins,
        x_label_text=x_label_text,
        y_label_text=y_label_text,
        colorbar_label_text=colorbar_label_text,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        cbar_label_offset_mm=cbar_label_offset_mm,
        font_family=font_family,
        dpi=dpi,
    )

    nrows = len(panel_heights_mm)
    ncols = len(panel_widths_mm)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=layout["figsize_in"],
        dpi=dpi,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )

    if transparent:
        fig.patch.set_alpha(0)
        for ax in axes.ravel():
            ax.patch.set_alpha(0)

    _place_panels_variable(
        fig,
        axes,
        rect=layout["rect"],
        panel_widths_in=layout["panel_widths_in"],
        panel_heights_in=layout["panel_heights_in"],
        gap_x_in=layout["gap_x_in"],
        gap_y_in=layout["gap_y_in"],
        align="left",
    )

    return fig, axes, layout


def _draw_global_labels_and_title(
    fig,
    layout,
    *,
    x_label: Optional[str],
    y_label: Optional[str],
    axis_label_fontsize: float,
    title: Optional[str],
    title_fontsize: float = 18.0,
    single_x_label: bool = True,
    single_y_label: bool = True,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
):
    """Draw global X/Y labels (using mm offsets) and a title."""
    _configure_plot_fonts(font_family)
    L, B, R, T = layout["rect"]
    fig_w_in = float(layout["fig_w_in"])
    fig_h_in = float(layout["fig_h_in"])

    cx = L + (R - L) / 2.0
    cy = B + (T - B) / 2.0

    if single_x_label and x_label:
        x_off_frac = float(layout["x_off_in"]) / fig_h_in
        fig.text(
            cx,
            B - x_off_frac,
            x_label,
            ha="center",
            va="top",
            fontsize=axis_label_fontsize,
        )

    if single_y_label and y_label:
        y_off_frac = float(layout["y_off_in"]) / fig_w_in
        fig.text(
            L - y_off_frac,
            cy,
            y_label,
            ha="right",
            va="center",
            rotation=90,
            multialignment="center",
            fontsize=axis_label_fontsize,
        )

    if title:
        fig.suptitle(title, fontsize=title_fontsize, y=min(0.995, T + 0.01))


def _add_global_colorbar(
    fig,
    layout,
    mappable,
    *,
    right_edge: float,
    colorbar_label: Optional[str],
    tick_label_fontsize: float,
    axis_label_fontsize: float,
    colorbar_length_mm: Optional[float] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
):
    """Add a global colorbar using mm-based width/padding and label offset."""
    _configure_plot_fonts(font_family)
    L, B, R, T = layout["rect"]
    fig_w_in = float(layout["fig_w_in"])

    cb_pad_in = float(layout["cb_pad_in"])
    cb_w_in = float(layout["cb_w_in"])
    if cb_w_in <= 0:
        return None

    cb_pad_frac = cb_pad_in / fig_w_in
    cb_w_frac = cb_w_in / fig_w_in

    x0 = right_edge + cb_pad_frac
    # Keep within figure bounds
    if x0 + cb_w_frac > 0.99:
        x0 = max(0.0, 0.99 - cb_w_frac)

    cbar_y = B
    cbar_h = T - B
    if colorbar_length_mm is not None:
        length_mm = float(colorbar_length_mm)
        if not np.isfinite(length_mm) or length_mm <= 0:
            raise ValueError(
                f"colorbar_length_mm must be a positive finite number; got {colorbar_length_mm!r}."
            )
        fig_h_in = float(layout["fig_h_in"])
        length_frac = _mm_to_in(length_mm) / fig_h_in
        cbar_h = min(length_frac, T - B)
        cbar_y = B + ((T - B) - cbar_h) / 2.0

    cax = fig.add_axes([x0, cbar_y, cb_w_frac, cbar_h])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(labelsize=tick_label_fontsize)

    if colorbar_label:
        cbar_off_frac = float(layout["cbar_off_in"]) / fig_w_in
        cx_right = x0 + cb_w_frac
        fig.text(
            cx_right + cbar_off_frac,
            cbar_y + cbar_h / 2.0,
            colorbar_label,
            ha="left",
            va="center",
            rotation=90,
            fontsize=axis_label_fontsize,
        )

    return cbar


_OUTSIDE_LEGEND_SPECS: Dict[str, Tuple[str, Tuple[float, float]]] = {
    # New (preferred) API, matching visualdf_old.
    "outside_top": ("upper center", (0.5, 1.02)),
    "outside_bottom": ("lower center", (0.5, -0.04)),
    "outside_right": ("center left", (1.02, 0.5)),
    "outside_left": ("center right", (-0.02, 0.5)),
    # Legacy shorthands kept for backward compatibility.
    "right": ("right", (1.02, 0.5)),
    # "left" is not a valid Matplotlib loc string; interpret it as outside_left.
    "left": ("center right", (-0.02, 0.5)),
    "upper right": ("upper right", (1.02, 1.0)),
    "lower right": ("lower right", (1.02, 0.0)),
    "upper left": ("upper left", (-0.02, 1.0)),
    "lower left": ("lower left", (-0.02, 0.0)),
}


def _legend_outside_spec(loc: str) -> Optional[Tuple[str, Tuple[float, float]]]:
    """Return (legend_loc, bbox_to_anchor) for supported outside legend placements."""
    if loc is None:
        return None
    key = str(loc).strip().lower()
    return _OUTSIDE_LEGEND_SPECS.get(key)


def _is_outside_legend_loc(loc: str) -> bool:
    """True if loc is one of the supported outside legend placements."""
    return _legend_outside_spec(loc) is not None


def _place_legend(
    fig,
    ax,
    handles,
    labels,
    *,
    legend_loc: str,
    legend_fontsize: float,
    legend_ncol: int,
    legend_framealpha: float,
    inside_map: bool,
):
    """Standardized legend placement (inside or outside an anchor axis)."""
    if inside_map:
        ax.legend(
            handles,
            labels,
            loc=legend_loc,
            fontsize=legend_fontsize,
            framealpha=legend_framealpha,
            ncol=legend_ncol,
        )
        return

    spec = _legend_outside_spec(legend_loc)
    if spec is None:
        out_loc, anchor = (str(legend_loc), (1.02, 0.5))
    else:
        out_loc, anchor = spec

    fig.legend(
        handles,
        labels,
        loc=out_loc,
        bbox_to_anchor=anchor,
        fontsize=legend_fontsize,
        framealpha=legend_framealpha,
        ncol=legend_ncol,
    )


def _ensure_strip_background_opaque(fig):
    """
    If figure background is transparent, ensure strip axes are fully opaque
    (matplotlib sometimes inherits alpha unexpectedly).
    """
    if fig.get_facecolor()[-1] < 1.0:
        for ax in fig.axes:
            if (
                getattr(ax, "_visualdf_strip_axis", False)
                and ax.get_facecolor()[-1] < 1.0
            ):
                ax.set_facecolor((*ax.get_facecolor()[:3], 1.0))


def _parse_whiskers(w):
    if isinstance(w, (list, tuple)) and len(w) == 2:
        return (float(w[0]), float(w[1]))
    if isinstance(w, (int, float)):
        return float(w)
    if not isinstance(w, str):
        return 1.5
    s = w.strip().lower()
    if s in ("tukey", "iqr", "iqr1.5"):
        return 1.5
    if s == "minmax":
        return (0, 100)
    if s.startswith("iqr"):
        try:
            return float(s.replace("iqr", ""))
        except Exception:
            return 1.5
    if "," in s:
        try:
            lo, hi = s.split(",", 1)
            return (float(lo), float(hi))
        except Exception:
            return 1.5
    return 1.5


# bracket drawer (uses absolute y based on range fractions)
def _draw_tukey_brackets(
    ax,
    tuk_cell: pd.DataFrame,
    x_to_num: Dict,
    *,
    y_limits_local: Optional[Tuple[float, float]],
    hide_ns: bool,
    y_start: float,
    y_end: float,
    y_step: Optional[float],  # set to None to auto-compute
    bracket_height_frac: float,
    color: str,
    lw: float,
    text_size: int,
    label_pad_frac: float = 0.015,
    p_value_column: Optional[str] = None,
    star_thresholds: Optional[List[Tuple[float, str]]] = None,
    text_offset_fraction: float = -0.035,
):
    """
    Draw Tukey-style brackets with a minimal number of vertical layers (no horizontal overlap).

    Notes
    -----
    - Overlap is inclusive on endpoints (touching counts as overlap).
    - If y_step is None, the function packs all layers uniformly between y_start and y_end.
    - Coordinates y_start/y_end/bracket_height_frac/label_pad_frac are FRACTIONS of the axis y-range.
    - p-values are resolved with the following priority:
        p_tukey -> p_across -> p.value -> p_raw
      If a candidate is missing/empty/non-finite, the next one is tried.

    Parameters
    ----------
    ax : matplotlib Axes
    tuk_cell : DataFrame with at least columns {group1, group2} and any p-value column.
              'contrast' like 'A - B' is also accepted.
              If a 'stars' column exists, it is used directly.
    x_to_num : dict mapping each x-level (str) -> numeric x position on the axis
    y_limits_local : (ymin, ymax) or None to use ax.get_ylim()
    hide_ns : bool, if True drop non-significant labels ('ns', 'n.s.', empty)
    y_start, y_end : float in [0,1], bottom/top of the bracket zone, as fractions of y-range
    y_step : float or None. If None, computed automatically to fit the minimal number of layers
    bracket_height_frac : float of y-range, vertical size of the bracket (leg height)
    color, lw, text_size : styling
    label_pad_frac : extra vertical pad (in y-range fraction) between bracket and its text
    """
    if tuk_cell is None or tuk_cell.empty:
        return

    # --- normalize Tukey columns ---
    tk = tuk_cell.copy()

    # group1/group2 from contrast if needed
    if "group1" not in tk.columns or "group2" not in tk.columns:
        if "contrast" in tk.columns:
            parts = tk["contrast"].astype(str).str.split("-", n=1, expand=True)
            if parts.shape[1] == 2:
                tk["group1"] = parts[0].str.strip()
                tk["group2"] = parts[1].str.strip()

    if not {"group1", "group2"}.issubset(tk.columns):
        return

    if p_value_column is None:
        # Preserve the historical fallback behavior for existing callers.
        tk["p.value"] = _resolve_p_value_series(tk)
    else:
        if p_value_column not in tk.columns:
            raise KeyError(f"Tukey results do not contain {p_value_column!r}.")
        tk["p.value"] = pd.to_numeric(tk[p_value_column], errors="coerce")

    if star_thresholds is not None:

        def _explicit_stars(value):
            if not np.isfinite(value):
                return "ns"
            for threshold, label in star_thresholds:
                if value < threshold:
                    return label
            return "ns"

        tk["stars"] = tk["p.value"].apply(_explicit_stars)
    else:
        # Preserve explicit legacy labels and fill only missing values.
        if "stars" not in tk.columns:
            tk["stars"] = np.nan
        missing_star = tk["stars"].isna() | (tk["stars"].astype(str).str.strip() == "")
        if bool(missing_star.any()):
            tk["stars"] = tk["stars"].astype("object")
            tk.loc[missing_star, "stars"] = tk.loc[missing_star, "p.value"].apply(
                p_to_stars
            )
    # Optional hide ns
    if hide_ns:
        tk = tk[~tk["stars"].apply(_is_ns_label)]
    if tk.empty:
        return

    # Deduplicate symmetric pairs (A-B vs B-A): keep the most significant (smallest p);
    # if p is unavailable, fall back to star count.
    tk["group1"] = tk["group1"].astype(str).str.strip()
    tk["group2"] = tk["group2"].astype(str).str.strip()

    tk[["g_lo", "g_hi"]] = tk.apply(
        lambda r: pd.Series(sorted([str(r["group1"]), str(r["group2"])])),
        axis=1,
    )

    p_rank = pd.to_numeric(tk["p.value"], errors="coerce").astype(float)
    p_rank = p_rank.where(np.isfinite(p_rank), np.inf)
    tk["_p_rank"] = p_rank
    tk["_star_rank"] = tk["stars"].apply(_stars_sort_rank)

    tk = tk.sort_values(["_p_rank", "_star_rank"], kind="mergesort").drop_duplicates(
        subset=["g_lo", "g_hi"], keep="first"
    )

    # Build interval list using numeric x positions; drop pairs not on axis
    intervals = []
    for _, r in tk.iterrows():
        g1, g2 = str(r["g_lo"]), str(r["g_hi"])
        if g1 not in x_to_num or g2 not in x_to_num:
            continue
        x1, x2 = float(x_to_num[g1]), float(x_to_num[g2])
        if x1 == x2:
            continue
        lo, hi = (x1, x2) if x1 < x2 else (x2, x1)
        intervals.append(
            {
                "lo": lo,
                "hi": hi,
                "label": str(r["stars"]),
                "group1": g1,
                "group2": g2,
            }
        )
    if not intervals:
        return

    # --- Interval partitioning (touching counts as overlap) ---
    intervals.sort(key=lambda d: (d["lo"], d["hi"]))

    # Assign layers greedily with a min-heap of (end, layer_id)
    heap: List[Tuple[float, int]] = []  # (current_end, layer_id)
    next_layer_id = 0
    for d in intervals:
        lo, hi = d["lo"], d["hi"]
        # Reuse only if current_end < lo  (strict, because touching = overlap)
        if heap and heap[0][0] < lo:
            _, lid = heapq.heappop(heap)
        else:
            lid = next_layer_id
            next_layer_id += 1
        d["layer"] = lid
        heapq.heappush(heap, (hi, lid))
    n_layers = max(d["layer"] for d in intervals) + 1

    # --- y geometry ---
    if y_limits_local is None:
        y0, y1 = ax.get_ylim()
    else:
        y0, y1 = y_limits_local
    y_range = float(y1 - y0)
    if y_range <= 0:
        return

    # Bracket “tick” (vertical leg height) and text pad in data units
    tick = y_range * float(bracket_height_frac)
    text_pad = y_range * float(label_pad_frac)

    # Compute the step between layers
    top = y0 + y_range * float(y_end)
    base = y0 + y_range * float(y_start)
    usable = max(0.0, top - base)

    if y_step is None:
        # Pack all layers evenly in the available zone
        if n_layers <= 1:
            step = 0.0
        else:
            step = usable / (n_layers - 1)
        # Ensure some minimum vertical separation so text doesn't sit on the bracket below
        min_step = max(0.0, tick + text_pad * 0.8)
        if n_layers > 1 and step < min_step:
            step = min_step
    else:
        step = y_range * abs(float(y_step))

    # --- Draw brackets, top layer first (more readable) ---
    intervals.sort(key=lambda d: d["layer"])  # small layer id first (top)

    for d in intervals:
        lo, hi, lab, layer = d["lo"], d["hi"], d["label"], d["layer"]
        y = top - layer * step
        # Clamp if we're out of room
        if y < base:
            y = base

        ax.plot(
            [lo, lo, hi, hi],
            [y, y + tick, y + tick, y],
            color=color,
            lw=lw,
            zorder=5,
            clip_on=False,
        )
        ax.text(
            (lo + hi) / 2.0,
            y + tick + (text_offset_fraction * y_range),
            lab,
            ha="center",
            va="bottom",
            fontsize=text_size,
            color=color,
            zorder=6,
            clip_on=False,
        )


# --------------------- main functions ---------------------
# %%


# --------------------- scalar plotting ---------------------


def _normalize_emm_ci_columns(emw: pd.DataFrame) -> pd.DataFrame:
    """Normalize common CI column name variants to lower.CL / upper.CL."""
    if emw.empty:
        return emw
    if "lower.CL" in emw.columns and "upper.CL" in emw.columns:
        return emw
    alt = {
        "lower": "lower.CL",
        "upper": "upper.CL",
        "LCL": "lower.CL",
        "UCL": "upper.CL",
        "asymp.LCL": "lower.CL",
        "asymp.UCL": "upper.CL",
    }
    out = emw.copy()
    for old, new in alt.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})
    return out


def _auto_y_limits_scalar(
    dfw: pd.DataFrame,
    emw: pd.DataFrame,
    *,
    value_col: str,
    y_limits: Optional[Tuple[float, float]],
    lower_padding_fraction: float = 0.10,
    upper_padding_fraction: float = 0.30,
    reference_values: Optional[Union[List, np.ndarray]] = None,
) -> Tuple[float, float]:
    """Compute reasonable y-limits for scalar plots based on raw + EMM CI."""
    if y_limits is not None:
        return y_limits

    raw = (
        dfw[value_col].to_numpy(dtype=float)
        if value_col in dfw.columns
        else np.array([0.0, 1.0])
    )
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        raw = np.array([0.0, 1.0])

    ymin = float(np.min(raw))
    ymax = float(np.max(raw))

    if "lower.CL" in emw.columns:
        v = emw["lower.CL"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            ymin = min(ymin, float(np.min(v)))
    if "upper.CL" in emw.columns:
        v = emw["upper.CL"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            ymax = max(ymax, float(np.max(v)))

    if reference_values is not None:
        references = np.asarray(reference_values, dtype=float)
        references = references[np.isfinite(references)]
        if references.size:
            ymin = min(ymin, float(np.min(references)))
            ymax = max(ymax, float(np.max(references)))

    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
        return (0.0, 1.0)

    yr = ymax - ymin
    return (
        ymin - lower_padding_fraction * yr,
        ymax + upper_padding_fraction * yr,
    )


def _draw_emmean_vs_null_stars(
    ax,
    null_tests: pd.DataFrame,
    raw: pd.DataFrame,
    emm: pd.DataFrame,
    x_to_num: Dict,
    *,
    x_var: str,
    value_col: str,
    p_value_column: str,
    y_limits_local: Tuple[float, float],
    hide_ns: bool,
    star_thresholds: List[Tuple[float, str]],
    text_size: int,
    color: str,
    offset_fraction: float,
) -> None:
    """Draw one significance label above each EMM tested against a fixed null."""

    if null_tests is None or null_tests.empty:
        return
    required = {x_var, p_value_column}
    missing = sorted(required.difference(null_tests.columns))
    if missing:
        raise KeyError(f"Null-test results are missing columns: {missing}")
    y0, y1 = y_limits_local
    y_range = float(y1 - y0)
    if y_range <= 0:
        return

    for level, x_position in x_to_num.items():
        selected = null_tests.loc[null_tests[x_var].astype(str).eq(level)]
        if len(selected) != 1:
            raise ValueError(
                f"Expected one null-test row for {x_var}={level}; found {len(selected)}."
            )
        p_value = pd.to_numeric(
            pd.Series([selected.iloc[0][p_value_column]]), errors="coerce"
        ).iloc[0]
        label = "ns"
        if np.isfinite(p_value):
            for threshold, candidate in star_thresholds:
                if p_value < threshold:
                    label = candidate
                    break
        if hide_ns and _is_ns_label(label):
            continue

        candidates: List[float] = []
        raw_values = raw.loc[raw[x_var].astype(str).eq(level), value_col].to_numpy(
            dtype=float
        )
        candidates.extend(raw_values[np.isfinite(raw_values)].tolist())
        selected_emm = emm.loc[emm[x_var].astype(str).eq(level)]
        for column in ("upper.CL", "emmean"):
            if column in selected_emm.columns:
                values = selected_emm[column].to_numpy(dtype=float)
                candidates.extend(values[np.isfinite(values)].tolist())
        if not candidates:
            continue
        ax.text(
            float(x_position),
            max(candidates) + float(offset_fraction) * y_range,
            label,
            ha="center",
            va="bottom",
            fontsize=text_size,
            color=color,
            zorder=6,
            clip_on=False,
        )


def _plot_interaction_scalar_grid(
    *,
    df: pd.DataFrame,
    emm: pd.DataFrame,
    tuk: pd.DataFrame,
    value_col: str,
    x_var: str,
    col_var: Optional[str],
    row_var: Optional[str],
    col_levels: Optional[List],
    row_levels: Optional[List],
    jitter_var: Optional[str],
    fill_var: Optional[str],
    outline_var: Optional[str],
    x_levels: Optional[List],
    # labels/scales
    x_label: Optional[str],
    single_x_label: bool,
    y_label: Optional[str],
    single_y_label: bool,
    x_limits: Optional[Tuple[float, float]],
    y_limits: Optional[Tuple[float, float]],
    x_log: bool,
    y_log: bool,
    xtick_rotation: float,
    ytick_rotation: float,
    # style
    title: Optional[str],
    font_family: str,
    jitter_palette: Union[str, List, Dict],
    fill_palette: Union[str, List, Dict],
    outline_palette: Union[str, List, Dict, None],
    jitter_width: float,
    jitter_alpha: float,
    jitter_size: float,
    grid: bool,
    grid_alpha: float,
    dpi: int,
    seed: int,
    show_top_right_axes: bool,
    # box
    show_box: bool,
    box_width: float,
    fill_alpha: float,
    whiskers: Union[str, float, Tuple[float, float]],
    box_edge_color: str,
    box_edge_width: float,
    median_color: str,
    median_linewidth: float,
    whisker_color: str,
    whisker_linewidth: float,
    cap_linewidth: float,
    outlier_marker: str,
    outlier_markersize: float,
    # overlays
    show_emm_line: bool,
    show_emm_ci: bool,
    show_raw_mean_line: bool,
    emm_line_width: float,
    emm_line_color: str,
    emm_line_style: str,
    raw_mean_line_width: float,
    raw_mean_line_color: str,
    raw_mean_line_style: str,
    error_bar_linewidth: float,
    error_bar_cap: float,
    error_bar_color: str,
    # refs
    vertical_lines: Optional[Union[List, np.ndarray]],
    vline_color: str,
    vline_style: str,
    vline_width: float,
    vline_alpha: float,
    horizontal_lines: Optional[Union[List, np.ndarray]],
    hline_color: str,
    hline_style: str,
    hline_width: float,
    hline_alpha: float,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]],
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]],
    # strips
    label_fontsize: int,
    label_top_bg_color: str,
    label_right_bg_color: str,
    label_text_color: str,
    label_fontweight: str,
    strip_top_height_mm: float,
    strip_right_width_mm: float,
    strip_pad_mm: float,
    # text sizes
    title_fontsize: int,
    axis_label_fontsize: int,
    tick_label_fontsize: int,
    # legend
    legend_loc: str,
    legend_ncol: Optional[int],
    legend_framealpha: float,
    legend_fontsize: int,
    # geometry (mm)
    boxsize: Tuple[float, float],
    panel_gap: Tuple[float, float],
    include_global_label_margins: bool,
    x_label_offset_mm: float,
    y_label_offset_mm: float,
    # brackets
    show_brackets: bool,
    hide_ns: bool,
    y_start: float,
    y_end: float,
    y_step: Optional[float],
    bracket_height_frac: float,
    bracket_color: str,
    bracket_linewidth: float,
    bracket_text_size: int,
    # transparent
    transparent: bool,
    # optional study-specific presentation controls
    top_strip_labels: Optional[List[str]] = None,
    right_strip_labels: Optional[List[str]] = None,
    sample_size_id_col: Optional[str] = None,
    sample_size_template: str = "{n_ID}",
    sample_size_color: str = "#2C2C2C",
    sample_size_fontsize: int = 6,
    sample_size_y_axes: float = 0.01,
    sample_size_vertical_alignment: str = "bottom",
    axis_linewidth: Optional[float] = None,
    tick_linewidth: Optional[float] = None,
    jitter_linewidth: float = 0.0,
    p_value_column: Optional[str] = None,
    star_thresholds: Optional[List[Tuple[float, str]]] = None,
    bracket_text_offset_fraction: float = -0.035,
    lower_y_padding_fraction: float = 0.10,
    upper_y_padding_fraction: float = 0.30,
    error_bar_marker: str = "o",
    error_bar_marker_size: float = 0.0,
    error_bar_marker_color: Optional[str] = None,
    error_bar_marker_edge_width: float = 0.0,
    null_tests: Optional[pd.DataFrame] = None,
    show_null_stars: bool = False,
    null_p_value_column: Optional[str] = None,
    null_star_offset_fraction: float = 0.035,
    hline_zorder: float = 4.0,
) -> plt.Figure:
    """Core scalar grid plotter (supports 1D/2D faceting)."""
    rng = np.random.default_rng(seed)
    _configure_plot_fonts(font_family)

    dfw = df.copy()
    emw = _normalize_emm_ci_columns(emm.copy())
    tkw = tuk.copy()
    ntw = null_tests.copy() if null_tests is not None else pd.DataFrame()

    # Defaults for roles
    if jitter_var is None:
        jitter_var = x_var
    if fill_var is None:
        fill_var = x_var

    # Levels
    if x_levels is None:
        x_levels = _ordered_levels(dfw[x_var], None)
    if col_var is not None:
        if col_levels is None:
            col_levels = _ordered_levels(dfw[col_var], None)
    else:
        col_levels = [None]
    if row_var is not None:
        if row_levels is None:
            row_levels = _ordered_levels(dfw[row_var], None)
    else:
        row_levels = [None]

    # Enforce categorical ordering
    if x_var in dfw.columns:
        dfw[x_var] = pd.Categorical(dfw[x_var], categories=x_levels, ordered=True)
    if x_var in emw.columns:
        emw[x_var] = pd.Categorical(emw[x_var], categories=x_levels, ordered=True)
    if x_var in ntw.columns:
        ntw[x_var] = pd.Categorical(ntw[x_var], categories=x_levels, ordered=True)
    if col_var and col_var in dfw.columns:
        dfw[col_var] = pd.Categorical(dfw[col_var], categories=col_levels, ordered=True)
    if col_var and col_var in emw.columns:
        emw[col_var] = pd.Categorical(emw[col_var], categories=col_levels, ordered=True)
    if col_var and col_var in ntw.columns:
        ntw[col_var] = pd.Categorical(ntw[col_var], categories=col_levels, ordered=True)
    if row_var and row_var in dfw.columns:
        dfw[row_var] = pd.Categorical(dfw[row_var], categories=row_levels, ordered=True)
    if row_var and row_var in emw.columns:
        emw[row_var] = pd.Categorical(emw[row_var], categories=row_levels, ordered=True)
    if row_var and row_var in ntw.columns:
        ntw[row_var] = pd.Categorical(ntw[row_var], categories=row_levels, ordered=True)

    # Color maps
    jitter_levels = (
        _ordered_levels(dfw[jitter_var], None)
        if jitter_var in dfw.columns
        else x_levels
    )
    fill_levels = (
        _ordered_levels(dfw[fill_var], None) if fill_var in dfw.columns else x_levels
    )

    outline_levels: List = []
    if outline_var is not None and outline_var in dfw.columns:
        outline_levels = _ordered_levels(dfw[outline_var], None)

    jitter_cmap = _build_color_map(jitter_levels, jitter_palette)
    fill_cmap = _build_color_map(fill_levels, fill_palette)
    outline_cmap = (
        _build_color_map(outline_levels, outline_palette)
        if (outline_palette is not None and outline_levels)
        else {}
    )

    x_to_num = {str(lv): i for i, lv in enumerate(x_levels)}

    y_limits_use = _auto_y_limits_scalar(
        dfw,
        emw,
        value_col=value_col,
        y_limits=y_limits,
        lower_padding_fraction=lower_y_padding_fraction,
        upper_padding_fraction=upper_y_padding_fraction,
        reference_values=horizontal_lines,
    )

    nrows = len(row_levels)
    ncols = len(col_levels)

    # Layout: strips are shown whenever the dimension exists, including singleton facets.
    top_h = (
        strip_top_height_mm
        if top_strip_labels is not None or col_var is not None
        else 0.0
    )
    right_w = (
        strip_right_width_mm
        if right_strip_labels is not None or row_var is not None
        else 0.0
    )
    if top_strip_labels is not None and len(top_strip_labels) != ncols:
        raise ValueError("top_strip_labels must have one label per plot column.")
    if right_strip_labels is not None and len(right_strip_labels) != nrows:
        raise ValueError("right_strip_labels must have one label per plot row.")

    fig, axes, layout = _init_box_figure(
        nrows=nrows,
        ncols=ncols,
        boxsize_mm=boxsize,
        panel_gap_mm=panel_gap,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        colorbar_width_mm=0.0,
        colorbar_pad_mm=0.0,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        axis_label_fontsize=axis_label_fontsize,
        include_global_label_margins=include_global_label_margins,
        x_label_text=(x_label or x_var),
        y_label_text=(y_label or value_col),
        colorbar_label_text=None,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        cbar_label_offset_mm=0.0,
        dpi=dpi,
        font_family=font_family,
        sharex=True,
        sharey=True,
        transparent=transparent,
    )

    # --------------------- plotting ---------------------
    for i, row_lv in enumerate(row_levels):
        for j, col_lv in enumerate(col_levels):
            ax = axes[i, j]

            # background shadows
            if vertical_shadows:
                for (x0, x1), c in vertical_shadows.items():
                    ax.axvspan(x0, x1, color=_rgba_color(c), zorder=0)
            if horizontal_shadows:
                for (y0, y1), c in horizontal_shadows.items():
                    ax.axhspan(y0, y1, color=_rgba_color(c), zorder=0)

            cell_raw = dfw
            if col_var is not None and col_lv is not None:
                cell_raw = cell_raw[cell_raw[col_var] == col_lv]
            if row_var is not None and row_lv is not None:
                cell_raw = cell_raw[cell_raw[row_var] == row_lv]

            cell_emm = emw
            if (
                col_var is not None
                and col_lv is not None
                and col_var in cell_emm.columns
            ):
                cell_emm = cell_emm[cell_emm[col_var] == col_lv]
            if (
                row_var is not None
                and row_lv is not None
                and row_var in cell_emm.columns
            ):
                cell_emm = cell_emm[cell_emm[row_var] == row_lv]

            # Boxplot
            if show_box:
                data_for_boxes: List[np.ndarray] = []
                pos_for_boxes: List[float] = []
                box_levels_used: List = []

                for lv in x_levels:
                    sub = (
                        cell_raw[cell_raw[x_var] == lv]
                        if x_var in cell_raw.columns
                        else cell_raw.iloc[0:0]
                    )
                    vals = sub[value_col].to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        data_for_boxes.append(vals)
                        pos_for_boxes.append(float(x_to_num[str(lv)]))
                        box_levels_used.append(lv)

                if data_for_boxes:
                    whis = _parse_whiskers(whiskers)
                    bp = ax.boxplot(
                        data_for_boxes,
                        positions=pos_for_boxes,
                        widths=box_width,
                        patch_artist=True,
                        showfliers=True,
                        whis=whis,
                        medianprops=dict(
                            color=median_color, linewidth=median_linewidth
                        ),
                        whiskerprops=dict(
                            color=whisker_color, linewidth=whisker_linewidth
                        ),
                        capprops=dict(color=whisker_color, linewidth=cap_linewidth),
                        flierprops=dict(
                            marker=outlier_marker,
                            markersize=outlier_markersize,
                            markerfacecolor=whisker_color,
                            markeredgecolor=whisker_color,
                            alpha=0.7,
                        ),
                    )

                    use_outline_for_lines = (
                        outline_var is not None and outline_var in cell_raw.columns
                    )

                    for k, b in enumerate(bp["boxes"]):
                        lv = box_levels_used[k] if k < len(box_levels_used) else None

                        # fill by fill_var (typically x_var)
                        if fill_var in cell_raw.columns and lv is not None:
                            sub_x = cell_raw[cell_raw[x_var] == lv]
                            fill_level = (
                                sub_x[fill_var].iloc[0] if not sub_x.empty else lv
                            )
                        else:
                            fill_level = lv
                        face = fill_cmap.get(fill_level, "gray")
                        b.set_facecolor(mpl.colors.to_rgba(face, fill_alpha))

                        # outline by outline_var (optional)
                        # IMPORTANT: when outline_var is provided, it drives the color of:
                        #   - box outline
                        #   - median line
                        #   - whiskers + caps + fliers
                        outline_c = box_edge_color
                        if use_outline_for_lines and lv is not None:
                            sub_x = cell_raw[cell_raw[x_var] == lv]
                            if not sub_x.empty:
                                edge_level = sub_x[outline_var].iloc[0]
                                outline_c = outline_cmap.get(edge_level, box_edge_color)

                        b.set_edgecolor(outline_c)
                        b.set_linewidth(box_edge_width)

                        if use_outline_for_lines:
                            # median is 1 per box
                            if k < len(bp.get("medians", [])):
                                bp["medians"][k].set_color(outline_c)
                                bp["medians"][k].set_linewidth(median_linewidth)

                            # whiskers/caps are 2 per box (low/high)
                            for idx in (2 * k, 2 * k + 1):
                                if idx < len(bp.get("whiskers", [])):
                                    bp["whiskers"][idx].set_color(outline_c)
                                    bp["whiskers"][idx].set_linewidth(whisker_linewidth)
                                if idx < len(bp.get("caps", [])):
                                    bp["caps"][idx].set_color(outline_c)
                                    bp["caps"][idx].set_linewidth(cap_linewidth)

                            # fliers are 1 per box
                            if k < len(bp.get("fliers", [])):
                                fl = bp["fliers"][k]
                                fl.set_markerfacecolor(outline_c)
                                fl.set_markeredgecolor(outline_c)

            # Jitter points
            if not cell_raw.empty:
                xs = cell_raw[x_var].astype(str).map(x_to_num).to_numpy(dtype=float)
                xs_j = xs + (rng.random(xs.shape[0]) - 0.5) * jitter_width

                if jitter_var in cell_raw.columns:
                    j_levels = cell_raw[jitter_var].tolist()
                    cols = [jitter_cmap.get(lv, "gray") for lv in j_levels]
                else:
                    cols = "gray"

                ax.scatter(
                    xs_j,
                    cell_raw[value_col].to_numpy(dtype=float),
                    c=cols,
                    alpha=jitter_alpha,
                    s=jitter_size,
                    linewidths=jitter_linewidth,
                    zorder=2.5,
                )

            # EMM CI + mean line
            if (
                (show_emm_ci or show_emm_line)
                and not cell_emm.empty
                and "emmean" in cell_emm.columns
            ):
                # order by x_levels
                cell_emm = cell_emm.copy()
                cell_emm["_x_str"] = cell_emm[x_var].astype(str)
                cell_emm["_x_num"] = cell_emm["_x_str"].map(x_to_num)
                cell_emm = cell_emm.dropna(subset=["_x_num"]).sort_values("_x_num")
                x_sorted = cell_emm["_x_num"].to_numpy(dtype=float)
                emmean = cell_emm["emmean"].to_numpy(dtype=float)

                if show_emm_ci and {"lower.CL", "upper.CL"}.issubset(cell_emm.columns):
                    lower = cell_emm["lower.CL"].to_numpy(dtype=float)
                    upper = cell_emm["upper.CL"].to_numpy(dtype=float)
                    ax.errorbar(
                        x_sorted,
                        emmean,
                        yerr=[emmean - lower, upper - emmean],
                        fmt=error_bar_marker,
                        ms=error_bar_marker_size,
                        markerfacecolor=error_bar_marker_color or error_bar_color,
                        markeredgecolor=error_bar_marker_color or error_bar_color,
                        markeredgewidth=error_bar_marker_edge_width,
                        elinewidth=error_bar_linewidth,
                        ecolor=error_bar_color,
                        capsize=error_bar_cap,
                        zorder=3.0,
                    )

                if show_emm_line:
                    ax.plot(
                        x_sorted,
                        emmean,
                        color=emm_line_color,
                        lw=emm_line_width,
                        ls=emm_line_style,
                        zorder=3.1,
                        label="EMM",
                    )

            # Raw mean line
            if show_raw_mean_line and not cell_raw.empty:
                grp = (
                    cell_raw.groupby(x_var, observed=True)[value_col]
                    .mean()
                    .reindex(x_levels)
                )
                x_pos = np.array([x_to_num[str(k)] for k in grp.index], dtype=float)
                ax.plot(
                    x_pos,
                    grp.to_numpy(dtype=float),
                    color=raw_mean_line_color,
                    lw=raw_mean_line_width,
                    ls=raw_mean_line_style,
                    zorder=2.9,
                    label="Raw mean",
                )

            # references
            if vertical_lines is not None:
                for xv in vertical_lines:
                    ax.axvline(
                        x=xv,
                        color=vline_color,
                        linestyle=vline_style,
                        linewidth=vline_width,
                        alpha=vline_alpha,
                        zorder=4,
                    )
            if horizontal_lines is not None:
                for yv in horizontal_lines:
                    ax.axhline(
                        y=yv,
                        color=hline_color,
                        linestyle=hline_style,
                        linewidth=hline_width,
                        alpha=hline_alpha,
                        zorder=hline_zorder,
                    )

            # axis cosmetics
            if y_limits_use is not None:
                ax.set_ylim(*y_limits_use)
            if y_log:
                ax.set_yscale("log")
            if x_limits is not None:
                ax.set_xlim(*x_limits)
            if x_log:
                ax.set_xscale("log")

            ax.set_xlim(-0.5, len(x_levels) - 0.5)
            ax.set_xticks(range(len(x_levels)))
            ax.set_xticklabels(
                [str(x) for x in x_levels],
                fontsize=tick_label_fontsize,
                rotation=xtick_rotation,
            )
            for t in ax.get_yticklabels():
                t.set_fontsize(tick_label_fontsize)
                t.set_rotation(ytick_rotation)
            ax.tick_params(labelsize=tick_label_fontsize)
            if tick_linewidth is not None:
                ax.tick_params(width=tick_linewidth)
            if axis_linewidth is not None:
                for spine in ax.spines.values():
                    spine.set_linewidth(axis_linewidth)

            ax.xaxis.get_offset_text().set_fontsize(tick_label_fontsize)
            ax.yaxis.get_offset_text().set_fontsize(tick_label_fontsize)

            if grid:
                ax.grid(True, alpha=grid_alpha, linestyle="--", zorder=0)

            ax.spines["top"].set_visible(show_top_right_axes)
            ax.spines["right"].set_visible(show_top_right_axes)

            if (i == nrows - 1) and (not single_x_label):
                ax.set_xlabel(x_label or x_var, fontsize=axis_label_fontsize)
            if (j == 0) and (not single_y_label):
                ax.set_ylabel(y_label or value_col, fontsize=axis_label_fontsize)

            # Tukey brackets
            if show_brackets:
                cell_tk = tkw
                if (
                    col_var is not None
                    and col_lv is not None
                    and col_var in cell_tk.columns
                ):
                    cell_tk = cell_tk[cell_tk[col_var] == col_lv]
                if (
                    row_var is not None
                    and row_lv is not None
                    and row_var in cell_tk.columns
                ):
                    cell_tk = cell_tk[cell_tk[row_var] == row_lv]

                _draw_tukey_brackets(
                    ax,
                    cell_tk,
                    x_to_num,
                    y_limits_local=y_limits_use,
                    hide_ns=hide_ns,
                    y_start=y_start,
                    y_end=y_end,
                    y_step=y_step,
                    bracket_height_frac=bracket_height_frac,
                    color=bracket_color,
                    lw=bracket_linewidth,
                    text_size=bracket_text_size,
                    p_value_column=p_value_column,
                    star_thresholds=star_thresholds,
                    text_offset_fraction=bracket_text_offset_fraction,
                )

            if show_null_stars:
                if null_p_value_column is None:
                    raise ValueError(
                        "null_p_value_column is required when show_null_stars is true."
                    )
                cell_null = ntw
                if (
                    col_var is not None
                    and col_lv is not None
                    and col_var in cell_null.columns
                ):
                    cell_null = cell_null[cell_null[col_var] == col_lv]
                if (
                    row_var is not None
                    and row_lv is not None
                    and row_var in cell_null.columns
                ):
                    cell_null = cell_null[cell_null[row_var] == row_lv]
                _draw_emmean_vs_null_stars(
                    ax,
                    cell_null,
                    cell_raw,
                    cell_emm,
                    x_to_num,
                    x_var=x_var,
                    value_col=value_col,
                    p_value_column=null_p_value_column,
                    y_limits_local=y_limits_use,
                    hide_ns=hide_ns,
                    star_thresholds=star_thresholds or [],
                    text_size=bracket_text_size,
                    color=bracket_color,
                    offset_fraction=null_star_offset_fraction,
                )

            if sample_size_id_col is not None:
                if sample_size_id_col not in cell_raw.columns:
                    raise KeyError(
                        f"Raw data do not contain sample ID column {sample_size_id_col!r}."
                    )
                for level in x_levels:
                    subset = cell_raw[cell_raw[x_var] == level]
                    n_id = subset[sample_size_id_col].nunique(dropna=True)
                    ax.text(
                        x_to_num[str(level)],
                        sample_size_y_axes,
                        sample_size_template.format(n_ID=n_id),
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va=sample_size_vertical_alignment,
                        fontsize=sample_size_fontsize,
                        color=sample_size_color,
                        zorder=7,
                    )

    # strips
    col_labs = (
        top_strip_labels
        if top_strip_labels is not None
        else ([f"{lv}" for lv in col_levels] if col_var is not None else None)
    )
    row_labs = (
        right_strip_labels
        if right_strip_labels is not None
        else ([f"{lv}" for lv in row_levels] if row_var is not None else None)
    )
    _ = _add_strips_mm(
        fig,
        axes,
        col_labels=col_labs,
        row_labels=row_labs,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
    )

    if transparent:
        _ensure_strip_background_opaque(fig)

    _draw_global_labels_and_title(
        fig,
        layout,
        x_label=(x_label or x_var),
        y_label=(y_label or value_col),
        axis_label_fontsize=axis_label_fontsize,
        title=title,
        title_fontsize=title_fontsize,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        font_family=font_family,
    )

    # legend
    if legend_loc != "none":
        handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=mpl.colors.to_rgba("gray", fill_alpha),
                markeredgecolor=box_edge_color,
                label="Box",
            ),
        ]
        if show_emm_line:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=emm_line_color,
                    lw=emm_line_width,
                    ls=emm_line_style,
                    label="EMM",
                )
            )
        if show_raw_mean_line:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=raw_mean_line_color,
                    lw=raw_mean_line_width,
                    ls=raw_mean_line_style,
                    label="Raw mean",
                )
            )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="gray",
                alpha=jitter_alpha,
                markersize=float(np.sqrt(jitter_size)),
                label=f"Jitter ({jitter_var})",
            )
        )

        if legend_ncol is None:
            legend_ncol = 1

        _place_legend(
            fig,
            axes[0, 0],
            handles,
            [h.get_label() for h in handles],
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            legend_ncol=legend_ncol,
            legend_framealpha=legend_framealpha,
            inside_map=(
                legend_loc
                not in {
                    "right",
                    "left",
                    "upper right",
                    "lower right",
                    "upper left",
                    "lower left",
                }
            ),
        )

    return fig


def plot_triple_interaction_scalar(
    df: pd.DataFrame,
    emm: pd.DataFrame,
    tuk: pd.DataFrame,
    value_col: str = "value",
    x_var: str = "phase",
    panel_var: str = "region",
    facet_var: str = "lat",
    jitter_var: Optional[str] = None,
    fill_var: Optional[str] = None,
    outline_var: Optional[str] = None,
    x_levels: Optional[List] = None,
    panel_levels: Optional[List] = None,
    facet_levels: Optional[List] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    xtick_rotation: float = 0.0,
    ytick_rotation: float = 0.0,
    title: Optional[str] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    jitter_palette: Union[str, List, Dict] = "viridis",
    fill_palette: Union[str, List, Dict] = "viridis",
    outline_palette: Union[str, List, Dict, None] = None,
    jitter_width: float = 0.18,
    jitter_alpha: float = 0.35,
    jitter_size: float = 12.0,
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    seed: int = 1,
    show_top_right_axes: bool = True,
    show_box: bool = True,
    box_width: float = 0.55,
    fill_alpha: float = 0.40,
    whiskers: Union[str, float, Tuple[float, float]] = "tukey",
    box_edge_color: str = "black",
    box_edge_width: float = 1.0,
    median_color: str = "black",
    median_linewidth: float = 1.0,
    whisker_color: str = "black",
    whisker_linewidth: float = 1.0,
    cap_linewidth: float = 1.0,
    outlier_marker: str = "o",
    outlier_markersize: float = 3.0,
    show_emm_line: bool = True,
    show_emm_ci: bool = True,
    show_raw_mean_line: bool = True,
    emm_line_width: float = 1.0,
    emm_line_color: str = "black",
    emm_line_style: str = "-",
    raw_mean_line_width: float = 1.0,
    raw_mean_line_color: str = "#404040",
    raw_mean_line_style: str = "--",
    error_bar_linewidth: float = 1.0,
    error_bar_cap: float = 3.0,
    error_bar_color: str = "black",
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    label_fontsize: int = 16,
    label_top_bg_color: str = "lightgray",
    label_right_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_right_width_mm: float = 3.0,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    legend_loc: str = "none",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (0.0, 0.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    show_brackets: bool = True,
    hide_ns: bool = True,
    y_start: float = 0.70,
    y_end: float = 0.95,
    y_step: Optional[float] = 0.08,
    bracket_height_frac: float = 0.018,
    bracket_color: str = "black",
    bracket_linewidth: float = 1.2,
    bracket_text_size: int = 12,
    transparent: bool = False,
) -> plt.Figure:
    """Triple interaction scalar plot (panel × facet)."""
    return _plot_interaction_scalar_grid(
        df=df,
        emm=emm,
        tuk=tuk,
        value_col=value_col,
        x_var=x_var,
        col_var=panel_var,
        row_var=facet_var,
        col_levels=panel_levels,
        row_levels=facet_levels,
        jitter_var=jitter_var,
        fill_var=fill_var,
        outline_var=outline_var,
        x_levels=x_levels,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        xtick_rotation=xtick_rotation,
        ytick_rotation=ytick_rotation,
        title=title,
        font_family=font_family,
        jitter_palette=jitter_palette,
        fill_palette=fill_palette,
        outline_palette=outline_palette,
        jitter_width=jitter_width,
        jitter_alpha=jitter_alpha,
        jitter_size=jitter_size,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        seed=seed,
        show_top_right_axes=show_top_right_axes,
        show_box=show_box,
        box_width=box_width,
        fill_alpha=fill_alpha,
        whiskers=whiskers,
        box_edge_color=box_edge_color,
        box_edge_width=box_edge_width,
        median_color=median_color,
        median_linewidth=median_linewidth,
        whisker_color=whisker_color,
        whisker_linewidth=whisker_linewidth,
        cap_linewidth=cap_linewidth,
        outlier_marker=outlier_marker,
        outlier_markersize=outlier_markersize,
        show_emm_line=show_emm_line,
        show_emm_ci=show_emm_ci,
        show_raw_mean_line=show_raw_mean_line,
        emm_line_width=emm_line_width,
        emm_line_color=emm_line_color,
        emm_line_style=emm_line_style,
        raw_mean_line_width=raw_mean_line_width,
        raw_mean_line_color=raw_mean_line_color,
        raw_mean_line_style=raw_mean_line_style,
        error_bar_linewidth=error_bar_linewidth,
        error_bar_cap=error_bar_cap,
        error_bar_color=error_bar_color,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=strip_right_width_mm,
        strip_pad_mm=strip_pad_mm,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=panel_gap,
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        show_brackets=show_brackets,
        hide_ns=hide_ns,
        y_start=y_start,
        y_end=y_end,
        y_step=y_step,
        bracket_height_frac=bracket_height_frac,
        bracket_color=bracket_color,
        bracket_linewidth=bracket_linewidth,
        bracket_text_size=bracket_text_size,
        transparent=transparent,
    )


def plot_double_interaction_scalar(
    df: pd.DataFrame,
    emm: pd.DataFrame,
    tuk: pd.DataFrame,
    value_col: str = "value",
    x_var: str = "phase",
    panel_var: str = "region",
    jitter_var: Optional[str] = None,
    fill_var: Optional[str] = None,
    outline_var: Optional[str] = None,
    x_levels: Optional[List] = None,
    panel_levels: Optional[List] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    xtick_rotation: float = 0.0,
    ytick_rotation: float = 0.0,
    title: Optional[str] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    jitter_palette: Union[str, List, Dict] = "viridis",
    fill_palette: Union[str, List, Dict] = "viridis",
    outline_palette: Union[str, List, Dict, None] = None,
    jitter_width: float = 0.18,
    jitter_alpha: float = 0.35,
    jitter_size: float = 12.0,
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    seed: int = 1,
    show_top_right_axes: bool = True,
    show_box: bool = True,
    box_width: float = 0.55,
    fill_alpha: float = 0.40,
    whiskers: Union[str, float, Tuple[float, float]] = "tukey",
    box_edge_color: str = "black",
    box_edge_width: float = 1.0,
    median_color: str = "black",
    median_linewidth: float = 1.0,
    whisker_color: str = "black",
    whisker_linewidth: float = 1.0,
    cap_linewidth: float = 1.0,
    outlier_marker: str = "o",
    outlier_markersize: float = 3.0,
    show_emm_line: bool = True,
    show_emm_ci: bool = True,
    show_raw_mean_line: bool = True,
    emm_line_width: float = 1.0,
    emm_line_color: str = "black",
    emm_line_style: str = "-",
    raw_mean_line_width: float = 1.0,
    raw_mean_line_color: str = "#404040",
    raw_mean_line_style: str = "--",
    error_bar_linewidth: float = 1.0,
    error_bar_cap: float = 3.0,
    error_bar_color: str = "black",
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    label_fontsize: int = 16,
    label_top_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    legend_loc: str = "none",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (0.0, 0.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    show_brackets: bool = True,
    hide_ns: bool = True,
    y_start: float = 0.70,
    y_end: float = 0.95,
    y_step: Optional[float] = 0.08,
    bracket_height_frac: float = 0.018,
    bracket_color: str = "black",
    bracket_linewidth: float = 1.2,
    bracket_text_size: int = 12,
    transparent: bool = False,
) -> plt.Figure:
    """Double interaction scalar plot (panel only)."""
    return _plot_interaction_scalar_grid(
        df=df,
        emm=emm,
        tuk=tuk,
        value_col=value_col,
        x_var=x_var,
        col_var=panel_var,
        row_var=None,
        col_levels=panel_levels,
        row_levels=None,
        jitter_var=jitter_var,
        fill_var=fill_var,
        outline_var=outline_var,
        x_levels=x_levels,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        xtick_rotation=xtick_rotation,
        ytick_rotation=ytick_rotation,
        title=title,
        font_family=font_family,
        jitter_palette=jitter_palette,
        fill_palette=fill_palette,
        outline_palette=outline_palette,
        jitter_width=jitter_width,
        jitter_alpha=jitter_alpha,
        jitter_size=jitter_size,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        seed=seed,
        show_top_right_axes=show_top_right_axes,
        show_box=show_box,
        box_width=box_width,
        fill_alpha=fill_alpha,
        whiskers=whiskers,
        box_edge_color=box_edge_color,
        box_edge_width=box_edge_width,
        median_color=median_color,
        median_linewidth=median_linewidth,
        whisker_color=whisker_color,
        whisker_linewidth=whisker_linewidth,
        cap_linewidth=cap_linewidth,
        outlier_marker=outlier_marker,
        outlier_markersize=outlier_markersize,
        show_emm_line=show_emm_line,
        show_emm_ci=show_emm_ci,
        show_raw_mean_line=show_raw_mean_line,
        emm_line_width=emm_line_width,
        emm_line_color=emm_line_color,
        emm_line_style=emm_line_style,
        raw_mean_line_width=raw_mean_line_width,
        raw_mean_line_color=raw_mean_line_color,
        raw_mean_line_style=raw_mean_line_style,
        error_bar_linewidth=error_bar_linewidth,
        error_bar_cap=error_bar_cap,
        error_bar_color=error_bar_color,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_top_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=0.0,
        strip_pad_mm=strip_pad_mm,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=panel_gap,
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        show_brackets=show_brackets,
        hide_ns=hide_ns,
        y_start=y_start,
        y_end=y_end,
        y_step=y_step,
        bracket_height_frac=bracket_height_frac,
        bracket_color=bracket_color,
        bracket_linewidth=bracket_linewidth,
        bracket_text_size=bracket_text_size,
        transparent=transparent,
    )


def plot_single_effect_scalar(
    df: pd.DataFrame,
    emm: pd.DataFrame,
    tuk: pd.DataFrame,
    value_col: str = "value",
    x_var: str = "phase",
    jitter_var: Optional[str] = None,
    fill_var: Optional[str] = None,
    outline_var: Optional[str] = None,
    x_levels: Optional[List] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    xtick_rotation: float = 0.0,
    ytick_rotation: float = 0.0,
    title: Optional[str] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    jitter_palette: Union[str, List, Dict] = "viridis",
    fill_palette: Union[str, List, Dict] = "viridis",
    outline_palette: Union[str, List, Dict, None] = None,
    jitter_width: float = 0.18,
    jitter_alpha: float = 0.35,
    jitter_size: float = 12.0,
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    seed: int = 1,
    show_top_right_axes: bool = True,
    show_box: bool = True,
    box_width: float = 0.55,
    fill_alpha: float = 0.40,
    whiskers: Union[str, float, Tuple[float, float]] = "tukey",
    box_edge_color: str = "black",
    box_edge_width: float = 1.0,
    median_color: str = "black",
    median_linewidth: float = 1.0,
    whisker_color: str = "black",
    whisker_linewidth: float = 1.0,
    cap_linewidth: float = 1.0,
    outlier_marker: str = "o",
    outlier_markersize: float = 3.0,
    show_emm_line: bool = True,
    show_emm_ci: bool = True,
    show_raw_mean_line: bool = True,
    emm_line_width: float = 1.0,
    emm_line_color: str = "black",
    emm_line_style: str = "-",
    raw_mean_line_width: float = 1.0,
    raw_mean_line_color: str = "#404040",
    raw_mean_line_style: str = "--",
    error_bar_linewidth: float = 1.0,
    error_bar_cap: float = 3.0,
    error_bar_color: str = "black",
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    legend_loc: str = "none",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    show_brackets: bool = True,
    hide_ns: bool = True,
    y_start: float = 0.70,
    y_end: float = 0.95,
    y_step: Optional[float] = 0.08,
    bracket_height_frac: float = 0.018,
    bracket_color: str = "black",
    bracket_linewidth: float = 1.2,
    bracket_text_size: int = 12,
    transparent: bool = False,
) -> plt.Figure:
    """Single effect scalar plot (no faceting)."""
    return _plot_interaction_scalar_grid(
        df=df,
        emm=emm,
        tuk=tuk,
        value_col=value_col,
        x_var=x_var,
        col_var=None,
        row_var=None,
        col_levels=None,
        row_levels=None,
        jitter_var=jitter_var,
        fill_var=fill_var,
        outline_var=outline_var,
        x_levels=x_levels,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        xtick_rotation=xtick_rotation,
        ytick_rotation=ytick_rotation,
        title=title,
        font_family=font_family,
        jitter_palette=jitter_palette,
        fill_palette=fill_palette,
        outline_palette=outline_palette,
        jitter_width=jitter_width,
        jitter_alpha=jitter_alpha,
        jitter_size=jitter_size,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        seed=seed,
        show_top_right_axes=show_top_right_axes,
        show_box=show_box,
        box_width=box_width,
        fill_alpha=fill_alpha,
        whiskers=whiskers,
        box_edge_color=box_edge_color,
        box_edge_width=box_edge_width,
        median_color=median_color,
        median_linewidth=median_linewidth,
        whisker_color=whisker_color,
        whisker_linewidth=whisker_linewidth,
        cap_linewidth=cap_linewidth,
        outlier_marker=outlier_marker,
        outlier_markersize=outlier_markersize,
        show_emm_line=show_emm_line,
        show_emm_ci=show_emm_ci,
        show_raw_mean_line=show_raw_mean_line,
        emm_line_width=emm_line_width,
        emm_line_color=emm_line_color,
        emm_line_style=emm_line_style,
        raw_mean_line_width=raw_mean_line_width,
        raw_mean_line_color=raw_mean_line_color,
        raw_mean_line_style=raw_mean_line_style,
        error_bar_linewidth=error_bar_linewidth,
        error_bar_cap=error_bar_cap,
        error_bar_color=error_bar_color,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=16,
        label_top_bg_color="lightgray",
        label_right_bg_color="lightgray",
        label_text_color="black",
        label_fontweight="normal",
        strip_top_height_mm=0.0,
        strip_right_width_mm=0.0,
        strip_pad_mm=0.0,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=(0.0, 0.0),
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        show_brackets=show_brackets,
        hide_ns=hide_ns,
        y_start=y_start,
        y_end=y_end,
        y_step=y_step,
        bracket_height_frac=bracket_height_frac,
        bracket_color=bracket_color,
        bracket_linewidth=bracket_linewidth,
        bracket_text_size=bracket_text_size,
        transparent=transparent,
    )


# --------------------- series plotting ---------------------


def _auto_limits_series(
    dfw: pd.DataFrame,
    *,
    value_col: str,
    y_limits: Optional[Tuple[float, float]],
    x_limits: Optional[Tuple[float, float]],
    line_var: str,
    col_var: Optional[str] = None,
    row_var: Optional[str] = None,
    line_levels: Optional[List] = None,
    col_levels: Optional[List] = None,
    row_levels: Optional[List] = None,
    ribbon: str = "sem",
    include_ribbon: bool = True,
    n_grid: int = 400,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Auto-compute x/y limits for series plots based on what is actually rendered.

    The limits are computed from:
      - the mean curve (per `line_var`, per facet cell)
      - the ribbon envelope (low/high; e.g., SEM / SD / 95% CI) when `include_ribbon` is True

    This avoids a single outlier trace inside `dfw[value_col]` blowing up the axis range
    when the visualization is meant to emphasize the group mean ± uncertainty.
    """
    if x_limits is not None and y_limits is not None:
        return x_limits, y_limits

    # Infer levels if not provided (must match plotting logic).
    if line_levels is None:
        if line_var in dfw.columns:
            line_levels = _ordered_levels(dfw[line_var], None)
        else:
            line_levels = []

    if col_var is None:
        col_levels_use: List = [None]
    else:
        col_levels_use = (
            list(col_levels)
            if col_levels is not None
            else _ordered_levels(dfw[col_var], None)
        )

    if row_var is None:
        row_levels_use: List = [None]
    else:
        row_levels_use = (
            list(row_levels)
            if row_levels is not None
            else _ordered_levels(dfw[row_var], None)
        )

    xmins: List[float] = []
    xmaxs: List[float] = []
    ymins: List[float] = []
    ymaxs: List[float] = []

    # Prefer limits based on the rendered mean/ribbon curves.
    for row_lv in row_levels_use:
        for col_lv in col_levels_use:
            cell = dfw
            if col_var is not None and col_lv is not None and col_var in cell.columns:
                cell = cell[cell[col_var] == col_lv]
            if row_var is not None and row_lv is not None and row_var in cell.columns:
                cell = cell[cell[row_var] == row_lv]
            if cell.empty:
                continue

            for lv in line_levels:
                if line_var in cell.columns:
                    sub = cell[cell[line_var] == lv]
                else:
                    sub = cell

                series_list = [
                    s
                    for s in sub[value_col].tolist()
                    if isinstance(s, pd.Series) and not s.empty
                ]
                xg, mean, low, high = _series_grid_stats(
                    series_list, n_grid=n_grid, ribbon=ribbon
                )
                if xg.size == 0:
                    continue

                xg_f = xg[np.isfinite(xg)]
                if xg_f.size:
                    xmins.append(float(np.min(xg_f)))
                    xmaxs.append(float(np.max(xg_f)))

                if include_ribbon:
                    y_cat = np.concatenate([mean, low, high])
                else:
                    y_cat = mean
                y_f = y_cat[np.isfinite(y_cat)]
                if y_f.size:
                    ymins.append(float(np.min(y_f)))
                    ymaxs.append(float(np.max(y_f)))

    # Fallback: if nothing was rendered (e.g., all series empty), fall back to raw series.
    if (not xmins or not xmaxs) or (not ymins or not ymaxs):
        for s in dfw[value_col]:
            if not isinstance(s, pd.Series) or s.empty:
                continue
            x = _to_numeric_index(s.index)
            y = s.to_numpy(dtype=float)
            y = y[np.isfinite(y)]
            x = x[np.isfinite(x)]
            if x.size:
                xmins.append(float(np.min(x)))
                xmaxs.append(float(np.max(x)))
            if y.size:
                ymins.append(float(np.min(y)))
                ymaxs.append(float(np.max(y)))

    if x_limits is None:
        if xmins and xmaxs:
            x0, x1 = float(np.min(xmins)), float(np.max(xmaxs))
        else:
            x0, x1 = 0.0, 1.0
        xr = x1 - x0
        if (not np.isfinite(xr)) or xr == 0:
            xr = 1.0
        x_limits = (x0 - 0.02 * xr, x1 + 0.02 * xr)

    if y_limits is None:
        if ymins and ymaxs:
            y0, y1 = float(np.min(ymins)), float(np.max(ymaxs))
        else:
            y0, y1 = 0.0, 1.0
        yr = y1 - y0
        if (not np.isfinite(yr)) or yr == 0:
            yr = 1.0
        y_limits = (y0 - 0.05 * yr, y1 + 0.05 * yr)

    return x_limits, y_limits


def _plot_interaction_series_grid(
    *,
    df: pd.DataFrame,
    value_col: str,
    line_var: str,
    col_var: Optional[str],
    row_var: Optional[str],
    line_levels: Optional[List],
    col_levels: Optional[List],
    row_levels: Optional[List],
    x_label: Optional[str],
    single_x_label: bool,
    y_label: Optional[str],
    single_y_label: bool,
    x_limits: Optional[Tuple[float, float]],
    y_limits: Optional[Tuple[float, float]],
    x_log: bool,
    y_log: bool,
    title: Optional[str],
    font_family: str,
    line_palette: Union[str, List, Dict],
    line_width: float,
    line_alpha: float,
    show_ribbon: bool,
    ribbon_alpha: float,
    ribbon: str,
    n_grid: int,
    grid: bool,
    grid_alpha: float,
    dpi: int,
    show_top_right_axes: bool,
    vertical_lines: Optional[Union[List, np.ndarray]],
    vline_color: str,
    vline_style: str,
    vline_width: float,
    vline_alpha: float,
    horizontal_lines: Optional[Union[List, np.ndarray]],
    hline_color: str,
    hline_style: str,
    hline_width: float,
    hline_alpha: float,
    hline_zorder: float,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]],
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]],
    # strips
    label_fontsize: int,
    label_top_bg_color: str,
    label_right_bg_color: str,
    label_text_color: str,
    label_fontweight: str,
    strip_top_height_mm: float,
    strip_right_width_mm: float,
    strip_pad_mm: float,
    # text sizes
    title_fontsize: int,
    axis_label_fontsize: int,
    tick_label_fontsize: int,
    # legend
    legend_loc: str,
    legend_ncol: Optional[int],
    legend_framealpha: float,
    legend_fontsize: int,
    # geometry
    boxsize: Tuple[float, float],
    panel_gap: Tuple[float, float],
    include_global_label_margins: bool,
    x_label_offset_mm: float,
    y_label_offset_mm: float,
    transparent: bool,
) -> plt.Figure:
    """Core series grid plotter."""
    _configure_plot_fonts(font_family)
    dfw = df.copy()

    # Levels
    if line_levels is None:
        line_levels = _ordered_levels(dfw[line_var], None)
    if col_var is not None:
        if col_levels is None:
            col_levels = _ordered_levels(dfw[col_var], None)
    else:
        col_levels = [None]
    if row_var is not None:
        if row_levels is None:
            row_levels = _ordered_levels(dfw[row_var], None)
    else:
        row_levels = [None]

    if line_var in dfw.columns:
        dfw[line_var] = pd.Categorical(
            dfw[line_var], categories=line_levels, ordered=True
        )
    if col_var and col_var in dfw.columns:
        dfw[col_var] = pd.Categorical(dfw[col_var], categories=col_levels, ordered=True)
    if row_var and row_var in dfw.columns:
        dfw[row_var] = pd.Categorical(dfw[row_var], categories=row_levels, ordered=True)

    line_cmap = _build_color_map(line_levels, line_palette)

    x_limits_use, y_limits_use = _auto_limits_series(
        dfw,
        value_col=value_col,
        x_limits=x_limits,
        y_limits=y_limits,
        line_var=line_var,
        col_var=col_var,
        row_var=row_var,
        line_levels=line_levels,
        col_levels=col_levels,
        row_levels=row_levels,
        ribbon=ribbon,
        include_ribbon=show_ribbon,
        n_grid=n_grid,
    )

    nrows = len(row_levels)
    ncols = len(col_levels)

    top_h = strip_top_height_mm if col_var is not None else 0.0
    right_w = strip_right_width_mm if row_var is not None else 0.0

    fig, axes, layout = _init_box_figure(
        nrows=nrows,
        ncols=ncols,
        boxsize_mm=boxsize,
        panel_gap_mm=panel_gap,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        colorbar_width_mm=0.0,
        colorbar_pad_mm=0.0,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        axis_label_fontsize=axis_label_fontsize,
        include_global_label_margins=include_global_label_margins,
        x_label_text=(x_label or ""),
        y_label_text=(y_label or value_col),
        colorbar_label_text=None,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        dpi=dpi,
        font_family=font_family,
        sharex=True,
        sharey=True,
        transparent=transparent,
    )

    for i, row_lv in enumerate(row_levels):
        for j, col_lv in enumerate(col_levels):
            ax = axes[i, j]

            if vertical_shadows:
                for (x0, x1), c in vertical_shadows.items():
                    _safe_axvspan(ax, x0, x1, color=_rgba_color(c), zorder=0)
            if horizontal_shadows:
                for (y0, y1), c in horizontal_shadows.items():
                    _safe_axhspan(ax, y0, y1, color=_rgba_color(c), zorder=0)

            cell = dfw
            if col_var is not None and col_lv is not None:
                cell = cell[cell[col_var] == col_lv]
            if row_var is not None and row_lv is not None:
                cell = cell[cell[row_var] == row_lv]

            for lv in line_levels:
                sub = cell[cell[line_var] == lv]
                series_list = [
                    s for s in sub[value_col].tolist() if isinstance(s, pd.Series)
                ]
                xg, mean, low, high = _series_grid_stats(
                    series_list, n_grid=n_grid, ribbon=ribbon
                )
                if xg.size == 0:
                    continue

                c = line_cmap.get(lv, "black")
                ax.plot(
                    xg,
                    mean,
                    color=c,
                    lw=line_width,
                    alpha=line_alpha,
                    label=str(lv),
                    zorder=3,
                )

                if show_ribbon:
                    ax.fill_between(
                        xg,
                        low,
                        high,
                        color=c,
                        alpha=ribbon_alpha,
                        linewidth=0,
                        zorder=2,
                    )

            if vertical_lines is not None:
                for xv in vertical_lines:
                    ax.axvline(
                        x=xv,
                        color=vline_color,
                        linestyle=vline_style,
                        linewidth=vline_width,
                        alpha=vline_alpha,
                        zorder=4,
                    )

            if horizontal_lines is not None:
                for yv in horizontal_lines:
                    ax.axhline(
                        y=yv,
                        color=hline_color,
                        linestyle=hline_style,
                        linewidth=hline_width,
                        alpha=hline_alpha,
                        zorder=hline_zorder,
                    )

            ax.set_xlim(*x_limits_use)
            ax.set_ylim(*y_limits_use)

            if x_log:
                ax.set_xscale("log")
            if y_log:
                ax.set_yscale("log")

            ax.tick_params(labelsize=tick_label_fontsize)
            if grid:
                ax.grid(True, alpha=grid_alpha, linestyle="--", zorder=0)

            ax.spines["top"].set_visible(show_top_right_axes)
            ax.spines["right"].set_visible(show_top_right_axes)

            if (i == nrows - 1) and (not single_x_label):
                ax.set_xlabel(x_label or "", fontsize=axis_label_fontsize)
            if (j == 0) and (not single_y_label):
                ax.set_ylabel(y_label or value_col, fontsize=axis_label_fontsize)

    # strips
    col_labs = [f"{lv}" for lv in col_levels] if col_var is not None else None
    row_labs = [f"{lv}" for lv in row_levels] if row_var is not None else None
    _ = _add_strips_mm(
        fig,
        axes,
        col_labels=col_labs,
        row_labels=row_labs,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
    )

    if transparent:
        _ensure_strip_background_opaque(fig)

    _draw_global_labels_and_title(
        fig,
        layout,
        x_label=x_label,
        y_label=(y_label or value_col),
        axis_label_fontsize=axis_label_fontsize,
        title=title,
        title_fontsize=title_fontsize,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        font_family=font_family,
    )

    if legend_loc != "none":
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if legend_ncol is None:
            legend_ncol = 1

        inside_map = not _is_outside_legend_loc(legend_loc)
        _place_legend(
            fig,
            axes[0, 0],
            handles,
            labels,
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            legend_ncol=legend_ncol,
            legend_framealpha=legend_framealpha,
            inside_map=inside_map,
        )

    return fig


def plot_triple_interaction_series(
    df: pd.DataFrame,
    value_col: str = "value",
    x_var: str = "phase",
    panel_var: str = "region",
    facet_var: str = "lat",
    x_levels: Optional[List] = None,
    panel_levels: Optional[List] = None,
    facet_levels: Optional[List] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    line_palette: Union[str, List, Dict] = "viridis",
    line_width: float = 2.0,
    line_alpha: float = 0.95,
    show_ribbon: bool = True,
    ribbon_alpha: float = 0.25,
    ribbon: str = "sem",
    n_grid: int = 400,
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    show_top_right_axes: bool = True,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    label_fontsize: int = 16,
    label_top_bg_color: str = "lightgray",
    label_right_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_right_width_mm: float = 3.0,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    legend_loc: str = "right",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (3.0, 3.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    transparent: bool = False,
    hline_zorder: float = 4.0,
) -> plt.Figure:
    """Triple interaction series plot (panel × facet)."""
    return _plot_interaction_series_grid(
        df=df,
        value_col=value_col,
        line_var=x_var,
        col_var=panel_var,
        row_var=facet_var,
        line_levels=x_levels,
        col_levels=panel_levels,
        row_levels=facet_levels,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        line_palette=line_palette,
        line_width=line_width,
        line_alpha=line_alpha,
        show_ribbon=show_ribbon,
        ribbon_alpha=ribbon_alpha,
        ribbon=ribbon,
        n_grid=n_grid,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        show_top_right_axes=show_top_right_axes,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        hline_zorder=hline_zorder,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=strip_right_width_mm,
        strip_pad_mm=strip_pad_mm,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=panel_gap,
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        transparent=transparent,
    )


def plot_double_interaction_series(
    df: pd.DataFrame,
    value_col: str = "value",
    x_var: str = "phase",
    panel_var: str = "region",
    x_levels: Optional[List] = None,
    panel_levels: Optional[List] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    line_palette: Union[str, List, Dict] = "viridis",
    line_width: float = 2.0,
    line_alpha: float = 0.95,
    show_ribbon: bool = True,
    ribbon_alpha: float = 0.25,
    ribbon: str = "sem",
    n_grid: int = 400,
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    show_top_right_axes: bool = True,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    label_fontsize: int = 16,
    label_top_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    legend_loc: str = "right",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (3.0, 3.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    transparent: bool = False,
    hline_zorder: float = 4.0,
) -> plt.Figure:
    """Double interaction series plot (panel only)."""
    return _plot_interaction_series_grid(
        df=df,
        value_col=value_col,
        line_var=x_var,
        col_var=panel_var,
        row_var=None,
        line_levels=x_levels,
        col_levels=panel_levels,
        row_levels=None,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        line_palette=line_palette,
        line_width=line_width,
        line_alpha=line_alpha,
        show_ribbon=show_ribbon,
        ribbon_alpha=ribbon_alpha,
        ribbon=ribbon,
        n_grid=n_grid,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        show_top_right_axes=show_top_right_axes,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        hline_zorder=hline_zorder,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_top_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=0.0,
        strip_pad_mm=strip_pad_mm,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=panel_gap,
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        transparent=transparent,
    )


def plot_single_effect_series(
    df: pd.DataFrame,
    value_col: str = "value",
    x_var: str = "phase",
    x_levels: Optional[List] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    line_palette: Union[str, List, Dict] = "viridis",
    line_width: float = 2.0,
    line_alpha: float = 0.95,
    show_ribbon: bool = True,
    ribbon_alpha: float = 0.25,
    ribbon: str = "sem",
    n_grid: int = 400,
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    show_top_right_axes: bool = True,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    legend_loc: str = "right",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    transparent: bool = False,
    hline_zorder: float = 4.0,
) -> plt.Figure:
    """Single effect series plot (no faceting)."""
    return _plot_interaction_series_grid(
        df=df,
        value_col=value_col,
        line_var=x_var,
        col_var=None,
        row_var=None,
        line_levels=x_levels,
        col_levels=None,
        row_levels=None,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        line_palette=line_palette,
        line_width=line_width,
        line_alpha=line_alpha,
        show_ribbon=show_ribbon,
        ribbon_alpha=ribbon_alpha,
        ribbon=ribbon,
        n_grid=n_grid,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        show_top_right_axes=show_top_right_axes,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        hline_zorder=hline_zorder,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=16,
        label_top_bg_color="lightgray",
        label_right_bg_color="lightgray",
        label_text_color="black",
        label_fontweight="normal",
        strip_top_height_mm=0.0,
        strip_right_width_mm=0.0,
        strip_pad_mm=0.0,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=(0.0, 0.0),
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        transparent=transparent,
    )


# --------------------- categorical summary line plotting ---------------------


def _validate_distinct_plot_roles(**roles: Optional[str]) -> None:
    """Validate that a plotting variable is not assigned to multiple roles."""
    provided = [(role, var) for role, var in roles.items() if var is not None]
    seen: Dict[str, str] = {}
    for role, var in provided:
        if var in seen:
            raise ValueError(f"{var!r} is assigned to both {seen[var]!r} and {role!r}.")
        seen[var] = role


def _categorical_summary_line_stats(
    df: pd.DataFrame,
    *,
    x_var: str,
    value_col: str,
    line_var: Optional[str] = None,
    panel_var: Optional[str] = None,
    facet_var: Optional[str] = None,
    x_levels: Optional[List] = None,
    line_levels: Optional[List] = None,
    panel_levels: Optional[List] = None,
    facet_levels: Optional[List] = None,
) -> pd.DataFrame:
    """Aggregate finite values as mean and SEM for categorical line plots."""
    _validate_distinct_plot_roles(
        x_var=x_var,
        line_var=line_var,
        panel_var=panel_var,
        facet_var=facet_var,
    )
    required = [value_col, x_var]
    required.extend(var for var in (line_var, panel_var, facet_var) if var is not None)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns for categorical summary line plot: {missing}"
        )

    dfw = df.loc[:, list(dict.fromkeys(required))].copy()
    dfw[value_col] = pd.to_numeric(dfw[value_col], errors="coerce")
    dfw = dfw[np.isfinite(dfw[value_col].to_numpy(dtype=float))]

    levels_by_var: Dict[str, List] = {
        x_var: _ordered_levels(df[x_var], x_levels),
    }
    if line_var is not None:
        levels_by_var[line_var] = _ordered_levels(df[line_var], line_levels)
    if panel_var is not None:
        levels_by_var[panel_var] = _ordered_levels(df[panel_var], panel_levels)
    if facet_var is not None:
        levels_by_var[facet_var] = _ordered_levels(df[facet_var], facet_levels)

    if not levels_by_var[x_var]:
        raise ValueError(f"No levels available for x_var={x_var!r}.")

    for var, levels in levels_by_var.items():
        dfw[var] = pd.Categorical(dfw[var], categories=levels, ordered=True)

    group_vars = [
        var for var in (facet_var, panel_var, line_var, x_var) if var is not None
    ]
    summary = (
        dfw.groupby(group_vars, observed=False, sort=False)[value_col]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
        .sort_values(group_vars)
        .reset_index(drop=True)
    )
    summary["sem"] = np.where(
        summary["n"].to_numpy(dtype=float) >= 2.0,
        summary["sd"].to_numpy(dtype=float)
        / np.sqrt(summary["n"].to_numpy(dtype=float)),
        np.nan,
    )
    return summary.drop(columns=["sd"])


def _auto_y_limits_categorical_summary(
    summary: pd.DataFrame,
    *,
    y_limits: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    """Compute y limits from rendered mean +/- SEM values."""
    if y_limits is not None:
        return tuple(y_limits)

    mean = summary["mean"].to_numpy(dtype=float)
    sem = summary["sem"].to_numpy(dtype=float)
    low = mean - np.where(np.isfinite(sem), sem, 0.0)
    high = mean + np.where(np.isfinite(sem), sem, 0.0)
    values = np.concatenate([low, high])
    values = values[np.isfinite(values)]

    if values.size == 0:
        return (0.0, 1.0)

    y0 = float(np.min(values))
    y1 = float(np.max(values))
    yr = y1 - y0
    if not np.isfinite(yr) or yr == 0:
        pad = max(abs(y0) * 0.05, 0.5)
        return (y0 - pad, y1 + pad)

    pad = 0.05 * yr
    return (y0 - pad, y1 + pad)


def plot_categorical_summary_line(
    df: pd.DataFrame,
    value_col: str = "value",
    x_var: str = "group",
    line_var: Optional[str] = None,
    panel_var: Optional[str] = None,
    facet_var: Optional[str] = None,
    x_levels: Optional[List] = None,
    line_levels: Optional[List] = None,
    panel_levels: Optional[List] = None,
    facet_levels: Optional[List] = None,
    x_tick_labels: Optional[Dict] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    y_limits: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    line_palette: Union[str, List, Dict] = "viridis",
    line_width: float = 2.0,
    line_alpha: float = 0.95,
    marker: Optional[str] = "o",
    marker_size: float = 3.0,
    error_bar_linewidth: float = 1.0,
    error_bar_cap: float = 2.0,
    error_bar_color: Optional[str] = None,
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    show_top_right_axes: bool = True,
    xtick_rotation: float = 0.0,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    label_fontsize: int = 16,
    label_top_bg_color: str = "lightgray",
    label_right_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_right_width_mm: float = 3.0,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    legend_loc: str = "right",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (3.0, 3.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    transparent: bool = False,
) -> plt.Figure:
    """Plot categorical mean +/- SEM lines from long-form data."""
    _configure_plot_fonts(font_family)
    _validate_boxsize(boxsize)

    x_levels_use = _ordered_levels(df[x_var], x_levels)
    if not x_levels_use:
        raise ValueError(f"No levels available for x_var={x_var!r}.")

    if line_var is None:
        line_levels_use = [None]
    else:
        line_levels_use = _ordered_levels(df[line_var], line_levels)
        if not line_levels_use:
            raise ValueError(f"No levels available for line_var={line_var!r}.")

    if panel_var is None:
        panel_levels_use = [None]
    else:
        panel_levels_use = _ordered_levels(df[panel_var], panel_levels)
    if facet_var is None:
        facet_levels_use = [None]
    else:
        facet_levels_use = _ordered_levels(df[facet_var], facet_levels)

    summary = _categorical_summary_line_stats(
        df,
        x_var=x_var,
        value_col=value_col,
        line_var=line_var,
        panel_var=panel_var,
        facet_var=facet_var,
        x_levels=x_levels_use,
        line_levels=line_levels_use if line_var is not None else None,
        panel_levels=panel_levels_use if panel_var is not None else None,
        facet_levels=facet_levels_use if facet_var is not None else None,
    )

    line_cmap = (
        _build_color_map(line_levels_use, line_palette)
        if line_var is not None
        else {None: "black"}
    )
    y_limits_use = _auto_y_limits_categorical_summary(summary, y_limits=y_limits)

    nrows = len(facet_levels_use)
    ncols = len(panel_levels_use)
    top_h = strip_top_height_mm if panel_var is not None else 0.0
    right_w = strip_right_width_mm if facet_var is not None else 0.0
    x_label_use = x_label or x_var
    y_label_use = y_label or value_col

    fig, axes, layout = _init_box_figure(
        nrows=nrows,
        ncols=ncols,
        boxsize_mm=boxsize,
        panel_gap_mm=panel_gap,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        colorbar_width_mm=0.0,
        colorbar_pad_mm=0.0,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        axis_label_fontsize=axis_label_fontsize,
        include_global_label_margins=include_global_label_margins,
        x_label_text=x_label_use,
        y_label_text=y_label_use,
        colorbar_label_text=None,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        dpi=dpi,
        font_family=font_family,
        sharex=True,
        sharey=True,
        transparent=transparent,
    )

    x_positions = np.arange(len(x_levels_use), dtype=float)
    x_label_map = x_tick_labels or {}
    x_tick_text = [str(x_label_map.get(level, level)) for level in x_levels_use]

    for i, facet_lv in enumerate(facet_levels_use):
        for j, panel_lv in enumerate(panel_levels_use):
            ax = axes[i, j]
            cell = summary
            if facet_var is not None:
                cell = cell[cell[facet_var] == facet_lv]
            if panel_var is not None:
                cell = cell[cell[panel_var] == panel_lv]

            for line_lv in line_levels_use:
                sub = cell
                if line_var is not None:
                    sub = sub[sub[line_var] == line_lv]
                if sub.empty:
                    continue

                sub = sub.set_index(x_var).reindex(x_levels_use)
                y = sub["mean"].to_numpy(dtype=float)
                sem = sub["sem"].to_numpy(dtype=float)
                finite_mean = np.isfinite(y)
                if not finite_mean.any():
                    continue

                color = line_cmap.get(line_lv, "black")
                ax.plot(
                    x_positions[finite_mean],
                    y[finite_mean],
                    color=color,
                    lw=line_width,
                    alpha=line_alpha,
                    marker=marker,
                    markersize=marker_size,
                    label=str(line_lv) if line_var is not None else None,
                    zorder=3,
                )

                finite_sem = finite_mean & np.isfinite(sem)
                if finite_sem.any():
                    ax.errorbar(
                        x_positions[finite_sem],
                        y[finite_sem],
                        yerr=sem[finite_sem],
                        fmt="none",
                        ecolor=error_bar_color or color,
                        elinewidth=error_bar_linewidth,
                        capsize=error_bar_cap,
                        zorder=4,
                    )

            if vertical_lines is not None:
                for xv in vertical_lines:
                    ax.axvline(
                        x=xv,
                        color=vline_color,
                        linestyle=vline_style,
                        linewidth=vline_width,
                        alpha=vline_alpha,
                        zorder=2,
                    )
            if horizontal_lines is not None:
                for yv in horizontal_lines:
                    ax.axhline(
                        y=yv,
                        color=hline_color,
                        linestyle=hline_style,
                        linewidth=hline_width,
                        alpha=hline_alpha,
                        zorder=2,
                    )

            ax.set_xlim(-0.5, len(x_levels_use) - 0.5)
            ax.set_ylim(*y_limits_use)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_tick_text)
            for label in ax.get_xticklabels():
                label.set_rotation(xtick_rotation)
                label.set_ha("right" if xtick_rotation else "center")
            ax.tick_params(labelsize=tick_label_fontsize)

            if grid:
                ax.grid(True, axis="y", alpha=grid_alpha, linestyle="--", zorder=0)

            ax.spines["top"].set_visible(show_top_right_axes)
            ax.spines["right"].set_visible(show_top_right_axes)

            if (i == nrows - 1) and (not single_x_label):
                ax.set_xlabel(x_label_use, fontsize=axis_label_fontsize)
            if (j == 0) and (not single_y_label):
                ax.set_ylabel(y_label_use, fontsize=axis_label_fontsize)

    col_labs = [f"{lv}" for lv in panel_levels_use] if panel_var is not None else None
    row_labs = [f"{lv}" for lv in facet_levels_use] if facet_var is not None else None
    _add_strips_mm(
        fig,
        axes,
        col_labels=col_labs,
        row_labels=row_labs,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
    )

    if transparent:
        _ensure_strip_background_opaque(fig)

    _draw_global_labels_and_title(
        fig,
        layout,
        x_label=x_label_use,
        y_label=y_label_use,
        axis_label_fontsize=axis_label_fontsize,
        title=title,
        title_fontsize=title_fontsize,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        font_family=font_family,
    )

    if line_var is not None and legend_loc != "none":
        if legend_ncol is None:
            legend_ncol = 1
        handles = [
            Line2D(
                [0],
                [0],
                color=line_cmap.get(level, "black"),
                lw=line_width,
                marker=marker,
                markersize=marker_size,
                label=str(level),
            )
            for level in line_levels_use
        ]
        labels = [str(level) for level in line_levels_use]
        inside_map = not _is_outside_legend_loc(legend_loc)
        _place_legend(
            fig,
            axes[0, 0],
            handles,
            labels,
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            legend_ncol=legend_ncol,
            legend_framealpha=legend_framealpha,
            inside_map=inside_map,
        )

    return fig


def plot_donut_grid_df(
    df: pd.DataFrame,
    *,
    category_var: str,
    value_col: str,
    panel_var: Optional[str] = None,
    category_levels: Optional[List] = None,
    panel_levels: Optional[List] = None,
    palette: Union[str, List, Dict] = "tab10",
    boxsize: Tuple[float, float] = (45.0, 45.0),
    panel_gap: Tuple[float, float] = (6.0, 6.0),
    ncols: Optional[int] = None,
    ring_width: float = 0.35,
    startangle: float = 90.0,
    counterclock: bool = False,
    label_format: Optional[str] = "{category} {count:g} ({percent:.1f}%)",
    label_distance: float = 1.10,
    label_fontsize: float = 6.0,
    center_format: Optional[str] = "N = {total:g}",
    center_fontsize: float = 7.0,
    label_top: Optional[List[str]] = None,
    panel_label_fontsize: float = 7.0,
    legend_loc: str = "outside_bottom",
    legend_ncol: int = 1,
    legend_fontsize: float = 6.0,
    legend_labels: Optional[Dict] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    transparent: bool = False,
    dpi: int = 600,
) -> plt.Figure:
    """Render one nonnegative count per panel/category as a donut grid.

    Explicit levels must include all observations. Missing categories are zero
    and remain in the legend, without wedges or labels. Each panel must have a
    positive total. The input table is neither filtered nor modified.

    Label templates accept category, count, percent, and total; center templates
    accept total. None hides either text layer or the optional panel headings.
    Legend locations are the four outside_* positions or none. legend_labels
    optionally maps every category to display text without changing wedge labels.

    boxsize is the inner axes size in mm. Measured labels add space around each
    axes; panel_gap separates those annotated panels. The canvas also reserves
    the legend's measured size, preserving axes dimensions at export.
    """
    _validate_distinct_plot_roles(
        category_var=category_var, value_col=value_col, panel_var=panel_var
    )
    identity = [category_var] if panel_var is None else [panel_var, category_var]
    required = identity + [value_col]
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise KeyError(f"Donut table is missing columns: {missing}")
    plotted = df[required].copy()
    if plotted.empty or plotted[identity].isna().any().any():
        raise ValueError("Donut categories and panels must be nonempty and nonnull.")
    if plotted.duplicated(identity).any():
        raise ValueError("Donut table contains duplicate panel/category cells.")
    plotted[value_col] = pd.to_numeric(plotted[value_col], errors="raise")
    values = plotted[value_col].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Donut counts must be finite and nonnegative.")

    categories = _ordered_levels(plotted[category_var], category_levels)
    panels = (
        [None]
        if panel_var is None
        else _ordered_levels(plotted[panel_var], panel_levels)
    )
    for column, levels in [(category_var, categories), (panel_var, panels)]:
        if column is not None and (
            len(levels) != len(set(levels)) or not set(plotted[column]).issubset(levels)
        ):
            raise ValueError(f"Donut levels must uniquely cover column {column!r}.")
    if label_top is not None and len(label_top) != len(panels):
        raise ValueError("label_top must provide one heading per donut panel.")
    if ncols is not None and (
        isinstance(ncols, bool) or not isinstance(ncols, int) or ncols < 1
    ):
        raise ValueError("ncols must be a positive integer.")
    if not 0 < ring_width <= 1:
        raise ValueError("ring_width must be greater than zero and at most one.")
    gap = np.asarray(panel_gap, dtype=float)
    if gap.shape != (2,) or not np.isfinite(gap).all() or (gap < 0).any():
        raise ValueError("panel_gap must contain two finite nonnegative mm values.")
    if legend_loc not in {
        "none",
        "outside_bottom",
        "outside_top",
        "outside_left",
        "outside_right",
    }:
        raise ValueError("legend_loc must be an outside_* position or none.")
    box_w_mm, box_h_mm = _validate_boxsize(boxsize)
    counts = []
    for panel in panels:
        group = plotted if panel_var is None else plotted[plotted[panel_var] == panel]
        vector = (
            group.set_index(category_var)[value_col]
            .reindex(categories, fill_value=0)
            .to_numpy(dtype=float)
        )
        if vector.sum() <= 0:
            raise ValueError(f"Donut panel {panel!r} must have a positive total.")
        counts.append(vector)

    colors = _build_color_map(categories, palette)
    columns = min(ncols or len(panels), len(panels))
    rows = (len(panels) + columns - 1) // columns
    fig, axes, _ = _init_box_figure(
        nrows=rows,
        ncols=columns,
        boxsize_mm=(box_w_mm, box_h_mm),
        panel_gap_mm=panel_gap,
        include_global_label_margins=False,
        font_family=font_family,
        dpi=dpi,
        transparent=transparent,
    )
    fig.patch.set_facecolor("white")
    panel_axes = list(axes.flat)[: len(panels)]
    for index, (ax, vector) in enumerate(zip(panel_axes, counts)):
        total = float(vector.sum())
        visible = np.flatnonzero(vector > 0)
        labels = (
            None
            if label_format is None
            else [
                label_format.format(
                    category=categories[i],
                    count=vector[i],
                    percent=100 * vector[i] / total,
                    total=total,
                )
                for i in visible
            ]
        )
        ax.pie(
            vector[visible],
            colors=[colors[categories[i]] for i in visible],
            labels=labels,
            startangle=startangle,
            counterclock=counterclock,
            labeldistance=label_distance,
            wedgeprops={"width": ring_width, "edgecolor": "white", "linewidth": 0.6},
            textprops={"fontsize": label_fontsize, "color": "black"},
        )
        scale = min(box_w_mm, box_h_mm)
        ax.set_xlim(-1.25 * box_w_mm / scale, 1.25 * box_w_mm / scale)
        ax.set_ylim(-1.25 * box_h_mm / scale, 1.25 * box_h_mm / scale)
        if center_format is not None:
            ax.text(
                0,
                0,
                center_format.format(total=total),
                ha="center",
                va="center",
                fontsize=center_fontsize,
                color="black",
            )
        if label_top is not None:
            ax.text(
                0.5,
                1.03,
                label_top[index],
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=panel_label_fontsize,
                color="black",
            )
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)

    legend = None
    if legend_loc != "none":
        _place_legend(
            fig,
            panel_axes[0],
            [
                Rectangle((0, 0), 1, 1, facecolor=colors[c], edgecolor="none")
                for c in categories
            ],
            [str(c) if legend_labels is None else legend_labels[c] for c in categories],
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            legend_ncol=legend_ncol,
            legend_framealpha=0,
            inside_map=False,
        )
        legend = fig.legends[0]
        legend.set_frame_on(False)
        legend.borderaxespad = 0

    # Measure outside artists once, then place axes without changing their mm size.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overhangs = []
    for ax in panel_axes:
        box = ax.get_window_extent(renderer)
        tight = ax.get_tightbbox(renderer)
        overhangs.append(
            [
                max(0, box.x0 - tight.x0),
                max(0, tight.x1 - box.x1),
                max(0, box.y0 - tight.y0),
                max(0, tight.y1 - box.y1),
            ]
        )
    left, right, bottom, top = np.max(overhangs, axis=0) / dpi
    box_w, box_h = _tuple_mm_to_in((box_w_mm, box_h_mm))
    gap_x, gap_y = gap / MM_PER_INCH
    grid_w = columns * (box_w + left + right) + (columns - 1) * gap_x
    grid_h = rows * (box_h + bottom + top) + (rows - 1) * gap_y
    padding, legend_gap = _mm_to_in(0.75), _mm_to_in(2)
    figure_w, figure_h = grid_w + 2 * padding, grid_h + 2 * padding
    grid_left, grid_bottom = padding, padding
    if legend is not None:
        bounds = legend.get_window_extent(renderer)
        legend_w, legend_h = bounds.width / dpi, bounds.height / dpi
        if legend_loc in {"outside_top", "outside_bottom"}:
            figure_w = max(grid_w, legend_w) + 2 * padding
            figure_h += legend_h + legend_gap
            grid_left = (figure_w - grid_w) / 2
            if legend_loc == "outside_bottom":
                grid_bottom += legend_h + legend_gap
            anchor = (
                0.5,
                (
                    padding / figure_h
                    if legend_loc == "outside_bottom"
                    else 1 - padding / figure_h
                ),
            )
        else:
            figure_w += legend_w + legend_gap
            figure_h = max(grid_h, legend_h) + 2 * padding
            grid_bottom = (figure_h - grid_h) / 2
            if legend_loc == "outside_left":
                grid_left += legend_w + legend_gap
            anchor = (
                (
                    (padding + legend_w) / figure_w
                    if legend_loc == "outside_left"
                    else 1 - (padding + legend_w) / figure_w
                ),
                0.5,
            )

    fig.set_size_inches(figure_w, figure_h, forward=True)
    _place_panels_fixed(
        fig,
        axes,
        rect=[
            (grid_left + left) / figure_w,
            (grid_bottom + bottom) / figure_h,
            (grid_left + grid_w - right) / figure_w,
            (grid_bottom + grid_h - top) / figure_h,
        ],
        box_w_in=box_w,
        box_h_in=box_h,
        gap_x_in=gap_x + left + right,
        gap_y_in=gap_y + bottom + top,
    )
    if legend is not None:
        legend.set_bbox_to_anchor(anchor, transform=fig.transFigure)
    return fig


# --------------------- heatmap (DataFrame-in-DataFrame) plotting ---------------------


def _cell_aggregate_xyz(df_list: List[pd.DataFrame], mode: str = "mean"):
    """
    Aggregate a list of 2D DataFrames onto a common numeric X/Y grid.

    Each df in df_list is expected to be a DataFrame with:
      - columns representing x coordinates
      - index representing y coordinates
      - values are Z

    Returns:
        X (2D), Y (2D), Z (2D), x_coords (1D), y_coords (1D)

    Notes:
        Missing points are left as NaN before aggregation.
    """
    if len(df_list) == 0:
        return None, None, None, None, None

    x_set = set()
    y_set = set()
    for d in df_list:
        if not isinstance(d, pd.DataFrame) or d.empty:
            continue
        x_set.update(list(d.columns))
        y_set.update(list(d.index))

    if len(x_set) == 0 or len(y_set) == 0:
        return None, None, None, None, None

    x_labels = sorted(list(x_set), key=lambda v: _to_numeric_index([v])[0])
    y_labels = sorted(list(y_set), key=lambda v: _to_numeric_index([v])[0])

    x_coords = _to_numeric_index(x_labels)
    y_coords = _to_numeric_index(y_labels)

    X, Y = np.meshgrid(x_coords, y_coords)

    stack = []
    for d in df_list:
        if not isinstance(d, pd.DataFrame) or d.empty:
            continue
        # Reindex to full grid
        dd = d.reindex(index=y_labels, columns=x_labels)
        stack.append(dd.to_numpy(dtype=float))

    if len(stack) == 0:
        return None, None, None, None, None

    A = np.stack(stack, axis=0)
    if mode == "median":
        Z = np.nanmedian(A, axis=0)
    else:
        Z = np.nanmean(A, axis=0)

    return X, Y, Z, x_coords, y_coords


def _compute_global_vrange(
    cells: List[np.ndarray], vmin: Optional[float], vmax: Optional[float], vmode: str
):
    """Compute vmin/vmax from a list of Z arrays."""
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)

    vals = []
    for z in cells:
        if z is None:
            continue
        f = z[np.isfinite(z)]
        if f.size:
            vals.append(f)
    if not vals:
        vmin0, vmax0 = 0.0, 1.0
    else:
        allv = np.concatenate(vals)
        vmin0, vmax0 = float(np.min(allv)), float(np.max(allv))

    if vmode == "sym":
        m = max(abs(vmin0), abs(vmax0))
        return -m, m

    return vmin0, vmax0


def _plot_interaction_df_grid(
    *,
    df: pd.DataFrame,
    value_col: str,
    mode: str,
    col_var: Optional[str],
    row_var: Optional[str],
    col_levels: Optional[List],
    row_levels: Optional[List],
    x_label: Optional[str],
    single_x_label: bool,
    y_label: Optional[str],
    single_y_label: bool,
    x_limits: Optional[Tuple[float, float]],
    y_limits: Optional[Tuple[float, float]],
    x_log: bool,
    y_log: bool,
    title: Optional[str],
    font_family: str,
    cmap: Union[str, plt.Colormap],
    vmin: Optional[float],
    vmax: Optional[float],
    vmode: str,
    dpi: int,
    grid: bool,
    grid_alpha: float,
    # geometry
    boxsize: Tuple[float, float],
    panel_gap: Tuple[float, float],
    include_global_label_margins: bool,
    x_label_offset_mm: float,
    y_label_offset_mm: float,
    cbar_label_offset_mm: float,
    # refs/shadows
    vertical_lines: Optional[Union[List, np.ndarray]],
    vline_color: str,
    vline_style: str,
    vline_width: float,
    vline_alpha: float,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]],
    horizontal_lines: Optional[Union[List, np.ndarray]],
    hline_color: str,
    hline_style: str,
    hline_width: float,
    hline_alpha: float,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]],
    # strips
    label_fontsize: int,
    label_top_bg_color: str,
    label_right_bg_color: str,
    label_text_color: str,
    label_fontweight: str,
    strip_top_height_mm: float,
    strip_right_width_mm: float,
    strip_pad_mm: float,
    # text sizes
    title_fontsize: int,
    axis_label_fontsize: int,
    tick_label_fontsize: int,
    # colorbar
    colorbar_label: Optional[str],
    colorbar_width_mm: float,
    colorbar_pad_mm: float,
    # annotation
    annotate_input_vars: bool,
    input_vars_box_loc: str,
    input_vars_fontsize: int,
    input_vars_facecolor: str,
    input_vars_text_color: str,
    # transparent
    transparent: bool,
) -> plt.Figure:
    """Core heatmap grid plotter where each row holds a 2D DataFrame."""
    _configure_plot_fonts(font_family)
    dfx = df.copy()

    if col_var is not None:
        if col_levels is None:
            col_levels = _ordered_levels(dfx[col_var], None)
    else:
        col_levels = [None]

    if row_var is not None:
        if row_levels is None:
            row_levels = _ordered_levels(dfx[row_var], None)
    else:
        row_levels = [None]

    nrows = len(row_levels)
    ncols = len(col_levels)

    top_h = strip_top_height_mm if col_var is not None else 0.0
    right_w = strip_right_width_mm if row_var is not None else 0.0

    fig, axes, layout = _init_box_figure(
        nrows=nrows,
        ncols=ncols,
        boxsize_mm=boxsize,
        panel_gap_mm=panel_gap,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        colorbar_width_mm=colorbar_width_mm,
        colorbar_pad_mm=colorbar_pad_mm,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        axis_label_fontsize=axis_label_fontsize,
        include_global_label_margins=include_global_label_margins,
        x_label_text=(x_label or "x"),
        y_label_text=(y_label or value_col),
        colorbar_label_text=colorbar_label,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        cbar_label_offset_mm=cbar_label_offset_mm,
        dpi=dpi,
        font_family=font_family,
        sharex=True,
        sharey=True,
        transparent=transparent,
    )

    # Pre-compute Z arrays for vmin/vmax
    Z_cells: List[np.ndarray] = []
    cell_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for i, row_lv in enumerate(row_levels):
        for j, col_lv in enumerate(col_levels):
            cell = dfx
            if col_var is not None and col_lv is not None:
                cell = cell[cell[col_var] == col_lv]
            if row_var is not None and row_lv is not None:
                cell = cell[cell[row_var] == row_lv]

            df_list = [
                d for d in cell[value_col].tolist() if isinstance(d, pd.DataFrame)
            ]
            X, Y, Z, _, _ = _cell_aggregate_xyz(df_list, mode=mode)
            if Z is not None:
                Z_cells.append(Z)
                cell_cache[(i, j)] = (X, Y, Z)

    vmin_use, vmax_use = _compute_global_vrange(
        Z_cells, vmin=vmin, vmax=vmax, vmode=vmode
    )

    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    norm = mpl.colors.Normalize(vmin=vmin_use, vmax=vmax_use)

    last_mappable = None

    for i, row_lv in enumerate(row_levels):
        for j, col_lv in enumerate(col_levels):
            ax = axes[i, j]

            if vertical_shadows:
                for (x0, x1), c in vertical_shadows.items():
                    _safe_axvspan(ax, x0, x1, color=_rgba_color(c), zorder=2)
            if horizontal_shadows:
                for (y0, y1), c in horizontal_shadows.items():
                    _safe_axhspan(ax, y0, y1, color=_rgba_color(c), zorder=2)

            if (i, j) in cell_cache:
                X, Y, Z = cell_cache[(i, j)]
                m = ax.pcolormesh(
                    X, Y, Z, shading="auto", cmap=cmap_obj, norm=norm, zorder=1
                )
                last_mappable = m

            if vertical_lines is not None:
                for xv in vertical_lines:
                    ax.axvline(
                        x=xv,
                        color=vline_color,
                        linestyle=vline_style,
                        linewidth=vline_width,
                        alpha=vline_alpha,
                        zorder=3,
                    )

            if horizontal_lines is not None:
                for yv in horizontal_lines:
                    ax.axhline(
                        y=yv,
                        color=hline_color,
                        linestyle=hline_style,
                        linewidth=hline_width,
                        alpha=hline_alpha,
                        zorder=3,
                    )

            if x_log:
                ax.set_xscale("log")
            if y_log:
                ax.set_yscale("log")
            if x_limits:
                ax.set_xlim(*x_limits)
            if y_limits:
                ax.set_ylim(*y_limits)

            if i == nrows - 1:
                ax.set_xlabel(
                    "" if single_x_label else (x_label or "x"),
                    fontsize=axis_label_fontsize,
                )
            if j == 0:
                ax.set_ylabel(
                    "" if single_y_label else (y_label or value_col),
                    fontsize=axis_label_fontsize,
                )

            ax.tick_params(labelsize=tick_label_fontsize)
            if grid:
                ax.grid(True, alpha=grid_alpha, linestyle="--", zorder=4)

            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)

    # strips
    col_labs = [f"{lv}" for lv in col_levels] if col_var is not None else None
    row_labs = [f"{lv}" for lv in row_levels] if row_var is not None else None
    right_edge = _add_strips_mm(
        fig,
        axes,
        col_labels=col_labs,
        row_labels=row_labs,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
    )

    if transparent:
        _ensure_strip_background_opaque(fig)

    _draw_global_labels_and_title(
        fig,
        layout,
        x_label=(x_label or "x"),
        y_label=(y_label or value_col),
        axis_label_fontsize=axis_label_fontsize,
        title=title,
        title_fontsize=title_fontsize,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        font_family=font_family,
    )

    if annotate_input_vars:
        mapping_text = f"Heatmap: {value_col}  |  Columns: {col_var or '-'}  |  Rows: {row_var or '-'}  |  agg: {mode}"
        loc_map = {
            "upper left": (0.01, 0.995, "top", "left"),
            "upper right": (0.99, 0.995, "top", "right"),
            "lower left": (0.01, 0.01, "bottom", "left"),
            "lower right": (0.99, 0.01, "bottom", "right"),
        }
        x0, y0, va, ha = loc_map.get(input_vars_box_loc, (0.01, 0.995, "top", "left"))
        fig.text(
            x0,
            y0,
            mapping_text,
            ha=ha,
            va=va,
            fontsize=input_vars_fontsize,
            color=input_vars_text_color,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=input_vars_facecolor,
                edgecolor="none",
                alpha=0.9,
            ),
        )

    # colorbar
    if last_mappable is not None and colorbar_width_mm > 0:
        _add_global_colorbar(
            fig,
            layout,
            mappable=last_mappable,
            right_edge=float(right_edge),
            colorbar_label=colorbar_label,
            tick_label_fontsize=tick_label_fontsize,
            axis_label_fontsize=axis_label_fontsize,
            font_family=font_family,
        )

    return fig


def plot_triple_interaction_df(
    df: pd.DataFrame,
    panel_var: str = "region",
    facet_var: str = "lat",
    panel_levels: Optional[List] = None,
    facet_levels: Optional[List] = None,
    value_col: str = "value",
    mode: str = "mean",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    single_x_label: bool = True,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    cmap: Union[str, plt.Colormap] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmode: str = "auto",
    dpi: int = 300,
    grid: bool = False,
    grid_alpha: float = 0.3,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (3.0, 3.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    cbar_label_offset_mm: float = 1.5,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "white",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.8,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "white",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.8,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    label_fontsize: int = 16,
    label_top_bg_color: str = "lightgray",
    label_right_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_right_width_mm: float = 3.0,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    colorbar_label: Optional[str] = None,
    colorbar_width_mm: float = 3.0,
    colorbar_pad_mm: float = 1.5,
    annotate_input_vars: bool = False,
    input_vars_box_loc: str = "upper left",
    input_vars_fontsize: int = 10,
    input_vars_facecolor: str = "#f0f0f0",
    input_vars_text_color: str = "black",
    transparent: bool = False,
) -> plt.Figure:
    """Grid of heatmaps (panel × facet)."""
    return _plot_interaction_df_grid(
        df=df,
        value_col=value_col,
        mode=mode,
        col_var=panel_var,
        row_var=facet_var,
        col_levels=panel_levels,
        row_levels=facet_levels,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        vmode=vmode,
        dpi=dpi,
        grid=grid,
        grid_alpha=grid_alpha,
        boxsize=boxsize,
        panel_gap=panel_gap,
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        cbar_label_offset_mm=cbar_label_offset_mm,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=strip_right_width_mm,
        strip_pad_mm=strip_pad_mm,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        colorbar_label=colorbar_label,
        colorbar_width_mm=colorbar_width_mm,
        colorbar_pad_mm=colorbar_pad_mm,
        annotate_input_vars=annotate_input_vars,
        input_vars_box_loc=input_vars_box_loc,
        input_vars_fontsize=input_vars_fontsize,
        input_vars_facecolor=input_vars_facecolor,
        input_vars_text_color=input_vars_text_color,
        transparent=transparent,
    )


def plot_double_interaction_df(
    df: pd.DataFrame,
    panel_var: str = "region",
    panel_levels: Optional[List] = None,
    value_col: str = "value",
    mode: str = "mean",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    single_x_label: bool = True,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    cmap: Union[str, plt.Colormap] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmode: str = "auto",
    dpi: int = 300,
    grid: bool = False,
    grid_alpha: float = 0.3,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (3.0, 3.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    cbar_label_offset_mm: float = 1.5,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "white",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.8,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "white",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.8,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    label_fontsize: int = 16,
    label_top_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    colorbar_label: Optional[str] = None,
    colorbar_width_mm: float = 3.0,
    colorbar_pad_mm: float = 1.5,
    annotate_input_vars: bool = False,
    input_vars_box_loc: str = "upper left",
    input_vars_fontsize: int = 10,
    input_vars_facecolor: str = "#f0f0f0",
    input_vars_text_color: str = "black",
    transparent: bool = False,
) -> plt.Figure:
    """Side-by-side heatmaps (panel only)."""
    return _plot_interaction_df_grid(
        df=df,
        value_col=value_col,
        mode=mode,
        col_var=panel_var,
        row_var=None,
        col_levels=panel_levels,
        row_levels=None,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        vmode=vmode,
        dpi=dpi,
        grid=grid,
        grid_alpha=grid_alpha,
        boxsize=boxsize,
        panel_gap=panel_gap,
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        cbar_label_offset_mm=cbar_label_offset_mm,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_top_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=0.0,
        strip_pad_mm=strip_pad_mm,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        colorbar_label=colorbar_label,
        colorbar_width_mm=colorbar_width_mm,
        colorbar_pad_mm=colorbar_pad_mm,
        annotate_input_vars=annotate_input_vars,
        input_vars_box_loc=input_vars_box_loc,
        input_vars_fontsize=input_vars_fontsize,
        input_vars_facecolor=input_vars_facecolor,
        input_vars_text_color=input_vars_text_color,
        transparent=transparent,
    )


def plot_single_effect_df(
    df: pd.DataFrame,
    value_col: str = "value",
    mode: str = "mean",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    single_x_label: bool = True,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    cmap: Union[str, plt.Colormap] = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmode: str = "auto",
    dpi: int = 300,
    grid: bool = False,
    grid_alpha: float = 0.3,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    cbar_label_offset_mm: float = 1.5,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "white",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.8,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "white",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.8,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    colorbar_label: Optional[str] = None,
    colorbar_width_mm: float = 3.0,
    colorbar_pad_mm: float = 1.5,
    annotate_input_vars: bool = False,
    input_vars_box_loc: str = "upper left",
    input_vars_fontsize: int = 10,
    input_vars_facecolor: str = "#f0f0f0",
    input_vars_text_color: str = "black",
    transparent: bool = False,
) -> plt.Figure:
    """Single heatmap (no faceting)."""
    return _plot_interaction_df_grid(
        df=df,
        value_col=value_col,
        mode=mode,
        col_var=None,
        row_var=None,
        col_levels=None,
        row_levels=None,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        vmode=vmode,
        dpi=dpi,
        grid=grid,
        grid_alpha=grid_alpha,
        boxsize=boxsize,
        panel_gap=(0.0, 0.0),
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        cbar_label_offset_mm=cbar_label_offset_mm,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=16,
        label_top_bg_color="lightgray",
        label_right_bg_color="lightgray",
        label_text_color="black",
        label_fontweight="normal",
        strip_top_height_mm=0.0,
        strip_right_width_mm=0.0,
        strip_pad_mm=0.0,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        colorbar_label=colorbar_label,
        colorbar_width_mm=colorbar_width_mm,
        colorbar_pad_mm=colorbar_pad_mm,
        annotate_input_vars=annotate_input_vars,
        input_vars_box_loc=input_vars_box_loc,
        input_vars_fontsize=input_vars_fontsize,
        input_vars_facecolor=input_vars_facecolor,
        input_vars_text_color=input_vars_text_color,
        transparent=transparent,
    )


# --------------------- fit plotting ---------------------


def _pick_p_column(slope_df: pd.DataFrame, preferred: str) -> Optional[str]:
    """Pick a p-value column name from slope_df with p_tukey as the first choice."""
    if slope_df is None or slope_df.empty:
        return None

    candidates = _candidate_p_value_columns(preferred=preferred)

    for cand in candidates:
        if cand not in slope_df.columns:
            continue
        v = pd.to_numeric(slope_df[cand], errors="coerce").astype(float).to_numpy()
        if np.any(np.isfinite(v)):
            return cand
    return None


def _format_slope_text(
    slope_row: pd.Series,
    *,
    beta_col: str,
    lower_col: str,
    upper_col: str,
    p_col: Optional[str],
    text_fmt: str,
) -> str:
    """Format slope annotation string."""
    beta = float(slope_row.get(beta_col, np.nan))
    lower = float(slope_row.get(lower_col, np.nan))
    upper = float(slope_row.get(upper_col, np.nan))

    # Resolve p-value row-wise with strict priority. `p_col` is treated as a preferred name.
    pval = _resolve_p_value_from_row(slope_row, preferred=p_col)
    stars = p_to_stars(pval)

    # Provide both p and q placeholders for convenience
    return text_fmt.format(
        beta=beta, lower=lower, upper=upper, p=pval, q=pval, stars=stars
    )


def _plot_interaction_fit_grid(
    *,
    df: pd.DataFrame,
    curve: pd.DataFrame,
    slope: pd.DataFrame,
    value_col: str,
    x_var: str,
    col_var: Optional[str],
    row_var: Optional[str],
    color_var: Optional[str],
    col_levels: Optional[List],
    row_levels: Optional[List],
    x_label: Optional[str],
    single_x_label: bool,
    y_label: Optional[str],
    single_y_label: bool,
    x_limits: Optional[Tuple[float, float]],
    y_limits: Optional[Tuple[float, float]],
    x_log: bool,
    y_log: bool,
    title: Optional[str],
    font_family: str,
    palette: Union[str, List, Dict],
    jitter_alpha: float,
    jitter_size: float,
    curve_line_width: float,
    curve_line_color: str,
    ribbon_alpha: float,
    show_ribbon: bool,
    grid: bool,
    grid_alpha: float,
    dpi: int,
    seed: int,
    show_top_right_axes: bool,
    vertical_lines: Optional[Union[List, np.ndarray]],
    vline_color: str,
    vline_style: str,
    vline_width: float,
    vline_alpha: float,
    horizontal_lines: Optional[Union[List, np.ndarray]],
    hline_color: str,
    hline_style: str,
    hline_width: float,
    hline_alpha: float,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]],
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]],
    label_fontsize: int,
    label_top_bg_color: str,
    label_right_bg_color: str,
    label_text_color: str,
    label_fontweight: str,
    strip_top_height_mm: float,
    strip_right_width_mm: float,
    strip_pad_mm: float,
    title_fontsize: int,
    axis_label_fontsize: int,
    tick_label_fontsize: int,
    legend_loc: str,
    legend_ncol: Optional[int],
    legend_framealpha: float,
    legend_fontsize: int,
    boxsize: Tuple[float, float],
    panel_gap: Tuple[float, float],
    include_global_label_margins: bool,
    x_label_offset_mm: float,
    y_label_offset_mm: float,
    slope_beta_col: str,
    slope_lower_col: str,
    slope_upper_col: str,
    slope_p_col: str,
    slope_text_fmt: str,
    slope_text_loc: Union[str, Tuple[float, float]],
    slope_text_coord: str,
    slope_text_offset: Tuple[float, float],
    slope_text_ha: Optional[str],
    slope_text_va: Optional[str],
    slope_text_box_alpha: float,
    share_x_axes: bool,
    transparent: bool,
) -> plt.Figure:
    """Core fit grid plotter."""
    _configure_plot_fonts(font_family)

    dfw = df.copy()
    curw = _normalize_emm_ci_columns(curve.copy())
    slw = slope.copy()

    if col_var is not None:
        if col_levels is None:
            col_levels = _ordered_levels(dfw[col_var], None)
    else:
        col_levels = [None]
    if row_var is not None:
        if row_levels is None:
            row_levels = _ordered_levels(dfw[row_var], None)
    else:
        row_levels = [None]

    nrows = len(row_levels)
    ncols = len(col_levels)

    top_h = strip_top_height_mm if col_var is not None else 0.0
    right_w = strip_right_width_mm if row_var is not None else 0.0

    # Color mapping for raw points
    if color_var is None or color_var not in dfw.columns:
        color_levels = ["raw"]
        cmap = {"raw": "gray"}
        dfw["_color_level"] = "raw"
    else:
        color_levels = _ordered_levels(dfw[color_var], None)
        cmap = _build_color_map(color_levels, palette)
        dfw["_color_level"] = dfw[color_var]

    # Auto limits
    if x_limits is None or y_limits is None:
        x_arrays = []
        y_arrays = []
        if x_var in dfw.columns:
            x_arrays.append(dfw[x_var].to_numpy(dtype=float))
        if x_var in curw.columns:
            x_arrays.append(curw[x_var].to_numpy(dtype=float))
        if value_col in dfw.columns:
            y_arrays.append(dfw[value_col].to_numpy(dtype=float))
        if "emmean" in curw.columns:
            y_arrays.append(curw["emmean"].to_numpy(dtype=float))
        if "lower.CL" in curw.columns:
            y_arrays.append(curw["lower.CL"].to_numpy(dtype=float))
        if "upper.CL" in curw.columns:
            y_arrays.append(curw["upper.CL"].to_numpy(dtype=float))

        x_lim_auto, y_lim_auto = _auto_limits(
            x_arrays, y_arrays, x_pad_frac=0.05, y_pad_frac=0.05, y_log=y_log
        )
        if x_limits is None:
            x_limits = x_lim_auto
        if y_limits is None:
            y_limits = y_lim_auto

    fig, axes, layout = _init_box_figure(
        nrows=nrows,
        ncols=ncols,
        boxsize_mm=boxsize,
        panel_gap_mm=panel_gap,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        colorbar_width_mm=0.0,
        colorbar_pad_mm=0.0,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        axis_label_fontsize=axis_label_fontsize,
        include_global_label_margins=include_global_label_margins,
        x_label_text=(x_label or x_var),
        y_label_text=(y_label or value_col),
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        dpi=dpi,
        font_family=font_family,
        sharex=share_x_axes,
        sharey=True,
        transparent=transparent,
    )

    p_col = _pick_p_column(slw, slope_p_col)

    for i, row_lv in enumerate(row_levels):
        for j, col_lv in enumerate(col_levels):
            ax = axes[i, j]

            # shadows
            if vertical_shadows:
                for (x0, x1), c in vertical_shadows.items():
                    _safe_axvspan(ax, x0, x1, color=_rgba_color(c), zorder=0)
            if horizontal_shadows:
                for (y0, y1), c in horizontal_shadows.items():
                    _safe_axhspan(ax, y0, y1, color=_rgba_color(c), zorder=0)

            cell_raw = dfw
            cell_curve = curw
            cell_slope = slw

            if col_var is not None and col_lv is not None and col_var in dfw.columns:
                cell_raw = cell_raw[cell_raw[col_var] == col_lv]
            if row_var is not None and row_lv is not None and row_var in dfw.columns:
                cell_raw = cell_raw[cell_raw[row_var] == row_lv]

            if col_var is not None and col_lv is not None and col_var in curw.columns:
                cell_curve = cell_curve[cell_curve[col_var] == col_lv]
            if row_var is not None and row_lv is not None and row_var in curw.columns:
                cell_curve = cell_curve[cell_curve[row_var] == row_lv]

            if col_var is not None and col_lv is not None and col_var in slw.columns:
                cell_slope = cell_slope[cell_slope[col_var] == col_lv]
            if row_var is not None and row_lv is not None and row_var in slw.columns:
                cell_slope = cell_slope[cell_slope[row_var] == row_lv]

            # raw scatter
            if not cell_raw.empty:
                xs = cell_raw[x_var].to_numpy(dtype=float)
                ys = cell_raw[value_col].to_numpy(dtype=float)
                cols = [
                    cmap.get(lv, "gray") for lv in cell_raw["_color_level"].tolist()
                ]
                ax.scatter(
                    xs,
                    ys,
                    c=cols,
                    alpha=jitter_alpha,
                    s=jitter_size,
                    linewidths=0,
                    zorder=2,
                )

            # curve ribbon and mean
            if (
                not cell_curve.empty
                and x_var in cell_curve.columns
                and "emmean" in cell_curve.columns
            ):
                cell_curve = cell_curve.sort_values(x_var)
                xg = cell_curve[x_var].to_numpy(dtype=float)
                em = cell_curve["emmean"].to_numpy(dtype=float)

                if show_ribbon and {"lower.CL", "upper.CL"}.issubset(
                    cell_curve.columns
                ):
                    lo = cell_curve["lower.CL"].to_numpy(dtype=float)
                    hi = cell_curve["upper.CL"].to_numpy(dtype=float)
                    ax.fill_between(
                        xg,
                        lo,
                        hi,
                        color=curve_line_color,
                        alpha=ribbon_alpha,
                        linewidth=0,
                        zorder=1.5,
                    )

                ax.plot(
                    xg,
                    em,
                    color=curve_line_color,
                    lw=curve_line_width,
                    zorder=3,
                    label="Fit",
                )

            # slope label
            if cell_slope is not None and not cell_slope.empty:
                row0 = cell_slope.iloc[0]
                text = _format_slope_text(
                    row0,
                    beta_col=slope_beta_col,
                    lower_col=slope_lower_col,
                    upper_col=slope_upper_col,
                    p_col=p_col,
                    text_fmt=slope_text_fmt,
                )

                if isinstance(slope_text_loc, tuple):
                    x0, y0 = slope_text_loc
                    coord_alias = str(slope_text_coord).strip().lower()
                    if coord_alias in {"axes", "axes fraction"}:
                        xycoords = "axes fraction"
                    elif coord_alias in {"data"}:
                        xycoords = "data"
                    elif coord_alias in {"figure", "figure fraction"}:
                        xycoords = "figure fraction"
                    else:
                        xycoords = "axes fraction"

                    ha = slope_text_ha or ("left" if x0 <= 0.5 else "right")
                    va = slope_text_va or ("top" if y0 >= 0.5 else "bottom")

                    ax.annotate(
                        text,
                        xy=(x0, y0),
                        xycoords=xycoords,
                        textcoords="offset points",
                        xytext=slope_text_offset,
                        ha=ha,
                        va=va,
                        fontsize=tick_label_fontsize,
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="white",
                            edgecolor="none",
                            alpha=slope_text_box_alpha,
                        ),
                        zorder=10,
                        clip_on=False,
                    )
                else:
                    anchors = {
                        "upper left": dict(x=0.05, y=0.95, ha="left", va="top"),
                        "upper right": dict(x=0.95, y=0.95, ha="right", va="top"),
                        "lower left": dict(x=0.05, y=0.05, ha="left", va="bottom"),
                        "lower right": dict(x=0.95, y=0.05, ha="right", va="bottom"),
                    }
                    an = anchors.get(str(slope_text_loc).lower(), anchors["upper left"])
                    ax.text(
                        an["x"],
                        an["y"],
                        text,
                        transform=ax.transAxes,
                        ha=an["ha"],
                        va=an["va"],
                        fontsize=tick_label_fontsize,
                        bbox=dict(
                            boxstyle="round,pad=0.25",
                            facecolor="white",
                            edgecolor="none",
                            alpha=slope_text_box_alpha,
                        ),
                        zorder=10,
                        clip_on=False,
                    )

            # refs
            if vertical_lines is not None:
                for xv in vertical_lines:
                    ax.axvline(
                        x=xv,
                        color=vline_color,
                        linestyle=vline_style,
                        linewidth=vline_width,
                        alpha=vline_alpha,
                        zorder=4,
                    )

            if horizontal_lines is not None:
                for yv in horizontal_lines:
                    ax.axhline(
                        y=yv,
                        color=hline_color,
                        linestyle=hline_style,
                        linewidth=hline_width,
                        alpha=hline_alpha,
                        zorder=4,
                    )

            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)

            if x_log:
                ax.set_xscale("log")
            if y_log:
                ax.set_yscale("log")

            ax.tick_params(labelsize=tick_label_fontsize)

            if grid:
                ax.grid(True, alpha=grid_alpha, linestyle="--", zorder=0)

            ax.spines["top"].set_visible(show_top_right_axes)
            ax.spines["right"].set_visible(show_top_right_axes)

            if (i == nrows - 1) and (not single_x_label):
                ax.set_xlabel(x_label or x_var, fontsize=axis_label_fontsize)
            if (j == 0) and (not single_y_label):
                ax.set_ylabel(y_label or value_col, fontsize=axis_label_fontsize)

    # strips
    col_labs = [f"{lv}" for lv in col_levels] if col_var is not None else None
    row_labs = [f"{lv}" for lv in row_levels] if row_var is not None else None
    _ = _add_strips_mm(
        fig,
        axes,
        col_labels=col_labs,
        row_labels=row_labs,
        strip_top_height_mm=top_h,
        strip_right_width_mm=right_w,
        strip_pad_mm=strip_pad_mm,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
    )

    if transparent:
        _ensure_strip_background_opaque(fig)

    _draw_global_labels_and_title(
        fig,
        layout,
        x_label=(x_label or x_var),
        y_label=(y_label or value_col),
        axis_label_fontsize=axis_label_fontsize,
        title=title,
        title_fontsize=title_fontsize,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        font_family=font_family,
    )

    if legend_loc != "none":
        handles = [
            Line2D([0], [0], color=curve_line_color, lw=curve_line_width, label="Fit"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="gray",
                alpha=jitter_alpha,
                markersize=float(np.sqrt(jitter_size)),
                label=f"Raw ({color_var or 'raw'})",
            ),
        ]
        if legend_ncol is None:
            legend_ncol = 1

        inside_map = not _is_outside_legend_loc(legend_loc)
        _place_legend(
            fig,
            axes[0, 0],
            handles,
            [h.get_label() for h in handles],
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            legend_ncol=legend_ncol,
            legend_framealpha=legend_framealpha,
            inside_map=inside_map,
        )

    return fig


def plot_triple_interaction_fit(
    df: pd.DataFrame,
    curve: pd.DataFrame,
    slope: pd.DataFrame,
    value_col: str = "Value",
    x_var: str = "LogI_c",
    panel_var: str = "Region",
    facet_var: str = "Stimulus",
    color_var: Optional[str] = "Subject",
    panel_levels: Optional[List] = None,
    facet_levels: Optional[List] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    palette: Union[str, List, Dict] = "viridis",
    jitter_alpha: float = 0.35,
    jitter_size: float = 10.0,
    curve_line_width: float = 2.0,
    curve_line_color: str = "black",
    ribbon_alpha: float = 0.20,
    show_ribbon: bool = True,
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    seed: int = 1,
    show_top_right_axes: bool = True,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    label_fontsize: int = 14,
    label_top_bg_color: str = "lightgray",
    label_right_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_right_width_mm: float = 3.0,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 18,
    axis_label_fontsize: int = 14,
    tick_label_fontsize: int = 12,
    legend_loc: str = "none",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (3.0, 3.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    slope_beta_col: str = "slope",
    slope_lower_col: str = "lower",
    slope_upper_col: str = "upper",
    slope_p_col: str = "q",
    slope_text_fmt: str = "β = {beta:.3f}\\nq = {q:.3f} ({stars})",
    slope_text_loc: Union[str, Tuple[float, float]] = "upper left",
    slope_text_coord: str = "axes fraction",
    slope_text_offset: Tuple[float, float] = (0.0, 0.0),
    slope_text_ha: Optional[str] = None,
    slope_text_va: Optional[str] = None,
    slope_text_box_alpha: float = 0.9,
    share_x_axes: bool = True,
    transparent: bool = False,
) -> plt.Figure:
    """Triple interaction fit plot (panel × facet)."""
    return _plot_interaction_fit_grid(
        df=df,
        curve=curve,
        slope=slope,
        value_col=value_col,
        x_var=x_var,
        col_var=panel_var,
        row_var=facet_var,
        color_var=color_var,
        col_levels=panel_levels,
        row_levels=facet_levels,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        palette=palette,
        jitter_alpha=jitter_alpha,
        jitter_size=jitter_size,
        curve_line_width=curve_line_width,
        curve_line_color=curve_line_color,
        ribbon_alpha=ribbon_alpha,
        show_ribbon=show_ribbon,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        seed=seed,
        show_top_right_axes=show_top_right_axes,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=strip_right_width_mm,
        strip_pad_mm=strip_pad_mm,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=panel_gap,
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        slope_beta_col=slope_beta_col,
        slope_lower_col=slope_lower_col,
        slope_upper_col=slope_upper_col,
        slope_p_col=slope_p_col,
        slope_text_fmt=slope_text_fmt,
        slope_text_loc=slope_text_loc,
        slope_text_coord=slope_text_coord,
        slope_text_offset=slope_text_offset,
        slope_text_ha=slope_text_ha,
        slope_text_va=slope_text_va,
        slope_text_box_alpha=slope_text_box_alpha,
        share_x_axes=share_x_axes,
        transparent=transparent,
    )


def plot_double_interaction_fit(
    df: pd.DataFrame,
    curve: pd.DataFrame,
    slope: pd.DataFrame,
    value_col: str = "Value",
    x_var: str = "LogI_c",
    panel_var: str = "Region",
    color_var: Optional[str] = "Subject",
    panel_levels: Optional[List] = None,
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    palette: Union[str, List, Dict] = "viridis",
    jitter_alpha: float = 0.35,
    jitter_size: float = 10.0,
    curve_line_width: float = 2.0,
    curve_line_color: str = "black",
    ribbon_alpha: float = 0.20,
    show_ribbon: bool = True,
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    seed: int = 1,
    show_top_right_axes: bool = True,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    label_fontsize: int = 14,
    label_top_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",
    strip_top_height_mm: float = 2.5,
    strip_pad_mm: float = 0.3,
    title_fontsize: int = 18,
    axis_label_fontsize: int = 14,
    tick_label_fontsize: int = 12,
    legend_loc: str = "none",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    panel_gap: Tuple[float, float] = (3.0, 3.0),
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    slope_beta_col: str = "slope",
    slope_lower_col: str = "lower",
    slope_upper_col: str = "upper",
    slope_p_col: str = "q",
    slope_text_fmt: str = "β = {beta:.3f}\\nq = {q:.3f} ({stars})",
    slope_text_loc: Union[str, Tuple[float, float]] = "upper left",
    slope_text_coord: str = "axes fraction",
    slope_text_offset: Tuple[float, float] = (0.0, 0.0),
    slope_text_ha: Optional[str] = None,
    slope_text_va: Optional[str] = None,
    slope_text_box_alpha: float = 0.9,
    transparent: bool = False,
) -> plt.Figure:
    """Double interaction fit plot (panel only)."""
    return _plot_interaction_fit_grid(
        df=df,
        curve=curve,
        slope=slope,
        value_col=value_col,
        x_var=x_var,
        col_var=panel_var,
        row_var=None,
        color_var=color_var,
        col_levels=panel_levels,
        row_levels=None,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        palette=palette,
        jitter_alpha=jitter_alpha,
        jitter_size=jitter_size,
        curve_line_width=curve_line_width,
        curve_line_color=curve_line_color,
        ribbon_alpha=ribbon_alpha,
        show_ribbon=show_ribbon,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        seed=seed,
        show_top_right_axes=show_top_right_axes,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_top_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=0.0,
        strip_pad_mm=strip_pad_mm,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=panel_gap,
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        slope_beta_col=slope_beta_col,
        slope_lower_col=slope_lower_col,
        slope_upper_col=slope_upper_col,
        slope_p_col=slope_p_col,
        slope_text_fmt=slope_text_fmt,
        slope_text_loc=slope_text_loc,
        slope_text_coord=slope_text_coord,
        slope_text_offset=slope_text_offset,
        slope_text_ha=slope_text_ha,
        slope_text_va=slope_text_va,
        slope_text_box_alpha=slope_text_box_alpha,
        share_x_axes=True,
        transparent=transparent,
    )


def plot_single_effect_fit(
    df: pd.DataFrame,
    curve: pd.DataFrame,
    slope: pd.DataFrame,
    value_col: str = "Value",
    x_var: str = "LogI_c",
    color_var: Optional[str] = "Subject",
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    x_log: bool = False,
    y_log: bool = False,
    title: Optional[str] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    palette: Union[str, List, Dict] = "viridis",
    jitter_alpha: float = 0.35,
    jitter_size: float = 10.0,
    curve_line_width: float = 2.0,
    curve_line_color: str = "black",
    ribbon_alpha: float = 0.20,
    show_ribbon: bool = True,
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 100,
    seed: int = 1,
    show_top_right_axes: bool = True,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    title_fontsize: int = 18,
    axis_label_fontsize: int = 14,
    tick_label_fontsize: int = 12,
    legend_loc: str = "none",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    slope_beta_col: str = "slope",
    slope_lower_col: str = "lower",
    slope_upper_col: str = "upper",
    slope_p_col: str = "q",
    slope_text_fmt: str = "β = {beta:.3f}\\nq = {q:.3f} ({stars})",
    slope_text_loc: Union[str, Tuple[float, float]] = "upper left",
    slope_text_coord: str = "axes fraction",
    slope_text_offset: Tuple[float, float] = (0.0, 0.0),
    slope_text_ha: Optional[str] = None,
    slope_text_va: Optional[str] = None,
    slope_text_box_alpha: float = 0.9,
    transparent: bool = False,
) -> plt.Figure:
    """Single fit plot (no faceting)."""
    return _plot_interaction_fit_grid(
        df=df,
        curve=curve,
        slope=slope,
        value_col=value_col,
        x_var=x_var,
        col_var=None,
        row_var=None,
        color_var=color_var,
        col_levels=None,
        row_levels=None,
        x_label=x_label,
        single_x_label=single_x_label,
        y_label=y_label,
        single_y_label=single_y_label,
        x_limits=x_limits,
        y_limits=y_limits,
        x_log=x_log,
        y_log=y_log,
        title=title,
        font_family=font_family,
        palette=palette,
        jitter_alpha=jitter_alpha,
        jitter_size=jitter_size,
        curve_line_width=curve_line_width,
        curve_line_color=curve_line_color,
        ribbon_alpha=ribbon_alpha,
        show_ribbon=show_ribbon,
        grid=grid,
        grid_alpha=grid_alpha,
        dpi=dpi,
        seed=seed,
        show_top_right_axes=show_top_right_axes,
        vertical_lines=vertical_lines,
        vline_color=vline_color,
        vline_style=vline_style,
        vline_width=vline_width,
        vline_alpha=vline_alpha,
        horizontal_lines=horizontal_lines,
        hline_color=hline_color,
        hline_style=hline_style,
        hline_width=hline_width,
        hline_alpha=hline_alpha,
        vertical_shadows=vertical_shadows,
        horizontal_shadows=horizontal_shadows,
        label_fontsize=14,
        label_top_bg_color="lightgray",
        label_right_bg_color="lightgray",
        label_text_color="black",
        label_fontweight="normal",
        strip_top_height_mm=0.0,
        strip_right_width_mm=0.0,
        strip_pad_mm=0.0,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_loc=legend_loc,
        legend_ncol=legend_ncol,
        legend_framealpha=legend_framealpha,
        legend_fontsize=legend_fontsize,
        boxsize=boxsize,
        panel_gap=(0.0, 0.0),
        include_global_label_margins=include_global_label_margins,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        slope_beta_col=slope_beta_col,
        slope_lower_col=slope_lower_col,
        slope_upper_col=slope_upper_col,
        slope_p_col=slope_p_col,
        slope_text_fmt=slope_text_fmt,
        slope_text_loc=slope_text_loc,
        slope_text_coord=slope_text_coord,
        slope_text_offset=slope_text_offset,
        slope_text_ha=slope_text_ha,
        slope_text_va=slope_text_va,
        slope_text_box_alpha=slope_text_box_alpha,
        transparent=transparent,
    )


def plot_triple_interaction_nifti(
    heat_img: Union[str, nib.Nifti1Image],
    bg_img: Optional[Union[str, nib.Nifti1Image]] = None,
    # Binary masks overlay (second layer)
    masks: Optional[List[Union[str, nib.Nifti1Image]]] = None,
    mask_colors: Optional[List[str]] = None,  # list of '#RRGGBBAA'
    mask_threshold: float = 0.5,
    mask_default_alpha: float = 0.4,
    # Slicing positions (percent along heatmap bbox on each axis)
    percent_step: float = 25.0,
    percent_list: Optional[List[float]] = None,
    facets: List[str] = ("Ax", "Cor", "Sag"),
    # Domain & sampling
    threshold: Optional[float] = 0.0,  # heat <= threshold -> masked
    res_mm: float = 0.2,
    # Optional user ranges (centers may be overridden by "valid heat" center when enforcing global span)
    box_ranges: Optional[
        Dict
    ] = None,  # {('ax','50%'):{'xlim':(..),'ylim':(..)}, 'cor':{..}}
    # Make all panels share the same (Δx, Δy) and match the panel aspect (b/a)
    enforce_global_box_span: bool = True,
    global_box_span_mm: Optional[
        Tuple[float, float]
    ] = None,  # (Δx, Δy); if None -> auto from data
    global_span_mode: str = "max",  # "min" or "max" (recommended "max" to fully cover all slices)
    # Axis labels / visibility
    x_label: Optional[str] = None,
    single_x_label: bool = True,
    y_label: Optional[str] = None,
    single_y_label: bool = True,
    show_x_label: bool = True,
    show_y_label: bool = True,
    show_tick_labels: bool = True,
    # Figure / appearance
    title: Optional[str] = "MNI-space heatmap (panel = percent, facet = plane)",
    cmap_heat: Union[str, plt.Colormap] = "viridis",
    vmin_heat: Optional[float] = None,
    vmax_heat: Optional[float] = None,
    vmode_heat: str = "auto",  # "auto" | "sym"
    heat_alpha: float = 0.95,
    # Background auto-contrast/brightness
    cmap_bg: Union[str, plt.Colormap] = "gray",
    bg_vmin: Optional[float] = None,
    bg_vmax: Optional[float] = None,
    bg_percentile: Tuple[float, float] = (2.0, 98.0),
    bg_gamma: float = 1.0,
    bg_alpha: float = 1.0,
    dpi: int = 300,
    # Deterministic panel geometry (mm)
    boxsize: Tuple[
        float, float
    ] = DEFAULT_BOXSIZE_MM,  # (width_mm, height_mm) per panel
    panel_gap: Tuple[float, float] = (3.0, 3.0),
    label_top_bg_color: str = "lightgray",
    label_right_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontweight: str = "normal",  # <— NEW: 'normal' | 'bold' | numeric
    strip_top_height_mm: float = 3.5,
    strip_right_width_mm: float = 3.5,
    strip_pad_mm: float = 0.35,
    label_fontsize: int = 16,
    axis_label_fontsize: int = 14,
    tick_label_fontsize: int = 10,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    # colorbar (single, global) — semantics identical to your DF plotter
    colorbar_label: Optional[str] = "Heat",
    colorbar_width_mm: float = 3.5,  # mm
    colorbar_pad_mm: float = 1.75,  # mm
    x_label_offset_mm: float = 7.0,
    y_label_offset_mm: float = 7.0,
    cbar_label_offset_mm: float = 7.0,
    transparent: bool = False,
    # Per-panel scale bar (drawn inside axes)
    draw_scale_bar: bool = False,
    scale_bar_length_mm: float = 5.0,  # real-world length along x (MNI mm)
    scale_bar_pos: Tuple[float, float] = (0.9, 0.1),  # axes fraction (x_frac,y_frac)
    scale_bar_color: str = "white",
    scale_bar_alpha: float = 1.0,
    scale_bar_height_mm: float = 0.5,  # thickness in FIGURE mm (not world mm)
    scale_bar_label: Optional[str] = None,  # if None -> "{actual_length} mm"
    scale_bar_label_fontsize: int = 10,
    scale_bar_label_color: str = "white",
    scale_bar_label_position: str = "above",  # "above" or "below"
    scale_bar_label_pad_mm: float = 1.0,  # label vertical pad in FIGURE mm
    scale_bar_auto_shrink: bool = True,  # shrink if bar would overflow the box
    # Figure-level (global) scale bar
    draw_global_scale_bar: bool = True,
    global_scale_bar_length_mm: float = 5.0,  # MNI mm
    global_scale_bar_pos: Tuple[float, float] = (0.9, 0.1),  # figure fraction
    global_scale_bar_color: str = "white",
    global_scale_bar_alpha: float = 1.0,
    global_scale_bar_height_mm: float = 0.5,  # FIGURE mm
    global_scale_bar_label: Optional[str] = None,  # None -> "{actual_length} mm"
    global_scale_bar_label_fontsize: int = 10,
    global_scale_bar_label_color: str = "white",
    global_scale_bar_label_position: str = "below",  # "above" | "below"
    global_scale_bar_label_pad_mm: float = 1.0,  # FIGURE mm
    global_scale_bar_auto_shrink: bool = True,
    global_scale_bar_ref: Optional[
        Tuple[str, str]
    ] = None,  # e.g., ("ax","50%"); default -> bottom-right panel
    global_scale_bar_anchor: str = "right",  # "right" | "center" | "left"
) -> plt.Figure:
    """
    Faceted MNI-space visualization with 3 layers (background -> masks -> heatmap).
    Slices are sampled in WORLD (MNI, mm) space using canonical RAS+ orientation.

    Key guarantees:
      - True axial/coronal/sagittal slicing in world coordinates.
      - All panels share a unified (Δx, Δy) span when `enforce_global_box_span=True`.
      - The unified span is adjusted so that Δy/Δx equals the panel aspect ratio b/a from `boxsize`.
      - Visual metric equality on x vs y via aspect='equal' (1 mm equals 1 mm).
    """

    # NIfTI plotting is an optional feature (nibabel is not a hard dependency).
    if nib is None:  # pragma: no cover
        raise ImportError(
            "plot_triple_interaction_nifti requires nibabel. Install it with `pip install nibabel`."
        )

    _validate_boxsize(boxsize)

    # Convert mm-based layout parameters to the internal *ratio* parameters used below.
    # This keeps the core NIfTI rendering logic intact while exposing a consistent mm API.
    box_w_mm, box_h_mm = float(boxsize[0]), float(boxsize[1])

    strip_top_height_ratio = float(strip_top_height_mm) / box_h_mm
    strip_right_width_ratio = float(strip_right_width_mm) / box_w_mm

    _ref_strip_mm = max(float(strip_top_height_mm), float(strip_right_width_mm), 1e-9)
    strip_pad_frac = float(strip_pad_mm) / _ref_strip_mm

    colorbar_width = float(colorbar_width_mm) / box_w_mm
    colorbar_pad = float(colorbar_pad_mm) / box_w_mm

    x_label_offset_ratio = float(x_label_offset_mm) / box_h_mm
    y_label_offset_ratio = float(y_label_offset_mm) / box_w_mm
    cbar_label_offset_ratio = float(cbar_label_offset_mm) / box_h_mm

    # --------------------------- small helpers ---------------------------

    def _mm_to_in(x: float) -> float:
        return float(x) / 25.4

    def _as_inches(size_mm: Tuple[float, float]) -> Tuple[float, float]:
        return _mm_to_in(size_mm[0]), _mm_to_in(size_mm[1])

    def _ordered_percents(step: float, plist: Optional[List[float]]) -> List[str]:
        if plist is not None:
            ps = sorted([float(p) for p in plist if 0.0 <= float(p) <= 100.0])
        else:
            s = float(step)
            n = int(np.floor(100.0 / s))
            ps = [i * s for i in range(n + 1)]
            if ps[-1] < 100.0 - 1e-9:
                ps.append(100.0)
        return [f"{int(p)}%" if abs(p - round(p)) < 1e-9 else f"{p:.1f}%" for p in ps]

    def _percent_to_t(p_label: str) -> float:
        return float(p_label.strip("%")) / 100.0

    def _as_canonical(img_or_path) -> nib.Nifti1Image:
        im = nib.load(img_or_path) if isinstance(img_or_path, str) else img_or_path
        return nib.as_closest_canonical(im)  # RAS+ orientation

    def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int, int, int]:
        xs, ys, zs = np.where(mask)
        return (
            int(xs.min()),
            int(xs.max()),
            int(ys.min()),
            int(ys.max()),
            int(zs.min()),
            int(zs.max()),
        )

    def _grid_from_ranges(
        xlim: Tuple[float, float], ylim: Tuple[float, float], res: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build an anisotropic grid at isotropic resolution `res` (mm/px)."""
        x0, x1 = float(xlim[0]), float(xlim[1])
        y0, y1 = float(ylim[0]), float(ylim[1])
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid range: xlim={xlim}, ylim={ylim}")
        Lx, Ly = x1 - x0, y1 - y0
        nx = max(2, int(round(Lx / res)) + 1)
        ny = max(2, int(round(Ly / res)) + 1)
        return np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)

    def _sample_slice(
        vol: np.ndarray,
        inv_aff: np.ndarray,
        plane: str,
        fixed_mm: float,
        xlim_mm: Tuple[float, float],
        ylim_mm: Tuple[float, float],
        res: float,
        order: int = 1,
        cval: float = np.nan,
    ) -> Tuple[np.ndarray, List[float]]:
        """Sample a slice in world (mm): returns (image [ny,nx], extent=[x0,x1,y0,y1])."""
        x_mm, y_mm = _grid_from_ranges(xlim_mm, ylim_mm, res)
        X, Y = np.meshgrid(x_mm, y_mm)
        if plane == "Ax":  # z fixed -> axes (x,y)
            Z = np.full_like(X, fixed_mm)
            world = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        elif plane == "Cor":  # y fixed -> axes (x,z)
            Z = Y
            Yfix = np.full_like(X, fixed_mm)
            world = np.stack([X, Yfix, Z], axis=-1).reshape(-1, 3)
        elif plane == "Sag":  # x fixed -> axes (y,z)
            Z = Y
            Xfix = np.full_like(X, fixed_mm)
            world = np.stack([Xfix, X, Z], axis=-1).reshape(-1, 3)
        else:
            raise ValueError("plane must be one of {'Ax','Cor','Sag'}")
        ones = np.ones((world.shape[0], 1), dtype=float)
        hom = np.concatenate([world, ones], axis=1)
        ijk = (inv_aff @ hom.T).T[:, :3]
        sampled = map_coordinates(
            vol,
            [ijk[:, 0], ijk[:, 1], ijk[:, 2]],
            order=order,
            mode="constant",
            cval=cval,
            prefilter=False,
        ).reshape(Y.shape)
        extent = [x_mm.min(), x_mm.max(), y_mm.min(), y_mm.max()]
        return sampled, extent

    def _measure_text_inches(
        text,
        fontsize,
        rotation=0,
        fontfamily=DEFAULT_TEXT_FONT_FAMILY,
        dpi=100,
    ):
        tmp = plt.figure(figsize=(2, 2), dpi=dpi)
        t = tmp.text(
            0, 0, text, fontsize=fontsize, rotation=rotation, fontfamily=fontfamily
        )
        tmp.canvas.draw()
        bb = t.get_window_extent(renderer=tmp.canvas.get_renderer())
        w_in, h_in = bb.width / dpi, bb.height / dpi
        plt.close(tmp)
        return w_in, h_in

    def _compute_fig_layout(
        nrows,
        ncols,
        boxsize_mm,
        gap_mm,
        strip_top_ratio,
        strip_right_ratio,
        strip_pad_frac,
        axis_label_fontsize,
        font_family,
        dpi,
        x_label,
        y_label,
        single_x_label,
        single_y_label,
        show_x_label,
        show_y_label,
        x_label_offset_ratio,
        y_label_offset_ratio,
        colorbar_label,
        cbar_label_offset_ratio,
        colorbar_width,
        colorbar_pad,
    ):
        """
        Deterministic page geometry:
          - Each panel's inner box equals boxsize_mm (W,H) on the page.
          - Gaps between panels are absolute (mm).
          - Strips scale from panel size.
          - Colorbar pad/width are fractions of PANEL WIDTH (same as DF plotter).
        """
        _configure_plot_fonts(font_family)
        box_w_in, box_h_in = _as_inches(boxsize_mm)
        gap_x_in, gap_y_in = _as_inches(gap_mm)

        grid_w_in = ncols * box_w_in + (ncols - 1) * gap_x_in
        grid_h_in = nrows * box_h_in + (nrows - 1) * gap_y_in

        right_strip_extra_in = strip_right_ratio * box_w_in * (1.0 + strip_pad_frac)
        top_strip_extra_in = strip_top_ratio * box_h_in * (1.0 + strip_pad_frac)

        x_off_in = (
            (x_label_offset_ratio * box_h_in)
            if (single_x_label and show_x_label and x_label)
            else 0.0
        )
        y_off_in = (
            (y_label_offset_ratio * box_w_in)
            if (single_y_label and show_y_label and y_label)
            else 0.0
        )
        cbar_off_in = (cbar_label_offset_ratio * box_w_in) if colorbar_label else 0.0

        left_margin_in = bottom_margin_in = 0.03
        if single_y_label and show_y_label and y_label:
            y_w_in, _ = _measure_text_inches(
                y_label,
                fontsize=axis_label_fontsize,
                rotation=90,
                fontfamily=font_family,
                dpi=dpi,
            )
            left_margin_in += y_off_in + y_w_in
        if single_x_label and show_x_label and x_label:
            _, x_h_in = _measure_text_inches(
                x_label,
                fontsize=axis_label_fontsize,
                rotation=0,
                fontfamily=font_family,
                dpi=dpi,
            )
            bottom_margin_in += x_off_in + x_h_in

        # panel-referenced colorbar pad/width
        cb_pad_in = float(colorbar_pad) * box_w_in
        cb_w_in = float(colorbar_width) * box_w_in

        cbl_w_in = 0.0
        if colorbar_label:
            cbl_w_in, _ = _measure_text_inches(
                colorbar_label,
                fontsize=axis_label_fontsize,
                rotation=270,
                fontfamily=font_family,
                dpi=dpi,
            )

        fig_w_in = (
            left_margin_in
            + grid_w_in
            + right_strip_extra_in
            + cb_pad_in
            + cb_w_in
            + cbar_off_in
            + cbl_w_in
        )
        fig_h_in = bottom_margin_in + grid_h_in + top_strip_extra_in

        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
        fig.set_constrained_layout(False)

        L = left_margin_in / fig_w_in
        B = bottom_margin_in / fig_h_in
        R = (
            1.0
            - (right_strip_extra_in + cb_pad_in + cb_w_in + cbar_off_in + cbl_w_in)
            / fig_w_in
        )
        T = 1.0 - top_strip_extra_in / fig_h_in

        axes = np.array(
            [[fig.add_axes([0, 0, 1, 1]) for _ in range(ncols)] for _ in range(nrows)]
        )
        ax_w_frac = box_w_in / fig_w_in
        ax_h_frac = box_h_in / fig_h_in
        gap_x_frac = gap_x_in / fig_w_in
        gap_y_frac = gap_y_in / fig_h_in

        for i in range(nrows):
            for j in range(ncols):
                x0 = L + j * (ax_w_frac + gap_x_frac)
                y0 = B + (nrows - 1 - i) * (ax_h_frac + gap_y_frac)
                axes[i, j].set_position([x0, y0, ax_w_frac, ax_h_frac])

        layout = dict(
            rect=[L, B, R, T],
            cb_pad_in=cb_pad_in,
            cb_w_in=cb_w_in,
            cbar_off_in=cbar_off_in,
            x_off_in=x_off_in,
            y_off_in=y_off_in,
        )
        return fig, axes, layout

    def _auto_contrast(
        img2d: np.ndarray,
        vmin: Optional[float],
        vmax: Optional[float],
        p: Tuple[float, float],
        gamma: float,
    ) -> Tuple[np.ndarray, float, float]:
        arr = np.asarray(img2d, float)
        if vmin is None or vmax is None:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                vmin_used, vmax_used = 0.0, 1.0
            else:
                vmin_used, vmax_used = np.percentile(finite, [p[0], p[1]])
                if vmin_used >= vmax_used:
                    vmin_used, vmax_used = float(finite.min()), float(finite.max())
        else:
            vmin_used, vmax_used = float(vmin), float(vmax)
        scaled = (arr - vmin_used) / max(vmax_used - vmin_used, 1e-12)
        scaled = np.clip(scaled, 0.0, 1.0)
        if gamma and abs(gamma - 1.0) > 1e-6:
            scaled = np.power(scaled, float(gamma))
        return scaled, vmin_used, vmax_used

    def _parse_hex_rgba8(
        s: str, default_alpha: float
    ) -> Tuple[float, float, float, float]:
        if s is None:
            raise ValueError("color is None")
        x = s.strip()
        if x.startswith("#"):
            x = x[1:]
        if len(x) == 8:
            r = int(x[0:2], 16)
            g = int(x[2:4], 16)
            b = int(x[4:6], 16)
            a = int(x[6:8], 16)
        elif len(x) == 6:
            r = int(x[0:2], 16)
            g = int(x[2:4], 16)
            b = int(x[4:6], 16)
            a = int(round(default_alpha * 255))
        else:
            raise ValueError("Color must be 6- or 8-digit hex (e.g., #FF000080).")
        return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    def _default_mask_rgba_list(
        n: int, default_alpha: float
    ) -> List[Tuple[float, float, float, float]]:
        cmap = plt.get_cmap("tab10")
        out = []
        for i in range(n):
            r, g, b, _ = cmap(i % 10)
            out.append((r, g, b, float(default_alpha)))
        return out

    def _draw_scale_bar(
        ax: plt.Axes,
        extent: List[float],
        length_mm: float,  # MNI mm along x
        pos_frac: Tuple[float, float],
        color: str,
        alpha: float,
        height_mm: float,  # FIGURE mm
        label: Optional[str],
        label_fs: int,
        label_color: str,
        label_pos: str,
        label_pad_mm: float,  # FIGURE mm
        auto_shrink: bool,
    ) -> float:
        """Draw a scale bar INSIDE a panel; thickness and label pad are in FIGURE mm."""
        fig = ax.figure
        fig_w_in, fig_h_in = fig.get_size_inches()
        pos = ax.get_position()  # axes bbox in figure fraction

        # Convert figure-mm to data-mm along Y (thickness, pad)
        def _fig_mm_to_data_mm_y(mm_fig: float) -> float:
            ax_h_in = pos.height * fig_h_in
            frac_ax_h = (mm_fig / 25.4) / max(ax_h_in, 1e-12)
            y0, y1 = extent[2], extent[3]
            return frac_ax_h * (y1 - y0)

        x0, x1, y0, y1 = extent
        W = x1 - x0
        H = y1 - y0
        xr = x0 + float(pos_frac[0]) * W
        yb = y0 + float(pos_frac[1]) * H

        height_data_mm = _fig_mm_to_data_mm_y(float(height_mm))
        pad_data_mm = _fig_mm_to_data_mm_y(float(label_pad_mm))

        length = float(length_mm)
        margin = max(0.5, 0.5 * height_data_mm)
        max_len = max(1e-6, (xr - (x0 + margin)))
        if auto_shrink and length > max_len:
            length = max_len
        if length <= 0:
            return 0.0

        xl = xr - length
        rect = Rectangle(
            (xl, yb),
            width=length,
            height=height_data_mm,
            facecolor=color,
            edgecolor="none",
            alpha=float(alpha),
        )
        ax.add_patch(rect)

        txt = label if (label is not None) else (f"{length:g} mm")
        if isinstance(label_pos, str) and label_pos.lower() == "below":
            y_text = yb - pad_data_mm
            va = "top"
        else:
            y_text = yb + height_data_mm + pad_data_mm
            va = "bottom"
        ax.text(
            (xl + xr) / 2.0,
            y_text,
            txt,
            ha="center",
            va=va,
            color=label_color,
            fontsize=int(label_fs),
        )
        return length

    def _draw_global_scale_bar(
        fig: plt.Figure,
        ref_ax: plt.Axes,
        ref_extent: List[float],  # [x0,x1,y0,y1] in MNI mm
        length_mm: float,  # MNI mm (bar width)
        height_mm: float,  # FIGURE mm (thickness)
        pos_fig: Tuple[float, float],  # (x_frac, y_frac) in figure coords
        color: str,
        alpha: float,
        label: Optional[str],
        label_fs: int,
        label_color: str,
        label_pos: str,
        label_pad_mm: float,  # FIGURE mm
        auto_shrink: bool,
        anchor: str = "right",
    ) -> float:
        """Draw a SINGLE scale bar in FIGURE coordinates."""
        x0, x1, y0, y1 = ref_extent
        xr = max(1e-12, x1 - x0)

        pos = ref_ax.get_position()  # axes bbox in figure fraction
        width_fig = pos.width

        w_req = (float(length_mm) / xr) * width_fig
        fig_w_in, fig_h_in = fig.get_size_inches()
        h_fig = (float(height_mm) / 25.4) / max(fig_h_in, 1e-12)
        pad_fig = (float(label_pad_mm) / 25.4) / max(fig_h_in, 1e-12)

        x_fig, y_fig = float(pos_fig[0]), float(pos_fig[1])
        anchor = str(anchor).lower()
        margin = 0.005

        if anchor == "right":
            max_w = max(0.0, x_fig - margin)
            w_use = min(w_req, max_w) if auto_shrink else w_req
            left = x_fig - w_use
        elif anchor == "left":
            max_w = max(0.0, 1.0 - margin - x_fig)
            w_use = min(w_req, max_w) if auto_shrink else w_req
            left = x_fig
        else:  # "center"
            max_w = 2.0 * min(max(0.0, x_fig - margin), max(0.0, 1.0 - margin - x_fig))
            w_use = min(w_req, max_w) if auto_shrink else w_req
            left = x_fig - 0.5 * w_use

        if w_use <= 0 or h_fig <= 0:
            return 0.0

        rect = Rectangle(
            (left, y_fig),
            width=w_use,
            height=h_fig,
            transform=fig.transFigure,
            facecolor=color,
            edgecolor="none",
            alpha=float(alpha),
        )
        fig.patches.append(rect)

        drawn_mm = float(length_mm) * (w_use / max(1e-12, w_req))
        txt = label if (label is not None) else (f"{drawn_mm:g} mm")
        if isinstance(label_pos, str) and label_pos.lower() == "below":
            y_text = y_fig - pad_fig
            va = "top"
        else:
            y_text = y_fig + h_fig + pad_fig
            va = "bottom"
        fig.text(
            left + 0.5 * w_use,
            y_text,
            txt,
            ha="center",
            va=va,
            color=label_color,
            fontsize=int(label_fs),
        )
        return drawn_mm

    # --------------------------- load & canonicalize ---------------------------

    heat = _as_canonical(heat_img)
    h_aff, h_inv = heat.affine, np.linalg.inv(heat.affine)
    h_vol = np.asarray(heat.get_fdata(), dtype=float)

    bg = None
    b_inv = None
    b_vol = None
    if bg_img is not None:
        bg = _as_canonical(bg_img)
        b_inv = np.linalg.inv(bg.affine)
        b_vol = np.asarray(bg.get_fdata(), dtype=float)

    # Load masks (canonicalize)
    mask_items: List[
        Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]
    ] = []
    if masks:
        if (mask_colors is not None) and (len(mask_colors) != len(masks)):
            raise ValueError(
                f"mask_colors length ({len(mask_colors)}) must equal masks length ({len(masks)})."
            )
        rgba_list: List[Tuple[float, float, float, float]]
        if mask_colors is None:
            rgba_list = _default_mask_rgba_list(len(masks), mask_default_alpha)
        else:
            rgba_list = [_parse_hex_rgba8(c, mask_default_alpha) for c in mask_colors]
        for i, m in enumerate(masks):
            m_img = _as_canonical(m)
            m_inv = np.linalg.inv(m_img.affine)
            m_vol = np.asarray(m_img.get_fdata(), dtype=float)
            mask_items.append((m_vol, m_inv, rgba_list[i]))

    # Heatmap valid mask
    if threshold is None:
        mask = np.isfinite(h_vol) & (h_vol != 0)
    else:
        mask = np.isfinite(h_vol) & (h_vol > float(threshold))
    if not mask.any():
        raise ValueError("No valid heat voxels under the given threshold.")

    # Bounding box in voxel index space
    xlo, xhi, ylo, yhi, zlo, zhi = _bbox_from_mask(mask)

    # Build world-mm centers along each axis for the bbox region
    dx, dy, dz = nib.affines.voxel_sizes(h_aff)  # positive sizes (RAS+)
    tx, ty, tz = h_aff[0, 3], h_aff[1, 3], h_aff[2, 3]
    x_mm_all = tx + (np.arange(xlo, xhi + 1) + 0.5) * dx
    y_mm_all = ty + (np.arange(ylo, yhi + 1) + 0.5) * dy
    z_mm_all = tz + (np.arange(zlo, zhi + 1) + 0.5) * dz

    panel_labels = _ordered_percents(percent_step, percent_list)
    # facet_levels = [f.lower() for f in facets]
    facet_levels = [f.capitalize() for f in facets]

    # Default (non-square) ranges per facet
    def _default_ranges_for_facet(facet: str) -> Dict[str, Tuple[float, float]]:
        if facet == "Ax":
            xmin, xmax = float(x_mm_all.min()), float(x_mm_all.max())
            ymin, ymax = float(y_mm_all.min()), float(y_mm_all.max())
        elif facet == "Cor":
            xmin, xmax = float(x_mm_all.min()), float(x_mm_all.max())
            ymin, ymax = float(z_mm_all.min()), float(z_mm_all.max())
        elif facet == "Sag":
            xmin, xmax = float(y_mm_all.min()), float(y_mm_all.max())
            ymin, ymax = float(z_mm_all.min()), float(z_mm_all.max())
        else:
            raise ValueError("facet must be 'Ax','Cor','Sag'")
        return {"xlim": (xmin, xmax), "ylim": (ymin, ymax)}

    # Initialize ranges (can be overridden by box_ranges; later may be overridden again by global-span logic)
    ranges: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
    for f in facet_levels:
        base = _default_ranges_for_facet(f)
        for p in panel_labels:
            ranges[(f, p)] = dict(xlim=base["xlim"], ylim=base["ylim"])

    if box_ranges:
        for k, v in box_ranges.items():
            if isinstance(k, tuple) and len(k) == 2:
                fkey, pkey = k
                fkey = fkey.lower()
                if fkey not in facet_levels or pkey not in panel_labels:
                    raise ValueError(f"Unknown facet/panel in box_ranges: {k}")
                if "xlim" in v:
                    ranges[(fkey, pkey)]["xlim"] = tuple(v["xlim"])
                if "ylim" in v:
                    ranges[(fkey, pkey)]["ylim"] = tuple(v["ylim"])
            else:
                fkey = str(k).lower()
                if fkey not in facet_levels:
                    raise ValueError(f"Unknown facet key in box_ranges: {k}")
                for p in panel_labels:
                    if "xlim" in v:
                        ranges[(fkey, p)]["xlim"] = tuple(v["xlim"])
                    if "ylim" in v:
                        ranges[(fkey, p)]["ylim"] = tuple(v["ylim"])

    # Sub-mask limited to the overall bbox indices
    mask_sub = mask[xlo : xhi + 1, ylo : yhi + 1, zlo : zhi + 1]

    # Helper: for each slice, get the valid center and raw spans (in mm)
    def _slice_valid_center_and_span(
        facet: str, fixed_mm: float
    ) -> Tuple[float, float, float, float]:
        if facet == "Ax":
            idx = int(np.argmin(np.abs(z_mm_all - fixed_mm)))
            sl = mask_sub[:, :, idx]  # (nx, ny) -> x,y
            x_centers = x_mm_all
            y_centers = y_mm_all
        elif facet == "Cor":
            idx = int(np.argmin(np.abs(y_mm_all - fixed_mm)))
            sl = mask_sub[:, idx, :]  # (nx, nz) -> x,z
            x_centers = x_mm_all
            y_centers = z_mm_all
        elif facet == "Sag":
            idx = int(np.argmin(np.abs(x_mm_all - fixed_mm)))
            sl = mask_sub[idx, :, :]  # (ny, nz) -> y,z
            x_centers = y_mm_all
            y_centers = z_mm_all
        else:
            raise ValueError("facet must be 'Ax','Cor','Sag'")

        if np.any(sl):
            coords = np.argwhere(sl)
            ix = coords[:, 0]
            iy = coords[:, 1]
            x_min = float(np.min(x_centers[ix]))
            x_max = float(np.max(x_centers[ix]))
            y_min = float(np.min(y_centers[iy]))
            y_max = float(np.max(y_centers[iy]))
        else:
            x_min = float(np.min(x_centers))
            x_max = float(np.max(x_centers))
            y_min = float(np.min(y_centers))
            y_max = float(np.max(y_centers))

        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        Lx = max(1e-9, x_max - x_min)
        Ly = max(1e-9, y_max - y_min)
        return cx, cy, Lx, Ly

    # PASS A: decide fixed_mm per panel and record slice centers & raw spans
    panel_geom: Dict[Tuple[str, str], Dict[str, float]] = {}
    for f in facet_levels:
        if f == "Ax":
            fixed_min, fixed_max = float(z_mm_all.min()), float(z_mm_all.max())
        elif f == "Cor":
            fixed_min, fixed_max = float(y_mm_all.min()), float(y_mm_all.max())
        else:
            fixed_min, fixed_max = float(x_mm_all.min()), float(x_mm_all.max())

        for p in panel_labels:
            t = _percent_to_t(p)
            fixed_mm = fixed_min + t * (fixed_max - fixed_min)
            cx, cy, Lx, Ly = _slice_valid_center_and_span(f, fixed_mm)
            panel_geom[(f, p)] = dict(cx=cx, cy=cy, Lx=Lx, Ly=Ly, fixed_mm=fixed_mm)

    # PASS A.1: unify global spans and match panel aspect (b/a from boxsize)
    if enforce_global_box_span:
        Lx_list = [v["Lx"] for v in panel_geom.values()]
        Ly_list = [v["Ly"] for v in panel_geom.values()]
        if global_box_span_mm is not None:
            Lx_raw, Ly_raw = float(global_box_span_mm[0]), float(global_box_span_mm[1])
        else:
            if str(global_span_mode).lower() == "min":
                Lx_raw, Ly_raw = float(np.min(Lx_list)), float(np.min(Ly_list))
            else:  # "max" recommended to fully cover all slices
                Lx_raw, Ly_raw = float(np.max(Lx_list)), float(np.max(Ly_list))

        # Match target aspect r = b/a (panel height / panel width)
        a, b = float(boxsize[0]), float(boxsize[1])
        r = b / max(a, 1e-12)  # desired Δy/Δx

        if (Ly_raw / max(Lx_raw, 1e-12)) <= r:
            Lx_final = Lx_raw
            Ly_final = r * Lx_raw
        else:
            Ly_final = Ly_raw
            Lx_final = Ly_raw / r
    else:
        Lx_final = None
        Ly_final = None

    # Update per-panel ranges so everyone uses the unified spans centered at its slice's valid center
    for key, v in panel_geom.items():
        cx, cy = v["cx"], v["cy"]
        if enforce_global_box_span:
            xr = (cx - 0.5 * Lx_final, cx + 0.5 * Lx_final)
            yr = (cy - 0.5 * Ly_final, cy + 0.5 * Ly_final)
        else:
            xr = (cx - 0.5 * v["Lx"], cx + 0.5 * v["Lx"])
            yr = (cy - 0.5 * v["Ly"], cy + 0.5 * v["Ly"])
        ranges[key]["xlim"] = xr
        ranges[key]["ylim"] = yr

    # PASS B: sample bg/heat using the final ranges, and collect scales
    cache: Dict[
        Tuple[str, str], Tuple[Optional[np.ndarray], np.ndarray, List[float], float]
    ] = {}
    heat_vals = []
    bg_vals = []
    for f in facet_levels:
        for p in panel_labels:
            fixed_mm = panel_geom[(f, p)]["fixed_mm"]
            xr = ranges[(f, p)]["xlim"]
            yr = ranges[(f, p)]["ylim"]

            bg_slice = None
            if bg is not None:
                bg_slice, _ = _sample_slice(
                    b_vol, b_inv, f, fixed_mm, xr, yr, res_mm, order=1, cval=np.nan
                )
                if (bg_vmin is None or bg_vmax is None) and np.isfinite(bg_slice).any():
                    bg_vals.append(bg_slice[np.isfinite(bg_slice)])

            heat_slice, extent = _sample_slice(
                h_vol, h_inv, f, fixed_mm, xr, yr, res_mm, order=1, cval=np.nan
            )
            if threshold is not None:
                heat_slice = np.ma.masked_where(
                    ~np.isfinite(heat_slice) | (heat_slice <= float(threshold)),
                    heat_slice,
                )
            else:
                heat_slice = np.ma.masked_where(~np.isfinite(heat_slice), heat_slice)

            if vmin_heat is None or vmax_heat is None:
                hvec = np.asarray(heat_slice).ravel()
                hvec = hvec[np.isfinite(hvec)]
                if hvec.size:
                    heat_vals.append(hvec)

            cache[(f, p)] = (bg_slice, heat_slice, extent, fixed_mm)

    if vmin_heat is None or vmax_heat is None:
        if heat_vals:
            H = np.concatenate(heat_vals)
            if vmin_heat is None:
                vmin_heat = float(np.nanmin(H))
            if vmax_heat is None:
                vmax_heat = float(np.nanmax(H))
        else:
            vmin_heat = 0.0 if vmin_heat is None else vmin_heat
            vmax_heat = 1.0 if vmax_heat is None else vmax_heat

    if vmode_heat == "sym":
        abs_max = max(abs(vmin_heat), abs(vmax_heat))
        vmin_heat, vmax_heat = -abs_max, abs_max

    if bg is not None and (bg_vmin is None or bg_vmax is None) and bg_vals:
        B = np.concatenate(bg_vals)
        lo, hi = np.percentile(B, list(bg_percentile))
        if bg_vmin is None:
            bg_vmin = float(lo)
        if bg_vmax is None:
            bg_vmax = float(hi)

    # --------------------------- figure & axes ---------------------------

    nrows, ncols = len(facet_levels), len(panel_labels)
    fig, axes, layout = _compute_fig_layout(
        nrows,
        ncols,
        boxsize,
        panel_gap,
        strip_top_height_ratio,
        strip_right_width_ratio,
        strip_pad_frac,
        axis_label_fontsize,
        font_family,
        dpi,
        x_label,
        y_label,
        single_x_label,
        single_y_label,
        show_x_label,
        show_y_label,
        x_label_offset_ratio,
        y_label_offset_ratio,
        colorbar_label,
        cbar_label_offset_ratio,
        colorbar_width,
        colorbar_pad,
    )

    if transparent:
        fig.patch.set_alpha(0)
        for ax in fig.get_axes():
            ax.set_facecolor("none")

    cmap_heat_obj = plt.get_cmap(cmap_heat) if isinstance(cmap_heat, str) else cmap_heat
    cmap_heat_obj = cmap_heat_obj.copy()
    if hasattr(cmap_heat_obj, "set_bad"):
        cmap_heat_obj.set_bad(alpha=0.0)

    # Draw cells: background -> masks -> heat -> (optional) per-panel scale bar
    for i, f in enumerate(facet_levels):
        for j, p in enumerate(panel_labels):
            ax = axes[i, j]
            bg_slice, heat_slice, extent, fixed_mm = cache[(f, p)]

            # 1) background
            if bg_slice is not None:
                if bg_vmin is None or bg_vmax is None or abs(bg_gamma - 1.0) > 1e-6:
                    scaled_bg, _, _ = _auto_contrast(
                        bg_slice, bg_vmin, bg_vmax, bg_percentile, bg_gamma
                    )
                    ax.imshow(
                        scaled_bg,
                        extent=extent,
                        origin="lower",
                        cmap=cmap_bg,
                        vmin=0.0,
                        vmax=1.0,
                        interpolation="nearest",
                        alpha=bg_alpha,
                    )
                else:
                    ax.imshow(
                        bg_slice,
                        extent=extent,
                        origin="lower",
                        cmap=cmap_bg,
                        vmin=bg_vmin,
                        vmax=bg_vmax,
                        interpolation="nearest",
                        alpha=bg_alpha,
                    )

            # 2) masks
            for m_vol, m_inv, rgba in mask_items:
                m_slice, _ = _sample_slice(
                    m_vol,
                    m_inv,
                    f,
                    fixed_mm,
                    ranges[(f, p)]["xlim"],
                    ranges[(f, p)]["ylim"],
                    res_mm,
                    order=0,
                    cval=0.0,
                )
                m_bin = np.isfinite(m_slice) & (m_slice > float(mask_threshold))
                if m_bin.any():
                    Hh, Ww = m_slice.shape
                    overlay = np.zeros((Hh, Ww, 4), dtype=float)
                    overlay[m_bin, 0] = rgba[0]
                    overlay[m_bin, 1] = rgba[1]
                    overlay[m_bin, 2] = rgba[2]
                    overlay[m_bin, 3] = rgba[3]
                    ax.imshow(
                        overlay, extent=extent, origin="lower", interpolation="nearest"
                    )

            # 3) heat overlay
            ax.imshow(
                heat_slice,
                extent=extent,
                origin="lower",
                cmap=cmap_heat_obj,
                vmin=vmin_heat,
                vmax=vmax_heat,
                interpolation="nearest",
                alpha=heat_alpha,
            )

            # Geometry and cosmetics
            ax.set_aspect("equal")
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

            if not show_tick_labels:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.tick_params(labelsize=tick_label_fontsize)
                offset_x = ax.xaxis.get_offset_text()
                offset_x.set_fontsize(tick_label_fontsize)
                offset_y = ax.yaxis.get_offset_text()
                offset_y.set_fontsize(tick_label_fontsize)

            if single_x_label:
                ax.set_xlabel("")
            else:
                if show_x_label and (i == nrows - 1):
                    ax.set_xlabel(
                        x_label if x_label is not None else "",
                        fontsize=axis_label_fontsize,
                    )
                else:
                    ax.set_xlabel("")
            if single_y_label:
                ax.set_ylabel("")
            else:
                if show_y_label and (j == 0):
                    ax.set_ylabel(
                        y_label if y_label is not None else "",
                        fontsize=axis_label_fontsize,
                    )
                else:
                    ax.set_ylabel("")

            # 4) per-panel scale bar
            if (
                draw_scale_bar
                and (scale_bar_length_mm is not None)
                and (scale_bar_length_mm > 0)
            ):
                _draw_scale_bar(
                    ax=ax,
                    extent=extent,
                    length_mm=float(scale_bar_length_mm),
                    pos_frac=scale_bar_pos,
                    color=scale_bar_color,
                    alpha=float(scale_bar_alpha),
                    height_mm=float(scale_bar_height_mm),
                    label=scale_bar_label,
                    label_fs=int(scale_bar_label_fontsize),
                    label_color=scale_bar_label_color,
                    label_pos=str(scale_bar_label_position),
                    label_pad_mm=float(scale_bar_label_pad_mm),
                    auto_shrink=bool(scale_bar_auto_shrink),
                )

    right_edge = _add_strips_mm(
        fig,
        axes,
        col_labels=[f"{lv}" for lv in panel_labels],
        row_labels=[f"{lv}" for lv in facet_levels],
        strip_top_height_mm=strip_top_height_mm,
        strip_right_width_mm=strip_right_width_mm,
        strip_pad_mm=strip_pad_mm,
        label_fontsize=label_fontsize,
        label_top_bg_color=label_top_bg_color,
        label_right_bg_color=label_right_bg_color,
        label_text_color=label_text_color,
        label_fontweight=label_fontweight,
    )

    if transparent:
        _ensure_strip_background_opaque(fig)

    # Global labels
    L, B, R, T = layout["rect"]
    fig_w_in, fig_h_in = fig.get_size_inches()
    x_off_frac = (
        (layout["x_off_in"] / fig_h_in)
        if (single_x_label and show_x_label and x_label)
        else 0.0
    )
    y_off_frac = (
        (layout["y_off_in"] / fig_w_in)
        if (single_y_label and show_y_label and y_label)
        else 0.0
    )
    grid_cx = L + (R - L) / 2.0
    grid_cy = B + (T - B) / 2.0

    if single_x_label and show_x_label and x_label:
        fig.text(
            grid_cx,
            B - x_off_frac,
            x_label,
            ha="center",
            va="center",
            fontsize=axis_label_fontsize,
        )
    if single_y_label and show_y_label and y_label:
        fig.text(
            L - y_off_frac,
            grid_cy,
            y_label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=axis_label_fontsize,
        )

    # Title
    if title:
        top_y = max(ax.get_position().y1 for ax in axes[0, :])
        fig.suptitle(
            title, fontsize=int(label_fontsize * 1.25), y=min(0.995, top_y + 0.03)
        )

    # --- Global scale bar (single, in figure coordinates) ---
    if (
        draw_global_scale_bar
        and (global_scale_bar_length_mm is not None)
        and (global_scale_bar_length_mm > 0)
    ):
        if global_scale_bar_ref is None:
            ref_f, ref_p = facet_levels[-1], panel_labels[-1]  # bottom-right by default
        else:
            ref_f, ref_p = global_scale_bar_ref
            if ref_f not in facet_levels or ref_p not in panel_labels:
                raise ValueError(
                    f"global_scale_bar_ref {global_scale_bar_ref} not found in (facet, panel)."
                )

        i_ref = facet_levels.index(ref_f)
        j_ref = panel_labels.index(ref_p)
        ref_ax = axes[i_ref, j_ref]
        ref_extent = cache[(ref_f, ref_p)][2]  # [x0,x1,y0,y1] in mm

        _ = _draw_global_scale_bar(
            fig=fig,
            ref_ax=ref_ax,
            ref_extent=ref_extent,
            length_mm=float(global_scale_bar_length_mm),
            height_mm=float(global_scale_bar_height_mm),
            pos_fig=global_scale_bar_pos,
            color=global_scale_bar_color,
            alpha=float(global_scale_bar_alpha),
            label=global_scale_bar_label,
            label_fs=int(global_scale_bar_label_fontsize),
            label_color=global_scale_bar_label_color,
            label_pos=global_scale_bar_label_position,
            label_pad_mm=float(global_scale_bar_label_pad_mm),
            auto_shrink=bool(global_scale_bar_auto_shrink),
            anchor=str(global_scale_bar_anchor),
        )

    # Global colorbar (heat only)
    fig.canvas.draw()
    y0 = min(ax.get_position().y0 for ax in axes.ravel())
    y1 = max(ax.get_position().y1 for ax in axes.ravel())
    height = y1 - y0

    # panel-referenced inches -> figure fraction
    cb_pad_frac = layout["cb_pad_in"] / fig_w_in
    cb_w_frac = layout["cb_w_in"] / fig_w_in

    x0 = min(0.98, right_edge + cb_pad_frac)
    cbar_w = cb_w_frac if x0 + cb_w_frac <= 0.99 else max(0.01, 0.99 - x0)

    cax = fig.add_axes([x0, y0, cbar_w, height])
    sm = plt.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=vmin_heat, vmax=vmax_heat), cmap=cmap_heat_obj
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=tick_label_fontsize)
    if colorbar_label:
        cpos = cax.get_position()
        cx_r = cpos.x1
        cy_m = cpos.y0 + cpos.height / 2.0

        cbar_off_frac = layout["cbar_off_in"] / fig_w_in
        fig.text(
            cx_r + cbar_off_frac,
            cy_m,
            colorbar_label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=axis_label_fontsize,
        )

    return fig


# %%


# --------------------- Prism-style one-way boxplot ---------------------


def plot_prism_boxplot(
    df: pd.DataFrame,
    tuk: Optional[pd.DataFrame] = None,
    *,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    group_order: Optional[List[str]] = None,
    x_levels: Optional[List[str]] = None,
    jitter_palette: Union[str, List, Dict] = "viridis",
    fill_palette: Union[str, List, Dict] = "viridis",
    outline_palette: Union[str, List, Dict] = "viridis",
    xtick_rotation: float = 0.0,
    ytick_rotation: float = 0.0,
    x_log: bool = False,
    y_log: bool = False,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    y_min_zero: bool = False,
    fill_alpha: float = 0.40,
    show_points: bool = True,
    jitter_alpha: float = 0.35,
    jitter_size: float = 12.0,
    jitter_width: float = 0.18,
    show_box: bool = True,
    box_width: float = 0.55,
    whiskers: Union[str, float, Tuple[float, float]] = "tukey",
    box_edge_color: str = "black",
    box_edge_width: float = 1.0,
    median_color: str = "black",
    median_linewidth: float = 1.0,
    whisker_color: str = "black",
    whisker_linewidth: float = 1.0,
    cap_linewidth: float = 1.0,
    outlier_marker: str = "o",
    outlier_markersize: float = 3.0,
    show_raw_mean_line: bool = True,
    raw_mean_line_width: float = 1.0,
    raw_mean_line_color: str = "#404040",
    raw_mean_line_style: str = "--",
    grid: bool = True,
    grid_alpha: float = 0.3,
    dpi: int = 150,
    seed: int = 1,
    show_top_right_axes: bool = True,
    vertical_lines: Optional[Union[List, np.ndarray]] = None,
    vline_color: str = "gray",
    vline_style: str = "--",
    vline_width: float = 1.0,
    vline_alpha: float = 0.6,
    horizontal_lines: Optional[Union[List, np.ndarray]] = None,
    hline_color: str = "gray",
    hline_style: str = "--",
    hline_width: float = 1.0,
    hline_alpha: float = 0.6,
    vertical_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    horizontal_shadows: Optional[Dict[Tuple[float, float], str]] = None,
    show_brackets: bool = True,
    hide_ns: bool = True,
    y_start: float = 0.70,
    y_end: float = 0.95,
    y_step: Optional[float] = 0.08,
    bracket_height_frac: float = 0.018,
    bracket_color: str = "black",
    bracket_linewidth: float = 1.2,
    bracket_text_size: int = 12,
    title_fontsize: int = 18,
    axis_label_fontsize: int = 14,
    tick_label_fontsize: int = 12,
    legend_loc: str = "none",
    legend_ncol: Optional[int] = None,
    legend_framealpha: float = 0.9,
    legend_fontsize: int = 10,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    transparent: bool = False,
) -> plt.Figure:
    """
    GraphPad/Prism-style one-factor boxplot.

    Parameters
    ----------
    df:
        Wide table; each column is a group, rows are replicates (NaN allowed).
    tuk:
        Prism multiple-comparisons table (optional). Expected:
          - first column contains comparisons like "A vs. B"
          - a column containing star summary ("*", "**", "ns", etc.)
        The function will try to infer which column has stars.
    """
    _configure_plot_fonts(font_family)
    rng = np.random.default_rng(seed)

    if df is None or df.empty:
        raise ValueError("df is empty; expected a wide table with group columns.")

    if x_levels is None:
        if group_order is not None:
            x_levels = list(group_order)
        else:
            x_levels = list(df.columns)
    groups = list(x_levels)
    missing = [g for g in groups if g not in df.columns]
    if missing:
        raise ValueError(f"x_levels/group_order contains missing groups: {missing}")

    jitter_cmap = _build_color_map(groups, jitter_palette)
    fill_cmap = _build_color_map(groups, fill_palette)
    outline_cmap = (
        _build_color_map(groups, outline_palette) if outline_palette is not None else {}
    )
    use_outline_for_lines = outline_palette is not None
    x_offset = 1.0 if x_log else 0.0
    x_to_num = {str(g): i + x_offset for i, g in enumerate(groups)}

    fig, axes, layout = _init_box_figure(
        nrows=1,
        ncols=1,
        boxsize_mm=boxsize,
        panel_gap_mm=(0.0, 0.0),
        strip_top_height_mm=0.0,
        strip_right_width_mm=0.0,
        strip_pad_mm=0.0,
        colorbar_width_mm=0.0,
        colorbar_pad_mm=0.0,
        single_x_label=True,
        single_y_label=True,
        axis_label_fontsize=axis_label_fontsize,
        include_global_label_margins=include_global_label_margins,
        x_label_text=(x_label or ""),
        y_label_text=(y_label or ""),
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        dpi=dpi,
        font_family=font_family,
        sharex=False,
        sharey=False,
        transparent=transparent,
    )
    ax = axes[0, 0]

    # background shadows
    if vertical_shadows:
        for (x0, x1), c in vertical_shadows.items():
            _safe_axvspan(ax, x0, x1, color=_rgba_color(c), zorder=0)
    if horizontal_shadows:
        for (y0, y1), c in horizontal_shadows.items():
            _safe_axhspan(ax, y0, y1, color=_rgba_color(c), zorder=0)

    # data
    data_for_boxes: List[np.ndarray] = []
    pos_for_boxes: List[float] = []
    all_vals = []

    for g in groups:
        vals = df[g].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        all_vals.append(vals)
        if vals.size:
            data_for_boxes.append(vals)
            pos_for_boxes.append(float(x_to_num[str(g)]))

    # boxplot
    if show_box and data_for_boxes:
        whis = _parse_whiskers(whiskers)
        bp = ax.boxplot(
            data_for_boxes,
            positions=pos_for_boxes,
            widths=box_width,
            patch_artist=True,
            showfliers=True,
            whis=whis,
            medianprops=dict(color=median_color, linewidth=median_linewidth),
            whiskerprops=dict(color=whisker_color, linewidth=whisker_linewidth),
            capprops=dict(color=whisker_color, linewidth=cap_linewidth),
            flierprops=dict(
                marker=outlier_marker,
                markersize=outlier_markersize,
                markerfacecolor=whisker_color,
                markeredgecolor=whisker_color,
                alpha=0.7,
            ),
        )

        # Style boxes in group order
        for k, b in enumerate(bp["boxes"]):
            g = groups[k] if k < len(groups) else None
            face = fill_cmap.get(g, "gray")
            outline_c = (
                outline_cmap.get(g, box_edge_color)
                if use_outline_for_lines
                else box_edge_color
            )
            b.set_facecolor(mpl.colors.to_rgba(face, fill_alpha))
            b.set_edgecolor(outline_c)
            b.set_linewidth(box_edge_width)

            if use_outline_for_lines:
                if k < len(bp.get("medians", [])):
                    bp["medians"][k].set_color(outline_c)
                    bp["medians"][k].set_linewidth(median_linewidth)

                for idx in (2 * k, 2 * k + 1):
                    if idx < len(bp.get("whiskers", [])):
                        bp["whiskers"][idx].set_color(outline_c)
                        bp["whiskers"][idx].set_linewidth(whisker_linewidth)
                    if idx < len(bp.get("caps", [])):
                        bp["caps"][idx].set_color(outline_c)
                        bp["caps"][idx].set_linewidth(cap_linewidth)

                if k < len(bp.get("fliers", [])):
                    fl = bp["fliers"][k]
                    fl.set_markerfacecolor(outline_c)
                    fl.set_markeredgecolor(outline_c)

    # jitter points
    if show_points:
        for g in groups:
            vals = df[g].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if not vals.size:
                continue
            x0 = float(x_to_num[str(g)])
            xs = x0 + (rng.random(vals.size) - 0.5) * jitter_width
            ax.scatter(
                xs,
                vals,
                color=jitter_cmap.get(g, "gray"),
                alpha=jitter_alpha,
                s=jitter_size,
                linewidths=0,
                zorder=3,
            )

    # optional raw mean line across groups
    if show_raw_mean_line:
        xs_mean: List[float] = []
        ys_mean: List[float] = []
        for g in groups:
            vals = df[g].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if not vals.size:
                continue
            xs_mean.append(float(x_to_num[str(g)]))
            ys_mean.append(float(np.mean(vals)))
        if ys_mean:
            ax.plot(
                np.array(xs_mean, dtype=float),
                np.array(ys_mean, dtype=float),
                color=raw_mean_line_color,
                lw=raw_mean_line_width,
                ls=raw_mean_line_style,
                zorder=2.9,
                label="Raw mean",
            )

    # guides
    if vertical_lines is not None:
        for xv in vertical_lines:
            ax.axvline(
                x=xv,
                color=vline_color,
                linestyle=vline_style,
                linewidth=vline_width,
                alpha=vline_alpha,
                zorder=4,
            )
    if horizontal_lines is not None:
        for yv in horizontal_lines:
            ax.axhline(
                y=yv,
                color=hline_color,
                linestyle=hline_style,
                linewidth=hline_width,
                alpha=hline_alpha,
                zorder=4,
            )

    # axes cosmetics
    default_x_limits = (-0.5 + x_offset, len(groups) - 0.5 + x_offset)
    x_limits_use = x_limits if x_limits is not None else default_x_limits
    ax.set_xlim(*x_limits_use)
    xtick_pos = [x_to_num[str(g)] for g in groups]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(
        [str(g) for g in groups], fontsize=tick_label_fontsize, rotation=xtick_rotation
    )
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    for t in ax.get_yticklabels():
        t.set_rotation(ytick_rotation)
    ax.tick_params(labelsize=tick_label_fontsize)

    if grid:
        ax.grid(True, alpha=grid_alpha, linestyle="--", zorder=0)

    ax.spines["top"].set_visible(show_top_right_axes)
    ax.spines["right"].set_visible(show_top_right_axes)

    # y-limits
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    elif all_vals and any(v.size for v in all_vals):
        ymin = float(np.min([np.min(v) for v in all_vals if v.size]))
        if y_min_zero and ymin > 0:
            ymin = 0.0
        ymax = float(np.max([np.max(v) for v in all_vals if v.size]))
        yr = ymax - ymin
        if yr <= 0:
            yr = 1.0

        ymin = ymin - 0.10 * yr
        ymax = ymax + 0.30 * yr

        ax.set_ylim(ymin - 0.10 * yr, ymax + 0.30 * yr)
    y_limits_use = ax.get_ylim()

    # Brackets from Prism table (legacy GraphPad parsing)
    if show_brackets and tuk is not None and not tuk.empty:
        cols = list(tuk.columns)
        pair_col = cols[0]
        # Prefer a column named like 'Summary' (case-insensitive), else second-to-last
        star_col = next(
            (
                c
                for c in reversed(cols)
                if re.search(r"(summary|p\s*value\s*summary)", str(c), flags=re.I)
            ),
            None,
        )
        if star_col is None and len(cols) >= 2:
            star_col = cols[-2]

        def _norm_label(s) -> str:
            txt = str(s).replace("\xa0", " ")
            txt = txt.replace("–", "-").replace("—", "-")
            return re.sub(r"\s+", " ", txt).strip()

        def _draw_brackets_from_pairs(
            ax,
            pairs: List[Tuple[str, str, str]],
            x_to_num: Dict[str, int],
            *,
            y_limits_local: Optional[Tuple[float, float]],
            y_start: float,
            y_end: float,
            y_step: Optional[float],
            bracket_height_frac: float,
            color: str,
            lw: float,
            text_size: int,
        ):
            if not pairs:
                return
            intervals = []
            for g1, g2, lab in pairs:
                if g1 not in x_to_num or g2 not in x_to_num:
                    continue
                x1, x2 = float(x_to_num[g1]), float(x_to_num[g2])
                if x1 == x2:
                    continue
                lo, hi = (x1, x2) if x1 < x2 else (x2, x1)
                intervals.append({"lo": lo, "hi": hi, "label": lab})
            if not intervals:
                return

            intervals.sort(key=lambda d: (d["lo"], d["hi"]))
            heap: List[Tuple[float, int]] = []
            next_layer_id = 0
            for d in intervals:
                lo, hi = d["lo"], d["hi"]
                if heap and heap[0][0] < lo:
                    _, lid = heapq.heappop(heap)
                else:
                    lid = next_layer_id
                    next_layer_id += 1
                d["layer"] = lid
                heapq.heappush(heap, (hi, lid))
            n_layers = max(d["layer"] for d in intervals) + 1

            if y_limits_local is None:
                y0, y1 = ax.get_ylim()
            else:
                y0, y1 = y_limits_local
            y_range = float(y1 - y0)
            if y_range <= 0:
                return
            tick = y_range * float(bracket_height_frac)
            top = y0 + y_range * float(y_end)
            base = y0 + y_range * float(y_start)
            usable = max(0.0, top - base)
            if y_step is None:
                step = (
                    0.0 if n_layers <= 1 else max(usable / (n_layers - 1), tick * 1.0)
                )
            else:
                step = y_range * abs(float(y_step))
            intervals.sort(key=lambda d: d["layer"])
            for d in intervals:
                lo, hi, lab, layer = d["lo"], d["hi"], d["label"], d["layer"]
                y = max(base, top - layer * step)
                ax.plot(
                    [lo, lo, hi, hi],
                    [y, y + tick, y + tick, y],
                    color=color,
                    lw=lw,
                    zorder=5,
                    clip_on=False,
                )
                ax.text(
                    (lo + hi) / 2.0,
                    y + tick - 0.035 * y_range,
                    str(lab),
                    ha="center",
                    va="bottom",
                    fontsize=text_size,
                    color=color,
                    zorder=6,
                    clip_on=False,
                )

        pairs = []
        norm_map = {_norm_label(g): g for g in x_levels}
        for _, row in tuk.iterrows():
            comp = row.get(pair_col, None)
            stars = row.get(star_col, None)
            if comp is None or pd.isna(comp) or stars is None or pd.isna(stars):
                continue
            s = _norm_label(stars)
            if s.lower() == "ns" and hide_ns:
                continue
            txt = _norm_label(comp)
            parts = re.split(r"\s+vs\.?\s+", txt, maxsplit=1, flags=re.I)
            if len(parts) != 2:
                continue
            a, b = _norm_label(parts[0]), _norm_label(parts[1])
            g1 = norm_map.get(a) or (a if a in x_levels else None)
            g2 = norm_map.get(b) or (b if b in x_levels else None)
            if g1 is None or g2 is None:
                continue
            pairs.append((g1, g2, s))

        _draw_brackets_from_pairs(
            ax,
            pairs,
            x_to_num,
            y_limits_local=y_limits_use,
            y_start=y_start,
            y_end=y_end,
            y_step=y_step,
            bracket_height_frac=bracket_height_frac,
            color=bracket_color,
            lw=bracket_linewidth,
            text_size=bracket_text_size,
        )

    _draw_global_labels_and_title(
        fig,
        layout,
        x_label=x_label,
        y_label=y_label,
        axis_label_fontsize=axis_label_fontsize,
        title=title,
        title_fontsize=title_fontsize,
        single_x_label=True,
        single_y_label=True,
        font_family=font_family,
    )

    # legend
    if legend_loc != "none":
        handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=mpl.colors.to_rgba("gray", fill_alpha),
                markeredgecolor=box_edge_color,
                label="Box",
            ),
        ]
        if show_raw_mean_line:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=raw_mean_line_color,
                    lw=raw_mean_line_width,
                    ls=raw_mean_line_style,
                    label="Raw mean",
                )
            )
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="gray",
                alpha=jitter_alpha,
                markersize=float(np.sqrt(jitter_size)),
                label="Points",
            )
        )
        if legend_ncol is None:
            legend_ncol = 1

        inside_map = not _is_outside_legend_loc(legend_loc)
        _place_legend(
            fig,
            ax,
            handles,
            [h.get_label() for h in handles],
            legend_loc=legend_loc,
            legend_fontsize=legend_fontsize,
            legend_ncol=legend_ncol,
            legend_framealpha=legend_framealpha,
            inside_map=inside_map,
        )

    return fig


# --------------------- plate / well heatmap plotting ---------------------


def _parse_plate_grid_lines(
    grid_lines: Optional[Union[List, np.ndarray]],
    *,
    ncols: int,
    nrows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Parse `grid_lines` into (x_lines, y_lines) for plate-style heatmaps.

    `grid_lines` is interpreted as:
      - None: draw the full grid (all cell boundaries).
      - 1D array-like: draw both vertical and horizontal lines at these edge coordinates.
      - (x_lines, y_lines): two array-likes specifying vertical and horizontal line positions separately.
      - Empty list/array: draw no grid lines.

    Line positions are in "cell-edge coordinates" where x spans [0..ncols], y spans [0..nrows].
    Returned arrays are unique-sorted and clipped into the visible range.
    """
    if ncols <= 0 or nrows <= 0:
        raise ValueError(
            f"ncols and nrows must be positive; got ncols={ncols}, nrows={nrows}."
        )

    if grid_lines is None:
        x_lines = np.arange(0, ncols + 1, dtype=float)
        y_lines = np.arange(0, nrows + 1, dtype=float)
        return x_lines, y_lines

    # Explicit (x_lines, y_lines)
    if isinstance(grid_lines, (tuple, list)) and len(grid_lines) == 2:
        xg, yg = grid_lines[0], grid_lines[1]
        if isinstance(xg, (list, tuple, np.ndarray)) and isinstance(
            yg, (list, tuple, np.ndarray)
        ):
            x_lines = np.asarray(xg, dtype=float).ravel()
            y_lines = np.asarray(yg, dtype=float).ravel()
        else:
            arr = np.asarray(grid_lines, dtype=float).ravel()
            x_lines = arr
            y_lines = arr
    else:
        arr = np.asarray(grid_lines, dtype=float).ravel()
        x_lines = arr
        y_lines = arr

    # Clip + de-duplicate (keeps plotting stable if users pass out-of-range values)
    x_lines = np.unique(x_lines[(x_lines >= 0) & (x_lines <= ncols)])
    y_lines = np.unique(y_lines[(y_lines >= 0) & (y_lines <= nrows)])
    return x_lines, y_lines


def _coerce_well_heatmap_values(df_value: pd.DataFrame) -> np.ndarray:
    if not isinstance(df_value, pd.DataFrame):
        raise TypeError("df_value must be a pandas DataFrame.")
    if df_value.ndim != 2:
        raise ValueError("df_value must be 2-dimensional.")

    dfv = df_value.copy()
    for column in dfv.columns:
        dfv[column] = pd.to_numeric(dfv[column], errors="coerce")
    z_values = dfv.to_numpy(dtype=float)

    nrows, ncols = z_values.shape
    if nrows == 0 or ncols == 0:
        raise ValueError("df_value must have at least one row and one column.")

    return z_values


def _align_well_heatmap_p_values(
    df_value: pd.DataFrame,
    df_p: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if df_p is None:
        return None
    if not isinstance(df_p, pd.DataFrame):
        raise TypeError("df_p must be a pandas DataFrame or None.")

    missing_rows = set(df_value.index) - set(df_p.index)
    missing_cols = set(df_value.columns) - set(df_p.columns)
    if missing_rows or missing_cols:
        raise ValueError(
            "df_p is missing labels present in df_value. "
            f"Missing rows: {sorted(missing_rows)}; missing cols: {sorted(missing_cols)}"
        )

    dfp_aligned = df_p.reindex(index=df_value.index, columns=df_value.columns).copy()
    for column in dfp_aligned.columns:
        dfp_aligned[column] = pd.to_numeric(dfp_aligned[column], errors="coerce")
    return dfp_aligned


def _draw_well_heatmap_panel(
    ax: plt.Axes,
    *,
    df_value: pd.DataFrame,
    z_values: np.ndarray,
    df_p_aligned: Optional[pd.DataFrame],
    cmap,
    norm: mpl.colors.Normalize,
    xtick_rotation: float,
    ytick_rotation: float,
    show_x_ticklabels: bool,
    show_y_ticklabels: bool,
    grid_lines: Optional[Union[List, np.ndarray]],
    gline_color: str,
    gline_width: float,
    gline_alpha: float,
    tick_label_fontsize: float,
    show_p: bool,
    hide_ns: bool,
    p_text_color: str,
    p_text_size: float,
    na_color: Optional[str] = None,
) -> mpl.collections.QuadMesh:
    nrows, ncols = z_values.shape

    z_masked = np.ma.masked_invalid(z_values)
    if na_color is not None:
        cmap = cmap.with_extremes(bad=na_color)
    x_edges = np.arange(ncols + 1, dtype=float)
    y_edges = np.arange(nrows + 1, dtype=float)

    mappable = ax.pcolormesh(
        x_edges,
        y_edges,
        z_masked,
        shading="auto",
        cmap=cmap,
        norm=norm,
        zorder=1,
    )

    x_lines, y_lines = _parse_plate_grid_lines(grid_lines, ncols=ncols, nrows=nrows)
    if x_lines.size or y_lines.size:
        ax.vlines(
            x_lines,
            ymin=0,
            ymax=nrows,
            colors=gline_color,
            linewidth=gline_width,
            alpha=gline_alpha,
            zorder=2,
        )
        ax.hlines(
            y_lines,
            xmin=0,
            xmax=ncols,
            colors=gline_color,
            linewidth=gline_width,
            alpha=gline_alpha,
            zorder=2,
        )

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()

    x_centers = np.arange(ncols, dtype=float) + 0.5
    y_centers = np.arange(nrows, dtype=float) + 0.5
    ax.set_xticks(x_centers)
    ax.set_yticks(y_centers)

    x_labels = (
        [str(column) for column in df_value.columns]
        if show_x_ticklabels
        else [""] * ncols
    )
    y_labels = (
        [str(index) for index in df_value.index] if show_y_ticklabels else [""] * nrows
    )
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    ax.tick_params(labelsize=tick_label_fontsize)
    for label in ax.get_xticklabels():
        label.set_rotation(xtick_rotation)
    for label in ax.get_yticklabels():
        label.set_rotation(ytick_rotation)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    if na_color is not None:
        for row_index, column_index in np.argwhere(np.ma.getmaskarray(z_masked)):
            ax.text(
                column_index + 0.5,
                row_index + 0.5,
                "NA",
                ha="center",
                va="center",
                fontsize=4,
                color="#555555",
                zorder=3,
            )

    if show_p and df_p_aligned is not None:
        for row_index in range(nrows):
            for column_index in range(ncols):
                if not np.isfinite(z_values[row_index, column_index]):
                    continue
                stars = p_to_stars(df_p_aligned.iat[row_index, column_index])
                if hide_ns and stars == "n.s.":
                    continue
                ax.text(
                    column_index + 0.5,
                    row_index + 0.75,  # for asterisk vertical centering
                    stars,
                    ha="center",
                    va="center",
                    color=p_text_color,
                    fontsize=p_text_size,
                    zorder=3,
                )

    return mappable


def _validate_well_heatmap_grid(
    df_values: List[List[pd.DataFrame]],
    df_ps: Optional[List[List[Optional[pd.DataFrame]]]],
    *,
    allow_variable_shape: bool = False,
) -> Tuple[
    List[List[np.ndarray]],
    List[List[Optional[pd.DataFrame]]],
    int,
    int,
    List[int],
    List[int],
]:
    if not df_values or not all(isinstance(row, list) and row for row in df_values):
        raise ValueError(
            "df_values must be a non-empty rectangular list of DataFrame rows."
        )

    n_panel_rows = len(df_values)
    n_panel_cols = len(df_values[0])
    if any(len(row) != n_panel_cols for row in df_values):
        raise ValueError("df_values must be rectangular.")

    if df_ps is not None:
        if len(df_ps) != n_panel_rows or any(len(row) != n_panel_cols for row in df_ps):
            raise ValueError("df_ps must match the rectangular shape of df_values.")
        df_p_grid = df_ps
    else:
        df_p_grid = [[None for _ in range(n_panel_cols)] for _ in range(n_panel_rows)]

    z_grid: List[List[np.ndarray]] = []
    p_grid: List[List[Optional[pd.DataFrame]]] = []
    expected_shape: Optional[Tuple[int, int]] = None
    row_counts: List[Optional[int]] = [None] * n_panel_rows
    column_counts: List[Optional[int]] = [None] * n_panel_cols
    for row_index, row in enumerate(df_values):
        z_row: List[np.ndarray] = []
        p_row: List[Optional[pd.DataFrame]] = []
        for column_index, df_value in enumerate(row):
            z_values = _coerce_well_heatmap_values(df_value)
            nrows, ncols = z_values.shape
            if allow_variable_shape:
                expected_rows = row_counts[row_index]
                if expected_rows is None:
                    row_counts[row_index] = nrows
                elif nrows != expected_rows:
                    raise ValueError(
                        "All df_values panels in the same grid row must have the same row count "
                        "for variable-size grid plotting. "
                        f"Expected {expected_rows}; got {nrows} at row {row_index}, column {column_index}."
                    )

                expected_columns = column_counts[column_index]
                if expected_columns is None:
                    column_counts[column_index] = ncols
                elif ncols != expected_columns:
                    raise ValueError(
                        "All df_values panels in the same grid column must have the same column count "
                        "for variable-size grid plotting. "
                        f"Expected {expected_columns}; got {ncols} at row {row_index}, column {column_index}."
                    )
            else:
                if expected_shape is None:
                    expected_shape = z_values.shape
                elif z_values.shape != expected_shape:
                    raise ValueError(
                        "All df_values panels must have the same shape for fixed-size grid plotting. "
                        f"Expected {expected_shape}; got {z_values.shape} at "
                        f"row {row_index}, column {column_index}."
                    )
            z_row.append(z_values)
            p_row.append(
                _align_well_heatmap_p_values(
                    df_value, df_p_grid[row_index][column_index]
                )
            )
        z_grid.append(z_row)
        p_grid.append(p_row)

    if allow_variable_shape:
        return (
            z_grid,
            p_grid,
            n_panel_rows,
            n_panel_cols,
            [int(row_count) for row_count in row_counts if row_count is not None],
            [
                int(column_count)
                for column_count in column_counts
                if column_count is not None
            ],
        )

    if expected_shape is None:
        raise ValueError("df_values must include at least one panel.")

    panel_nrows, panel_ncols = expected_shape
    return (
        z_grid,
        p_grid,
        n_panel_rows,
        n_panel_cols,
        [panel_nrows] * n_panel_rows,
        [panel_ncols] * n_panel_cols,
    )


def _show_well_heatmap_ticklabels(
    policy: str, *, index: int, last_index: int, axis: str
) -> bool:
    valid = {"all", "none", "bottom", "left"}
    if policy not in valid:
        raise ValueError(
            f"show_{axis}_ticklabels must be one of {sorted(valid)}; got {policy!r}."
        )
    if policy == "all":
        return True
    if policy == "none":
        return False
    if policy == "bottom":
        return axis == "x" and index == last_index
    if policy == "left":
        return axis == "y" and index == 0
    return False


def plot_well_heatmap_df(
    df_value: pd.DataFrame,
    df_p: Optional[pd.DataFrame] = None,
    *,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    xtick_rotation: float = 0.0,
    ytick_rotation: float = 0.0,
    title: Optional[str] = None,
    label_top: Optional[str] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    cmap: Union[str, plt.Colormap] = cm.vik,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmode: str = "auto",
    dpi: int = 300,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    cellsize: Optional[Tuple[float, float]] = None,
    label_top_height_mm: float = 4.5,
    label_top_pad_mm: float = 2.0,
    label_top_bg_color: str = "lightgray",
    label_top_text_color: str = "black",
    label_top_fontsize: Optional[float] = None,
    label_top_fontweight: str = "bold",
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    cbar_label_offset_mm: float = 1.5,
    grid_lines: Optional[Union[List, np.ndarray]] = None,
    gline_color: str = "black",
    gline_width: float = 1.0,
    gline_alpha: float = 0.8,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    colorbar_label: Optional[str] = None,
    colorbar_width_mm: float = 3.0,
    colorbar_pad_mm: float = 1.5,
    colorbar_length_mm: Optional[float] = None,
    show_p: bool = True,
    hide_ns: bool = True,
    p_text_color: str = "black",
    p_text_size: int = 12,
    transparent: bool = False,
) -> plt.Figure:
    """Plot a plate-style (well-grid) heatmap using mm-based `boxsize` geometry.

    This function is designed for well/plate layouts (e.g., 96-well, 384-well), where each cell
    corresponds to a "well" and the axes ticks are discrete row/column labels.

    It reuses `visualdf.py`'s fixed-geometry helpers:
      - `_init_box_figure`
      - `_compute_global_vrange`
      - `_draw_global_labels_and_title`
      - `_add_global_colorbar`
      - `p_to_stars` (for significance)

    Parameters
    ----------
    df_value:
        A 2D DataFrame (rows x columns) holding the scalar value per well. Row labels become y-ticks,
        column labels become x-ticks.
    df_p:
        Optional 2D DataFrame of p-values aligned to `df_value`. If provided and `show_p=True`,
        each well is annotated with significance stars from `p_to_stars()`.
    boxsize:
        Inner heatmap panel size in millimeters, used when `cellsize` is None.
    cellsize:
        Optional `(cell_width_mm, cell_height_mm)` tuple. If provided, it overrides `boxsize`
        by deriving the panel size from the matrix shape.
    colorbar_length_mm:
        Optional vertical colorbar length in millimeters. If None, the colorbar spans
        the full heatmap panel height.
    label_top:
        Optional text for a top strip above the heatmap. If None, no top strip is drawn
        and no top-strip space is allocated.
    show_p:
        If True, annotate each cell using `df_p`.
    hide_ns:
        If True, do not annotate wells whose star label is "n.s.".

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    Z = _coerce_well_heatmap_values(df_value)
    nrows, ncols = Z.shape

    effective_boxsize = boxsize
    if cellsize is not None:
        cell_w_mm, cell_h_mm = _validate_cellsize(cellsize)
        effective_boxsize = (ncols * cell_w_mm, nrows * cell_h_mm)

    dfp_aligned = _align_well_heatmap_p_values(df_value, df_p)

    # Layout: only allocate global label margins when the label text is provided
    single_x_label = bool(x_label)
    single_y_label = bool(y_label)
    has_top_label = label_top is not None
    strip_top_height_use = float(label_top_height_mm) if has_top_label else 0.0
    strip_top_pad_use = float(label_top_pad_mm) if has_top_label else 0.0

    fig, axes, layout = _init_box_figure(
        nrows=1,
        ncols=1,
        boxsize_mm=effective_boxsize,
        panel_gap_mm=(0.0, 0.0),
        strip_top_height_mm=strip_top_height_use,
        strip_right_width_mm=0.0,
        strip_pad_mm=strip_top_pad_use,
        colorbar_width_mm=colorbar_width_mm,
        colorbar_pad_mm=colorbar_pad_mm,
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        axis_label_fontsize=float(axis_label_fontsize),
        include_global_label_margins=include_global_label_margins,
        x_label_text=(x_label or ""),
        y_label_text=(y_label or ""),
        colorbar_label_text=colorbar_label,
        x_label_offset_mm=x_label_offset_mm,
        y_label_offset_mm=y_label_offset_mm,
        cbar_label_offset_mm=cbar_label_offset_mm,
        dpi=dpi,
        font_family=font_family,
        sharex=False,
        sharey=False,
        transparent=transparent,
    )
    ax = axes[0, 0]
    right_edge = layout["rect"][2]

    if has_top_label:
        right_edge = _add_strips_mm(
            fig,
            axes,
            col_labels=[str(label_top)],
            row_labels=None,
            strip_top_height_mm=strip_top_height_use,
            strip_right_width_mm=0.0,
            strip_pad_mm=strip_top_pad_use,
            label_fontsize=float(label_top_fontsize or axis_label_fontsize),
            label_top_bg_color=label_top_bg_color,
            label_right_bg_color=label_top_bg_color,
            label_text_color=label_top_text_color,
            label_fontweight=label_top_fontweight,
        )

    # Colormap and normalization (matches plot_single_effect_df behavior)
    vmin_use, vmax_use = _compute_global_vrange([Z], vmin=vmin, vmax=vmax, vmode=vmode)
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    norm = mpl.colors.Normalize(vmin=vmin_use, vmax=vmax_use)

    mappable = _draw_well_heatmap_panel(
        ax,
        df_value=df_value,
        z_values=Z,
        df_p_aligned=dfp_aligned,
        cmap=cmap_obj,
        norm=norm,
        xtick_rotation=xtick_rotation,
        ytick_rotation=ytick_rotation,
        show_x_ticklabels=True,
        show_y_ticklabels=True,
        grid_lines=grid_lines,
        gline_color=gline_color,
        gline_width=gline_width,
        gline_alpha=gline_alpha,
        tick_label_fontsize=float(tick_label_fontsize),
        show_p=show_p,
        hide_ns=hide_ns,
        p_text_color=p_text_color,
        p_text_size=float(p_text_size),
    )

    _draw_global_labels_and_title(
        fig,
        layout,
        x_label=x_label,
        y_label=y_label,
        axis_label_fontsize=float(axis_label_fontsize),
        title=title,
        title_fontsize=float(title_fontsize),
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        font_family=font_family,
    )

    if mappable is not None and colorbar_width_mm > 0:
        _add_global_colorbar(
            fig,
            layout,
            mappable=mappable,
            right_edge=float(right_edge),
            colorbar_label=colorbar_label,
            tick_label_fontsize=float(tick_label_fontsize),
            axis_label_fontsize=float(axis_label_fontsize),
            colorbar_length_mm=colorbar_length_mm,
            font_family=font_family,
        )

    return fig


def plot_well_heatmap_grid_df(
    df_values: List[List[pd.DataFrame]],
    df_ps: Optional[List[List[Optional[pd.DataFrame]]]] = None,
    *,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    xtick_rotation: float = 0.0,
    ytick_rotation: float = 0.0,
    title: Optional[str] = None,
    label_top: Optional[List[str]] = None,
    label_right: Optional[List[str]] = None,
    font_family: str = DEFAULT_TEXT_FONT_FAMILY,
    cmap: Union[str, plt.Colormap] = cm.vik,
    na_color: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmode: str = "auto",
    dpi: int = 300,
    boxsize: Tuple[float, float] = DEFAULT_BOXSIZE_MM,
    cellsize: Optional[Tuple[float, float]] = None,
    panel_gap_mm: Tuple[float, float] = (0.0, 0.0),
    label_top_height_mm: float = 4.5,
    label_right_width_mm: float = 4.5,
    strip_pad_mm: float = 2.0,
    label_top_bg_color: str = "lightgray",
    label_right_bg_color: str = "lightgray",
    label_text_color: str = "black",
    label_fontsize: Optional[float] = None,
    label_fontweight: str = "bold",
    include_global_label_margins: bool = True,
    x_label_offset_mm: float = 1.25,
    y_label_offset_mm: float = 1.5,
    cbar_label_offset_mm: float = 1.5,
    grid_lines: Optional[Union[List, np.ndarray]] = None,
    gline_color: str = "black",
    gline_width: float = 1.0,
    gline_alpha: float = 0.8,
    title_fontsize: int = 20,
    axis_label_fontsize: int = 16,
    tick_label_fontsize: int = 12,
    colorbar_label: Optional[str] = None,
    colorbar_ticks: Optional[List[float]] = None,
    colorbar_width_mm: float = 3.0,
    colorbar_pad_mm: float = 1.5,
    colorbar_length_mm: Optional[float] = None,
    show_p: bool = True,
    hide_ns: bool = True,
    p_text_color: str = "black",
    p_text_size: int = 12,
    show_x_ticklabels: str = "all",
    show_y_ticklabels: str = "all",
    transparent: bool = False,
) -> plt.Figure:
    """Plot a rectangular grid of well heatmaps with shared normalization.

    When na_color is supplied, missing cells use that fill and centered NA text
    at 4 pt in #555555. Otherwise, the colormap's existing bad color is retained.
    """
    use_cellsize_layout = cellsize is not None
    (
        z_grid,
        p_grid,
        n_panel_rows,
        n_panel_cols,
        panel_row_counts,
        panel_column_counts,
    ) = _validate_well_heatmap_grid(
        df_values,
        df_ps,
        allow_variable_shape=use_cellsize_layout,
    )

    if label_top is not None and len(label_top) != n_panel_cols:
        raise ValueError(
            f"label_top must have {n_panel_cols} entries; got {len(label_top)}."
        )
    if label_right is not None and len(label_right) != n_panel_rows:
        raise ValueError(
            f"label_right must have {n_panel_rows} entries; got {len(label_right)}."
        )

    if use_cellsize_layout:
        cell_w_mm, cell_h_mm = _validate_cellsize(cellsize)
        panel_widths_mm = [
            column_count * cell_w_mm for column_count in panel_column_counts
        ]
        panel_heights_mm = [row_count * cell_h_mm for row_count in panel_row_counts]
    else:
        effective_boxsize = boxsize

    has_top_label = label_top is not None
    has_right_label = label_right is not None
    strip_top_height_use = float(label_top_height_mm) if has_top_label else 0.0
    strip_right_width_use = float(label_right_width_mm) if has_right_label else 0.0
    strip_pad_use = float(strip_pad_mm) if (has_top_label or has_right_label) else 0.0
    single_x_label = bool(x_label)
    single_y_label = bool(y_label)

    if use_cellsize_layout:
        fig, axes, layout = _init_variable_box_figure(
            panel_widths_mm=panel_widths_mm,
            panel_heights_mm=panel_heights_mm,
            panel_gap_mm=panel_gap_mm,
            strip_top_height_mm=strip_top_height_use,
            strip_right_width_mm=strip_right_width_use,
            strip_pad_mm=strip_pad_use,
            colorbar_width_mm=colorbar_width_mm,
            colorbar_pad_mm=colorbar_pad_mm,
            single_x_label=single_x_label,
            single_y_label=single_y_label,
            axis_label_fontsize=float(axis_label_fontsize),
            include_global_label_margins=include_global_label_margins,
            x_label_text=(x_label or ""),
            y_label_text=(y_label or ""),
            colorbar_label_text=colorbar_label,
            x_label_offset_mm=x_label_offset_mm,
            y_label_offset_mm=y_label_offset_mm,
            cbar_label_offset_mm=cbar_label_offset_mm,
            dpi=dpi,
            font_family=font_family,
            sharex=False,
            sharey=False,
            transparent=transparent,
        )
    else:
        fig, axes, layout = _init_box_figure(
            nrows=n_panel_rows,
            ncols=n_panel_cols,
            boxsize_mm=effective_boxsize,
            panel_gap_mm=panel_gap_mm,
            strip_top_height_mm=strip_top_height_use,
            strip_right_width_mm=strip_right_width_use,
            strip_pad_mm=strip_pad_use,
            colorbar_width_mm=colorbar_width_mm,
            colorbar_pad_mm=colorbar_pad_mm,
            single_x_label=single_x_label,
            single_y_label=single_y_label,
            axis_label_fontsize=float(axis_label_fontsize),
            include_global_label_margins=include_global_label_margins,
            x_label_text=(x_label or ""),
            y_label_text=(y_label or ""),
            colorbar_label_text=colorbar_label,
            x_label_offset_mm=x_label_offset_mm,
            y_label_offset_mm=y_label_offset_mm,
            cbar_label_offset_mm=cbar_label_offset_mm,
            dpi=dpi,
            font_family=font_family,
            sharex=False,
            sharey=False,
            transparent=transparent,
        )

    right_edge = layout["rect"][2]
    if has_top_label or has_right_label:
        right_edge = _add_strips_mm(
            fig,
            axes,
            col_labels=(
                [str(label) for label in label_top] if label_top is not None else None
            ),
            row_labels=(
                [str(label) for label in label_right]
                if label_right is not None
                else None
            ),
            strip_top_height_mm=strip_top_height_use,
            strip_right_width_mm=strip_right_width_use,
            strip_pad_mm=strip_pad_use,
            label_fontsize=float(label_fontsize or axis_label_fontsize),
            label_top_bg_color=label_top_bg_color,
            label_right_bg_color=label_right_bg_color,
            label_text_color=label_text_color,
            label_fontweight=label_fontweight,
        )

    z_arrays = [z_values for row in z_grid for z_values in row]
    vmin_use, vmax_use = _compute_global_vrange(
        z_arrays, vmin=vmin, vmax=vmax, vmode=vmode
    )
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    norm = mpl.colors.Normalize(vmin=vmin_use, vmax=vmax_use)

    mappable = None
    for row_index, row in enumerate(df_values):
        for column_index, df_value in enumerate(row):
            ax = axes[row_index, column_index]
            x_ticklabels_visible = _show_well_heatmap_ticklabels(
                show_x_ticklabels,
                index=row_index,
                last_index=n_panel_rows - 1,
                axis="x",
            )
            y_ticklabels_visible = _show_well_heatmap_ticklabels(
                show_y_ticklabels,
                index=column_index,
                last_index=n_panel_cols - 1,
                axis="y",
            )
            mappable = _draw_well_heatmap_panel(
                ax,
                df_value=df_value,
                z_values=z_grid[row_index][column_index],
                df_p_aligned=p_grid[row_index][column_index],
                cmap=cmap_obj,
                na_color=na_color,
                norm=norm,
                xtick_rotation=xtick_rotation,
                ytick_rotation=ytick_rotation,
                show_x_ticklabels=x_ticklabels_visible,
                show_y_ticklabels=y_ticklabels_visible,
                grid_lines=grid_lines,
                gline_color=gline_color,
                gline_width=gline_width,
                gline_alpha=gline_alpha,
                tick_label_fontsize=float(tick_label_fontsize),
                show_p=show_p,
                hide_ns=hide_ns,
                p_text_color=p_text_color,
                p_text_size=float(p_text_size),
            )

    _draw_global_labels_and_title(
        fig,
        layout,
        x_label=x_label,
        y_label=y_label,
        axis_label_fontsize=float(axis_label_fontsize),
        title=title,
        title_fontsize=float(title_fontsize),
        single_x_label=single_x_label,
        single_y_label=single_y_label,
        font_family=font_family,
    )

    if mappable is not None and colorbar_width_mm > 0:
        colorbar = _add_global_colorbar(
            fig,
            layout,
            mappable=mappable,
            right_edge=float(right_edge),
            colorbar_label=colorbar_label,
            tick_label_fontsize=float(tick_label_fontsize),
            axis_label_fontsize=float(axis_label_fontsize),
            colorbar_length_mm=colorbar_length_mm,
            font_family=font_family,
        )
        if colorbar is not None and colorbar_ticks is not None:
            colorbar.set_ticks(colorbar_ticks)

    return fig
