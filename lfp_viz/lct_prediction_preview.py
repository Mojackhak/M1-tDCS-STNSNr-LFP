"""Provide raw-score LOOCV rendering helpers and focused previews."""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from lfp_clinical.lct_prediction import compute_baseline_only
from lfp_clinical.runner import _trash_existing
from lfp_cohort.io import atomic_csv
from . import visualdf
from .config import VizConfig
from .lct_preview import heatmap_plan
from .correlation_heatmap import render_correlation_heatmap_figure
from .correlation_fit import _raw_linear_model_curve


def fold_coefficients(points, model):
    """Recover raw focal coefficients from the original LOOCV training designs."""
    columns = ['Baseline']
    if model != 'baseline-only':
        columns.append('LCT' if model == 'LCT' else 'PredictorValue')
    design = np.column_stack([np.ones(len(points)), points[columns].to_numpy(float)])
    outcome = points.Outcome.to_numpy(float)
    return np.array([np.linalg.lstsq(np.delete(design, i, axis=0),
                                    np.delete(outcome, i), rcond=None)[0][-1]
                     for i in range(len(points))])


def draw_prediction(points, model, style, reference):
    """Draw held-out predictions and a descriptive OLS calibration band."""
    points = points.copy()
    row = points.iloc[0]
    top, right = '', ''
    if model == 'tDCS':
        region = str(row.RegionValue).replace('SNr-STN', 'SNr–STN')
        polar = {'Anodal': '⊕', 'Cathodal': '⊖'}[row.Polar]
        top = f'{region} | {polar} | {row.PhaseContrast}'
        right = f'{row.Lat} | {row.FeatureLabel}'
    model_label = 'Baseline' if model == 'baseline-only' else model
    endpoint = {'stn-3m': 'STN-DBS', 'stn-snr-3m': 'STN+SNr-DBS'}[row.EndpointID]
    points['Panel'] = top
    points['Facet'] = right
    def limits(values):
        low, high = float(values.min()), float(values.max())
        pad = .08 * (high - low) if high > low else 1.
        return (low - pad, high + pad)

    x_limits = limits(points.PredictedOutcome)
    curve = _raw_linear_model_curve(
        points, method='spearman', x_column='PredictedOutcome', y_column='Outcome',
        baseline_reference=None, confidence_level=.95,
    )
    curve['Panel'], curve['Facet'] = top, right
    y_limits = limits(pd.concat([points.Outcome, curve['lower.CL'], curve['upper.CL']]))
    error = points.Outcome - points.PredictedOutcome
    mae, rmse = error.abs().mean(), np.sqrt((error ** 2).mean())
    if model == 'baseline-only':
        annotation = f'MAE = {mae:.2f}\nRMSE = {rmse:.2f}'
    else:
        reference_error = reference.Outcome - reference.PredictedOutcome
        delta_mae = reference_error.abs().mean() - mae
        delta_rmse = np.sqrt((reference_error ** 2).mean()) - rmse
        annotation = f'ΔMAE = {delta_mae:+.2f}\nΔRMSE = {delta_rmse:+.2f}'
    font, strips = style['font'], style['strips']
    fig = visualdf.plot_triple_interaction_fit(
        points, curve, pd.DataFrame(), value_col='Outcome', x_var='PredictedOutcome',
        panel_var='Panel', facet_var='Facet', color_var='ID',
        panel_levels=[top], facet_levels=[right],
        x_label=f'Predicted {row.ScaleLabel} ({model_label})',
        y_label=f'Observed {row.ScaleLabel}\n({endpoint})',
        x_limits=x_limits, y_limits=y_limits, font_family=font['family'],
        palette=style['points']['palette'], jitter_alpha=1, jitter_size=10,
        show_ribbon=True, ribbon_alpha=style['fit']['ribbon_alpha'],
        curve_line_width=style['fit']['line_width'], grid=False,
        label_fontsize=font['strip_pt'], label_top_bg_color=strips['top_background'],
        label_right_bg_color=strips['right_background'], label_fontweight='bold',
        strip_top_height_mm=4.5, strip_right_width_mm=4.5, strip_pad_mm=2,
        axis_label_fontsize=7, tick_label_fontsize=6, legend_loc='none',
        boxsize=(30, 25), x_label_offset_mm=5, y_label_offset_mm=5,
        show_top_right_axes=False, transparent=True, dpi=600,
    )
    if model != 'tDCS':
        for strip_axis in list(fig.axes):
            if getattr(strip_axis, '_visualdf_strip_axis', False):
                fig.delaxes(strip_axis)
    axis = next(ax for ax in fig.axes if not getattr(ax, '_visualdf_strip_axis', False))
    axis.plot(x_limits, x_limits, color='black', linewidth=.7, linestyle='--', zorder=1)
    positive_slope = curve.emmean.iloc[-1] > curve.emmean.iloc[0]
    annotation_y, alignment = (.96, 'top') if positive_slope else (.04, 'bottom')
    axis.text(.04, annotation_y, annotation,
              transform=axis.transAxes, ha='left', va=alignment, fontsize=6,
              family=font['family'])
    coefficients = (points.FoldCoefficient.to_numpy() if 'FoldCoefficient' in points
                    else fold_coefficients(points, model))
    term = {'baseline-only': 'B', 'LCT': 'LCT', 'tDCS': 'X'}[model]
    r = points.PredictedOutcome.corr(points.Outcome)
    diagnostics = (f'r = {r:+.2f}\n'
                   + rf'$\beta_{{{term}}}$' + f' = {np.median(coefficients):+.3f}')
    axis.text(.96, .04 if positive_slope else .96, diagnostics,
              transform=axis.transAxes, ha='right',
              va='bottom' if positive_slope else 'top', fontsize=6,
              family=font['family'])
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--fits-only', action='store_true')
    parser.add_argument('--heatmap-only', action='store_true')
    args = parser.parse_args()
    paths = yaml.safe_load(args.config.read_text())['paths']
    root, output = Path(paths['stats_root']), Path(paths['output_root'])
    if args.heatmap_only:
        cells = pd.read_csv(output / 'heatmap-cells.csv')
        style_path = Path(paths['heatmap_style'])
        data = yaml.safe_load(style_path.read_text())
        data['inputs']['value_column'] = 'MAEReduction'
        cells['p_holm'] = np.nan
        limit = float(np.ceil(cells.MAEReduction.abs().max()))
        data['style']['heatmap']['value_limits'] = [-limit, limit]
        data['style']['colorbar']['ticks'] = [-limit, 0, limit]
        spec = {'Layout': 'phase_rows_composite',
                'ColorbarLabel': 'MAE reduction vs Baseline (points)'}
        fig = render_correlation_heatmap_figure(cells, spec, VizConfig(style_path, data))
        target = output / 'UPDRS-III_SNr_local_MAE-vs-baseline_heatmap.pdf'
        if target.exists():
            _trash_existing([target])
        fig.savefig(target, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(target, flush=True)
        return
    if args.fits_only:
        points = pd.read_csv(output / 'fit-points.csv')
        reference = points[points.Model.eq('baseline-only')]
        style = yaml.safe_load(Path(paths['fit_style']).read_text())['style']
        for model in ['baseline-only', 'LCT', 'tDCS']:
            target = output / f'{model}_UPDRS-III_STN-SNr_theta-imcoh-abs_Cathodal_Ipsi_Post-Late.pdf'
            fig = draw_prediction(points[points.Model.eq(model)], model, style, reference)
            if target.exists():
                _trash_existing([target])
            fig.savefig(target, dpi=600, bbox_inches='tight')
            plt.close(fig)
            print(target, flush=True)
        return
    saved = pd.read_csv(root / 'prediction/predictions.csv')
    performance = pd.read_csv(root / 'prediction/performance.csv')
    baseline = compute_baseline_only(saved, performance)
    selection = (saved.Formulation.eq('baseline_adjusted') & saved.Metric.eq('imcoh_abs')
                 & saved.Band.eq('theta') & saved.RegionValue.eq('SNr-STN')
                 & saved.Polar.eq('Cathodal') & saved.Lat.eq('Ipsi')
                 & saved.Phase.eq('Post') & saved.ScaleLabel.eq('UPDRS-III')
                 & saved.EndpointID.eq('stn-snr-3m') & saved.Model.isin(['LCT', 'tDCS']))
    points = saved[selection].copy()
    reference = baseline['predictions.csv']
    reference = reference[reference.PredictorID.eq(points.PredictorID.iloc[0])
                          & reference.ScaleLabel.eq('UPDRS-III')
                          & reference.EndpointID.eq('stn-snr-3m')]
    points = pd.concat([reference, points], ignore_index=True)
    style = yaml.safe_load(Path(paths['fit_style']).read_text())['style']
    figures = {}
    for model in ['baseline-only', 'LCT', 'tDCS']:
        figures[f'{model}_UPDRS-III_STN-SNr_theta-imcoh-abs_Cathodal_Ipsi_Post-Late.pdf'] = draw_prediction(
            points[points.Model.eq(model)], model, style, reference)

    keys = ['PredictorID', 'ScaleLabel', 'EndpointID']
    tdcs = performance[performance.Formulation.eq('baseline_adjusted') & performance.Model.eq('tDCS')]
    matched = tdcs.merge(baseline['performance.csv'][keys + ['MAE_score', 'RMSE_score']],
                         on=keys, suffixes=('_tdcs', '_baseline'), validate='one_to_one')
    matched['MAEReduction'] = matched.MAE_score_baseline - matched.MAE_score_tdcs
    matched['RMSEReduction'] = matched.RMSE_score_baseline - matched.RMSE_score_tdcs
    inventory = pd.read_csv(root / 'tdcs-adjusted/results.csv')
    inventory = inventory.merge(matched[keys + ['MAEReduction', 'RMSEReduction']],
                                on=keys, how='left', validate='one_to_one')
    inventory['rho'] = inventory.MAEReduction
    inventory['p_raw'] = np.nan
    inventory['p_holm'] = np.nan
    inventory['Status'] = np.where(inventory.MAEReduction.notna(), 'ok', 'unavailable_prediction')
    config, plan = heatmap_plan(paths, inventory)
    spec = plan.figures[plan.figures.FigureKind.eq('composite')
                        & plan.figures.ScaleLabel.eq('UPDRS-III')
                        & plan.figures.RegionValue.eq('SNr')].iloc[0].copy()
    cells = plan.coverage[plan.coverage.FigureID.eq(spec.FigureID)].copy()
    limit = float(np.ceil(cells.MAEReduction.abs().max()))
    config.data['inputs']['value_column'] = 'MAEReduction'
    config.data['style']['heatmap']['value_limits'] = [-limit, limit]
    config.data['style']['colorbar']['ticks'] = [-limit, 0, limit]
    spec['ColorbarLabel'] = 'MAE reduction vs Baseline (points)'
    fig = render_correlation_heatmap_figure(cells, spec, config)
    figures['UPDRS-III_SNr_local_MAE-vs-baseline_heatmap.pdf'] = fig
    cells = cells.drop(columns=['rho', 'p_raw', 'p_holm'])
    cells['ValueMeaning'] = 'Baseline-only MAE minus tDCS MAE (raw scale points)'
    tables = {'fit-points.csv': points, 'heatmap-cells.csv': cells,
              'figures.csv': pd.DataFrame({'OutputPath': [str(output / name) for name in figures],
                                           'Status': 'preview', 'StatsRoot': str(root)})}
    existing = [output / name for name in [*figures, *tables] if (output / name).exists()]
    if existing:
        _trash_existing(existing)
    output.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(output / name, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(output / name, flush=True)
    for name, table in tables.items():
        atomic_csv(table, output / name, overwrite=False)


if __name__ == '__main__':
    main()
