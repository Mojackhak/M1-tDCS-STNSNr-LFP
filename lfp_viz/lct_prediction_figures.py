"""Export approved raw-score LOOCV fits and complete composite heatmaps."""

import argparse
import plistlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .lct_prediction_preview import draw_prediction, fold_coefficients
from .lct_preview import heatmap_plan
from .correlation_heatmap import render_correlation_heatmap_figure
from lfp_clinical.runner import _trash_existing
from lfp_cohort.io import atomic_csv
import matplotlib.pyplot as plt


KEYS = ['PredictorID', 'ScaleLabel', 'EndpointID']
ITEM_KEYS = ['Polar', 'Lat', 'RegionValue', 'Metric', 'FeatureOutput', 'Band']


def consolidate_references(output, style):
    """Merge cohort-identical reference figures and preserve their provenance."""
    figures = pd.read_csv(output / 'figures.csv')
    points = pd.read_csv(output / 'fit-points.csv')
    coverage = pd.read_csv(output / 'coverage.csv')
    refs = figures[figures.Model.isin(['baseline-only', 'LCT'])].copy()
    refs['SubjectIDs'] = refs.SubjectIDs.map(lambda value: ';'.join(sorted(value.split(';'))))
    point_groups = dict(tuple(points.groupby('OutputPath', sort=False)))
    groups = list(refs.groupby(['ScaleLabel', 'EndpointID', 'Model', 'SubjectIDs'], sort=False))
    plans = []
    numeric = ['Baseline', 'Outcome', 'LCT', 'PredictedOutcome', 'FoldCoefficient']
    for (scale, endpoint, model, subjects), rows in groups:
        source = point_groups[rows.OutputPath.iloc[0]].sort_values('ID').copy()
        for path in rows.OutputPath:
            other = point_groups[path].sort_values('ID')
            if source.ID.tolist() != other.ID.tolist():
                raise ValueError(f'Reference patient mismatch: {path}')
            np.testing.assert_allclose(source[numeric], other[numeric], atol=1e-8, rtol=1e-10)
        target = output / 'fit' / scale / endpoint / 'reference' / f'{model}_subjects-{subjects.replace(";", "-")}.pdf'
        plans.append((rows, source, target))
    path_map, records, clean_points = {}, [], []
    tag_key = 'com.apple.metadata:_kMDItemUserTags'
    for rows, source, target in plans:
        row = rows.iloc[0]
        baseline_row = refs[refs.ScaleLabel.eq(row.ScaleLabel) & refs.EndpointID.eq(row.EndpointID)
                            & refs.SubjectIDs.eq(row.SubjectIDs) & refs.Model.eq('baseline-only')].iloc[0]
        baseline = point_groups[baseline_row.OutputPath]
        tags = []
        old_paths = list(dict.fromkeys([Path(p) for p in rows.OutputPath] + [target]))
        for path in old_paths:
            if path.exists():
                attrs = subprocess.check_output(['/usr/bin/xattr', str(path)], text=True).splitlines()
                if tag_key in attrs:
                    payload = subprocess.check_output(['/usr/bin/xattr', '-px', tag_key, str(path)], text=True)
                    tags.extend(plistlib.loads(bytes.fromhex(payload)))
        fig = draw_prediction(source, row.Model, style, baseline)
        if target.exists():
            _trash_existing([target])
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=600, bbox_inches='tight')
        plt.close(fig)
        if tags:
            payload = plistlib.dumps(list(dict.fromkeys(tags)), fmt=plistlib.FMT_BINARY).hex()
            subprocess.run(['/usr/bin/xattr', '-wx', tag_key, payload, str(target)], check=True)
        path_map.update({path: str(target) for path in rows.OutputPath})
        keep = ['ScaleLabel', 'EndpointID', 'Model', 'SubjectIDs', 'n', 'FigureType', 'Status',
                'MAE_score', 'RMSE_score', 'DeltaMAE_score', 'DeltaRMSE_score']
        record = row[keep].to_dict()
        record['OutputPath'] = str(target)
        records.append(record)
        point_columns = ['ScaleLabel', 'EndpointID', 'Formulation', 'Model', 'ID', 'n',
                         'TrainingN', 'TrainingRank', 'Parameters', 'TrainingResidualDF', 'Status',
                         'Baseline', 'Outcome', 'LCT', 'PredictedOutcome', 'AbsoluteError_score',
                         'FoldCoefficient', 'CoefficientTerm']
        source = source[point_columns].copy()
        source['OutputPath'], source['SubjectIDs'] = str(target), row.SubjectIDs
        clean_points.append(source)
    old_targets = [Path(path) for path in path_map if path not in set(path_map.values()) and Path(path).exists()]
    _trash_existing(old_targets)
    coverage['OutputPath'] = coverage.OutputPath.map(lambda path: path_map.get(path, path))
    mapping = coverage[coverage.Model.isin(['baseline-only', 'LCT']) & coverage.Status.eq('ok')].copy()
    mapping = mapping[KEYS + ['Model', 'SubjectIDs', 'OutputPath']].rename(columns={'OutputPath': 'ReferencePath'})
    tdcs_paths = coverage[coverage.Model.eq('tDCS') & coverage.Status.eq('ok')][KEYS + ['OutputPath']]
    mapping = mapping.merge(tdcs_paths, on=KEYS, validate='many_to_one').rename(columns={'OutputPath': 'TDCSPath'})
    tables = {'figures.csv': pd.concat([figures[~figures.Model.isin(['baseline-only', 'LCT'])], pd.DataFrame(records)], ignore_index=True),
              'fit-points.csv': pd.concat([points[points.Model.eq('tDCS')], *clean_points], ignore_index=True),
              'coverage.csv': coverage, 'reference-mapping.csv': mapping}
    for name, table in tables.items():
        target = output / name
        if target.exists():
            _trash_existing([target])
        atomic_csv(table, target, overwrite=False)
    print(f'Consolidated {len(refs)} reference records to {len(records)} PDFs; trashed {len(old_targets)} old PDFs.', flush=True)


def selected_candidates(candidates, items):
    """Retain requested order and unavailable candidates in the inventory."""
    inventory = pd.DataFrame(items, columns=ITEM_KEYS)
    inventory['ItemOrder'] = np.arange(1, len(inventory) + 1)
    return inventory.merge(candidates, on=ITEM_KEYS, validate='one_to_many')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--references-only', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    paths = config['paths']
    root, output = Path(paths['stats_root']), Path(paths['output_root'])
    if args.references_only:
        style = yaml.safe_load(Path(paths['fit_style']).read_text())['style']
        consolidate_references(output, style)
        return
    prediction_root = root / 'prediction'
    predictions = pd.read_csv(prediction_root / 'predictions.csv')
    performance = pd.read_csv(prediction_root / 'performance.csv')
    baseline_points = pd.read_csv(prediction_root / 'baseline-only/predictions.csv')
    baseline_scores = pd.read_csv(prediction_root / 'baseline-only/performance.csv')
    candidates = pd.read_csv(prediction_root / 'candidates.csv')
    selected = selected_candidates(candidates, config['fit_items'])
    points = pd.concat([baseline_points, predictions[
        predictions.Formulation.eq('baseline_adjusted') & predictions.Model.isin(['LCT', 'tDCS'])]],
        ignore_index=True)
    groups = {key: group for key, group in points.groupby(KEYS, sort=False)}
    style = yaml.safe_load(Path(paths['fit_style']).read_text())['style']
    records, coverage, plotted_points, heat_cells = [], [], [], []

    def export(fig, target):
        if target.exists():
            _trash_existing([target])
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=600, bbox_inches='tight')
        plt.close(fig)

    for row in selected.to_dict('records'):
        key = tuple(row[k] for k in KEYS)
        if not row['Eligible']:
            for model in ['baseline-only', 'LCT', 'tDCS']:
                coverage.append(dict(row, Model=model, Status='insufficient_n', OutputPath=''))
            continue
        group = groups[key]
        reference = group[group.Model.eq('baseline-only')]
        ref_error = reference.Outcome - reference.PredictedOutcome
        for model in ['baseline-only', 'LCT', 'tDCS']:
            model_points = group[group.Model.eq(model)].copy()
            model_points['FoldCoefficient'] = fold_coefficients(model_points, model)
            model_points['CoefficientTerm'] = {'baseline-only': 'B', 'LCT': 'LCT', 'tDCS': 'X'}[model]
            if set(model_points.ID) != set(reference.ID):
                raise ValueError(f'Model patients do not match baseline: {key}/{model}')
            error = model_points.Outcome - model_points.PredictedOutcome
            name = '_'.join(str(row[k]).replace('_', '-').replace('→', '-to-').replace('−', '-')
                            for k in ['Band', 'FeatureOutput', 'RegionValue', 'Polar', 'Lat', 'PhaseContrast'])
            target = output / 'fit' / row['ScaleLabel'] / row['EndpointID'] / row['Domain'] / row['Metric'] / f'{name}_{model}.pdf'
            if model == 'tDCS':
                fig = draw_prediction(model_points, model, style, reference)
                export(fig, target)
            record = dict(row, Model=model, FigureType='fit', OutputPath=str(target), Status='ok',
                          MAE_score=float(error.abs().mean()), RMSE_score=float(np.sqrt((error**2).mean())),
                          DeltaMAE_score=float(ref_error.abs().mean()-error.abs().mean()),
                          DeltaRMSE_score=float(np.sqrt((ref_error**2).mean())-np.sqrt((error**2).mean())))
            records.append(record)
            coverage.append(record)
            model_points['OutputPath'] = str(target)
            model_points['ItemOrder'] = row['ItemOrder']
            plotted_points.append(model_points)
            if len(records) % 30 == 0:
                print(f'Fit model combinations processed: {len(records)}', flush=True)

    perf = performance[performance.Formulation.eq('baseline_adjusted')]
    tdcs = perf[perf.Model.eq('tDCS')]
    inventory = pd.read_csv(root / 'tdcs-adjusted/results.csv')
    inventory = inventory[[column for column in candidates.columns if column in inventory.columns]].copy()
    comparisons = []
    for comparator, ref in [('baseline', baseline_scores), ('lct', perf[perf.Model.eq('LCT')])]:
        values = tdcs.merge(ref[KEYS + ['MAE_score', 'RMSE_score']], on=KEYS,
                            suffixes=('_tdcs', '_reference'), validate='one_to_one')
        values['Comparator'] = comparator
        for metric in ['MAE', 'RMSE']:
            values[metric+'Reduction'] = values[metric+'_score_reference'] - values[metric+'_score_tdcs']
        comparisons.append(values)
    comparisons = pd.concat(comparisons, ignore_index=True)
    for comparator in ['baseline', 'lct']:
        values = comparisons[comparisons.Comparator.eq(comparator)]
        layout = inventory.merge(values[KEYS + ['MAEReduction', 'RMSEReduction']],
                                  on=KEYS, how='left', validate='one_to_one')
        layout['p_raw'], layout['p_holm'] = np.nan, np.nan
        layout['Status'] = np.where(layout.MAEReduction.notna(), 'ok', 'unavailable_prediction')
        for metric in ['MAE', 'RMSE']:
            layout['rho'] = layout[metric+'Reduction']
            heat_config, plan = heatmap_plan(paths, layout)
            heat_config.data['inputs']['value_column'] = metric+'Reduction'
            for _, spec in plan.figures[plan.figures.FigureKind.eq('composite')].iterrows():
                cells = plan.coverage[plan.coverage.FigureID.eq(spec.FigureID)].copy()
                limit = float(np.ceil(comparisons.loc[comparisons.ScaleLabel.eq(spec.ScaleLabel), metric+'Reduction'].abs().max()))
                heat_config.data['style']['heatmap']['value_limits'] = [-limit, limit]
                heat_config.data['style']['colorbar']['ticks'] = [-limit, 0, limit]
                label = 'Baseline' if comparator == 'baseline' else 'LCT'
                spec = spec.copy()
                spec['ColorbarLabel'] = f'{metric} reduction vs {label} (points)'
                target = output / 'heatmap' / f'vs-{comparator}' / metric.lower() / spec.ScaleLabel / spec.Domain / f'{spec.ScaleLabel}_{spec.RegionValue}_{metric}-vs-{comparator}.pdf'
                fig = render_correlation_heatmap_figure(cells, spec, heat_config)
                export(fig, target)
                record = spec.to_dict()
                record.update(FigureType='heatmap', OutputPath=str(target), Status='ok',
                              Comparator=comparator, ErrorMetric=metric, ColorLimit=limit)
                records.append(record)
                cells = cells.drop(columns=['rho', 'p_raw', 'p_holm'])
                cells['Comparator'], cells['ErrorMetric'] = comparator, metric
                cells['OutputPath'] = str(target)
                cells['ValueMeaning'] = f'{label} {metric} minus tDCS {metric} (raw scale points)'
                heat_cells.append(cells)
                print(f'Heatmap: {comparator}/{metric}/{spec.ScaleLabel}/{spec.RegionValue}', flush=True)
    tables = {'figures.csv': pd.DataFrame(records), 'coverage.csv': pd.DataFrame(coverage),
              'fit-points.csv': pd.concat(plotted_points, ignore_index=True),
              'heatmap-cells.csv': pd.concat(heat_cells, ignore_index=True)}
    for name, table in tables.items():
        target = output / name
        if target.exists():
            _trash_existing([target])
        atomic_csv(table, target, overwrite=False)
    consolidate_references(output, style)
    print(f'Completed figure export: {output}', flush=True)


if __name__ == '__main__':
    main()
