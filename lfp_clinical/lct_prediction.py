"""Exploratory matched-patient LCT and tDCS leave-one-out prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lfp_cohort.io import atomic_csv
from .runner import _trash_existing


def loo(design, outcome):
    """Return held-out OLS predictions and per-fold rank status."""
    predicted = np.full(len(outcome), np.nan)
    ranks = np.zeros(len(outcome), dtype=int)
    for i in range(len(outcome)):
        train = np.arange(len(outcome)) != i
        ranks[i] = np.linalg.matrix_rank(design[train])
        if ranks[i] == design.shape[1]:
            beta = np.linalg.lstsq(design[train], outcome[train], rcond=None)[0]
            predicted[i] = design[i] @ beta
    return predicted, ranks


def compute(lct, endpoints, predictors):
    """Compute all eligible candidates without significance filtering."""
    if lct.ID.duplicated().any() or endpoints.duplicated(['ID', 'ScaleLabel']).any():
        raise ValueError('Clinical patient keys must be unique.')
    if predictors.duplicated(['ID', 'PredictorID']).any():
        raise ValueError('Predictor patient keys must be unique.')
    meta = predictors.drop(columns=['ID', 'Value', 'FromValue', 'ToValue']).drop_duplicates()
    if meta.PredictorID.duplicated().any():
        raise ValueError('Predictor metadata must be invariant across patients.')
    pivot = predictors.pivot(index='ID', columns='PredictorID', values='Value')
    ids = lct.ID.to_numpy()
    l = lct.LCT_ImprovementPercent.to_numpy(float)
    candidates, coverage, predictions, performance, comparisons = [], [], [], [], []
    for scale in ['UPDRS-III', 'PDQ-39']:
        clinical = endpoints[endpoints.ScaleLabel.eq(scale)].set_index('ID').reindex(ids)
        b = clinical.Baseline.to_numpy(float)
        for endpoint, column in [('stn-3m', 'STN3m'), ('stn-snr-3m', 'STNSNr3m')]:
            y = clinical[column].to_numpy(float)
            for metadata in meta.to_dict('records'):
                key = dict(metadata, ScaleLabel=scale, EndpointID=endpoint)
                x = pivot[metadata['PredictorID']].reindex(ids).to_numpy(float)
                checks = {'missing_baseline': np.isfinite(b), 'nonpositive_baseline': b > 0,
                          'missing_outcome': np.isfinite(y), 'missing_lct': np.isfinite(l),
                          'missing_predictor': np.isfinite(x)}
                mask = np.logical_and.reduce(list(checks.values()))
                n = int(mask.sum())
                candidates.append(dict(key, n=n, Eligible=n >= 6,
                                       SubjectIDs=';'.join(ids[mask])))
                for j, subject in enumerate(ids):
                    coverage.append(dict(key, ID=subject, Baseline=b[j], Outcome=y[j],
                                         LCT=l[j], PredictorValue=x[j], Included=bool(mask[j]),
                                         CandidateEligible=n >= 6,
                                         MissingReason=';'.join(k for k, v in checks.items() if not v[j])))
                if n < 6:
                    continue
                baseline, outcome, lct_value, predictor = b[mask], y[mask], l[mask], x[mask]
                response = 100 * (baseline - outcome) / baseline
                for formulation in ['improvement', 'baseline_adjusted']:
                    prefix = [np.ones(n)] if formulation == 'improvement' else [np.ones(n), baseline]
                    target = response if formulation == 'improvement' else outcome
                    models = {'LCT': [lct_value], 'tDCS': [predictor], 'joint': [lct_value, predictor]}
                    errors, scores = {}, {}
                    for model, columns in models.items():
                        design = np.column_stack(prefix + columns)
                        pred, ranks = loo(design, target)
                        pred_response = pred if formulation == 'improvement' else 100 * (baseline - pred) / baseline
                        error = np.abs(response - pred_response)
                        ok = bool(np.isfinite(pred_response).all())
                        mae = float(error.mean()) if ok else np.nan
                        rmse = float(np.sqrt(np.mean((response - pred_response) ** 2))) if ok else np.nan
                        errors[model], scores[model] = error, (mae, rmse)
                        performance.append(dict(key, Formulation=formulation, Model=model, n=n,
                                                Status='ok' if ok else 'rank_deficient_fold',
                                                ValidFolds=int(np.isfinite(pred).sum()),
                                                MAE_pp=mae, RMSE_pp=rmse,
                                                MAE_score=float(np.mean(np.abs(outcome-pred))) if ok and formulation == 'baseline_adjusted' else np.nan,
                                                RMSE_score=float(np.sqrt(np.mean((outcome-pred)**2))) if ok and formulation == 'baseline_adjusted' else np.nan))
                        for j, subject in enumerate(ids[mask]):
                            predictions.append(dict(key, Formulation=formulation, Model=model, ID=subject,
                                                    n=n, TrainingN=n-1, TrainingRank=int(ranks[j]),
                                                    Parameters=design.shape[1],
                                                    TrainingResidualDF=n-1-int(ranks[j]),
                                                    Status='ok' if np.isfinite(pred[j]) else 'rank_deficient_fold',
                                                    Baseline=baseline[j], Outcome=outcome[j], LCT=lct_value[j],
                                                    PredictorValue=predictor[j], ObservedImprovement=response[j],
                                                    PredictedImprovement=pred_response[j], AbsoluteError_pp=error[j],
                                                    PredictedOutcome=pred[j] if formulation == 'baseline_adjusted' else np.nan))
                    for candidate in ['tDCS', 'joint']:
                        ok = np.isfinite(scores['LCT'][0]) and np.isfinite(scores[candidate][0])
                        comparisons.append(dict(key, Formulation=formulation, Comparator='LCT', Candidate=candidate,
                                                n=n, Status='ok' if ok else 'rank_deficient_fold',
                                                DeltaMAE_pp=scores['LCT'][0]-scores[candidate][0],
                                                DeltaRMSE_pp=scores['LCT'][1]-scores[candidate][1],
                                                PatientsLowerError=int(np.sum(errors[candidate] < errors['LCT'])) if ok else np.nan))
    return {name+'.csv': pd.DataFrame(rows) for name, rows in [
        ('candidates', candidates), ('coverage', coverage), ('predictions', predictions),
        ('performance', performance), ('comparisons', comparisons)]}


def compute_null(predictions, performance):
    """Compare saved improvement models with matched training-only means."""
    keys = ['PredictorID', 'ScaleLabel', 'EndpointID']
    saved = predictions[predictions.Formulation.eq('improvement')].copy()
    null = saved[saved.Model.eq('LCT')].copy().reset_index(drop=True)
    if null.duplicated(keys + ['ID']).any():
        raise ValueError('Saved LCT patient keys must be unique.')
    null['Model'] = 'null'
    for _, group in null.groupby(keys, sort=False):
        predicted, _ = loo(np.ones((len(group), 1)), group.ObservedImprovement.to_numpy())
        null.loc[group.index, 'PredictedImprovement'] = predicted
    null['AbsoluteError_pp'] = abs(null.ObservedImprovement - null.PredictedImprovement)
    null['TrainingRank'] = 1
    null['Parameters'] = 1
    null['TrainingResidualDF'] = null.TrainingN - 1
    null['Status'] = 'ok'
    squared = null.assign(SquaredError=null.AbsoluteError_pp ** 2)
    scores = squared.groupby(keys).agg(
        NullMAE=('AbsoluteError_pp', 'mean'), NullMSE=('SquaredError', 'mean'))
    scores['NullRMSE'] = np.sqrt(scores.pop('NullMSE'))
    perf = performance[performance.Formulation.eq('improvement')].copy()
    perf = perf.merge(scores, on=keys, validate='many_to_one')
    null_perf = perf[perf.Model.eq('LCT')].copy()
    null_perf['Model'] = 'null'
    null_perf['Status'] = 'ok'
    null_perf['ValidFolds'] = null_perf.n
    null_perf['MAE_pp'] = null_perf.NullMAE
    null_perf['RMSE_pp'] = null_perf.NullRMSE
    null_perf = null_perf.drop(columns=['NullMAE', 'NullRMSE'])
    paired = saved.merge(null[keys + ['ID', 'AbsoluteError_pp']], on=keys + ['ID'],
                         suffixes=('', '_null'), validate='many_to_one')
    wins = paired.assign(Win=paired.AbsoluteError_pp < paired.AbsoluteError_pp_null)
    wins = wins.groupby(keys + ['Model']).Win.sum().rename('PatientsLowerError')
    comparisons = perf.merge(wins, on=keys + ['Model'], validate='one_to_one')
    comparisons['Comparator'] = 'null'
    comparisons['DeltaMAE_pp'] = comparisons.NullMAE - comparisons.MAE_pp
    comparisons['DeltaRMSE_pp'] = comparisons.NullRMSE - comparisons.RMSE_pp
    comparisons.loc[comparisons.Status.ne('ok'), 'PatientsLowerError'] = np.nan
    comparisons = comparisons.rename(columns={'Model': 'Candidate'}).drop(
        columns=['NullMAE', 'NullRMSE', 'MAE_pp', 'RMSE_pp', 'MAE_score', 'RMSE_score', 'ValidFolds'])
    return {'predictions.csv': null, 'performance.csv': null_perf,
            'comparisons.csv': comparisons}


def compute_baseline_only(predictions, performance):
    """Predict raw outcomes from baseline on each saved matched cohort."""
    keys = ['PredictorID', 'ScaleLabel', 'EndpointID']
    pred = predictions[predictions.Formulation.eq('baseline_adjusted')
                       & predictions.Model.eq('LCT')].copy().reset_index(drop=True)
    if pred.duplicated(keys + ['ID']).any():
        raise ValueError('Saved LCT patient keys must be unique.')
    pred = pred.drop(columns=['ObservedImprovement', 'PredictedImprovement', 'AbsoluteError_pp'])
    pred['Model'] = 'baseline-only'
    pred['Parameters'] = 2
    for _, group in pred.groupby(keys, sort=False):
        design = np.column_stack([np.ones(len(group)), group.Baseline.to_numpy()])
        values, ranks = loo(design, group.Outcome.to_numpy())
        pred.loc[group.index, 'PredictedOutcome'] = values
        pred.loc[group.index, 'TrainingRank'] = ranks
    pred['TrainingResidualDF'] = pred.TrainingN - pred.TrainingRank
    pred['Status'] = np.where(np.isfinite(pred.PredictedOutcome), 'ok', 'rank_deficient_fold')
    pred['AbsoluteError_score'] = abs(pred.Outcome - pred.PredictedOutcome)
    scores = pred.assign(SquaredError=pred.AbsoluteError_score ** 2).groupby(keys).agg(
        MAE_score=('AbsoluteError_score', 'mean'), MSE=('SquaredError', 'mean'),
        ValidFolds=('PredictedOutcome', 'count'))
    scores['RMSE_score'] = np.sqrt(scores.pop('MSE'))
    perf = performance[performance.Formulation.eq('baseline_adjusted')
                       & performance.Model.eq('LCT')].copy()
    perf = perf.drop(columns=['MAE_pp', 'RMSE_pp', 'MAE_score', 'RMSE_score', 'ValidFolds'])
    perf = perf.merge(scores, on=keys, validate='one_to_one')
    perf['Model'] = 'baseline-only'
    ok = perf.ValidFolds.eq(perf.n)
    perf['Status'] = np.where(ok, 'ok', 'rank_deficient_fold')
    perf.loc[~ok, ['MAE_score', 'RMSE_score']] = np.nan
    return {'predictions.csv': pred, 'performance.csv': perf}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    supplement = parser.add_mutually_exclusive_group()
    supplement.add_argument('--null-only', action='store_true',
                        help='Supplement saved improvement predictions with an intercept-only model.')
    supplement.add_argument('--baseline-only', action='store_true',
                            help='Supplement saved raw-outcome predictions with a baseline-only model.')
    args = parser.parse_args()
    paths = yaml.safe_load(args.config.read_text())['paths']
    root = Path(paths['output_root']) / 'prediction'
    if args.null_only or args.baseline_only:
        inputs = {name: str(root / (name + '.csv')) for name in ['predictions', 'performance']}
        tables = {name: pd.read_csv(path) for name, path in inputs.items()}
        outputs = compute_baseline_only(**tables) if args.baseline_only else compute_null(**tables)
        root = root / ('baseline-only' if args.baseline_only else 'null-model')
    else:
        inputs = {name: paths[name] for name in ['lct', 'endpoints', 'predictors']}
        tables = {name: pd.read_csv(path) for name, path in inputs.items()}
        outputs = compute(**tables)
    outputs['manifest.csv'] = pd.DataFrame(
        [{'Role': 'input', 'Path': inputs[k], 'Rows': len(v)} for k, v in tables.items()]
        + [{'Role': 'config', 'Path': str(args.config.resolve())},
           {'Role': 'code', 'Path': str(Path(__file__).resolve())}]
        + [{'Role': 'output', 'Path': str(root/k), 'Rows': len(v)} for k, v in outputs.items()])
    existing = [root/name for name in outputs if (root/name).exists()]
    if existing:
        _trash_existing(existing)
    root.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        atomic_csv(frame, root/name, overwrite=False)
    if not (args.null_only or args.baseline_only):
        print(outputs['candidates.csv'].groupby(['n', 'Eligible']).size().to_string())
    print(outputs['performance.csv'].groupby(['Formulation', 'Model', 'Status']).size().to_string())
    print(f'Output: {root}')


if __name__ == '__main__':
    main()
