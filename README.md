# M1 tDCS: STN and SNr LFP analysis

Code supporting the manuscript's LFP, clinical, and structural-connectivity
analyses. The repository contains analysis code, figure renderers, and example
configurations. Participant data, private rosters, and
generated study results are not included.

## Inputs and setup

Analysis starts with LFP-TensorPipe exports and `summary/cohort` tables. Signal
preprocessing, localization, tensor estimation, alignment, and feature export
are performed in LFP-TensorPipe before cohort analysis.

Import the supplied JSON files into the corresponding LFP-TensorPipe stages:

| Configuration | Purpose |
|---|---|
| [Tensor settings](configs/lfptensorpipe/lfptensorpipe_tensor_config.json) | Metric selection, spectral estimation, channel pairs, and burst detection. |
| [Feature settings](configs/lfptensorpipe/lfptensorpipe_features_config.json) | Frequency bands and aligned Pre, Early, Late, and Post windows for feature extraction. |

Apply the feature settings after epoch alignment, then export the features used
by cohort analysis below. Channel and pair names describe the configured
bipolar montage; select matching channels for each recording before computing
tensors. The supplied files preserve the original calculation parameters and
LFP-TensorPipe configuration format versions.

| Input | Content |
|---|---|
| `derivatives/lfptensorpipe/sub-*/tDCS_*/features/*/<metric>/*.pkl` | Scalar, spectral, trace, and raw feature tables; nested values and transform metadata must be preserved. |
| `derivatives/lfptensorpipe/sub-*/tDCS_*/localize/channel_representative_coords.pkl` | Channel coordinates and STN/SNr membership flags. |
| `derivatives/lfptensorpipe/sub-*/tDCS_*/tensor/<metric>/tensor.pkl` and `alignment/<trial>/warp_labels.pkl` | Native tensors and alignment timing for the response-profile baseline bootstrap. |
| `summary/cohort/subj/subject_effect_origin.xlsx` | Clinical table: `ID`, `Protocol`, `Phase`, `Scale`, `Value`, `Baseline`. |
| `summary/cohort/subj/structured/*.csv` | `programming_parameters.csv`, `baseline_covariates.csv`, `tdcs_execution_status.csv`, `contact_pair_regions.csv`; column selections are in `lfp_clinical/programming.py` and `cohort_tables.py`. |
| `summary/cohort/lead/lead_coords.csv` | Right-normalized coordinates and `Region` for the lead-coordinate view. |

Cohort CSVs are read directly; they do not replace nested LFP feature PKLs.
Load only trusted PKL exports. Participant identifiers are supplied at runtime
to join records and define repeated measurements.

```bash
conda env create -f environment.yml
conda activate lfptp
```

Run commands from the repository root. Install LFP-TensorPipe separately.
The following MATLAB tools are included:

| Tool | Purpose and dependencies |
|---|---|
| `lfp_cohort/matlab/flip_coords_lr_nonlinear.m` | Nonlinear hemisphere normalization using MATLAB and Lead-DBS; supporting `require_functions.m` and `table_find_var.m` are in its `helper/` directory. |
| `lfp_viz/matlab/plot_lead_coords_right_atlas.m` | Figure 1c source view from the enriched cohort coordinate CSV; requires MATLAB, Lead-DBS, and the configured atlas. |
| `lfp_viz/matlab/ea_add_ras_triad.m` | R/A/S orientation arrows used by the atlas view; requires MATLAB graphics. |

Structural extraction also requires the relevant Sextant, SimNIBS, atlas,
connectome, and imaging inputs;
these cannot be reconstructed from LFP and clinical tables alone.

Configurations use `/path/to/data/` for private inputs and intermediate tables,
`/path/to/results/` for submission outputs, and `/path/to/` placeholders for
external software. Replace these with your own absolute paths before execution.
Keep private configurations under
ignored `configs/private/` and update their referenced configurations together.
The structural `subjects` entry is fictional and must be replaced privately.
Data, results, and private configurations are excluded from Git. Install the
configured Arial fonts to reproduce the original typography.

## Analysis workflow

1. `lfp_cohort`: merge features, assign anatomy/laterality, apply metric-specific
   transforms, center on Pre, aggregate observations, and split connectivity
   endpoints. The publication configuration disables outlier removal.
2. `lfp_stats`: fit mixed models and export estimates, contrasts, pairing, and
   coverage. Phase models pair contacts; polarity and laterality have their own
   registered comparisons.
3. `lfp_clinical`: derive adjacent-phase changes, join clinical/programming
   tables, and compute correlations, multiplicity adjustment, and programming
   jitter sensitivity. Contact matching uses recorded physical contacts.
4. `lfp_structural`: extract field/connectivity predictors and fit contact-level
   models. Saved model rows and predictions can be used for rendering.
5. `lfp_viz`: render component panels; final Illustrator assembly is separate.

```bash
python -B -m pipeline.cohort_preproc --config configs/cohort_preproc.yaml --stage all
python -B -m pipeline.stats --config configs/stats/phase_by_lat_contact_paired.yaml
python -B -m pipeline.submission_polar --config configs/submission_polar.yaml --stage all
python -B -m pipeline.stats --config configs/stats/laterality_within_polar.yaml
python -B -m pipeline.clinical_correlation_stats --config configs/stats/clinical_correlation.yaml
python -B -m pipeline.clinical_correlation_stats --config configs/stats/clinical_correlation_adjacent_phase.yaml
python -B -m pipeline.lfp_programming_intensity_stats --config configs/stats/lfp_programming_intensity_contact_matched.yaml
python -B -m pipeline.viz --config configs/viz/submission_phase_contact_scalar.yaml
```

Render other panel families using `pipeline.viz --config` and the configurations
below. For laterality, render `submission_lat_region_scalar.yaml` before the
trajectory configuration, which consumes its statistics and pairing tables.
Trajectory counts show unique participants per phase; Pre uses the paired
component-model population.

Figure 3 uses a 60-second Pre reconstruction, 10-second circular block bootstrap,
and Q75 thresholds to classify the significant Ipsi Phase response profiles:

```bash
python -B -m lfp_viz.phase_response_schematic --output-root /path/to/results/response_schematic
python -B -m lfp_stats.phase_response_bootstrap --data-root /path/to/data/derivatives/lfptensorpipe --stats-root /path/to/results/submission/data/Phase/scalar --output-root /path/to/results/response_bootstrap
python -B -m lfp_viz.phase_response_profiles --stats-root /path/to/results/submission/data/Phase/scalar --calibration-root /path/to/results/response_bootstrap --output-root /path/to/results/submission/data/Phase/response_profiles
```

For structural analysis, use `pipeline.structural_connectivity --config
configs/structural_connectivity.yaml --stage STAGE`. Following predictor
extraction, run `contact-lmm-adjacent`, then
`contact-lmm-adjacent-field-excess-local`. Render using
`contact-lmm-adjacent-field-excess-heatmaps` and
`contact-lmm-adjacent-field-excess-atlas-grid-fits`. Earlier extraction stages
are listed by this command's `--help` and `pipeline.structural_simnibs --help`.
Use fresh output directories or explicitly request replacement where supported.

## LCT associations, prediction, and cohort tables

The adjacent-phase clinical workflow first exports `clinical_endpoints.csv` and
`adjacent_phase_predictors.csv`. LCT analyses join these with
`baseline_covariates.csv`, including `LCT_ImprovementPercent`. LCT–outcome partial
associations (Supplementary Figure 20) control for Baseline only. LCT-adjusted
tDCS–outcome partial associations (Figure 7d, Supplementary Figure 27, and
Supplementary Figures 21–26) control for both Baseline and LCT improvement;
each covariate is ranked separately. Prediction uses
leave-one-participant-out OLS with matched complete cases across compared models.
Baseline-only and LCT reference panels are shared only for identical endpoint,
scale, model, and participant membership. The renderers reuse the clinical fit
style and its participant colors.

```bash
python -B -m lfp_clinical.lct_runner --config configs/stats/clinical_correlation_lct.yaml
python -B -m lfp_clinical.lct_prediction --config configs/stats/clinical_correlation_lct.yaml
python -B -m lfp_clinical.lct_prediction --config configs/stats/clinical_correlation_lct.yaml --baseline-only
python -B -m lfp_viz.lct_preview --config configs/viz/clinical_correlation_lct.yaml --all
python -B -m lfp_viz.lct_prediction_figures --config configs/viz/clinical_prediction.yaml
python -B -m lfp_viz.phase_contact_trajectories --stats-config configs/stats/phase_by_lat_contact_paired.yaml --profiles /path/to/results/submission/data/Phase/response_profiles/profiles.csv --output-root /path/to/results/contact_paired_trajectories
python -B -m lfp_clinical.cohort_tables --cohort-root /path/to/data/summary/cohort/subj/structured --output-root /path/to/results/cohort_tables
```

The contact trajectory renderer rebuilds paired model inputs using the existing
stats planner, selects significant Ipsi models from `profiles.csv`, and preserves
Pre, Early, Late, and Post observations without refitting models. It does not
require temporary LOSO outputs. Cohort table export reads the four structured
CSVs listed below and writes five CSV tables and a matching XLSX workbook.
Missing programming entries remain missing. All runtime identifiers and outputs
belong in private data/results directories.

Rerun LCT associations and predictions when their endpoint, predictor, or LCT
inputs change; rerender their downstream panels after corresponding results
change. Trajectories depend on paired feature inputs, the Phase configuration,
and the selected profiles. Cohort tables depend only on their corresponding CSV.
Display changes require rendering only. No global invalidation is introduced.

## Manuscript figures and tables

This mapping follows 56 figure YAMLs (153 panel records): Figures 1–8 and Supplementary
Figures 1–48. Private source-image paths are omitted. Configuration names below
are relative to `configs/viz/`. Final artwork assembly remains external.

Figure 7a–c show baseline-adjusted coupling–outcome associations: UPDRS-III in
panels a–b and PDQ-39 in panel c. Panel d additionally adjusts the association
in panel b for LCT improvement, following the order of the manuscript narrative.

Supplementary Figure 19 contains baseline-adjusted theta-coupling association
examples, and Supplementary Figure 27 contains the baseline- and LCT-adjusted
delta-ciPLV example. Supplementary Figures 28–39 show the complete prediction-error
heatmaps; Supplementary Figures 40–41 show prediction examples. Supplementary Figure 40 places
clinical-reference panels a–d before connectivity panels e–h, in a 3–3–2 layout
with the final row centered. Figure assembly and numbering changes require only
artwork and document updates; they do not invalidate statistical results.

| Figure/panels | Rendering source |
|---|---|
| 1d–i; 2a–o; S4a–i | `submission_phase_region_raw.yaml`, `submission_phase_region_spectral.yaml`, `submission_phase_region_trace.yaml`, `submission_phase_contact_scalar.yaml`; `lfp_viz/raw.py`, `series.py`, `scalar.py`. Figure 2c uses the trace configuration. |
| 1c | `lfp_viz/matlab/plot_lead_coords_right_atlas.m`, from the cohort coordinate CSV and Lead-DBS atlas. |
| 3a | `lfp_viz/phase_response_schematic.py`. |
| 3b | `lfp_stats/phase_response_bootstrap.py` and `lfp_viz/phase_response_profiles.py`. |
| 4a–f | `submission_polar_trajectory.yaml`; `lfp_viz/lat_trajectory.py`. |
| 4g–j | `submission_lat_trajectory.yaml`; panel h: Anodal STN high-beta periodic power; panel i: Cathodal SNr high-beta periodic power. |
| 6a–b; S12d–g | Structural atlas-grid fits: `lfp_structural/contact_lmm_adjacent.py` and `contact_lmm.py`. |
| 7a–c; S19a–c | `submission_clinical_correlation_adjacent_phase_fit.yaml`; `lfp_viz/correlation_fit.py`. |
| 7d; S20a–d; S27 | `clinical_correlation_lct.yaml`; `lfp_viz/lct_preview.py`. |
| 7e–l; S40a–h; S41a–h | `clinical_prediction.yaml`; `lfp_viz/lct_prediction_figures.py`, using `lct_prediction_preview.py` rendering helpers. |
| 8a–b | `lfp_programming_intensity_contact_matched.yaml`; `lfp_viz/programming.py`. |
| S1–S3 | `submission_phase_contact_scalar_heatmap.yaml`; `lfp_viz/heatmap.py`. |
| S5a–p | `lfp_viz/phase_contact_trajectories.py`; paired observations for significant Ipsi profiles. |
| S6 | `submission_polar_contact_scalar_heatmap.yaml`. |
| S7 | `submission_lat_region_scalar_heatmap.yaml`. |
| S8–S11 | Structural atlas heatmaps: `contact-lmm-adjacent-field-excess-heatmaps` stage. |
| S12a–c | `submission_phase_contact_scalar.yaml`. |
| S13–S18 | `submission_clinical_correlation_scale_heatmap.yaml`; adjacent-phase clinical heatmaps. |
| S21–S26 | `clinical_correlation_lct.yaml`; LCT-adjusted association heatmaps. |
| S28–S33 | `clinical_prediction.yaml`; UPDRS-III MAE reduction versus Baseline (S28–S30) and LCT (S31–S33), for SNr, STN, and connectivity. |
| S34–S39 | The corresponding PDQ-39 MAE reduction heatmaps versus Baseline (S34–S36) and LCT (S37–S39). |
| S42–S47 | `lfp_programming_intensity_contact_matched.yaml`; unadjusted, UPDRS-III adjusted, and PDQ-39 adjusted local/connectivity heatmaps. |
| S48a–d | Adjusted fits and jitter panels from the same programming configuration. |
| 1a–b; 5a; 8c | Study/anatomy diagrams and structural QC composites assembled with external software. |

| Supplementary table | Input to `lfp_clinical/cohort_tables.py` |
|---|---|
| 1 | `baseline_covariates.csv`: age, sex, disease duration, Med-OFF UPDRS-III, PDQ-39. |
| 2 | `tdcs_execution_status.csv`: stimulation side and completion by polarity. |
| 3 | `contact_pair_regions.csv`: region by bipolar contact pair and lead side. |
| 4 | `programming_parameters.csv`: STN protocol settings. |
| 5 | `programming_parameters.csv`: STN+SNr protocol settings. |
