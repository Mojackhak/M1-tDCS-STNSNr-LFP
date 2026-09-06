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
| `summary/cohort/subj/structured/*.csv` | `programming_parameters.csv`, `baseline_covariates.csv`, `tdcs_execution_status.csv`, `contact_pair_regions.csv`; column contracts are in `lfp_clinical/programming.py`. |
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

## Manuscript figures

This mapping follows the current 32 manuscript panel YAMLs: Figures 1–8 and
Supplementary Figures 1–24. Private source-image paths are omitted.
Configuration names below are relative to `configs/viz/`.

| Figure/panels | Rendering source |
|---|---|
| 1d–i; 2a–o; S4a–i | `submission_phase_region_raw.yaml`, `submission_phase_region_spectral.yaml`, `submission_phase_region_trace.yaml`, `submission_phase_contact_scalar.yaml`; `lfp_viz/raw.py`, `series.py`, `scalar.py`. Figure 2c uses the trace configuration. |
| 1c | `lfp_viz/matlab/plot_lead_coords_right_atlas.m`, from the cohort coordinate CSV and Lead-DBS atlas. |
| 3a | `lfp_viz/phase_response_schematic.py`. |
| 3b | `lfp_stats/phase_response_bootstrap.py` and `lfp_viz/phase_response_profiles.py`. |
| 4a–f | `submission_polar_trajectory.yaml`; `lfp_viz/lat_trajectory.py`. |
| 4g–j | `submission_lat_trajectory.yaml`; `lfp_viz/lat_trajectory.py`. Panel h: Anodal STN high-beta periodic power; panel i: Cathodal SNr high-beta periodic power. |
| 6a–b; S11d–g | Structural atlas-grid fits: `lfp_structural/contact_lmm_adjacent.py` and `contact_lmm.py`. |
| 7a–f | `submission_clinical_correlation_adjacent_phase_fit.yaml`; `lfp_viz/correlation_fit.py`. |
| 8a–b | `lfp_programming_intensity_contact_matched.yaml`; `lfp_viz/programming.py`. |
| S1–S3 | `submission_phase_contact_scalar_heatmap.yaml`; `lfp_viz/heatmap.py`. |
| S5 | `submission_polar_contact_scalar_heatmap.yaml`. |
| S6 | `submission_lat_region_scalar_heatmap.yaml`. |
| S7–S10 | Structural atlas heatmaps: `contact-lmm-adjacent-field-excess-heatmaps` stage. |
| S11a–c | `submission_phase_contact_scalar.yaml`. |
| S12–S17 | `submission_clinical_correlation_scale_heatmap.yaml`; `lfp_viz/correlation_heatmap.py`, using the adjacent-phase clinical analysis. |
| S18–S23 | `lfp_programming_intensity_contact_matched.yaml`; unadjusted, UPDRS-III adjusted, and PDQ-39 adjusted local/connectivity heatmaps. |
| S24a–d | Adjusted fits and jitter panels from the same programming configuration. |
| 1a–b; 5a; 8c | Study/anatomy diagrams and structural QC composites assembled with external software. Their final artwork is outside the Python rendering commands. |
