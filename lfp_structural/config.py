"""Load the structural-connectivity YAML at the external-input boundary."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SubjectConfig:
    """One registered patient and actual HD-tDCS montage side."""

    subject_id: str
    stimulation_side: str
    acquisition: str
    normalization_approval: float


@dataclass(frozen=True)
class ConnectomeConfig:
    """One registered streamline source."""

    connectome_id: str
    representation: str
    path: Path | None


@dataclass(frozen=True)
class StructuralConnectivityConfig(Mapping[str, Any]):
    """Validated project-specific structural-connectivity configuration."""

    path: Path
    data: dict[str, Any]
    subjects: tuple[SubjectConfig, ...]
    connectomes: tuple[ConnectomeConfig, ...]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def analysis_id(self) -> str:
        return str(self.data["analysis"]["id"])

    def path_value(self, key: str) -> Path:
        return Path(self.data["paths"][key]).expanduser()

    @property
    def leaddbs_root(self) -> Path:
        return self.path_value("leaddbs_root")

    @property
    def sextant_python_root(self) -> Path:
        return self.path_value("sextant_python_root")

    @property
    def leaddbs_derivatives_root(self) -> Path:
        return self.path_value("leaddbs_derivatives_root")

    @property
    def lfptensorpipe_root(self) -> Path:
        return self.path_value("lfptensorpipe_root")

    @property
    def derivative_root(self) -> Path:
        return self.path_value("derivative_root")

    @property
    def summary_root(self) -> Path:
        return self.path_value("summary_root")

    @property
    def mni_reference(self) -> Path:
        return self.path_value("mni_reference")

    @property
    def region_atlas_root(self) -> Path:
        return self.path_value("region_atlas_root")

    @property
    def cortical_region_atlas_root(self) -> Path:
        return self.path_value("cortical_region_atlas_root")

    def output_path(self, key: str) -> Path:
        return self.summary_root / str(self.data["outputs"][key])

    @property
    def contact_lmm_root(self) -> Path:
        return self.output_path("contact_lmm_root")

    @property
    def contact_lmm_adjacent_root(self) -> Path:
        return self.output_path("contact_lmm_adjacent_root")

    @property
    def contact_lmm_adjacent_id_random_intercept_root(self) -> Path:
        return self.output_path(
            "contact_lmm_adjacent_id_random_intercept_root"
        )

    @property
    def cortical_allocation_lmm_root(self) -> Path:
        return self.output_path("cortical_allocation_lmm_root")

    @property
    def cortical_allocation_lmm_adjacent_root(self) -> Path:
        return self.output_path("cortical_allocation_lmm_adjacent_root")

    def subject(self, subject_id: str) -> SubjectConfig:
        for subject in self.subjects:
            if subject.subject_id == subject_id:
                return subject
        raise KeyError(f"Unknown subject: {subject_id}")

    def subject_leaddbs_root(self, subject_id: str) -> Path:
        return self.leaddbs_derivatives_root / subject_id

    def anchor_t1(self, subject_id: str) -> Path:
        subject = self.subject(subject_id)
        return (
            self.subject_leaddbs_root(subject_id)
            / "coregistration"
            / "anat"
            / (
                f"{subject_id}_ses-preop_space-anchorNative_desc-preproc_"
                f"acq-{subject.acquisition}_T1w.nii"
            )
        )

    def anchor_t2(self, subject_id: str) -> Path:
        subject = self.subject(subject_id)
        return (
            self.subject_leaddbs_root(subject_id)
            / "coregistration"
            / "anat"
            / (
                f"{subject_id}_ses-preop_space-anchorNative_desc-preproc_"
                f"acq-{subject.acquisition}_T2w.nii"
            )
        )

    def reconstruction(self, subject_id: str) -> Path:
        return (
            self.subject_leaddbs_root(subject_id)
            / "reconstruction"
            / f"{subject_id}_desc-reconstruction.mat"
        )

    def forward_deformation(self, subject_id: str) -> Path:
        return (
            self.subject_leaddbs_root(subject_id)
            / "normalization"
            / "transformations"
            / (
                f"{subject_id}_from-anchorNative_to-"
                "MNI152NLin2009bAsym_desc-ants.nii.gz"
            )
        )

    def localization_table(self, subject_id: str) -> Path:
        side = self.subject(subject_id).stimulation_side
        return (
            self.lfptensorpipe_root
            / subject_id
            / f"tDCS_excite_{side}"
            / "localize"
            / "channel_representative_coords.csv"
        )

    def individualized_tck(self, subject_id: str) -> Path:
        side = self.subject(subject_id).stimulation_side
        hemisphere = "lh" if side == "L" else "rh"
        return (
            self.subject_leaddbs_root(subject_id)
            / "connectomics"
            / "dMRI"
            / "mrtrix_seed_target"
            / "tractograms"
            / "MNI152NLin2009bAsym"
            / hemisphere
            / "STNSNrplus"
            / "seedwide.tck"
        )

    def region_mask(self, subject_id: str, region: str) -> Path:
        side = self.subject(subject_id).stimulation_side
        hemisphere = "lh" if side == "L" else "rh"
        return self.region_atlas_root / hemisphere / f"{region}.nii.gz"

    def cortical_region_mask(self, subject_id: str, region: str) -> Path:
        side = self.subject(subject_id).stimulation_side
        hemisphere = "lh" if side == "L" else "rh"
        return self.cortical_region_atlas_root / hemisphere / f"{region}.nii.gz"

    def patient_derivative_root(self, subject_id: str) -> Path:
        return self.derivative_root / subject_id

    def native_target(self, subject_id: str) -> Path:
        side = self.subject(subject_id).stimulation_side
        return (
            self.patient_derivative_root(subject_id)
            / "masks"
            / "anchorNative"
            / f"{subject_id}_side-{side}_space-anchorNative_desc-Tstim.nii.gz"
        )

    def mni_target(self, subject_id: str) -> Path:
        side = self.subject(subject_id).stimulation_side
        return (
            self.patient_derivative_root(subject_id)
            / "masks"
            / "MNI152NLin2009bAsym"
            / (f"{subject_id}_side-{side}_space-MNI152NLin2009bAsym_desc-Tstim.nii.gz")
        )

    def native_magnitude(self, subject_id: str) -> Path:
        side = self.subject(subject_id).stimulation_side
        return (
            self.patient_derivative_root(subject_id)
            / "fields"
            / "anchorNative"
            / (
                f"{subject_id}_side-{side}_polarity-Anodal_"
                "space-anchorNative_desc-magnE.nii.gz"
            )
        )

    def native_cortical_ribbon(self, subject_id: str) -> Path:
        return (
            self.patient_derivative_root(subject_id)
            / "masks"
            / "anchorNative"
            / f"{subject_id}_space-anchorNative_desc-corticalRibbon.nii.gz"
        )

    def mni_cortical_magnitude(self, subject_id: str) -> Path:
        side = self.subject(subject_id).stimulation_side
        return (
            self.patient_derivative_root(subject_id)
            / "fields"
            / "MNI152NLin2009bAsym"
            / (
                f"{subject_id}_side-{side}_space-MNI152NLin2009bAsym_"
                "desc-corticalMagnE.nii.gz"
            )
        )

    def field_exposure_manifest(self, subject_id: str) -> Path:
        return (
            self.patient_derivative_root(subject_id)
            / "fields"
            / "field_exposure_manifest.json"
        )

    def membership_archive(self, subject_id: str, connectome_id: str) -> Path:
        return (
            self.patient_derivative_root(subject_id)
            / "fibers"
            / connectome_id
            / "memberships.npz"
        )

    def field_exposure_archive(self, subject_id: str, connectome_id: str) -> Path:
        return (
            self.patient_derivative_root(subject_id)
            / "fibers"
            / connectome_id
            / "field_exposure.npz"
        )

    def field_exposure_archive_manifest(
        self,
        subject_id: str,
        connectome_id: str,
    ) -> Path:
        return (
            self.patient_derivative_root(subject_id)
            / "fibers"
            / connectome_id
            / "field_exposure_manifest.json"
        )

    def native_contact_seed(self, subject_id: str, channel: str) -> Path:
        return (
            self.patient_derivative_root(subject_id)
            / "masks"
            / "anchorNative"
            / (
                f"{subject_id}_channel-{channel}_space-anchorNative_"
                "desc-contactSeed.nii.gz"
            )
        )

    def mni_contact_seed(self, subject_id: str, channel: str) -> Path:
        return (
            self.patient_derivative_root(subject_id)
            / "masks"
            / "MNI152NLin2009bAsym"
            / (
                f"{subject_id}_channel-{channel}_space-MNI152NLin2009bAsym_"
                "desc-contactSeed.nii.gz"
            )
        )


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration key {key!r} must contain a mapping.")
    return value


def _nonempty_string(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"Configuration key {key!r} must be a non-empty string.")
    return value


def _load_subjects(data: Mapping[str, Any]) -> tuple[SubjectConfig, ...]:
    rows = data.get("subjects")
    if not isinstance(rows, list) or not rows:
        raise TypeError("subjects must be a non-empty list.")
    subjects: list[SubjectConfig] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Each subject must be a mapping.")
        subject = SubjectConfig(
            subject_id=_nonempty_string(row, "id"),
            stimulation_side=_nonempty_string(row, "stimulation_side"),
            acquisition=_nonempty_string(row, "acquisition"),
            normalization_approval=float(row["normalization_approval"]),
        )
        if subject.stimulation_side not in {"L", "R"}:
            raise ValueError("stimulation_side must be L or R.")
        subjects.append(subject)
    identifiers = [subject.subject_id for subject in subjects]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Subject identifiers must be unique.")
    return tuple(subjects)


def _load_connectomes(data: Mapping[str, Any]) -> tuple[ConnectomeConfig, ...]:
    rows = data.get("connectomes")
    if not isinstance(rows, list) or not rows:
        raise TypeError("connectomes must be a non-empty list.")
    connectomes: list[ConnectomeConfig] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Each connectome must be a mapping.")
        representation = _nonempty_string(row, "representation")
        path = Path(row["path"]).expanduser() if row.get("path") else None
        if representation not in {"tck", "leaddbs_hdf5"}:
            raise ValueError(f"Unsupported connectome representation: {representation}")
        if representation == "leaddbs_hdf5" and path is None:
            raise ValueError("A Lead-DBS HDF5 connectome requires path.")
        connectomes.append(
            ConnectomeConfig(
                connectome_id=_nonempty_string(row, "id"),
                representation=representation,
                path=path,
            )
        )
    identifiers = [source.connectome_id for source in connectomes]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Connectome identifiers must be unique.")
    return tuple(connectomes)


def _validate_frozen_contract(data: Mapping[str, Any]) -> None:
    masks = _mapping(data, "masks")
    if float(masks.get("contact_radius_mm", -1)) != 2.4:
        raise ValueError("The contact seed radius must be 2.4 mm.")
    if float(masks.get("region_probability_threshold", -1)) != 0.05:
        raise ValueError("The Region probability threshold must be 0.05.")
    if float(masks.get("field_threshold_v_per_m", -1)) != 0.10:
        raise ValueError("The field threshold must be 0.10 V/m.")
    inference = _mapping(data, "inference")
    if inference.get("lift_metrics") != ["BinaryLift", "FieldWeightedLift"]:
        raise ValueError("The Lift metrics must be BinaryLift and FieldWeightedLift.")
    if int(inference.get("minimum_n", -1)) != 3:
        raise ValueError("The minimum analysis sample size must be 3.")
    if int(inference.get("bh_planned_tests", -1)) != 21:
        raise ValueError("Each BH family must contain 21 planned tests.")
    contact_lmm = _mapping(inference, "contact_lmm")
    if contact_lmm.get("formula") != (
        "Value ~ Phase * X_z + (1 | ID) + (1 | ID:ContactUnitID)"
    ):
        raise ValueError(
            "The contact-level mixed-model formula does not match the contract."
        )
    if contact_lmm.get("predictor_metrics") != [
        "BinaryLift",
        "FieldWeightedLift",
        "FieldExcess",
        "SeedFieldExposure",
    ]:
        raise ValueError(
            "The contact-level predictor metrics must be BinaryLift, "
            "FieldWeightedLift, FieldExcess, and SeedFieldExposure."
        )
    if int(contact_lmm.get("minimum_n", -1)) != 3:
        raise ValueError(
            "The contact-level mixed-model minimum patient count must be 3."
        )
    if contact_lmm.get("phase_levels") != ["Early", "Late", "Post"]:
        raise ValueError("The contact-level Phase order must be Early, Late, and Post.")
    if contact_lmm.get("reml") is not True:
        raise ValueError("The contact-level mixed model must use REML.")
    if contact_lmm.get("degrees_of_freedom") != "kenward-roger":
        raise ValueError(
            "The contact-level mixed model must use Kenward-Roger degrees of freedom."
        )
    if contact_lmm.get("slope_p_adjust") != "holm":
        raise ValueError(
            "The contact-level within-model Phase-slope adjustment must use Holm."
        )
    if contact_lmm.get("phase_slope_contrasts") is not False:
        raise ValueError(
            "The contact-level model must not calculate Phase-slope contrasts."
        )
    if contact_lmm.get("prediction_quantiles") != [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]:
        raise ValueError("The contact-level prediction grid must use deciles.")
    if float(contact_lmm.get("display_limit", -1)) != 1.0:
        raise ValueError("The standardized-slope display limit must be 1.0.")
    visualization = _mapping(contact_lmm, "visualization")
    expected_visualization = {
        "backbone": "lfp_viz.visualdf",
        "phase_scalar_style_config": (
            "configs/viz/submission_phase_contact_scalar.yaml"
        ),
        "phase_heatmap_style_config": (
            "configs/viz/submission_phase_contact_scalar_heatmap.yaml"
        ),
        "slope_panel_width_mm": 30.0,
        "connectome_labels": {
            "individualized_dti": "individualized DTI",
            "ppmi_85_ewert_2017": "PPMI",
            "mgh_usc_hcp_32_horn_2017": "HCP32",
            "dtor_985_full_elias_2024": "dTOR985",
        },
        "lift_metric_labels": {
            "BinaryLift": "Binary Lift",
            "FieldWeightedLift": "Field-weighted Lift",
            "FieldExcess": "Field excess (S - R)",
            "SeedFieldExposure": "Seed field exposure (S)",
        },
        "lift_predictor_labels": {
            "STN": "STN",
            "SNr": "SNr",
            "RegionMean": "Region mean",
        },
    }
    if visualization != expected_visualization:
        raise ValueError(
            "The contact-level VisualDF figure contract does not match the "
            "registered Phase reference styles."
        )
    adjacent_lmm = _mapping(inference, "contact_lmm_adjacent")
    paths = _mapping(data, "paths")
    local_root = Path(_nonempty_string(paths, "local_lfp")).parent.parent
    expected_adjacent = {
        "analysis_id": "structural-connectivity-contact-lmm-adjacent-phase",
        "outcome_change_basis": "adjacent_phase",
        "previous_phases": {
            "Early": "Pre",
            "Late": "Early",
            "Post": "Late",
        },
        "formula": "Value ~ Phase * X_z + (1 | ID) + (1 | ID:ContactUnitID)",
        "predictor_metrics": [
            "FieldWeightedLift",
            "FieldExcess",
            "SeedFieldExposure",
        ],
        "field_excess_local_outcomes": [
            {
                "metric": "raw_power",
                "feature_output": "mean-scalar",
                "path": str(local_root / "raw_power/mean-scalar.pkl"),
                "transform": "dB",
                "transform_policy_status": "matched",
                "outcome_scale": "db_pre_centered",
                "bands": [
                    "delta",
                    "theta",
                    "alpha",
                    "beta_low",
                    "beta_high",
                    "gamma_low",
                    "gamma_high",
                ],
            },
            {
                "metric": "burst",
                "feature_output": "mean-scalar",
                "path": str(local_root / "burst/mean-scalar.pkl"),
                "transform": "log10",
                "transform_policy_status": "matched",
                "outcome_scale": "log10_pre_centered",
                "bands": [
                    "delta",
                    "theta",
                    "alpha",
                    "beta_low",
                    "beta_high",
                    "gamma_low",
                    "gamma_high",
                ],
            },
            {
                "metric": "burst",
                "feature_output": "duration-scalar",
                "path": str(local_root / "burst/duration-scalar.pkl"),
                "transform": "log10",
                "transform_policy_status": "configured_override",
                "outcome_scale": "log10_pre_centered",
                "bands": [
                    "delta",
                    "theta",
                    "alpha",
                    "beta_low",
                    "beta_high",
                    "gamma_low",
                    "gamma_high",
                ],
            },
            {
                "metric": "burst",
                "feature_output": "rate-scalar",
                "path": str(local_root / "burst/rate-scalar.pkl"),
                "transform": "asinh",
                "transform_policy_status": "configured_override",
                "outcome_scale": "asinh_pre_centered",
                "bands": [
                    "delta",
                    "theta",
                    "alpha",
                    "beta_low",
                    "beta_high",
                    "gamma_low",
                    "gamma_high",
                ],
            },
            {
                "metric": "burst",
                "feature_output": "occupancy-scalar",
                "path": str(local_root / "burst/occupancy-scalar.pkl"),
                "transform": "none",
                "transform_policy_status": "matched",
                "outcome_scale": "native_pre_centered",
                "bands": [
                    "delta",
                    "theta",
                    "alpha",
                    "beta_low",
                    "beta_high",
                    "gamma_low",
                    "gamma_high",
                ],
            },
            {
                "metric": "aperiodic",
                "feature_output": "mean-scalar",
                "path": str(local_root / "aperiodic/mean-scalar.pkl"),
                "transform": "none",
                "transform_policy_status": "matched",
                "outcome_scale": "native_pre_centered",
                "bands": ["exponent", "offset"],
            },
        ],
        "minimum_n": 3,
        "phase_levels": ["Early", "Late", "Post"],
        "reml": True,
        "degrees_of_freedom": "kenward-roger",
        "slope_p_adjust": "holm",
        "phase_slope_contrasts": False,
        "prediction_quantiles": [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ],
        "display_limit": 1.0,
        "visualization": {
            "backbone": "lfp_viz.visualdf",
            "phase_scalar_style_config": (
                "configs/viz/submission_phase_contact_scalar.yaml"
            ),
            "phase_heatmap_style_config": (
                "configs/viz/submission_phase_contact_scalar_heatmap.yaml"
            ),
            "slope_panel_width_mm": 30.0,
            "connectome_labels": expected_visualization["connectome_labels"],
            "lift_metric_labels": {
                "FieldWeightedLift": "Field-weighted Lift",
                "FieldExcess": "Field excess (S - R)",
                "SeedFieldExposure": "Seed field exposure (S)",
            },
            "lift_predictor_labels": expected_visualization[
                "lift_predictor_labels"
            ],
            "phase_display_labels": {
                "Early": "Early - Pre",
                "Late": "Late - Early",
                "Post": "Post - Late",
            },
            "atlas_grid_heatmap": {
                "lift_metric": "FieldExcess",
                "orientation": "metric-by-band",
                "connectome_order": [
                    "individualized_dti",
                    "mgh_usc_hcp_32_horn_2017",
                    "ppmi_85_ewert_2017",
                    "dtor_985_full_elias_2024",
                ],
                "max_width_mm": 150.0,
                "max_height_mm": 210.0,
            },
        },
    }
    if adjacent_lmm != expected_adjacent:
        raise ValueError(
            "The adjacent-Phase contact-level mixed-model contract is invalid."
        )
    _nonempty_string(paths, "contact_lmm_adjacent_temp_root")
    outputs = _mapping(data, "outputs")
    if _nonempty_string(outputs, "contact_lmm_adjacent_root") != (
        "contact_lmm/adjacent_phase"
    ):
        raise ValueError("The adjacent-Phase output root is invalid.")
    adjacent_id_lmm = _mapping(
        inference,
        "contact_lmm_adjacent_id_random_intercept",
    )
    expected_adjacent_id = {
        "analysis_id": (
            "structural-connectivity-contact-lmm-adjacent-phase-"
            "id-random-intercept"
        ),
        "outcome_change_basis": "adjacent_phase",
        "previous_phases": {
            "Early": "Pre",
            "Late": "Early",
            "Post": "Late",
        },
        "formula": "Value ~ Phase * X_z + (1 | ID)",
        "predictor_metrics": [
            "FieldWeightedLift",
            "FieldExcess",
            "SeedFieldExposure",
        ],
        "minimum_n": 3,
        "phase_levels": ["Early", "Late", "Post"],
        "reml": True,
        "degrees_of_freedom": "kenward-roger",
        "slope_p_adjust": "holm",
        "phase_slope_contrasts": False,
        "prediction_quantiles": [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ],
        "display_limit": 1.0,
    }
    if adjacent_id_lmm != expected_adjacent_id:
        raise ValueError(
            "The adjacent-Phase ID-random-intercept mixed-model contract "
            "is invalid."
        )
    _nonempty_string(paths, "contact_lmm_adjacent_id_random_intercept_temp_root")
    if _nonempty_string(
        outputs,
        "contact_lmm_adjacent_id_random_intercept_root",
    ) != "contact_lmm/adjacent_phase_id_random_intercept":
        raise ValueError(
            "The adjacent-Phase ID-random-intercept output root is invalid."
        )
    cortical_lmm = _mapping(inference, "cortical_allocation_lmm")
    if cortical_lmm.get("analysis_id") != "cortical-allocation-lmm":
        raise ValueError("The cortical-allocation analysis ID is invalid.")
    if cortical_lmm.get("candidate_set_id") != (
        "STNSNrplus-connected-regions-cortical-four"
    ):
        raise ValueError("The cortical-allocation candidate-set ID is invalid.")
    if cortical_lmm.get("atlas_name") != "STNSNrplus-connected regions":
        raise ValueError("The cortical-allocation atlas name is invalid.")
    if cortical_lmm.get("regions") != ["M1", "premotor", "SMA", "DLPFC"]:
        raise ValueError("The cortical-allocation region order is invalid.")
    if cortical_lmm.get("formula") != (
        "Value ~ Phase * RelativeAllocation + (1 | ID) + (1 | ID:ContactUnitID)"
    ):
        raise ValueError("The cortical-allocation model formula is invalid.")
    if cortical_lmm.get("predictor_column") != "RelativeAllocation":
        raise ValueError("The cortical-allocation predictor is invalid.")
    if cortical_lmm.get("joint_term") != "Phase:RelativeAllocation":
        raise ValueError("The cortical-allocation joint term is invalid.")
    if int(cortical_lmm.get("minimum_n", -1)) != 3:
        raise ValueError("The cortical-allocation minimum patient count must be 3.")
    if cortical_lmm.get("phase_levels") != ["Early", "Late", "Post"]:
        raise ValueError("The cortical-allocation Phase order is invalid.")
    if cortical_lmm.get("reml") is not True:
        raise ValueError("The cortical-allocation model must use REML.")
    if cortical_lmm.get("degrees_of_freedom") != "kenward-roger":
        raise ValueError(
            "The cortical-allocation model must use Kenward-Roger degrees of freedom."
        )
    if cortical_lmm.get("slope_p_adjust") != "holm":
        raise ValueError(
            "The cortical-allocation within-model slope adjustment must use Holm."
        )
    if cortical_lmm.get("phase_slope_contrasts") is not True:
        raise ValueError(
            "The cortical-allocation model must retain Phase-slope contrasts."
        )
    if cortical_lmm.get("prediction_quantiles") != [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]:
        raise ValueError("The cortical-allocation prediction grid must use deciles.")
    if int(cortical_lmm.get("bh_joint_planned_tests", -1)) != 28:
        raise ValueError(
            "The cortical-allocation joint-test BH family must contain 28 tests."
        )
    cortical_adjacent = _mapping(
        inference,
        "cortical_allocation_lmm_adjacent",
    )
    if cortical_adjacent.get("analysis_id") != (
        "cortical-allocation-lmm-adjacent-phase"
    ):
        raise ValueError(
            "The adjacent-Phase cortical-allocation analysis ID is invalid."
        )
    if cortical_adjacent.get("outcome_change_basis") != "adjacent_phase":
        raise ValueError(
            "The adjacent-Phase cortical-allocation outcome basis is invalid."
        )
    if cortical_adjacent.get("previous_phases") != {
        "Early": "Pre",
        "Late": "Early",
        "Post": "Late",
    }:
        raise ValueError(
            "The adjacent-Phase cortical-allocation contrasts are invalid."
        )
    shared_cortical_keys = (
        "candidate_set_id",
        "atlas_name",
        "regions",
        "formula",
        "predictor_column",
        "joint_term",
        "minimum_n",
        "phase_levels",
        "reml",
        "degrees_of_freedom",
        "slope_p_adjust",
        "phase_slope_contrasts",
        "prediction_quantiles",
        "bh_joint_planned_tests",
    )
    changed_cortical_keys = [
        key
        for key in shared_cortical_keys
        if cortical_adjacent.get(key) != cortical_lmm.get(key)
    ]
    if changed_cortical_keys:
        raise ValueError(
            "The adjacent-Phase cortical-allocation model differs from the "
            f"registered Pre-centered contract: {changed_cortical_keys}"
        )
    if cortical_adjacent.get("additional_local_outcomes") != adjacent_lmm.get(
        "field_excess_local_outcomes"
    ):
        raise ValueError(
            "The adjacent-Phase cortical-allocation local outcomes must match "
            "the registered adjacent Contact-LMM local outcomes."
        )
    _nonempty_string(paths, "cortical_allocation_lmm_adjacent_temp_root")
    if _nonempty_string(
        outputs,
        "cortical_allocation_lmm_adjacent_root",
    ) != "cortical_allocation_lmm/adjacent_phase":
        raise ValueError(
            "The adjacent-Phase cortical-allocation output root is invalid."
        )
    outcomes = _mapping(data, "outcomes")
    if outcomes.get("phases") != ["Early", "Late", "Post"]:
        raise ValueError("The analysis phases must be Early, Late, and Post.")
    if outcomes.get("bands") != [
        "delta",
        "theta",
        "alpha",
        "beta_low",
        "beta_high",
        "gamma_low",
        "gamma_high",
    ]:
        raise ValueError("The seven-band order does not match the frozen contract.")
    if outcomes.get("connectivity_pair_regions") != {
        "ciplv": "SNr-STN",
        "imcoh_abs": "SNr-STN",
        "psi": "SNr→STN",
        "trgc": "SNr→STN",
        "wpli": "SNr-STN",
    }:
        raise ValueError(
            "Connectivity PairRegion values do not match the frozen contract."
        )


def load_config(path: Path | str) -> StructuralConnectivityConfig:
    """Read and validate the structural-connectivity configuration once."""

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError("Structural-connectivity configuration must be a mapping.")
    required = {
        "analysis",
        "paths",
        "subjects",
        "connectomes",
        "montages",
        "masks",
        "outcomes",
        "inference",
        "execution",
        "outputs",
    }
    if set(loaded) != required:
        raise ValueError(f"Top-level configuration keys must be {sorted(required)}.")
    _validate_frozen_contract(loaded)
    return StructuralConnectivityConfig(
        path=source,
        data=loaded,
        subjects=_load_subjects(loaded),
        connectomes=_load_connectomes(loaded),
    )
