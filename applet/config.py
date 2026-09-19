"""
Configuration schema and parser for uplinked mission parameters.

All radiometric thresholds are in reflectance units (0..1) so they carry over
between sensors; pixel-size thresholds are derived from metres via the scene GSD.
"""

import os
import yaml
from typing import Optional
from pydantic import BaseModel, Field


class MissionConfig(BaseModel):
    mission_name: str = "PULSAR-SPACEHACK-2026"
    satellite_id: str = "MOBIUS-1"
    orbital_pass_id: str = "PASS_001"
    gsd_meters: float = 4.75  # default when the bundle manifest does not specify one


class ScreeningConfig(BaseModel):
    cloud_cover_max_pct: float = 60.0
    blur_min_laplacian_var: float = 2.0e-6  # variance of Laplacian on reflectance
    min_valid_data_pct: float = 75.0
    min_water_pct: float = 5.0  # below this there is no sea to search

    # Water / land
    ndwi_water_min: float = 0.0
    nir_water_max: float = 0.12
    nir_deep_water_max: float = 0.02  # darker than this in NIR is water even when NDWI is undefined
    land_min_area_m2: float = 60000.0  # non-water blobs larger than this are land, not ships
    land_buffer_m: float = 60.0

    # Cloud
    cloud_brightness_min: float = 0.22
    cloud_whiteness_max: float = 0.30  # (max-min)/mean across bands
    cloud_min_area_m2: float = 40000.0
    cloud_buffer_m: float = 40.0


class DetectionConfig(BaseModel):
    # CFAR (constant false alarm rate) local-contrast detector on the NIR band
    cfar_k_sigma: float = 5.0
    cfar_min_contrast: float = 0.02  # absolute reflectance above local background
    cfar_guard_m: float = 120.0
    cfar_background_m: float = 400.0
    cfar_sigma_floor: float = 0.0015

    min_target_area_m2: float = 90.0
    max_target_area_m2: float = 60000.0
    link_gap_m: float = 15.0  # closing radius joining hull and wake fragments
    shore_exclusion_m: float = 200.0  # blobs reaching this close to the land mask are port clutter

    # Rough-sea adaptation: when candidates are dense they are treated as a clutter
    # population and targets must be outliers of it
    clutter_density_per_km2: float = 3.0
    clutter_outlier_mads: float = 5.0

    # Wake ray transform
    wake_search_radius_m: float = 900.0
    wake_min_length_m: float = 60.0
    wake_min_snr: float = 3.5
    kelvin_half_angle_deg: float = 19.47
    kelvin_tolerance_deg: float = 6.0

    parallax_reject_px: float = 3.0  # band-to-band displacement that marks an aircraft, not a vessel
    min_physics_score: float = 0.35
    chip_crop_size_px: int = 64
    max_candidates_per_scene: int = 400


class VerifierConfig(BaseModel):
    enabled: bool = True
    model_path: str = "applet/models/verifier_int8.onnx"
    fallback_model_path: str = "applet/models/verifier_fp32.onnx"
    reject_below: float = 0.80  # CNN probability under which a candidate is dropped ...
    physics_override_score: float = 0.95  # ... unless the physics evidence is near-certain
    intra_op_threads: int = 4


class AISCorrelationConfig(BaseModel):
    spatial_gating_radius_nm: float = 1.0  # outer gate: beyond this a broadcaster is unrelated
    tight_gate_nm: float = 0.15  # inner gate at zero fix age (geolocation + AIS GPS error)
    gate_growth_fraction: float = 0.25  # inner gate grows by this fraction of distance run since the fix
    max_heading_delta_deg: float = 35.0
    speed_tolerance_knots: float = 6.0
    max_fix_age_hours: float = 3.0  # older AIS fixes cannot identify a contact
    static_mismatch_min_wake_m: float = 300.0  # "wake but AIS says stopped" needs at least this much wake ...
    static_mismatch_min_wake_snr: float = 8.0  # ... at this strength; piers beside berthed ships mimic short wakes
    speed_can_raise_anomaly: bool = False  # speed disagreement alone is advisory until validated on real wakes
    default_delta_hours: float = 0.0
    unknown_memory_enabled: bool = True
    unknown_memory_radius_nm: float = 0.08
    unknown_memory_required_passes: int = 3
    unknown_memory_path: Optional[str] = None


class RuntimeConfig(BaseModel):
    # Scenes processed concurrently. Each in-flight 4096x4096 scene costs ~1 GB of
    # temporaries, so 3 workers keeps a full-swath pass far inside the 14 GB envelope.
    scene_workers: int = 3


class DownlinkConfig(BaseModel):
    max_downlink_budget_kb: int = 500
    include_target_chips: bool = True
    chip_jpeg_quality: int = 70
    chip_min_priority: float = 0.5
    # Full-context crops for the file-queue downlink scheduler (src/pyFlows). None keeps them with
    # the rest of the pass under <output_dir>/queues; flight code never writes into its own source tree.
    write_queues: bool = True
    queue_dir: Optional[str] = None


class AppletConfig(BaseModel):
    mission: MissionConfig = Field(default_factory=MissionConfig)
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    ais_correlation: AISCorrelationConfig = Field(default_factory=AISCorrelationConfig)
    downlink: DownlinkConfig = Field(default_factory=DownlinkConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @classmethod
    def load_from_yaml(cls, path: Optional[str] = None) -> "AppletConfig":
        if not path or not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
