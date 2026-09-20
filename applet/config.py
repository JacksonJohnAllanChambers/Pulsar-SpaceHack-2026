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

    # Sea ice. HyperScape100 stops at 860 nm, so the usual NDSI (green vs 1610 nm SWIR) does not
    # exist for us and ice has to be separated from cloud inside the VNIR range alone. It can be:
    # ice absorbs toward 865 nm while cloud droplets scatter almost neutrally, so at equal
    # brightness ice keeps a positive NDWI and cloud sits at zero. Measured on four real Arctic
    # Sentinel-2 scenes (median NDWI: water +0.52..+0.64, ice +0.10..+0.23, cloud -0.01..+0.04,
    # land -0.53..-0.08), the ice/cloud boundary lands at +0.05 in every one of them.
    ice_ndwi_min: float = 0.05  # bright AND above this NDWI is ice, not cloud
    ice_ndwi_max: float = 0.40  # above this it is water, not ice
    ice_brightness_min: float = 0.18


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

    # Ice regime. Sea ice is a field of bright, high-contrast, ship-sized objects on dark water --
    # the exact signature the CFAR exists to find -- so in ice a bright blob is no longer evidence
    # of a hull and the detector has to ask for something ice cannot produce. Density alone does
    # not catch it: the four Arctic probe scenes run at 0.06-1.45 candidates/km2, well under the
    # rough-sea gate, while being far more ship-like than any whitecap field.
    ice_background_fraction: float = 0.05  # ice this fraction of a candidate's surroundings = ice regime
    ice_background_radius_m: float = 600.0  # annulus in which that fraction is measured
    # A vessel under way in pack ice leaves an open-water channel astern: at 865 nm that channel is
    # ~1 % reflectance against 20-30 % for the floes, a far larger contrast than the wake it would
    # leave in open water. Same ray transform, opposite sign.
    lead_min_length_m: float = 150.0
    lead_min_snr: float = 3.0
    lead_max_width_m: float = 400.0

    parallax_reject_px: float = 3.0  # band-to-band displacement that marks an aircraft, not a vessel
    # 0.20 was tried ("let the CNN rescue borderline candidates"): on the 16 real scenes it found no
    # extra AIS-confirmed ship and added 13 dark-vessel alerts, ~9 of them visibly jetties, reef surf,
    # islets and piers; held-out synthetic scores were identical. The CNN passes that clutter at > 0.8.
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


class ArcticConfig(BaseModel):
    enabled: bool = False  # opt in: it renames and demotes contacts, so it is never on by surprise
    min_scene_ice_pct: float = 2.0  # screener ice cover below which the call is not made (US max 0.7, Svalbard min 7.9)
    min_neighbours: int = 3  # other bright objects sharing the contact's chip ...
    min_neighbours_per_km2: float = 7.0  # ... and the same count as a density, so a finer GSD cannot fire on one
    blob_sigma: float = 4.0  # robust sigmas over the chip's own border for a pixel to count as bright
    iceberg_priority: float = 0.02  # under a charted structure (0.05): still downlinked, last in the queue


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
    # Off unless the uplinked config turns it on: it is state that outlives a pass, and a default
    # run must be a pure function of its input bundle.
    unknown_memory_enabled: bool = False
    unknown_memory_radius_nm: float = 0.08
    unknown_memory_required_passes: int = 3  # distinct acquisition dates, not runs
    unknown_memory_max_age_days: float = 180.0  # an area that stops recurring is forgotten
    unknown_memory_path: Optional[str] = None


class ThermalConfig(BaseModel):
    """
    Orbital thermal model and the cascade governor it drives. See `applet/core/thermal.py`
    for the physics and `applet/core/governor.py` for the control law.

    Two switches, deliberately separate, because they carry very different risk:

    `report_thermal` (ON by default) only *models and reports*. It adds a predicted
    junction temperature and power budget to `edge_telemetry.json` beside the measured
    RAM and CPU figures, and changes no decision anywhere. Nothing downstream can
    observe it, so a pass stays a pure function of its input bundle.

    `governor_enabled` (OFF by default) lets the model *act*: it re-profiles the cascade
    mid-pass from accumulated stage timings. Those timings depend on the host, so two
    runs of the same bundle on different machines can legitimately produce different
    output. That is correct behaviour for a spacecraft and wrong behaviour for a
    reproducibility test, so it is opt-in -- same reasoning as `unknown_memory_enabled`.
    `scripts/orbit_pass_sim.py` and the ground console turn it on explicitly.
    """

    report_thermal: bool = True
    governor_enabled: bool = False
    start_profile: str = "FULL"
    # Hold one rung and never re-assess. Two uses: measuring what each rung actually
    # costs in detection terms (scripts/orbit_pass_sim.py), and a ground operator
    # commanding "stay in BEACON" because they know something the model does not.
    pin_profile: Optional[str] = None

    # Orbit. The default is a mid-beta sun-synchronous orbit; set eclipse_fraction to 0.0
    # for dawn-dusk SSO, which never gets thermal relief and is the real stress case.
    orbit_period_s: float = 94.6 * 60.0
    eclipse_fraction: float = 0.35

    # Spacecraft thermal design. All [ASSUMED] -- we have no bus. Exposed here so a
    # reviewer can substitute a real one and re-derive every conclusion downstream.
    radiator_area_m2: float = 0.090
    radiator_emissivity: float = 0.85
    radiator_absorptivity: float = 0.20
    parasitic_load_w: float = 12.0
    soc_heat_capacity_j_per_k: float = 120.0
    soc_to_chassis_r_k_per_w: float = 0.9
    chassis_heat_capacity_j_per_k: float = 1800.0

    # Jetson limits. throttle_c is where WE act; Orin's own TJ_max is 105 C.
    throttle_c: float = 95.0
    target_c: float = 80.0
    initial_junction_c: float = 25.0
    initial_chassis_c: float = 20.0


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
    arctic: ArcticConfig = Field(default_factory=ArcticConfig)
    ais_correlation: AISCorrelationConfig = Field(default_factory=AISCorrelationConfig)
    downlink: DownlinkConfig = Field(default_factory=DownlinkConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    thermal: ThermalConfig = Field(default_factory=ThermalConfig)

    @classmethod
    def load_from_yaml(cls, path: Optional[str] = None) -> "AppletConfig":
        if not path or not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
