"""
Single entry point for executing a pass: used by the CLI, the benchmark harness,
the ground-station GUI and the tests, so they all measure exactly the same code path.
"""

import os
import shutil
from typing import Dict, Any, Optional, Tuple

from applet.config import AppletConfig
from applet.core.validator import InputBundleValidator
from applet.core.telemetry import EdgeTelemetryTracker
from applet.pipelines.pipeline_registry import PipelineDispatcher
from applet.packaging.downlink import DownlinkPackager
from src.pyFlows.process import route_classified_targets

_RASTER_KEYS = ("array", "nodata_mask", "cloud_mask", "land_mask", "sea_mask", "ndwi", "zmap", "det_mask")


def run_pass(
    input_dir: str,
    output_dir: str,
    config: Optional[AppletConfig] = None,
    track: str = "tactical",
    keep_rasters: bool = False,
    log=lambda msg: None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Returns (context, telemetry_summary, downlink_stats)."""
    config = config or AppletConfig()

    with EdgeTelemetryTracker(label=f"{config.mission.mission_name}_{track}") as tracker:
        with tracker.stage("Ingest+Validate"):
            validator = InputBundleValidator(input_dir, default_gsd_m=config.mission.gsd_meters)
            ingest = validator.validate()
        log(
            f"[INFO] Ingested {ingest['valid_scenes_count']} valid scenes, "
            f"{ingest['rejected_scenes_count']} rejected, {ingest['ais_vessels_in_catalog']} AIS records."
        )

        context: Dict[str, Any] = {
            "valid_scenes": validator.valid_scenes,
            "rejected_scenes": validator.rejected_scenes,
            "ais_catalog": validator.ais_catalog,
            "known_structures": validator.known_structures,
            "ingest_stats": ingest,
        }

        for stage in PipelineDispatcher.get_pipeline(track, config):
            name = stage.__class__.__name__
            log(f"[STAGE] {name}")
            with tracker.stage(name):
                context = stage.process(context)

        context.setdefault("classified_targets", context.get("detected_vessels", []))

        if config.downlink.write_queues:
            with tracker.stage("QueueRouter"):
                queue_dir = config.downlink.queue_dir
                if queue_dir is None:
                    # This pass owns the default queue, so it holds this pass's crops only. A
                    # configured queue belongs to whatever scheduler is draining it: leave it alone.
                    queue_dir = os.path.join(output_dir, "queues")
                    shutil.rmtree(queue_dir, ignore_errors=True)
                route_classified_targets(
                    context.get("screened_scenes", []), context["classified_targets"], queue_dir
                )

        if not keep_rasters:
            # Free the big arrays before packaging; onboard nothing downstream needs them
            for scene in context.get("screened_scenes", []):
                for key in _RASTER_KEYS:
                    scene.pop(key, None)
            context.pop("valid_scenes", None)

        with tracker.stage("DownlinkPackager"):
            packager = DownlinkPackager(config, output_dir)
            downlink = packager.package(context, tracker.snapshot())
        tracker.set_io_metrics(
            input_bytes=ingest["total_file_bytes"],
            output_bytes=downlink["final_bundle_bytes"],
            input_raw_bytes=ingest["total_raw_bytes"],
        )

    telemetry = tracker.get_summary()
    telemetry["verifier"] = context.get("verifier_info", {})
    telemetry["funnel"] = context.get("detection_funnel", {})
    DownlinkPackager.write_telemetry(downlink["telemetry_path"], telemetry)
    return context, telemetry, downlink
