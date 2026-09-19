"""
Ground-station console (development / demo tool -- not part of the flight applet).

Runs the exact onboard code path (`applet.runner.run_pass`) on a chosen input bundle and
serves an interactive view of every stage: imagery layers, masks, the CFAR z-score map,
detections with heading / wake geometry, AIS correlation, the downlink artefact and the
edge telemetry. No CDN or internet resources are used, so it works on an air-gapped laptop.

    python -m ground.server            # then open http://127.0.0.1:8050
"""

import os
import sys
import json
import math
import argparse
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional
from xml.etree import ElementTree

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, JSONResponse
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from applet.config import AppletConfig  # noqa: E402
from applet.runner import run_pass  # noqa: E402
from applet.core.exceptions import InvalidManifestError  # noqa: E402
from ground.fleets import FLEETS, dispatch_sent_alerts, load_alerts, parse_alert_filename  # noqa: E402
from scripts.evaluate import score_context  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs", "gui")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
# The console is the one caller that feeds the file-queue downlink scheduler (src/pyFlows/workflow.py),
# which drains src/downlink/queues into src/sent. Every other caller keeps crops under its output dir.
QUEUE_DIR = os.path.join(ROOT, "src", "downlink", "queues")
SENT_DIR = os.path.join(ROOT, "src", "sent")
FLEET_ALERT_PATH = os.path.join(DATA_DIR, "outputs", "fleet_alerts.json")
FLEET_OUTPUT_DIR = os.path.join(ROOT, "src", "fleet_alerts")
TRANSFER_ROOT = os.path.join(ROOT, "src")
MAX_LAYER_PX = 2048
TRANSFER_EVENT_LIMIT = 100
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TRANSFER_STAGE_LABELS = {
    "incoming": "Incoming",
    "processing": "Processing",
    "priority": "Priority queue",
    "standard": "Standard queue",
    "no_ship": "No ship",
    "sent": "Downlink sent",
}
TRANSFER_EVENT_TYPES = {
    "incoming": "arrived",
    "processing": "processing",
    "priority": "queued_priority",
    "standard": "queued_standard",
    "no_ship": "no_ship",
    "sent": "downlinked",
}

app = FastAPI(title="Tactical Edge Sentinel - Ground Console")
_state: Dict[str, Any] = {"context": None, "scenes": {}, "chips": {}, "downlink": None}
_lock = threading.Lock()
_transfer_lock = threading.Lock()
_transfer_previous: Dict[str, Dict[str, Any]] = {}
_transfer_events = deque(maxlen=TRANSFER_EVENT_LIMIT)


class RunRequest(BaseModel):
    bundle: str
    cfar_k_sigma: Optional[float] = None
    min_physics_score: Optional[float] = None
    cloud_cover_max_pct: Optional[float] = None
    verifier_enabled: Optional[bool] = None
    verifier_reject_below: Optional[float] = None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not isinstance(v, (bytes, np.ndarray)) and k != "georef"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def list_bundles():
    out = []
    for dirpath, dirnames, filenames in os.walk(DATA_DIR, followlinks=True):  # data/real may be a symlink
        dirnames[:] = [d for d in dirnames if d not in ("outputs",)]
        if "manifest.json" in filenames:
            out.append(os.path.relpath(dirpath, ROOT).replace("\\", "/"))
    return sorted(out)


def transfer_stage_dirs() -> Dict[str, Path]:
    root = Path(TRANSFER_ROOT)
    return {
        "incoming": root / "rawImages",
        "processing": root / "processing",
        "priority": root / "downlink" / "queues" / "priority",
        "standard": root / "downlink" / "queues" / "nonPriority",
        "no_ship": root / "noShipDetected",
        "sent": root / "sent",
    }


def _transfer_file(stage: str, path: Path) -> Dict[str, Any]:
    stat = path.stat()
    alert = parse_alert_filename(path)
    return {
        "id": f"{stage}:{path.name}",
        "stage": stage,
        "stage_label": TRANSFER_STAGE_LABELS[stage],
        "filename": path.name,
        "bytes": stat.st_size,
        "modified_at": stat.st_mtime,
        "image_url": f"/api/transfer-image/{stage}/{path.name}",
        "alert": alert,
    }


def _fleet_ping(path: Path) -> Dict[str, Any]:
    info_path = path / "info.xml"
    record: Dict[str, Any] = {"path": str(path.relative_to(Path(TRANSFER_ROOT))).replace("\\", "/"),
                              "folder": path.name, "warning": None}
    try:
        root = ElementTree.parse(info_path).getroot()
        fleet = root.find("fleet")
        record.update({
            "detection_id": root.findtext("detection_id"),
            "scene_id": root.findtext("scene_id"),
            "classification": root.findtext("classification"),
            "latitude": root.findtext("latitude"),
            "longitude": root.findtext("longitude"),
            "distance_nm": root.findtext("distance_nm"),
            "source_filename": root.findtext("source_filename"),
            "dispatched_at": root.findtext("dispatched_at"),
            "fleet": {"id": fleet.findtext("id"), "name": fleet.findtext("name"),
                      "latitude": fleet.findtext("latitude"), "longitude": fleet.findtext("longitude")}
            if fleet is not None else None,
        })
    except (OSError, ElementTree.ParseError) as error:
        record["warning"] = f"Unable to read info.xml: {type(error).__name__}"
    image_path = path / "image.jpg"
    if image_path.is_file():
        record["image_url"] = f"/api/transfer-image/fleet_ping/{path.parent.name}/{path.name}/image.jpg"
    else:
        record["warning"] = record["warning"] or "Ping image is missing"
    return record


def _transfer_snapshot() -> Dict[str, Any]:
    stages = []
    files = []
    for stage, directory in transfer_stage_dirs().items():
        records = []
        if directory.is_dir():
            for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    records.append(_transfer_file(stage, path))
                except OSError:
                    continue  # A workflow moved the file between iterdir and stat.
        files.extend(records)
        stages.append({"id": stage, "label": TRANSFER_STAGE_LABELS[stage], "count": len(records), "files": records})

    pings = []
    fleet_root = Path(TRANSFER_ROOT) / "fleet_alerts"
    if fleet_root.is_dir():
        for info_path in sorted(fleet_root.glob("*/ping_*/info.xml")):
            pings.append(_fleet_ping(info_path.parent))
    return {"timestamp": time.time(), "stages": stages, "files": files, "fleet_pings": pings}


def transfer_state() -> Dict[str, Any]:
    """Return a filesystem snapshot plus transitions observed since this server started."""
    global _transfer_previous
    with _transfer_lock:
        snapshot = _transfer_snapshot()
        current = {item["filename"]: item for item in snapshot["files"]}
        for filename, item in current.items():
            previous = _transfer_previous.get(filename)
            if previous is None or previous["stage"] != item["stage"]:
                _transfer_events.appendleft({
                    "event": TRANSFER_EVENT_TYPES[item["stage"]], "timestamp": snapshot["timestamp"],
                    "filename": filename, "stage": item["stage"], "stage_label": item["stage_label"],
                    "from_stage": previous["stage"] if previous else None, "alert": item["alert"],
                })
        _transfer_previous = current
        snapshot["events"] = list(_transfer_events)
        return snapshot


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"), headers={"Cache-Control": "no-store"})


@app.get("/transfer")
def transfer_index():
    return FileResponse(os.path.join(STATIC_DIR, "transfer.html"), headers={"Cache-Control": "no-store"})


@app.get("/explainer")
def explainer():
    return FileResponse(os.path.join(STATIC_DIR, "explainer.html"), headers={"Cache-Control": "no-store"})


@app.get("/api/bundles")
def bundles():
    return {"bundles": list_bundles()}


@app.get("/api/fleet-alerts")
def fleet_alerts():
    return {"fleets": FLEETS, "alerts": load_alerts(FLEET_ALERT_PATH)}


@app.get("/api/transfer-state")
def transfer_state_api():
    return transfer_state()


@app.get("/api/transfer-image/{stage}/{filename:path}")
def transfer_image(stage: str, filename: str):
    stages = transfer_stage_dirs()
    if stage == "fleet_ping":
        root = Path(TRANSFER_ROOT) / "fleet_alerts"
    else:
        root = stages.get(stage)
    if root is None or not filename or Path(filename).is_absolute():
        raise HTTPException(400, "invalid transfer image path")
    path = (root / filename).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise HTTPException(400, "invalid transfer image path")
    if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
        raise HTTPException(404, "transfer image not found")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.post("/api/fleet-alerts/dispatch")
def dispatch_fleet_alerts():
    with _lock:
        dispatched = dispatch_sent_alerts(SENT_DIR, FLEET_ALERT_PATH, FLEET_OUTPUT_DIR)
        return {"fleets": FLEETS, "dispatched": dispatched, "alerts": load_alerts(FLEET_ALERT_PATH)}


@app.get("/api/sent/{filename}")
def sent_image(filename: str):
    if os.path.basename(filename) != filename:
        raise HTTPException(400, "invalid filename")
    path = os.path.join(SENT_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "sent image not found")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/run")
def run(req: RunRequest):
    bundle_dir = os.path.abspath(os.path.join(ROOT, req.bundle))
    if not bundle_dir.startswith(os.path.abspath(DATA_DIR)) or not os.path.isdir(bundle_dir):
        raise HTTPException(400, "bundle must be a directory under data/")

    config = AppletConfig.load_from_yaml(os.path.join(ROOT, "config.example.yaml"))
    config.downlink.queue_dir = QUEUE_DIR
    if req.cfar_k_sigma is not None:
        config.detection.cfar_k_sigma = req.cfar_k_sigma
    if req.min_physics_score is not None:
        config.detection.min_physics_score = req.min_physics_score
    if req.cloud_cover_max_pct is not None:
        config.screening.cloud_cover_max_pct = req.cloud_cover_max_pct
    if req.verifier_enabled is not None:
        config.verifier.enabled = req.verifier_enabled
    if req.verifier_reject_below is not None:
        config.verifier.reject_below = req.verifier_reject_below

    with _lock:
        try:
            context, telemetry, downlink = run_pass(bundle_dir, OUTPUT_DIR, config, keep_rasters=True)
        except InvalidManifestError as e:
            raise HTTPException(400, str(e))

        manifest = {}
        try:
            with open(os.path.join(bundle_dir, "manifest.json"), "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            pass
        truth_by_scene = {s.get("id"): s.get("ground_truth") for s in manifest.get("scenes", []) if isinstance(s, dict)}

        _state.update(context=context, downlink=downlink,
                      scenes={s["id"]: s for s in context["screened_scenes"]},
                      chips={t["detection_id"]: t.get("chip_jpeg", b"") for t in context["classified_targets"]})
        for s in context["screened_scenes"]:
            for t in s.get("rejected_candidates", []):
                _state["chips"][t["detection_id"] + "_rej"] = t.get("chip_jpeg", b"")

        scenes_out = []
        for s in context["screened_scenes"]:
            h, w = s["array"].shape[:2]
            georef = s["georef"]
            ais_px = []
            for a in context.get("ais_predictions", {}).get(s["id"], []):
                x, y = georef.lonlat_to_pixel(a["longitude"], a["latitude"])
                if -50 <= x < w + 50 and -50 <= y < h + 50:
                    ais_px.append(dict(a, x=round(x, 1), y=round(y, 1)))
            scenes_out.append({
                "id": s["id"], "file": s.get("file"), "description": s.get("description", ""),
                "width": w, "height": h, "gsd_m": s["gsd_m"],
                "display_scale": min(1.0, MAX_LAYER_PX / max(h, w)),
                "corners": {"ul": georef.ul, "ur": georef.ur, "lr": georef.lr, "ll": georef.ll},
                "quality": s["quality_metrics"], "detector": s.get("detector_stats"),
                "targets": [t["detection_id"] for t in s.get("detections", [])],
                "rejected_candidates": _jsonable([dict(t, detection_id=t["detection_id"] + "_rej")
                                                  for t in s.get("rejected_candidates", [])]),
                "ais": ais_px, "truth": truth_by_scene.get(s["id"]),
            })
        for s in context.get("rejected_scenes", []):
            scenes_out.append({"id": s.get("id"), "file": s.get("file"), "description": s.get("description", ""),
                               "ingest_error": s.get("error"), "quality": {"is_usable": False,
                               "rejection_reasons": [s.get("error")]}, "targets": [], "ais": []})

        accuracy = None
        if any(truth_by_scene.values()):
            accuracy = score_context(manifest, context, telemetry, config.mission.gsd_meters)

        return JSONResponse(_jsonable({
            "bundle": req.bundle, "scenes": scenes_out, "targets": context["classified_targets"],
            "ais_not_observed": context.get("ais_not_observed", []),
            "telemetry": telemetry, "downlink": downlink, "accuracy": accuracy,
            "config": {"cfar_k_sigma": config.detection.cfar_k_sigma,
                       "min_physics_score": config.detection.min_physics_score,
                       "cloud_cover_max_pct": config.screening.cloud_cover_max_pct,
                       "verifier_enabled": config.verifier.enabled,
                       "verifier_reject_below": config.verifier.reject_below},
        }))


def _stretch(a: np.ndarray, hi: float) -> np.ndarray:
    return (np.clip(a / hi, 0.0, 1.0) ** (1 / 2.2) * 255.0 + 0.5).astype(np.uint8)


def render_layer(scene: Dict[str, Any], layer: str) -> np.ndarray:
    arr = scene["array"]
    if layer == "rgb":
        img = _stretch(arr[:, :, [2, 1, 0]], 0.30)  # -> BGR for OpenCV
    elif layer == "nir":
        img = cv2.cvtColor(_stretch(arr[:, :, 3], 0.25), cv2.COLOR_GRAY2BGR)
    elif layer == "false":
        img = _stretch(arr[:, :, [1, 0, 3]], 0.30)  # B<-green, G<-red, R<-NIR
    elif layer == "ndwi":
        ndwi = scene.get("ndwi")
        if ndwi is None:
            raise HTTPException(404, "layer unavailable")
        img = cv2.applyColorMap(((np.clip(ndwi, -1, 1) + 1) * 127.5).astype(np.uint8), cv2.COLORMAP_OCEAN)
    elif layer == "zscore":
        z = scene.get("zmap")
        if z is None:
            raise HTTPException(404, "layer unavailable (scene was not searched)")
        img = cv2.applyColorMap((np.clip(z, 0, 20) / 20 * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    elif layer == "masks":
        h, w = arr.shape[:2]
        img = np.zeros((h, w, 4), dtype=np.uint8)
        if scene.get("land_mask") is not None:
            img[scene["land_mask"] > 0] = (70, 140, 190, 120)
        if scene.get("cloud_mask") is not None:
            img[scene["cloud_mask"] > 0] = (255, 255, 255, 110)
        img[scene["nodata_mask"]] = (60, 0, 90, 200)
        if scene.get("det_mask") is not None:
            img[scene["det_mask"] > 0] = (60, 60, 255, 255)
    else:
        raise HTTPException(404, "unknown layer")

    h, w = img.shape[:2]
    scale = min(1.0, MAX_LAYER_PX / max(h, w))
    if scale < 1.0:
        interp = cv2.INTER_NEAREST if layer == "masks" else cv2.INTER_AREA
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interp)
    return img


@app.get("/api/layer/{scene_id}/{layer}.png")
def layer_png(scene_id: str, layer: str):
    with _lock:
        scene = _state["scenes"].get(scene_id)
        if scene is None or "array" not in scene:
            raise HTTPException(404, "scene not loaded")
        img = render_layer(scene, layer)
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise HTTPException(500, "encode failed")
    return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/chip/{detection_id}.jpg")
def chip(detection_id: str):
    data = _state["chips"].get(detection_id)
    if not data:
        raise HTTPException(404, "no chip")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/download")
def download():
    d = _state.get("downlink")
    if not d or not os.path.exists(d["downlink_tarball_path"]):
        raise HTTPException(404, "run a pass first")
    return FileResponse(d["downlink_tarball_path"], filename=os.path.basename(d["downlink_tarball_path"]))


@app.get("/api/model_card")
def model_card():
    path = os.path.join(ROOT, "applet", "models", "model_card.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8050)))
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
