"""
What would a passive RF payload beside the imager actually buy us? -- a MODELLED study.

    python scripts/rf_cue_study.py                        # everything
    python scripts/rf_cue_study.py --bundle arctic
    python scripts/rf_cue_study.py --bundle us --us-trials 500
    python scripts/rf_cue_study.py --out data/outputs/rf_cue_study.json

=======================================================================================
EVERY RF NUMBER BELOW IS SIMULATED. NO RF RECEIVER WAS INVOLVED.
=======================================================================================

What is MODELLED: the cue, drawn from `applet.core.rf_cue`. Its default parameters are
fitted to ESA's independent assessment of 14 real Unseenlabs products against AIS
(median 2.5 km, mean 5.4 km, std 6.7 km) -- third-party measurements of a flown system,
but not ours and not of our imagery.

What is REAL: everything optical. The contacts, their positions, the searched water area
and the human adjudication all come from running the UNMODIFIED applet on the unmodified
real bundles through `applet.runner.run_pass`, scored with `scripts/scorecard.py`.

So this measures: given contacts we really produced at positions we really computed, how
much disambiguation does a cue of a given quality provide? The clutter geometry is
measured; the cue is modelled. Do not let the two swap labels.

WHAT THIS DOES NOT PROVE -- read before quoting any number
----------------------------------------------------------
A synthesised cue has to be placed somewhere we believe an emitter is. The only vessel
ground truth we hold comes from AIS or from a human looking at pixels -- COOPERATIVE or
reviewer-confirmed vessels. A dark vessel by definition has no AIS, so no AIS-derived cue
is ever placed on one.

This study therefore DOES NOT and CANNOT show that an RF cue improves dark-vessel
detection. Experiment 3 reports its vessel-retention rate as "by construction" rather
than as a result, because that is exactly what it is.

What transfers to the non-cooperative case is the CLUTTER side. The detector never looks
at AIS; ice floes and reef surf do not know whether a nearby ship is broadcasting. So
"how many false alarms does a gate of radius r sweep up, given the real spatial
distribution of our real false alarms" is a measurement that holds whoever is emitting.

THE FOUR EXPERIMENTS
--------------------
0. CUE -> SWATH. Pure geometry, no bundle needed. `scripts/cue_geometry.py` asks how
   STALE a cue can be before the swath misses the ship. This asks the other half: how
   ACCURATE must it be? A 19.4 km swath pointed at a fix catches the emitter only if the
   fix error is under ~9.7 km, and at the ESA-measured error distribution it very often
   is not.

1. ARCTIC, EMPTY EMITTER SET. Four real Sentinel-2 Arctic scenes. Hand adjudication of
   all baseline contacts found ZERO vessels -- ice floes, ridges and melt features. Ice
   carries no navigation radar, so the modelled RF picture is no fixes at all and an
   RF-gated pipeline rejects every contact. This is the cleanest form of the claim and it
   is the one result here that does not depend on the error model in any way.

2. ARCTIC, ONE HYPOTHETICAL EMITTER. The case that costs RF something. Suppose one ship
   really is transiting that ice. Its gate sweeps up whatever ice happens to be inside.
   We measure how many, from the real contact positions, with the emitter placed
   uniformly over searched water and again placed inside the clutter field.

3. US BUNDLE, COOPERATIVE EMITTERS. Sixteen real US coastal scenes with the team's hand
   adjudication from data/outputs/review/labels_team.json, carried onto this run's
   contacts BY POSITION with scripts/transfer_labels.py. Detection ids renumber whenever
   the contact set changes, so scoring by id is unstable -- it disagreed with itself by
   0.14 in precision across two runs -- and this study does not offer it.

   Vessel retention here is >= 95 % BY CONSTRUCTION and is NOT a result: the gate IS the
   95 % containment radius of a fix placed on that vessel, so the fix falls inside its own
   gate 95 % of the time by definition. The results are the false-alarm retention rate and
   the per-cue contamination, plus a SOLAS sensitivity run showing what an RF requirement
   costs for small craft that are not required to carry radar at all.

Nothing here touches the pipeline. `run_pass` runs with the default config, so the pass
this study scores is byte-identical to a normal one; the gate is applied afterwards to
the contacts it returned.
"""

import os
import sys
import json
import math
import random
import argparse
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from applet.config import AppletConfig  # noqa: E402
from applet.runner import run_pass  # noqa: E402
from applet.core.rf_cue import (  # noqa: E402
    ESA_EDAP_MEDIAN_ERROR_M,
    RFCueModel,
    associate,
    expected_contacts_in_gate,
    haversine_m,
)
from scripts.cue_geometry import SWATH_KM  # noqa: E402
from scripts.scorecard import score as scorecard_score  # noqa: E402
from scripts.transfer_labels import transfer as transfer_labels  # noqa: E402

MATCHED = ("CONFIRMED_KNOWN_VESSEL", "AIS_KINEMATIC_MISMATCH")
SEED = 20260919

# The run the team's hand verdicts were made on. Labels are keyed by detection_id, and ids
# renumber whenever the contact set changes, so keying a verdict to an id on a DIFFERENT
# run silently lands it on a different object. That is not a small effect here: only 266 of
# this run's 375 contacts share an id with labels_team.json, 195 labels point at contacts
# that no longer exist, and two direct-match runs disagreed on precision by 0.14 -- which
# is itself the proof that direct matching is unstable.
#
# scripts/transfer_labels.py carries verdicts BY POSITION instead, which is what the
# README's headline figure was produced with. This study therefore never keys on
# detection_id; it transfers first and scores second.
US_REFERENCE_CONTACTS = "data/outputs/us_scorecard/contacts.json"

# SOLAS V/19 requires a 9 GHz radar from 300 GT upwards. Our contacts carry an estimated
# hull length, not a tonnage, so this is a length proxy and a deliberately generous
# reading of "300 GT". The sensitivity run is about the DIRECTION of the effect.
SOLAS_LENGTH_PROXY_M = 50.0

# The sweep. Four synthetic Rayleigh operating points bracketing the question, plus the
# one distribution that is anchored to a third-party measurement of a flown system.
CueSpec = Tuple[str, Callable[[], RFCueModel]]


def cue_specs() -> List[CueSpec]:
    return [
        ("Rayleigh CEP 1 km", lambda: RFCueModel(cep_m=1000.0, seed=SEED)),
        ("Rayleigh CEP 2 km", lambda: RFCueModel(cep_m=2000.0, seed=SEED)),
        ("Rayleigh CEP 5 km", lambda: RFCueModel(cep_m=5000.0, seed=SEED)),
        ("Rayleigh CEP 10 km", lambda: RFCueModel(cep_m=10000.0, seed=SEED)),
        ("ESA-measured (2.5 km median)", lambda: RFCueModel.esa_edap_2023(seed=SEED)),
    ]


# --------------------------------------------------------------------------------------
# Running the real pass
# --------------------------------------------------------------------------------------

def run_bundle(input_dir: str, keep_rasters: bool = False
               ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Default config, unmodified pipeline. Whatever this returns is what the applet does."""
    config = AppletConfig()
    # Same reason scripts/evaluate.py does this: persistent suppression is state that
    # outlives a pass and would leak between measurements.
    config.ais_correlation.unknown_memory_enabled = False
    config.downlink.write_queues = False  # a study does not need the crops
    with tempfile.TemporaryDirectory() as out_dir:
        context, telemetry, _ = run_pass(input_dir, out_dir, config, keep_rasters=keep_rasters)
    return context, telemetry


def water_samples(context: Dict[str, Any], per_scene: int, rng: random.Random
                  ) -> List[Tuple[str, float, float]]:
    """
    (scene_id, lat, lon) at random points of the water the detector actually searched.

    Uses each scene's own sea mask, so a hypothetical vessel is only ever placed where a
    vessel could be. Falls back to the whole footprint if the mask was already freed --
    that fallback puts emitters on land and inflates apparent contamination, so it is
    reported loudly rather than used silently.
    """
    out: List[Tuple[str, float, float]] = []
    used_fallback = False
    for scene in context.get("screened_scenes", []):
        if not scene["quality_metrics"]["is_usable"]:
            continue
        georef = scene["georef"]
        mask = scene.get("sea_mask")
        if mask is None:
            used_fallback = True
            for _ in range(per_scene):
                lon, lat = georef.pixel_to_lonlat(rng.uniform(0, georef.width - 1),
                                                  rng.uniform(0, georef.height - 1))
                out.append((scene["id"], lat, lon))
            continue
        ys, xs = np.nonzero(np.asarray(mask))
        if len(xs) == 0:
            continue
        for _ in range(per_scene):
            i = rng.randrange(len(xs))
            lon, lat = georef.pixel_to_lonlat(float(xs[i]), float(ys[i]))
            out.append((scene["id"], lat, lon))
    if used_fallback:
        print("  [warn] sea_mask unavailable for at least one scene; emitters placed over the "
              "whole footprint, which OVERSTATES contamination")
    return out


# --------------------------------------------------------------------------------------
# Local tangent-plane helpers (vectorised gating)
# --------------------------------------------------------------------------------------

def to_local_m(points: Sequence[Tuple[float, float]], ref_lat: float, ref_lon: float) -> np.ndarray:
    """(lat, lon) -> local east/north metres about a reference. Valid over one scene."""
    if not points:
        return np.zeros((0, 2), dtype=float)
    arr = np.asarray(points, dtype=float)
    m_per_deg_lat = 111195.0
    return np.column_stack([
        (arr[:, 1] - ref_lon) * m_per_deg_lat * math.cos(math.radians(ref_lat)),
        (arr[:, 0] - ref_lat) * m_per_deg_lat,
    ])


def min_distance_m(contacts_m: np.ndarray, fixes_m: np.ndarray) -> np.ndarray:
    """Distance from every contact to its nearest fix. inf when there are no fixes."""
    if len(fixes_m) == 0:
        return np.full(len(contacts_m), np.inf)
    return np.hypot(contacts_m[:, None, 0] - fixes_m[None, :, 0],
                    contacts_m[:, None, 1] - fixes_m[None, :, 1]).min(axis=1)


def quant(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if len(values) == 0:
        return {"n": 0, "mean": None, "p90": None, "max": None}
    s = np.asarray(values, dtype=float)
    return {"n": int(s.size), "mean": round(float(s.mean()), 3),
            "p90": round(float(np.percentile(s, 90)), 3), "max": round(float(s.max()), 3)}


# --------------------------------------------------------------------------------------
# Experiment 0 -- cue accuracy against the swath
# --------------------------------------------------------------------------------------

def experiment_swath(swath_km: float = SWATH_KM) -> Dict[str, Any]:
    """
    Point the imager at the fix. Is the emitter in the frame?

    The companion question to `scripts/cue_geometry.py`, which asks how stale a cue can
    be. This asks how accurate it must be, and the two share one budget: a target at the
    swath centre may run half the swath width before it leaves, and a fix error consumes
    that same half-width before the vessel has moved at all.

    Pure geometry plus the fix distribution. No bundle, no detector, nothing fitted.
    """
    half_m = 0.5 * swath_km * 1000.0
    rows = []
    for label, make in cue_specs():
        model = make()
        # P(fix error <= half swath) -- invert the model's own containment function by
        # bisection on probability, which is monotone.
        lo, hi = 1e-6, 1.0 - 1e-9
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if model.containment_radius_m(mid) < half_m:
                lo = mid
            else:
                hi = mid
        rows.append({
            "cue": label,
            "cep_m": model.cep_m,
            "gate_r95_m": round(model.containment_radius_m(0.95), 1),
            "gate_r95_vs_swath": round(2.0 * model.containment_radius_m(0.95) / (swath_km * 1000.0), 2),
            "p_emitter_in_swath": round(0.5 * (lo + hi), 3),
        })
    return {"swath_km": swath_km, "half_swath_m": half_m, "by_cue": rows}


# --------------------------------------------------------------------------------------
# Experiment 1 -- Arctic, empty emitter set
# --------------------------------------------------------------------------------------

def experiment_empty(contacts: List[Dict[str, Any]], searched_km2: float) -> Dict[str, Any]:
    """
    Ice does not emit. So there are no fixes, and nothing survives the gate.

    Why this is not a trick: the emitter set is empty because a human looked at every one
    of these contacts and found no vessel, not because the model chose to emit nothing.
    The model's only job here is to refrain from inventing emitters. The result is
    identical at every cue accuracy, because an empty fix list has no accuracy -- which is
    exactly what makes this the most robust claim in the study.
    """
    rows = []
    for label, make in cue_specs():
        model = make()
        fixes = model.fixes([])  # no emitters
        retained = sum(a.retained for a in associate(contacts, fixes))
        rows.append({
            "cue": label,
            "gate_r95_m": round(model.containment_radius_m(), 1),
            "contacts_in": len(contacts),
            "contacts_retained": retained,
            "rejected_pct": round(100.0 * (len(contacts) - retained) / max(len(contacts), 1), 1),
        })
    return {
        "contacts": len(contacts),
        "searched_water_km2": round(searched_km2, 1),
        "false_alarms_per_1000km2_before": round(1000.0 * len(contacts) / searched_km2, 1)
        if searched_km2 else None,
        "false_alarms_per_1000km2_after": 0.0,
        "by_cue": rows,
    }


# --------------------------------------------------------------------------------------
# Experiment 2 -- Arctic, one hypothetical emitter
# --------------------------------------------------------------------------------------

def experiment_single_emitter(contacts: List[Dict[str, Any]], searched_km2: float,
                              water: List[Tuple[str, float, float]], trials: int
                              ) -> Dict[str, Any]:
    """
    Suppose ONE real ship is transiting the ice. How ambiguous is the cue that finds it?

    Two placements, because they answer different questions and they disagree:

      `water`      -- the emitter is anywhere in the searched water. The scene average,
                      and what the analytic density estimate predicts.
      `at_contact` -- the emitter sits at the position of a real contact: one of the
                      bright things we found really is the ship. Clutter is clustered
                      along ice edges, so this is the operationally relevant case. A ship
                      in pack ice is surrounded by pack ice, not by open water.

    Reported per trial: how many OTHER contacts (the ice) the gate admits alongside it.
    """
    positions = [(float(c["world_coordinates"]["latitude"]),
                  float(c["world_coordinates"]["longitude"])) for c in contacts]
    density = len(contacts) / searched_km2 if searched_km2 else 0.0

    rows = []
    for label, make in cue_specs():
        gate = make().containment_radius_m()
        row: Dict[str, Any] = {
            "cue": label, "gate_r95_m": round(gate, 1),
            "contacts_per_km2": round(density, 4),
            "analytic_expected_in_gate": round(expected_contacts_in_gate(density, gate), 3),
        }
        for placement in ("water", "at_contact"):
            model = make()
            rng = random.Random(SEED + len(placement))
            counts: List[int] = []
            for _ in range(trials):
                if placement == "water":
                    if not water:
                        continue
                    _, lat, lon = water[rng.randrange(len(water))]
                    exclude = -1
                else:
                    exclude = rng.randrange(len(positions))
                    lat, lon = positions[exclude]
                fix = model.fix(lat, lon)
                counts.append(sum(
                    1 for i, (clat, clon) in enumerate(positions)
                    if i != exclude and haversine_m(fix.latitude, fix.longitude, clat, clon) <= gate))
            arr = np.asarray(counts, dtype=float) if counts else np.zeros(0)
            row[placement] = {
                **quant(arr),
                "pct_trials_clean": round(100.0 * float((arr == 0).mean()), 1) if arr.size else None,
            }
        rows.append(row)
    return {"trials": trials, "contacts": len(contacts), "by_cue": rows}


# --------------------------------------------------------------------------------------
# Experiment 3 -- US bundle, cooperative emitters
# --------------------------------------------------------------------------------------

def load_us_labels(labels_path: str, reference_path: str, contacts: List[Dict[str, Any]]
                   ) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    Hand verdicts carried onto THIS run's contacts by position, never by detection id.

    Ids are only stable while the detector sees the same pixels, and this run's contact
    set is not the one the team labelled: 375 contacts against the reference run's 571.
    Keying verdicts by id therefore drops most of the adjudication and, worse, can land a
    verdict on a different object than the one a human looked at. Two direct-match runs
    disagreed on precision by 0.14, which is how the instability was caught.

    `scripts/transfer_labels.py` pairs contacts within a few pixels, closest first, each
    labelled contact claimed at most once, and refuses to guess anything with no labelled
    neighbour. It also carries the MISS_<scene>_<mmsi> ghost-fix verdicts verbatim -- those
    say "a reviewer looked and there was no vessel under this broadcast", and dropping them
    charges every ghost AIS fix to the detector as a miss. That was a real bug in this repo
    once, so the count is asserted here rather than trusted.
    """
    if not os.path.exists(labels_path):
        print(f"  [warn] no labels at {labels_path}; experiments 3 and 3b cannot run")
        return {}, {}
    with open(labels_path, "r", encoding="utf-8") as f:
        raw = json.load(f).get("labels", {})
    if not os.path.exists(reference_path):
        raise SystemExit(
            f"[fatal] reference contacts not found at {reference_path}.\n"
            "        Labels must be carried by POSITION from the run they were made on.\n"
            "        Scoring by detection_id is unstable and is deliberately not offered.")
    with open(reference_path, "r", encoding="utf-8") as f:
        reference = json.load(f)["contacts"]

    labels, stats = transfer_labels(reference, raw, contacts, radius_px=4.0)

    raw_miss = sum(1 for k in raw if k.startswith("MISS_"))
    carried_miss = sum(1 for k in labels if k.startswith("MISS_"))
    if carried_miss != raw_miss:
        raise SystemExit(f"[fatal] {raw_miss - carried_miss} MISS_* ghost-fix verdicts were "
                         "lost in transfer; recall would be understated")
    print(f"  [labels] {len(raw)} verdicts from {os.path.basename(labels_path)}, carried by "
          f"position from {os.path.basename(os.path.dirname(reference_path))} "
          f"({len(reference)} contacts)")
    print(f"  [labels] {stats['labels_carried']} carried, "
          f"{stats['contacts_without_a_labelled_counterpart']} contacts left unlabelled, "
          f"{carried_miss}/{raw_miss} MISS_* ghost-fix verdicts preserved")
    print(f"  [labels] not re-detected in this run: {stats['labelled_vessels_not_redetected']} "
          f"labelled vessels, {stats['labelled_clutter_not_redetected']} labelled clutter")
    stats["miss_verdicts_in_source"] = raw_miss
    return labels, stats


def adjudicate(contacts: List[Dict[str, Any]], labels: Dict[str, str]
               ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """(vessels, false_alarms, unlabelled). Same rules scripts/scorecard.py uses."""
    vessels, false_alarms, unlabelled = [], [], 0
    for det in contacts:
        if det["classification"] in MATCHED:
            vessels.append(det)  # a transponder match is a vessel by definition
            continue
        verdict = labels.get(det["detection_id"])
        if verdict == "vessel":
            vessels.append(det)
        elif verdict in ("not_vessel", "structure"):
            false_alarms.append(det)
        else:
            unlabelled += 1
    return vessels, false_alarms, unlabelled


def _by_scene(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for d in items:
        out.setdefault(d["scene_id"], []).append(d)
    return out


def _latlon(d: Dict[str, Any]) -> Tuple[float, float]:
    c = d["world_coordinates"]
    return float(c["latitude"]), float(c["longitude"])


def experiment_cooperative(vessels: List[Dict[str, Any]], false_alarms: List[Dict[str, Any]],
                           trials: int, emitter_filter: Optional[float] = None) -> Dict[str, Any]:
    """
    Gate the adjudicated US contacts against cues placed on the vessels we know about.

    READ THE LIMITATION. The emitters here are AIS broadcasters plus reviewer-confirmed
    vessels. Vessel retention therefore lands at 95 % or better BY CONSTRUCTION, because
    the gate IS the 95 % containment radius of a fix placed on that vessel. It is a
    consistency check on the model, not a detection result, and it says nothing at all
    about vessels that do not broadcast.

    It comes out ABOVE 95 % -- 98 to 100 % -- and that is worth understanding rather than
    rounding away. US coastal scenes carry dense traffic, so gates overlap: a vessel whose
    own fix strayed too far is still retained by a neighbour's gate. That is real, it is
    why the retention column climbs with gate size, and it is also exactly why the
    false-alarm column climbs with it. Overlapping gates help and hurt symmetrically, and
    only the hurt half is a measurement.

    The measured quantity is FALSE-ALARM retention: how often a real false alarm, at its
    real position in the real clutter geometry of a real US coastal scene, falls inside
    the gate of a nearby real vessel. Clutter does not care about AIS, so that number
    transfers to a non-cooperative emitter population.

    `emitter_filter` restricts the emitting set to vessels at or above a hull length --
    the SOLAS carriage proxy. Vessels below it are still scored as vessels; they simply
    never receive a cue, which is the entire point.
    """
    emitters = vessels if emitter_filter is None else [
        v for v in vessels if float(v.get("hull_length_m") or 0.0) >= emitter_filter]

    v_scene, f_scene, e_scene = _by_scene(vessels), _by_scene(false_alarms), _by_scene(emitters)
    scenes = sorted(set(v_scene) | set(f_scene))

    # Project each scene's contacts once: the gate is re-drawn every trial, the geometry
    # is not. Gating is per-scene because a local tangent plane is only valid over one
    # scene, and scenes are hundreds of km apart -- no gate can span two of them.
    proj: Dict[str, Dict[str, Any]] = {}
    for sid in scenes:
        v = [_latlon(d) for d in v_scene.get(sid, [])]
        f = [_latlon(d) for d in f_scene.get(sid, [])]
        e = [_latlon(d) for d in e_scene.get(sid, [])]
        anchor = (v + f + e)[0] if (v + f + e) else (0.0, 0.0)
        proj[sid] = {"ref": anchor, "v": to_local_m(v, *anchor), "f": to_local_m(f, *anchor), "e": e}

    rows = []
    for label, make in cue_specs():
        model = make()
        gate = model.containment_radius_m()
        v_keep, f_keep, per_cue = [], [], []
        for _ in range(trials):
            kept_v = kept_f = 0
            for sid in scenes:
                p = proj[sid]
                fixes = [model.fix(lat, lon) for lat, lon in p["e"]]
                fm = to_local_m([(f.latitude, f.longitude) for f in fixes], *p["ref"])
                kept_v += int((min_distance_m(p["v"], fm) <= gate).sum())
                kept_f += int((min_distance_m(p["f"], fm) <= gate).sum())
                if len(fm):
                    if len(p["f"]):
                        d = np.hypot(p["f"][:, None, 0] - fm[None, :, 0],
                                     p["f"][:, None, 1] - fm[None, :, 1])
                        per_cue.extend((d <= gate).sum(axis=0).tolist())
                    else:
                        per_cue.extend([0] * len(fm))
            v_keep.append(kept_v)
            f_keep.append(kept_f)
        v_arr, f_arr = np.asarray(v_keep, float), np.asarray(f_keep, float)
        precision = v_arr / np.maximum(v_arr + f_arr, 1)
        rows.append({
            "cue": label, "cep_m": model.cep_m, "gate_r95_m": round(gate, 1),
            "emitters": len(emitters),
            "vessels_retained_mean": round(float(v_arr.mean()), 1),
            "vessel_retention_pct_BY_CONSTRUCTION":
                round(100.0 * float(v_arr.mean()) / max(len(vessels), 1), 1),
            "false_alarms_retained_mean": round(float(f_arr.mean()), 1),
            "false_alarm_retention_pct_MEASURED":
                round(100.0 * float(f_arr.mean()) / max(len(false_alarms), 1), 1),
            "modelled_precision_mean": round(float(precision.mean()), 3),
            "false_alarms_admitted_per_cue": quant(per_cue),
        })
    return {
        "trials": trials, "vessels": len(vessels), "false_alarms": len(false_alarms),
        "baseline_precision": round(len(vessels) / max(len(vessels) + len(false_alarms), 1), 3),
        "emitter_filter_m": emitter_filter, "by_cue": rows,
    }


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def banner(text: str) -> None:
    print(f"\n{'=' * 90}\n  {text}\n{'=' * 90}")


def report_swath(r: Dict[str, Any]) -> None:
    banner("EXPERIMENT 0 -- CUE ACCURACY vs THE SWATH  [pure geometry + MODELLED fix error]")
    print(f"  Point the {r['swath_km']} km swath at the fix. A target at frame centre can be up to")
    print(f"  {r['half_swath_m'] / 1000:.1f} km off before it is outside the frame -- before it has moved at all.")
    print("  scripts/cue_geometry.py spends that same budget on cue AGE; this spends it on cue ERROR.\n")
    print(f"  {'cue':<32}{'CEP':>9}{'gate r95':>11}{'gate/swath':>13}{'P(in swath)':>14}")
    print("  " + "-" * 79)
    for row in r["by_cue"]:
        print(f"  {row['cue']:<32}{row['cep_m'] / 1000:>7.1f} km{row['gate_r95_m'] / 1000:>9.1f} km"
              f"{row['gate_r95_vs_swath']:>12.2f}x{row['p_emitter_in_swath'] * 100:>13.1f}%")
    print("\n  'gate/swath' is the 95 % containment DIAMETER over the swath width. Above 1.0, a single")
    print("  frame pointed at the fix cannot cover the uncertainty and the cue needs a mosaic.")


def report_arctic(empty: Dict[str, Any], single: Dict[str, Any]) -> None:
    banner("EXPERIMENT 1 -- ARCTIC, EMPTY EMITTER SET  [cue MODELLED, clutter REAL]")
    print(f"  {empty['contacts']} contacts over {empty['searched_water_km2']:,.0f} km2 of searched water.")
    print("  Hand adjudication of the baseline contacts found ZERO vessels: ice floes, ridges,")
    print("  melt features. Ice carries no navigation radar, so the modelled RF picture is empty.\n")
    print(f"  {'cue':<32}{'gate r95':>11}{'contacts in':>14}{'retained':>10}{'rejected':>11}")
    print("  " + "-" * 78)
    for row in empty["by_cue"]:
        print(f"  {row['cue']:<32}{row['gate_r95_m'] / 1000:>9.1f} km{row['contacts_in']:>14}"
              f"{row['contacts_retained']:>10}{row['rejected_pct']:>10.1f}%")
    print(f"\n  False alarms per 1000 km2: {empty['false_alarms_per_1000km2_before']} -> 0.0")
    print("  Cue accuracy is irrelevant here. An empty fix list has no CEP, which is what makes")
    print("  this the only result in the study that does not rest on the error model.")

    banner("EXPERIMENT 2 -- ARCTIC, ONE HYPOTHETICAL EMITTER  [cue MODELLED, clutter REAL]")
    print(f"  Suppose one ship really is out there. {single['trials']} trials per placement.")
    print("  Reported: how many OTHER contacts (ice) the gate admits alongside it.\n")
    print(f"  {'cue':<32}{'gate':>9}{'analytic':>10}"
          f"{'water mean':>12}{'clean':>8}{'at-ice mean':>13}{'p90':>6}{'clean':>8}")
    print("  " + "-" * 98)
    for row in single["by_cue"]:
        w, a = row["water"], row["at_contact"]
        print(f"  {row['cue']:<32}{row['gate_r95_m'] / 1000:>6.1f} km{row['analytic_expected_in_gate']:>10.2f}"
              f"{w['mean']:>12.2f}{w['pct_trials_clean']:>7.0f}%"
              f"{a['mean']:>13.2f}{a['p90']:>6.0f}{a['pct_trials_clean']:>7.0f}%")
    print("\n  'clean' = trials where the gate admitted no ice at all, i.e. the cue is unambiguous.")
    print("  'water' places the ship anywhere in searched water; 'at-ice' places it inside the")
    print("  clutter field, which is where a ship transiting pack ice actually is.")


def report_us(main: Dict[str, Any], solas: Dict[str, Any], unlabelled: int) -> None:
    banner("EXPERIMENT 3 -- US BUNDLE, COOPERATIVE EMITTERS  [cue MODELLED, clutter REAL]")
    print(f"  {main['vessels']} adjudicated vessels, {main['false_alarms']} adjudicated false alarms,")
    print(f"  {unlabelled} unlabelled contacts excluded. Baseline precision {main['baseline_precision']}.")
    print("  Verdicts carried onto this run BY POSITION (scripts/transfer_labels.py), never by")
    print("  detection id -- ids renumber with the contact set and id-matching is unstable.")
    print("  Emitters are AIS broadcasters and reviewer-confirmed vessels -- COOPERATIVE ONLY.")
    print("  Vessel retention is >= 95 % BY CONSTRUCTION: the gate IS the 95 % containment radius")
    print("  of a fix placed on that vessel, so the fix lands inside its own gate 95 % of the time")
    print("  by definition. It runs higher still because dense coastal traffic makes gates overlap.")
    print("  It is NOT a dark-vessel detection result and must never be quoted as one.\n")
    print(f"  {'cue':<32}{'gate':>9}{'vessels kept':>15}{'FA kept':>14}{'precision':>11}{'FA/cue':>9}")
    print("  " + "-" * 90)
    for row in main["by_cue"]:
        print(f"  {row['cue']:<32}{row['gate_r95_m'] / 1000:>6.1f} km"
              f"{row['vessel_retention_pct_BY_CONSTRUCTION']:>13.1f}% "
              f"{row['false_alarm_retention_pct_MEASURED']:>12.1f}%"
              f"{row['modelled_precision_mean']:>11.3f}"
              f"{row['false_alarms_admitted_per_cue']['mean']:>9.2f}")
    print("\n  Only 'FA kept' and 'FA/cue' are measurements. They are measured on real false alarms")
    print("  at real positions, and clutter does not know whether a nearby ship broadcasts -- so")
    print("  those two columns are the part that transfers to a dark emitter.")

    banner(f"EXPERIMENT 3b -- SOLAS SENSITIVITY: only vessels >= {solas['emitter_filter_m']:.0f} m emit")
    print("  SOLAS V/19 requires a 9 GHz radar from 300 GT upwards. Below that carriage is not")
    print("  required, so an RF gate used as a NECESSARY condition silently deletes small craft --")
    print("  which is most of what an IUU-fishing mission is looking for. ESA saw this in real data:")
    print("  43 % of AIS vessels in the East China Sea had no RF detection at all.\n")
    print(f"  emitting vessels: {solas['by_cue'][0]['emitters']} of {solas['vessels']}\n")
    print(f"  {'cue':<32}{'vessels kept':>15}{'FA kept':>14}{'precision':>11}")
    print("  " + "-" * 72)
    for row in solas["by_cue"]:
        print(f"  {row['cue']:<32}{row['vessel_retention_pct_BY_CONSTRUCTION']:>13.1f}% "
              f"{row['false_alarm_retention_pct_MEASURED']:>12.1f}%{row['modelled_precision_mean']:>11.3f}")
    print("\n  Precision goes UP and recall goes DOWN. That trade is the honest cost of treating RF")
    print("  as a requirement rather than as evidence.")


def report_conclusions() -> None:
    banner("WHAT THIS PROVES, AND WHAT IT DOES NOT")
    print("""  PROVES (clutter side -- measured on real contacts at real positions):
    * With no emitters, an RF gate rejects every Arctic contact. The ice field that drives
      our worst precision is invisible to RF, so RF removes it entirely. This holds at any
      cue accuracy, because the fix list is empty.
    * A cue is only unambiguous while its gate is small compared with the clutter spacing.
      Experiment 2 quantifies where that stops being true.
    * A gate sweeps up real false alarms at a rate set by real clutter density, and that
      rate does not depend on whether the emitter broadcasts AIS.
    * At the only independently-measured error distribution we could find, the 95 %
      containment gate is larger than the whole 19.4 km swath. Cue ERROR, not just cue
      AGE, can exhaust the pointing budget in scripts/cue_geometry.py.

  DOES NOT PROVE:
    * Any dark-vessel detection improvement. Every cue in experiment 3 is placed on a
      COOPERATIVE or reviewer-confirmed vessel, because that is the only vessel ground
      truth that exists. A dark vessel has no AIS and never received a cue here.
    * That we can build this. No public spaceborne RF-geolocation dataset exists; the fix
      error is drawn from ESA's assessment of someone else's flown system, and we found no
      published link budget for spaceborne marine-radar intercept at all.
    * That every vessel emits. SOLAS V/19 requires radar from 300 GT; experiment 3b shows
      what an RF requirement costs below that threshold.""")


# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", choices=("arctic", "us", "both", "none"), default="both")
    ap.add_argument("--arctic-input", default="data/real/arctic_probe")
    ap.add_argument("--us-input", default="data/real/s2_us_bundle")
    ap.add_argument("--labels", default="data/outputs/review/labels_team.json")
    ap.add_argument("--reference-contacts", default=US_REFERENCE_CONTACTS,
                    help="contacts.json of the run --labels was made on; verdicts are "
                         "carried onto this run BY POSITION, never by detection id")
    ap.add_argument("--trials", type=int, default=2000, help="Monte Carlo trials (experiment 2)")
    ap.add_argument("--us-trials", type=int, default=200, help="Monte Carlo trials (experiment 3)")
    ap.add_argument("--out", default=None, help="write the full result as JSON")
    args = ap.parse_args()

    print(__doc__.split("THE FOUR EXPERIMENTS")[0].rstrip())
    results: Dict[str, Any] = {
        "seed": SEED, "disclaimer": "RF cue is MODELLED; optical contacts are REAL",
        "cues": [label for label, _ in cue_specs()],
        "esa_measured_median_error_m": ESA_EDAP_MEDIAN_ERROR_M,
    }

    swath = experiment_swath()
    report_swath(swath)
    results["experiment_0_swath"] = swath

    if args.bundle in ("arctic", "both"):
        print(f"\n[run] {args.arctic_input} (default config, unmodified pipeline)")
        context, telemetry = run_bundle(args.arctic_input, keep_rasters=True)
        card = scorecard_score(context, telemetry)
        contacts = context["classified_targets"]
        water = water_samples(context, per_scene=20000, rng=random.Random(SEED))
        print(f"[run] {len(contacts)} contacts, {card['searched_water_km2']:,.0f} km2 water, "
              f"{len(water)} water sample points")
        empty = experiment_empty(contacts, card["searched_water_km2"])
        single = experiment_single_emitter(contacts, card["searched_water_km2"], water, args.trials)
        report_arctic(empty, single)
        results["arctic"] = {"scorecard": {k: v for k, v in card.items() if k != "per_scene"},
                             "experiment_1_empty": empty, "experiment_2_single_emitter": single}
        del context, water

    if args.bundle in ("us", "both"):
        print(f"\n[run] {args.us_input} (default config, unmodified pipeline)")
        context, telemetry = run_bundle(args.us_input)
        contacts = context["classified_targets"]
        labels, transfer_stats = load_us_labels(args.labels, args.reference_contacts, contacts)
        card = scorecard_score(context, telemetry, labels or None)
        vessels, false_alarms, unlabelled = adjudicate(contacts, labels)
        print(f"[run] {len(contacts)} contacts, {card['searched_water_km2']:,.0f} km2 water, "
              f"{len(vessels)} vessels / {len(false_alarms)} false alarms / {unlabelled} unlabelled")
        if vessels and false_alarms:
            m = experiment_cooperative(vessels, false_alarms, args.us_trials)
            s = experiment_cooperative(vessels, false_alarms, args.us_trials,
                                       emitter_filter=SOLAS_LENGTH_PROXY_M)
            report_us(m, s, unlabelled)
            results["us"] = {"scorecard": {k: v for k, v in card.items() if k != "per_scene"},
                             "label_transfer": transfer_stats,
                             "experiment_3_cooperative": m, "experiment_3b_solas": s,
                             "unlabelled_excluded": unlabelled}

    report_conclusions()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"\n[done] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
