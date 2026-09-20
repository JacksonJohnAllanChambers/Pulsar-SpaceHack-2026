"""
Why stationary vessels are missed -- an investigation, including the two hypotheses that were wrong.

THE CASE
--------
LE BOREAL, a 142 m cruise ship sitting at 0.1 kn in Isfjorden, Svalbard. Hand-adjudicated as a GENUINE
miss (`MISS_SVA_ISFJORDEN_578000500` = "vessel" in data/outputs/svalbard_review/labels.json), so this is
not a stale AIS fix. The sensor saw it and the pipeline dropped it.

THE ANSWER, measured
--------------------
It scored **physics 0.315 against a 0.35 gate**. It was missed by 0.035.

The full chain, each step verified with scripts/diagnose_miss.py:

  1. CFAR fires HARD at the predicted pixel: peak z = 16.1 against a gate of 5.0. Not a
     sensitivity problem.
  2. The linked component is 21 px, bbox 40 x 130 m -- about the right size for a 142 m hull.
     It passes the area gate, is not in the land or cloud mask, and does not touch the 200 m
     shore keep-out.
  3. But `_analyse_candidate` resolves the hull to **37.1 x 23.1 m**. A 142 m ship was
     under-segmented to a quarter of its length, and what survives is nearly square
     (elongation 1.61).
  4. A stubby hull scores badly on the shape term.
  5. `wake_length_m = 0.0`, so `s_wake` and `s_kelvin` are both zero -- **40 % of the physics
     score is unavailable before anything else happens** (0.30 wake + 0.10 kelvin).
  6. 0.315 < 0.35. Dropped, and dropped SILENTLY: a candidate that fails the physics gate is
     `continue`d and never appears in `detections`, `rejected_candidates` or
     `verifier_rejected`. The funnel shows 254 candidates -> 196 accepted; the other 58 leave
     no record at all.

**The real lesson is the error budget.** A moving vessel with exactly this bad a segmentation would
have cleared the gate on wake evidence alone and nobody would ever have noticed. A stationary vessel
has no margin: every imperfection elsewhere in the cascade becomes fatal, because the 40 % that would
have absorbed it is structurally zero.

TWO HYPOTHESES THAT WERE WRONG -- recorded so nobody re-derives them
-------------------------------------------------------------------
1. **The `round_and_big` hard cap.** `_physics_score` caps a >=30 m, wakeless, elongation<1.5 blob at
   0.30 -- below the gate, so it is an outright rejection, and it reads like the obvious culprit. It is
   innocent here: LE BOREAL's elongation is 1.61, so the cap never fires. The sweep below adds an
   escape hatch for that cap across five contrast thresholds and recovers ZERO AIS vessels on either
   Arctic bundle. Note the escape only engages below 0.30, so it also cannot reach a score in the dead
   band between the cap (0.30) and the gate (0.35) -- which is exactly where LE BOREAL sits.
2. **The ice-regime gate.** `if det["ice_regime"] and not (lead_found or kelvin_arms_detected)` drops
   ice-regime candidates lacking motion evidence, and the code comment already predicts the cost:
   "a vessel stopped dead in the pack: no lead, no wake". Also innocent here. Sweeping
   `ice_background_fraction` from 0.05 to disabled changes the Svalbard result NOT AT ALL
   (49 contacts, 3 AIS-confirmed, LE BOREAL still missed at every setting) -- because the candidate
   dies at the physics gate before the ice rule is ever consulted. That gate does earn its keep
   elsewhere, though: on `arctic_probe` it holds contacts to 115 where disabling it gives 156 (+36 %).

WHAT THIS SCRIPT IS
-------------------
The `round_and_big` sweep from hypothesis 1, kept because a measured negative result is worth more
than an untested intuition, and because the harness is the right shape for testing the NEXT candidate
fix. It measures only quantities that survive detection-id renumbering:

  * AIS-confirmed vessels -- keyed on MMSI, immune to renumbering. The recall side.
  * total contacts        -- the cost side. Every extra contact is a potential false alarm.

NOTHING HERE CHANGES FLIGHT BEHAVIOUR. It monkey-patches a copy of the scoring function for the
duration of the run; `applet/` and the default config are untouched.

WHERE TO GO NEXT
----------------
The lead is hull under-segmentation (step 3), not the wake terms. Fixing a 142 m ship being resolved
as 37 m would raise the shape term for every stationary vessel at once, and unlike loosening a gate it
costs no false alarms in principle. Start at `_analyse_candidate`'s hull extraction, on this exact
candidate, with scripts/diagnose_miss.py.

    python scripts/diagnose_miss.py -i data/real/svalbard_poc --mmsi 578000500
    python scripts/stationary_study.py --bundles data/real/svalbard_poc
"""

import argparse
import collections
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from applet.config import AppletConfig  # noqa: E402
from applet.pipelines.vessel_detector import VesselDetector  # noqa: E402
from applet.runner import run_pass  # noqa: E402

# The stock rule, reproduced here so the patch is a visible diff rather than a rewrite.
# A staticmethod accessed on the class is already a plain function in Python 3.
_ORIGINAL = VesselDetector._physics_score


def make_scorer(contrast_escape):
    """
    Return a _physics_score that lets a strongly-contrasting blob out of the round_and_big cap.

    `contrast_escape` is a peak z-score. None reproduces stock behaviour exactly, which is what makes
    the baseline row of the table trustworthy: it runs through the same patched code path as every
    other row, so any difference in the table is the threshold and nothing else.
    """

    def scorer(peak_z, hull_len_m, hull_wid_m, wake, occluded):
        score = _ORIGINAL(peak_z, hull_len_m, hull_wid_m, wake, occluded)
        if contrast_escape is None or occluded:
            return score
        # Only ever RAISE a score that the cap pushed down, and only for the round_and_big case --
        # never for `impossible` (wider than 65 m, longer than 460 m, shorter than 6 m), which
        # encodes geometry no vessel can have and must stay absolute.
        impossible = hull_wid_m > 65.0 or hull_len_m > 460.0 or hull_len_m < 6.0
        if impossible or score > 0.30:
            return score
        if peak_z >= contrast_escape and not wake["found"]:
            # Recompute without the cap. The blob is round, big and still, but it is far brighter
            # than its surroundings -- which is the one piece of evidence ice cannot fake indefinitely.
            return _ORIGINAL(peak_z, hull_len_m, 1.0, wake, occluded)
        return score

    return scorer


def run(bundle, contrast_escape):
    VesselDetector._physics_score = staticmethod(make_scorer(contrast_escape))
    out = tempfile.mkdtemp(prefix="stat_")
    try:
        cfg = AppletConfig()
        cfg.downlink.write_queues = False
        ctx, tel, dl = run_pass(bundle, out, cfg)
    finally:
        shutil.rmtree(out, ignore_errors=True)
        VesselDetector._physics_score = staticmethod(_ORIGINAL)

    targets = ctx["classified_targets"]
    classes = collections.Counter(t.get("classification") for t in targets)
    matched = {t.get("matched_vessel") for t in targets if t.get("matched_vessel") is not None}
    missed = {g.get("mmsi"): g.get("reason") for g in ctx.get("ais_not_observed", [])}
    return {
        "contacts": len(targets),
        "ais_confirmed": classes.get("CONFIRMED_KNOWN_VESSEL", 0),
        "dark": classes.get("DARK_VESSEL", 0),
        "mismatch": classes.get("AIS_KINEMATIC_MISMATCH", 0),
        "matched_mmsi": matched,
        "missed": missed,
    }


# LE BOREAL: 142 m, 0.1 kn, Isfjorden. Hand-adjudicated as a genuine miss. If any threshold in this
# sweep recovers it without an unacceptable contact-count blowup, that is the headline of the study.
WATCH = {578000500: "LE BOREAL (142 m, 0.1 kn)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundles", nargs="+",
                    default=["data/real/s2_us_bundle", "data/real/svalbard_poc",
                             "data/real/fundy_bundle"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[8.0, 10.0, 12.0, 15.0, 20.0])
    ap.add_argument("--json", help="write full results here")
    args = ap.parse_args()

    sweep = [None] + list(args.thresholds)
    results = {}

    for bundle in args.bundles:
        if not os.path.isdir(bundle):
            print(f"[skip] {bundle} not found")
            continue
        name = os.path.basename(bundle)
        print(f"\n{'=' * 78}\n  {name}\n{'=' * 78}")
        print(f"  {'escape z':>9}{'contacts':>10}{'AIS conf':>10}{'dark':>8}{'d contacts':>12}{'d AIS':>8}   recovered")
        print("  " + "-" * 74)
        base = None
        rows = []
        for thr in sweep:
            r = run(bundle, thr)
            if base is None:
                base = r
            gained = r["matched_mmsi"] - base["matched_mmsi"]
            note = ", ".join(WATCH.get(m, str(m)) for m in sorted(gained)) if gained else ""
            label = "stock" if thr is None else f"{thr:.0f}"
            print(f"  {label:>9}{r['contacts']:>10}{r['ais_confirmed']:>10}{r['dark']:>8}"
                  f"{r['contacts'] - base['contacts']:>+12}{r['ais_confirmed'] - base['ais_confirmed']:>+8}   {note}")
            rows.append({"escape_z": thr, **{k: v for k, v in r.items()
                                             if k not in ("matched_mmsi", "missed")},
                         "recovered_mmsi": sorted(gained)})
        results[name] = rows
        still = base["missed"]
        for mmsi, why in still.items():
            if mmsi in WATCH:
                print(f"\n  stock verdict on {WATCH[mmsi]}: {why}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[done] {args.json}")

    print("\n  Read the trade: every +1 AIS-confirmed is real recall; every +N contacts is the price,")
    print("  paid mostly in ice. A threshold is only worth shipping if the first column moves and the")
    print("  fourth does not explode -- and on the Arctic bundle it almost certainly will explode.")


if __name__ == "__main__":
    main()
