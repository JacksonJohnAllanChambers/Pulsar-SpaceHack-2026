"""
How stale can a cue be before a 19.4 km swath can no longer catch the ship?

WHY THIS EXISTS
---------------
The standard architecture for satellite dark-vessel work is cue-and-confirm: a wide-area
sensor (SAR, or RF geolocation) finds a candidate, and a narrow high-resolution imager is
tasked to go look at it. Galaxia's HyperScape100 is firmly a confirm-tier instrument --
4.75 m GSD is excellent, 19.4 km of swath is not a search width.

That architecture has a geometric limit nobody states out loud, and it is the single most
important number for deciding what Galaxia's *next* spacecraft should carry:

    a cue is only useful while the target is still inside the swath you point at it.

A vessel under way does not wait. At 15 knots it covers 7.7 m every second. The cue gives
you a position that was true when the cueing sensor saw it, and by the time an imager is
tasked, uplinked, slewed and overhead, that position has aged. Age times speed is an
along-track displacement; when it exceeds half the swath, a perfectly-pointed image can
miss the ship entirely.

This script computes that budget. It is pure geometry -- no model of ours, nothing fitted,
nothing that needs hardware to validate. Run it and check the arithmetic.

    python scripts/cue_geometry.py

THE CONCLUSION IT SUPPORTS
--------------------------
Cross-platform cueing (someone else's SAR -> our optical) has to close in tens of minutes,
which is hard when it involves a ground segment and a separate operator. A cue sourced
*on the same spacecraft* has effectively zero age, and the whole budget problem disappears.
That is the argument for putting a passive RF payload on the same bus as the imager --
see docs/GALAXIA_ALIGNMENT.md.
"""

import argparse

KNOT_MS = 0.514444  # 1 knot in m/s, exactly 1852 m / 3600 s

# [DATASHEET] Simera Sense HyperScape100 at 500 km: 4.75 m GSD, 19.4 km swath.
SWATH_KM = 19.4
GSD_M = 4.75


def max_cue_age_s(speed_knots: float, swath_km: float = SWATH_KM,
                  pointing_margin: float = 1.0) -> float:
    """
    Seconds of cue age a swath can absorb.

    The target is assumed to start at the swath centre, so it may run half the swath width
    in any direction before it leaves. `pointing_margin` < 1 shrinks the usable width to
    account for pointing and ephemeris error, which is what a real tasking chain would use.
    """
    if speed_knots <= 0:
        return float("inf")
    usable_m = 0.5 * swath_km * 1000.0 * pointing_margin
    return usable_m / (speed_knots * KNOT_MS)


def displacement_km(speed_knots: float, age_minutes: float) -> float:
    return speed_knots * KNOT_MS * age_minutes * 60.0 / 1000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--swath-km", type=float, default=SWATH_KM)
    parser.add_argument("--margin", type=float, default=0.7,
                        help="usable fraction of the swath after pointing/ephemeris error")
    args = parser.parse_args()

    print()
    print("=" * 78)
    print(f"  CUE STALENESS BUDGET -- {args.swath_km} km swath, {GSD_M} m GSD (HyperScape100 @ 500 km)")
    print("=" * 78)
    print("  Pure geometry. A cue is spent once the target can no longer be inside the frame.")
    print()

    print(f"  {'vessel speed':>14}{'perfect pointing':>20}{'with ' + str(int(args.margin * 100)) + '% usable swath':>26}")
    print("  " + "-" * 60)
    for knots in (5, 10, 15, 20, 25):
        ideal = max_cue_age_s(knots, args.swath_km, 1.0) / 60.0
        real = max_cue_age_s(knots, args.swath_km, args.margin) / 60.0
        print(f"  {knots:>10} kn {ideal:>17.1f} min {real:>21.1f} min")

    print()
    print("  How far a contact runs while a cue ages:")
    print()
    print(f"  {'age':>8}" + "".join(f"{str(k) + ' kn':>10}" for k in (10, 15, 20)))
    print("  " + "-" * 40)
    for minutes in (5, 15, 30, 60, 120):
        row = "".join(f"{displacement_km(k, minutes):>9.1f}" + " " for k in (10, 15, 20))
        print(f"  {minutes:>5} min {row}")

    usable = max_cue_age_s(15.0, args.swath_km, args.margin) / 60.0
    print()
    print("  READ THIS OFF THE TABLE:")
    print(f"  * A 15-knot vessel spends the whole budget in {usable:.0f} minutes of cue age.")
    print("  * A ground-mediated cross-platform cue -- another operator's SAR pass, downlinked,")
    print("    correlated, uplinked as a tasking order -- rarely closes that fast.")
    print("  * A cue generated ON THE SAME BUS has ~zero age and no budget problem at all.")
    print("  * This is why the next spacecraft wants a passive RF payload beside the imager,")
    print("    not a bigger imager: the constraint is cue LATENCY, not pixels.")
    print()


if __name__ == "__main__":
    main()
