"""
Formatting utilities for tactical reports and tables.
"""

from typing import List, Dict, Any


def format_tactical_table(targets: List[Dict[str, Any]]) -> str:
    """Renders a clean ASCII table of detected vessels for CLI inspection."""
    if not targets:
        return "No targets detected in processed scenes."

    header = (
        f"{'ID':<18} | {'STATUS':<24} | {'LAT/LON':<20} | {'HDG':>5} | {'SPD':>6} | "
        f"{'LEN':>5} | {'PHYS':>4} | {'CNN':>4} | {'PRIO':>4}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for t in targets:
        coords = t.get("world_coordinates", {})
        pos = f"{coords.get('latitude', 0.0):.4f},{coords.get('longitude', 0.0):.4f}"
        hdg = f"{t.get('heading_deg', 0):.0f}" + ("?" if t.get("heading_ambiguous_180") else "")
        spd = t.get("estimated_speed_knots")
        spd_s = f"{spd:.0f}kt" if spd is not None else "n/a"
        cnn = t.get("verifier_prob")
        cnn_s = f"{cnn:.2f}" if cnn is not None else "n/a"
        lines.append(
            f"{t.get('detection_id', 'N/A')[:18]:<18} | {t.get('classification', 'UNKNOWN')[:24]:<24} | {pos:<20} | "
            f"{hdg:>5} | {spd_s:>6} | {t.get('hull_length_m', 0):>4.0f}m | {t.get('physics_score', 0):>4.2f} | "
            f"{cnn_s:>4} | {t.get('downlink_priority', 0.0):>4.2f}"
        )

    lines.append(sep)
    return "\n".join(lines)
