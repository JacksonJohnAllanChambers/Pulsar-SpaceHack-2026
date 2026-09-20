"""
Thermal model and EdgeGovernor.

These tests assert what SHOULD happen, because nothing here can be checked against
hardware -- we never had a Jetson. What they can and do pin down is that the physics is
self-consistent (energy balance, monotonicity, convergence) and that the control law
behaves the way its docstring claims, including the two ways an earlier version of it
was wrong. Both of those bugs have a named regression test below; neither was
hypothetical, both were caught by simulating an orbit and reading the trace.
"""

import math

import pytest

from applet.core.governor import (
    CASCADE_LADDER,
    EdgeGovernor,
    JetsonThermalSource,
    PROFILE_BY_NAME,
)
from applet.core.thermal import (
    OrbitModel,
    OrinThermalModel,
    ThermalParameters,
    describe_budget,
    power_for_mode_w,
)


# --------------------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------------------

def test_radiated_power_follows_stefan_boltzmann():
    """Radiation is quartic in absolute temperature, not linear in Celsius."""
    p = ThermalParameters()
    cold, hot = p.radiated_w(0.0), p.radiated_w(100.0)
    ratio = hot / cold
    expected = (373.15 / 273.15) ** 4
    assert ratio == pytest.approx(expected, rel=1e-3)


def test_eclipse_removes_the_solar_load_entirely():
    p = ThermalParameters()
    assert p.absorbed_solar_w(False) == 0.0
    # alpha * S * A, no geometry factor: we point the radiator at the Sun in the worst case.
    assert p.absorbed_solar_w(True) == pytest.approx(0.20 * 1361.0 * 0.09, rel=1e-6)


def test_eclipse_buys_back_roughly_a_whole_power_mode():
    """
    The headline claim in the module docstring. If this drifts, the pitch is wrong.

    The operative fact is the bracket: 25 W is affordable in eclipse and is not
    affordable in sunlight. That single inequality is why a governor exists at all --
    no static power mode is correct on both sides of the terminator.
    """
    p = ThermalParameters()
    sunlit = p.sustainable_soc_power_w(sunlit=True)
    eclipse = p.sustainable_soc_power_w(sunlit=False)

    assert eclipse > 25.0 > sunlit

    # The penalty is worth at least a full mode step (25 W -> 15 W), which is what makes
    # the decision meaningful rather than a rounding adjustment.
    assert eclipse - sunlit >= 10.0


def test_the_radiator_partly_compensates_for_its_own_solar_load():
    """
    Subtle, and worth pinning: the sunlit penalty is NOT the full solar load.

    Holding the junction at `target_c` with less SoC power means a smaller conducted
    drop across R_js, so the chassis is allowed to sit hotter -- and a hotter radiator
    is a quartically better radiator. That self-compensation returns a real fraction of
    the solar input, so the budget difference lands well below `absorbed_solar_w`.

    An earlier version of this test asserted the two were equal. They are not, and the
    gap is physics rather than error.
    """
    p = ThermalParameters()
    penalty = p.sustainable_soc_power_w(False) - p.sustainable_soc_power_w(True)
    solar = p.absorbed_solar_w(True)

    assert penalty < solar, "the radiator must claw back some of its own solar load"
    assert penalty > 0.5 * solar, "but it cannot claw back most of it"

    # The mechanism, asserted directly: the sunlit case runs the hotter chassis.
    chassis_sunlit = p.target_c - p.sustainable_soc_power_w(True) * p.soc_to_chassis_r_k_per_w
    chassis_eclipse = p.target_c - p.sustainable_soc_power_w(False) * p.soc_to_chassis_r_k_per_w
    assert chassis_sunlit > chassis_eclipse
    assert p.radiated_w(chassis_sunlit) > p.radiated_w(chassis_eclipse)


def test_sustainable_power_actually_settles_at_the_target():
    """The iterative solve has to agree with the forward integration, or it is fiction."""
    p = ThermalParameters()
    for sunlit in (True, False):
        budget = p.sustainable_soc_power_w(sunlit)
        settled = OrinThermalModel(p).steady_state_junction_c(budget, sunlit)
        assert settled == pytest.approx(p.target_c, abs=1.0)


def test_integration_converges_to_the_analytic_steady_state():
    """Forward-integrate for a long time and land where the closed form says we should."""
    model = OrinThermalModel()
    predicted = model.steady_state_junction_c(15.0, sunlit=True)
    for _ in range(4000):  # ~5.5 hours at 5 s steps, several chassis time constants
        model.step(5.0, 15.0, sunlit=True)
    assert model.state.junction_c == pytest.approx(predicted, abs=1.0)


def test_junction_sits_above_chassis_by_the_conducted_drop():
    """Tj - Tc == P * R_js once the fast node has settled."""
    p = ThermalParameters()
    model = OrinThermalModel(p)
    for _ in range(400):
        model.step(5.0, 20.0, sunlit=True)
    drop = model.state.junction_c - model.state.chassis_c
    assert drop == pytest.approx(20.0 * p.soc_to_chassis_r_k_per_w, rel=0.05)


def test_hotter_power_mode_is_never_cooler():
    model = OrinThermalModel()
    for sunlit in (True, False):
        temps = [model.steady_state_junction_c(w, sunlit) for w in (10, 15, 25)]
        assert temps == sorted(temps)


def test_substepping_keeps_integration_stable_for_long_stages():
    """
    A stage longer than the SoC time constant must not blow up. Explicit Euler at
    dt=600 s against a 108 s time constant diverges without the internal sub-step.
    """
    coarse = OrinThermalModel()
    coarse.step(600.0, 25.0, sunlit=True)
    fine = OrinThermalModel()
    for _ in range(120):
        fine.step(5.0, 25.0, sunlit=True)
    assert coarse.state.junction_c == pytest.approx(fine.state.junction_c, abs=0.5)
    assert coarse.state.junction_c < 150.0  # sanity: did not diverge


def test_power_mode_rejects_modes_the_orin_nx_does_not_have():
    for w in (10, 15, 25):
        assert power_for_mode_w(w) == float(w)
    with pytest.raises(ValueError):
        power_for_mode_w(30)


def test_describe_budget_reports_the_numbers_the_pitch_quotes():
    b = describe_budget()
    assert b["sustainable_soc_w_eclipse"] > b["sustainable_soc_w_sunlit"]
    assert b["throttle_c"] > b["target_c"]


# --------------------------------------------------------------------------------------
# Orbit
# --------------------------------------------------------------------------------------

def test_orbit_alternates_sunlight_and_eclipse():
    orbit = OrbitModel()
    phases = [orbit.sunlit_at(t) for t in range(0, int(orbit.period_s), 60)]
    assert True in phases and False in phases
    sunlit_fraction = sum(phases) / len(phases)
    assert sunlit_fraction == pytest.approx(1.0 - orbit.eclipse_fraction, abs=0.05)


def test_dawn_dusk_orbit_never_reaches_a_terminator():
    """
    REGRESSION. Returning "one orbit away" here let the governor burst forever in
    permanent sunlight, waiting for an eclipse that does not exist. Relief that never
    arrives has to be spelled `inf`, not a large finite number.
    """
    orbit = OrbitModel(eclipse_fraction=0.0)
    assert orbit.sunlit_at(0.0) and orbit.sunlit_at(12345.0)
    assert orbit.seconds_until_terminator(0.0) == math.inf
    assert orbit.seconds_until_terminator(50_000.0) == math.inf


def test_terminator_countdown_is_bounded_by_the_period():
    orbit = OrbitModel()
    for t in range(0, int(orbit.period_s * 2), 137):
        assert 0.0 <= orbit.seconds_until_terminator(t) <= orbit.period_s


# --------------------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------------------

def test_ladder_is_monotonically_cheaper():
    """Every rung must cost strictly less than the one above it, or it is not a ladder."""
    for shallower, deeper in zip(CASCADE_LADDER, CASCADE_LADDER[1:]):
        assert deeper.rank == shallower.rank + 1
        assert deeper.power_mode_w <= shallower.power_mode_w
        assert deeper.scene_workers <= shallower.scene_workers
        assert deeper.max_candidates_per_scene < shallower.max_candidates_per_scene
        assert int(deeper.wake_transform) <= int(shallower.wake_transform)
        assert int(deeper.verifier_enabled) <= int(shallower.verifier_enabled)
        assert int(deeper.include_chips) <= int(shallower.include_chips)


def test_every_rung_still_reports_contacts():
    """
    The invariant the whole design rests on: degrade the evidence, never the alert.
    No profile may cap candidates at zero or otherwise stop the pass producing output.
    """
    for profile in CASCADE_LADDER:
        assert profile.max_candidates_per_scene > 0
        assert profile.scene_workers >= 1


def test_the_first_rung_down_costs_no_detection_quality():
    """REDUCED must keep both the physics and the CNN; it may only shrink the payload."""
    reduced = PROFILE_BY_NAME["REDUCED"]
    assert reduced.wake_transform and reduced.verifier_enabled
    assert not reduced.include_chips


# --------------------------------------------------------------------------------------
# The control law
# --------------------------------------------------------------------------------------

def _run(governor, minutes, cores_busy=5.5):
    for i in range(minutes):
        governor.observe(f"scene{i}", 60.0, cores_busy=cores_busy)
    return governor


def test_governor_falls_back_to_the_model_without_jetson_sensors():
    g = EdgeGovernor(thermal_source=JetsonThermalSource(zone_glob="/nonexistent/zone*"))
    assert g.temperature_source == "model"
    assert g.summary()["validated_on_hardware"] is False


def test_governor_reads_real_sensors_when_they_exist(tmp_path):
    """The flight path we could never execute. Faked here so it is at least exercised."""
    zone = tmp_path / "thermal_zone0"
    zone.mkdir()
    (zone / "type").write_text("tj", encoding="utf-8")
    (zone / "temp").write_text("87500", encoding="utf-8")  # millidegrees
    g = EdgeGovernor(thermal_source=JetsonThermalSource(zone_glob=str(tmp_path / "thermal_zone*")))
    assert g.temperature_source == "jetson_sysfs"
    assert g._current_junction_c() == pytest.approx(87.5)


def test_hottest_zone_wins_and_disabled_sensors_are_ignored(tmp_path):
    for name, kind, milli in (("z0", "cpu-therm", "45000"),
                              ("z1", "gpu-therm", "91000"),
                              ("z2", "tj", "-256000")):   # disabled sensor sentinel
        zone = tmp_path / f"thermal_{name}"
        zone.mkdir()
        (zone / "type").write_text(kind, encoding="utf-8")
        (zone / "temp").write_text(milli, encoding="utf-8")
    source = JetsonThermalSource(zone_glob=str(tmp_path / "thermal_z*"))
    assert source.read_junction_c() == pytest.approx(91.0)


def test_cold_start_runs_the_full_cascade():
    g = EdgeGovernor()
    assert g.reassess("start").name == "FULL"


def test_mid_beta_orbit_sustains_the_full_cascade():
    """With a third of the orbit in shadow, the bus can afford full depth."""
    g = _run(EdgeGovernor(orbit=OrbitModel()), minutes=400)
    assert g.profile.name == "FULL"
    assert g.model.state.junction_c < g.model.parameters.throttle_c


def test_dawn_dusk_orbit_forces_a_permanently_shallower_cascade():
    """
    The result the demo is built on: same applet, same code, a hotter orbit, and it
    works out on its own that full depth is not affordable there.
    """
    g = _run(EdgeGovernor(orbit=OrbitModel(eclipse_fraction=0.0)), minutes=400)
    assert g.profile.name == "REDUCED"
    assert g.model.state.junction_c < g.model.parameters.throttle_c
    assert "REDUCED" in g.summary()["profiles_used"]


def test_governor_does_not_oscillate_between_rungs():
    """
    REGRESSION. BURST_GUARD_C and RECOVERY_MARGIN_C were once the same constant, which
    made the shed threshold and the recover threshold the same temperature. The governor
    then flapped FULL<->REDUCED every one to two minutes for the whole orbit.

    Bound the transition count well below the ~100 that bug produced, and require the
    two margins to stay distinct so nobody quietly collapses the band again.
    """
    assert EdgeGovernor.RECOVERY_MARGIN_C > EdgeGovernor.BURST_GUARD_C

    g = _run(EdgeGovernor(orbit=OrbitModel()), minutes=400)
    names = [d.profile for d in g.decisions]
    transitions = sum(1 for a, b in zip(names, names[1:]) if a != b)
    assert transitions <= 8, f"governor flapped {transitions} times in 400 minutes"


def test_governor_sheds_to_the_floor_once_past_the_throttle_point():
    g = EdgeGovernor()
    g.model.state = type(g.model.state)(
        junction_c=99.0, chassis_c=70.0, elapsed_s=0.0, sunlit=True, soc_power_w=25.0)
    assert g.reassess("hot").name == "BEACON"
    assert "throttle" in g.decisions[-1].reason


def test_shedding_is_immediate_but_recovery_needs_margin():
    """Safety direction is instant; the other direction has to earn it."""
    g = EdgeGovernor()
    State = type(g.model.state)

    g.model.state = State(junction_c=99.0, chassis_c=70.0, elapsed_s=0.0,
                          sunlit=True, soc_power_w=25.0)
    assert g.reassess("hot").name == "BEACON"

    # Cooled, but not yet by RECOVERY_MARGIN_C. Must hold the shallow profile.
    g.model.state = State(junction_c=95.0 - (EdgeGovernor.RECOVERY_MARGIN_C - 2.0),
                          chassis_c=60.0, elapsed_s=0.0, sunlit=True, soc_power_w=10.0)
    assert g.reassess("cooling").name == "BEACON"
    assert "headroom" in g.decisions[-1].reason


def test_disabled_governor_never_leaves_the_full_cascade():
    g = _run(EdgeGovernor(orbit=OrbitModel(eclipse_fraction=0.0), enabled=False), minutes=300)
    assert g.profile.name == "FULL"
    assert g.summary()["enabled"] is False


def test_idle_between_passes_cools_the_bus():
    g = _run(EdgeGovernor(), minutes=60)
    hot = g.model.state.junction_c
    for _ in range(60):
        g.advance_idle(60.0)
    assert g.model.state.junction_c < hot


def test_power_estimate_is_bounded_by_the_profile_envelope():
    g = EdgeGovernor()
    assert g.estimated_power_w(0.0) == pytest.approx(EdgeGovernor.IDLE_POWER_W)
    assert g.estimated_power_w(99.0) == pytest.approx(float(g.profile.power_mode_w))
    assert g.estimated_power_w(3.0) > g.estimated_power_w(1.0)


def test_summary_is_json_serialisable_and_carries_the_audit_trail():
    import json

    g = _run(EdgeGovernor(orbit=OrbitModel(eclipse_fraction=0.0)), minutes=200)
    summary = g.summary()
    json.dumps(summary)  # must not raise: this ships inside edge_telemetry.json
    assert summary["decisions"], "every decision must be auditable from the ground"
    assert summary["temperature_source"] == "model"
    assert summary["final_profile"] == g.profile.name
    for decision in summary["decisions"]:
        assert decision["reason"]
