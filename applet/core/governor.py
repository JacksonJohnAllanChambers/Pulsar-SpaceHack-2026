"""
EdgeGovernor: closes a control loop from predicted die temperature back onto which
stages of the detection cascade actually run.

THE IDEA IN ONE LINE
--------------------
The spacecraft knows where it is in its orbit, so the applet should too: spend the
eclipse, coast the sunlit arc, and when the budget runs out *degrade the evidence,
never the alert*.

WHY DEGRADE INSTEAD OF THROTTLE
-------------------------------
If we do nothing, the Jetson still protects itself -- Orin's own thermal management
drops clocks once the junction approaches TJ_max. But hardware throttling is
indiscriminate. It slows everything by the same fraction, which on a cascade means the
cheap screener and the expensive wake transform both get slower and the pass simply
takes longer. A pass that takes longer can miss its downlink window entirely, and the
Track Guide is blunt about that: "technically low-power but slow could still miss its
window."

A governor that acts *before* the hardware does gets to choose what to give up. The
ordering below is cheapest-value-first, and the invariant is that every profile still
reports every dark vessel it finds. What shrinks is the corroborating evidence --
chips, heading, wake measurements -- not the alert itself. A ground operator would
far rather have a position with a weak reason attached than nothing at all.

WHAT IS REAL AND WHAT IS MODELLED
---------------------------------
  REAL, measured in the container: per-stage wall-clock time, CPU cores busy,
      peak RSS. These come from `EdgeTelemetryTracker` and are honest numbers.
  REAL, read from hardware when present: `/sys/.../thermal_zone*/temp`. On a Jetson
      this is the actual die temperature and the model steps aside (see
      `JetsonThermalSource`). We have never been able to exercise this path.
  MODELLED: watts, and therefore degrees, everywhere else. Power is inferred from
      measured CPU utilisation against NVIDIA's published nvpmodel envelopes.

The pitch states this split in exactly these terms. The control logic is the
contribution; the temperature it consumes is a prediction we could not validate.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from applet.core.thermal import OrbitModel, OrinThermalModel, ThermalParameters, ThermalState


@dataclass(frozen=True)
class CascadeProfile:
    """
    One rung of the degradation ladder.

    The fields are exactly the knobs the pipeline consults; nothing here is advisory.
    `rank` orders them from deepest (0) to shallowest, which is also the order the
    governor walks when it needs to shed load.
    """

    name: str
    rank: int
    power_mode_w: int              # the nvpmodel envelope this profile is sized for
    scene_workers: int             # concurrency, and therefore most of the power draw
    max_candidates_per_scene: int
    wake_transform: bool           # heading + wake SNR + Kelvin check
    verifier_enabled: bool         # the 47k-param INT8 chip CNN
    include_chips: bool            # JPEG evidence crops in the downlink bundle
    rationale: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rank": self.rank,
            "power_mode_w": self.power_mode_w,
            "scene_workers": self.scene_workers,
            "max_candidates_per_scene": self.max_candidates_per_scene,
            "wake_transform": self.wake_transform,
            "verifier_enabled": self.verifier_enabled,
            "include_chips": self.include_chips,
        }


# Cheapest-value-first. Read top to bottom as "what we give up next, and what it costs".
CASCADE_LADDER: List[CascadeProfile] = [
    CascadeProfile(
        name="FULL",
        rank=0,
        power_mode_w=25,
        scene_workers=3,
        max_candidates_per_scene=400,
        wake_transform=True,
        verifier_enabled=True,
        include_chips=True,
        rationale="Full cascade. Every alert carries a chip, a heading and a wake measurement.",
    ),
    CascadeProfile(
        name="REDUCED",
        rank=1,
        power_mode_w=15,
        scene_workers=2,
        max_candidates_per_scene=250,
        wake_transform=True,
        verifier_enabled=True,
        include_chips=False,
        rationale=(
            "Drop the JPEG evidence crops and halve concurrency. Detection quality is "
            "UNCHANGED -- the same physics and the same CNN run; only the payload shrinks. "
            "This is the free rung, and it is first for that reason."
        ),
    ),
    CascadeProfile(
        name="SURVEY",
        rank=2,
        power_mode_w=15,
        scene_workers=1,
        max_candidates_per_scene=120,
        wake_transform=False,
        verifier_enabled=True,
        include_chips=False,
        rationale=(
            "Drop the wake ray transform: the most expensive per-candidate stage. Costs us "
            "heading, speed and the wake term of the physics score, so precision falls and "
            "under-way/at-anchor cannot be distinguished. The CNN still runs, so false "
            "alarms do not run away. Recall on bright hulls is essentially held."
        ),
    ),
    CascadeProfile(
        name="BEACON",
        rank=3,
        power_mode_w=10,
        scene_workers=1,
        max_candidates_per_scene=60,
        wake_transform=False,
        verifier_enabled=False,
        include_chips=False,
        rationale=(
            "Physics only, hard candidate cap, no CNN. False alarms rise sharply and every "
            "contact is UNVERIFIED -- but positions still go down. The applet never stops "
            "reporting; it only stops explaining."
        ),
    ),
]

PROFILE_BY_NAME = {p.name: p for p in CASCADE_LADDER}


def active_profile(context: Dict[str, Any]) -> Optional[CascadeProfile]:
    """
    The profile in force for this pass, or None when the governor is not governing.

    None means "defer entirely to the config", which is what every default run does.
    Deliberately not "return FULL": FULL is a specific set of caps, and applying them
    when nobody asked would silently override an uplinked config that wanted 500
    candidates per scene. A disabled governor has to be invisible, not merely harmless.
    """
    profile = context.get("cascade_profile")
    return profile if isinstance(profile, CascadeProfile) else None


def capped_scene_workers(context: Dict[str, Any], configured: int) -> int:
    """
    Scene concurrency after the governor has had its say.

    Concurrency is where most of the SoC power actually goes -- three scenes in flight
    keep three cores busy -- so it is the knob that most directly moves the temperature.
    Capped, never raised, and never below 1: a pass with zero workers is a pass that
    reports nothing, which the ladder is explicitly designed never to do.
    """
    profile = active_profile(context)
    if profile is None:
        return configured
    return max(1, min(configured, profile.scene_workers))


class JetsonThermalSource:
    """
    Reads the real die temperature on a Jetson, and admits when it cannot.

    On JetPack the SoC thermal zones appear as `/sys/devices/virtual/thermal/thermal_zone*/`
    with a `type` of `tj`, `CPU-therm`, `GPU-therm` and similar. We take the hottest
    relevant zone, because that is the one that will throttle first.

    Everywhere else -- including the judges' container, which has no such zones -- this
    returns None and the governor falls back to `OrinThermalModel`. That fallback is the
    only path we have ever executed.
    """

    ZONE_GLOB = "/sys/devices/virtual/thermal/thermal_zone*"
    INTERESTING = ("tj", "cpu-therm", "gpu-therm", "soc0-therm", "soc1-therm", "soc2-therm")

    def __init__(self, zone_glob: Optional[str] = None):
        self.zone_glob = zone_glob or self.ZONE_GLOB

    def available(self) -> bool:
        return self.read_junction_c() is not None

    def read_junction_c(self) -> Optional[float]:
        hottest: Optional[float] = None
        for zone in glob.glob(self.zone_glob):
            try:
                with open(os.path.join(zone, "type"), "r", encoding="utf-8") as f:
                    kind = f.read().strip().lower()
                if kind not in self.INTERESTING:
                    continue
                with open(os.path.join(zone, "temp"), "r", encoding="utf-8") as f:
                    milli_c = float(f.read().strip())
            except (OSError, ValueError):
                continue
            celsius = milli_c / 1000.0
            # Zones sometimes report -256000 when a sensor is disabled.
            if celsius < -50.0 or celsius > 150.0:
                continue
            if hottest is None or celsius > hottest:
                hottest = celsius
        return hottest


@dataclass
class GovernorDecision:
    """One entry in the governor's audit trail. Downlinked with the telemetry."""

    elapsed_s: float
    stage: str
    junction_c: float
    chassis_c: float
    sunlit: bool
    soc_power_w: float
    profile: str
    reason: str
    temperature_source: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "elapsed_s": round(self.elapsed_s, 2),
            "stage": self.stage,
            "junction_c": round(self.junction_c, 2),
            "chassis_c": round(self.chassis_c, 2),
            "sunlit": self.sunlit,
            "soc_power_w": round(self.soc_power_w, 2),
            "profile": self.profile,
            "reason": self.reason,
            "temperature_source": self.temperature_source,
        }


class EdgeGovernor:
    """
    Picks a `CascadeProfile` from predicted thermal state, and records why.

    Usage from the runner:

        governor = EdgeGovernor(config)
        ...
        with tracker.stage(name):
            context = stage.process(context)
        governor.observe(name, seconds, cores_busy)   # steps the model, may re-profile

    The profile is published into the pipeline context so stages can honour it, and the
    full decision log ships in `edge_telemetry.json` so a ground operator can see
    exactly why a pass came down thinner than the last one.
    """

    # Idle draw of the module with the SoC doing nothing useful. [ASSUMED], but bounded:
    # it has to be below the 10 W mode envelope and above zero.
    IDLE_POWER_W = 4.0

    # Two DIFFERENT margins, and they must stay different. Using one value for both
    # collapses the hysteresis band to zero width and the governor oscillates between
    # two rungs every couple of minutes -- which is exactly what the first version of
    # this class did. The band between them (75 C .. 83 C here) is the dead zone.
    BURST_GUARD_C = 12.0      # do not START a burst within this of the throttle point
    RECOVERY_MARGIN_C = 20.0  # do not step back UP without this much headroom

    # Cap on forward simulation when asking "how long could we hold this?". Two orbits
    # is far beyond any decision we make and keeps the probe loop bounded.
    BURST_HORIZON_S = 2 * 94.6 * 60.0

    # A burst has to survive until relief arrives, plus this much slack. Sizing the
    # slack off the SoC time constant (~108 s) rather than picking a round number:
    # it is the time the die itself needs to respond once we do step down.
    BURST_RELIEF_MARGIN_S = 300.0

    def __init__(self,
                 parameters: Optional[ThermalParameters] = None,
                 orbit: Optional[OrbitModel] = None,
                 thermal_source: Optional[JetsonThermalSource] = None,
                 start_profile: str = "FULL",
                 enabled: bool = True,
                 pinned: bool = False):
        self.model = OrinThermalModel(parameters)
        self.orbit = orbit or OrbitModel()
        self.source = thermal_source or JetsonThermalSource()
        self.enabled = enabled
        # Pinned means the ladder is held where it is: still modelled, still logged,
        # never changed. An operator override outranks the controller.
        self.pinned = pinned
        self.profile: CascadeProfile = PROFILE_BY_NAME[start_profile]
        self.decisions: List[GovernorDecision] = []
        self._hardware_backed = self.source.available()
        self._elapsed_s = 0.0

    @classmethod
    def from_config(cls, config) -> "EdgeGovernor":
        """
        Build from an `AppletConfig`. Takes the config object rather than importing
        `AppletConfig` for a type hint, so `applet.config` stays free of any dependency
        on this module.

        `report_thermal` without `governor_enabled` gives a governor that models and
        reports but never acts -- see `ThermalConfig` for why those are separate.
        """
        t = config.thermal
        parameters = ThermalParameters(
            soc_heat_capacity_j_per_k=t.soc_heat_capacity_j_per_k,
            soc_to_chassis_r_k_per_w=t.soc_to_chassis_r_k_per_w,
            chassis_heat_capacity_j_per_k=t.chassis_heat_capacity_j_per_k,
            radiator_area_m2=t.radiator_area_m2,
            radiator_emissivity=t.radiator_emissivity,
            radiator_absorptivity=t.radiator_absorptivity,
            parasitic_load_w=t.parasitic_load_w,
            throttle_c=t.throttle_c,
            target_c=t.target_c,
            initial_junction_c=t.initial_junction_c,
            initial_chassis_c=t.initial_chassis_c,
        )
        orbit = OrbitModel(period_s=t.orbit_period_s, eclipse_fraction=t.eclipse_fraction)
        pinned = bool(t.pin_profile)
        if pinned and t.pin_profile not in PROFILE_BY_NAME:
            raise ValueError(
                f"pin_profile={t.pin_profile!r} is not a rung; "
                f"expected one of {sorted(PROFILE_BY_NAME)}")
        return cls(
            parameters=parameters,
            orbit=orbit,
            start_profile=t.pin_profile if pinned else t.start_profile,
            # A pin has to act, so it implies enabled -- otherwise the profile would be
            # published nowhere and pinning would silently do nothing.
            enabled=t.governor_enabled or pinned,
            pinned=pinned,
        )

    # --- power model -----------------------------------------------------------------

    def estimated_power_w(self, cores_busy: float) -> float:
        """
        Infer module power from measured CPU utilisation.

        This is the one place where a measured quantity (cores busy, from psutil, which
        is real in the container) becomes a modelled one (watts, which is not). Linear
        interpolation between idle and the profile's nvpmodel envelope is crude, but it
        is monotonic in the thing we can actually observe, and it is stated as a model
        rather than presented as a measurement.
        """
        budget = float(self.profile.power_mode_w)
        utilisation = max(0.0, min(1.0, cores_busy / max(1, self.profile.scene_workers * 2)))
        return self.IDLE_POWER_W + (budget - self.IDLE_POWER_W) * utilisation

    # --- thermal state ---------------------------------------------------------------

    @property
    def temperature_source(self) -> str:
        return "jetson_sysfs" if self._hardware_backed else "model"

    def _current_junction_c(self) -> float:
        if self._hardware_backed:
            reading = self.source.read_junction_c()
            if reading is not None:
                return reading
            self._hardware_backed = False  # sensor went away mid-pass; fall back
        return self.model.state.junction_c

    def seconds_to_throttle(self, power_w: float, sunlit: bool, horizon_s: float) -> float:
        """
        Forward-simulate a copy of the model: how long can we hold this power?

        Returns `horizon_s` if it never throttles inside the horizon. Simulating rather
        than solving analytically keeps this honest to whatever the model actually does,
        including the quartic radiator term.

        One analytic short-circuit first, because it is both exact and worth a lot. At
        constant power and constant illumination both nodes approach their equilibria
        monotonically, so a junction currently below the throttle point whose ASYMPTOTE
        is also below it can never reach it -- there is nothing to simulate. That is the
        common case (sunlit FULL settles at 94.4 C against a 95 C limit), and without
        this guard the probe integrated two full orbits, every rung, every stage, to
        re-derive "no" each time. It took the ground console's orbit view from 3.0 s to
        well under a tenth of that.
        """
        junction = self._current_junction_c()
        throttle = self.model.parameters.throttle_c
        if junction >= throttle:
            return 0.0
        if self.model.steady_state_junction_c(power_w, sunlit) < throttle:
            return horizon_s

        probe = self.model.copy()
        elapsed = 0.0
        step = 5.0
        while elapsed < horizon_s:
            probe.step(step, power_w, sunlit)
            elapsed += step
            if probe.is_throttling():
                return elapsed
        return horizon_s

    # --- the control law -------------------------------------------------------------

    def _select(self, sunlit: bool) -> tuple[CascadeProfile, str]:
        """
        Deepest profile we can justify, looking ahead rather than only reacting.

        Reacting to the junction alone would be far too late: the chassis time constant
        is ~70 minutes, so by the time the die is hot the bus has already banked the
        heat and no profile change cools it quickly. We therefore test each profile
        against its *predicted steady state*, and allow a burst only when the model says
        we reach the horizon before the throttle point.
        """
        junction = self._current_junction_c()
        throttle = self.model.parameters.throttle_c

        # Emergency: already at or past the throttle point. Shed to the floor and let
        # the bus radiate. Nothing subtle is appropriate here.
        if junction >= throttle:
            return CASCADE_LADDER[-1], (
                f"junction {junction:.1f} C at or above throttle {throttle:.1f} C; "
                "shed to floor")

        phase = "sunlit" if sunlit else "eclipse"
        to_terminator = self.orbit.seconds_until_terminator(self._elapsed_s)

        for candidate in CASCADE_LADDER:
            power = float(candidate.power_mode_w)
            steady = self.model.steady_state_junction_c(power, sunlit)

            if steady < self.model.parameters.target_c:
                return candidate, (
                    f"{candidate.name} sustainable: steady state {steady:.1f} C < target "
                    f"{self.model.parameters.target_c:.0f} C ({phase})")

            # Too hot to hold forever. A burst is only justified when *relief is coming*
            # before the throttle is -- and relief means the terminator. Crossing into
            # eclipse buys back ~16 W, which is most of a power mode.
            #
            # In eclipse there is no relief to wait for: sunrise makes it strictly worse,
            # so an unsustainable profile in eclipse is simply unsustainable. Bursting
            # there would be borrowing against a debt that never comes due.
            if not sunlit:
                continue

            # Do not start a burst from a die that is already close to the limit; the
            # forward simulation is a model, and this is the margin for it being wrong.
            if throttle - junction < self.BURST_GUARD_C:
                continue

            holdable = self.seconds_to_throttle(power, sunlit, self.BURST_HORIZON_S)
            needed = to_terminator + self.BURST_RELIEF_MARGIN_S
            if holdable >= needed:
                return candidate, (
                    f"{candidate.name} as burst: steady state {steady:.1f} C exceeds target, "
                    f"but holdable {holdable / 60:.0f} min > eclipse in "
                    f"{to_terminator / 60:.0f} min + {self.BURST_RELIEF_MARGIN_S / 60:.0f} min slack")

            # Cannot reach the terminator on this rung. Try the next one down.

        return CASCADE_LADDER[-1], (
            f"nothing sustainable at junction {junction:.1f} C in {phase}; floor")

    def reassess(self, stage: str = "") -> CascadeProfile:
        """
        Re-pick the profile. Hysteresis stops it oscillating between rungs.

        When disabled we still evaluate and log what we *would* have done, because
        `ThermalConfig.report_thermal` promises a temperature in the telemetry without
        promising to act on it. The profile itself is left untouched, so a disabled
        governor cannot change a single byte of detector output.
        """
        sunlit = self.orbit.sunlit_at(self._elapsed_s)
        chosen, reason = self._select(sunlit)

        if not self.enabled:
            self._record(stage, sunlit, f"governor disabled (reporting only); would pick "
                                        f"{chosen.name}: {reason}")
            return self.profile

        if self.pinned:
            self._record(stage, sunlit, f"pinned to {self.profile.name}; would otherwise "
                                        f"pick {chosen.name}: {reason}")
            return self.profile

        # Going shallower (shedding load) is always allowed immediately -- that is the
        # safety direction. Going deeper needs real margin, or a pass spends its time
        # flapping between two rungs instead of detecting ships.
        if chosen.rank < self.profile.rank:
            headroom = self.model.parameters.throttle_c - self._current_junction_c()
            if headroom < self.RECOVERY_MARGIN_C:
                chosen, reason = self.profile, (
                    f"holding {self.profile.name}: only {headroom:.1f} C headroom, "
                    f"need {self.RECOVERY_MARGIN_C:.0f} C to step back up")

        if chosen.name != self.profile.name:
            self.profile = chosen

        self._record(stage, sunlit, reason)
        return self.profile

    def _record(self, stage: str, sunlit: bool, reason: str) -> None:
        state = self.model.state
        self.decisions.append(GovernorDecision(
            elapsed_s=self._elapsed_s,
            stage=stage,
            junction_c=self._current_junction_c(),
            chassis_c=state.chassis_c,
            sunlit=sunlit,
            soc_power_w=state.soc_power_w,
            profile=self.profile.name,
            reason=reason,
            temperature_source=self.temperature_source,
        ))

    def observe(self, stage: str, seconds: float, cores_busy: float = 1.0) -> CascadeProfile:
        """
        Feed one completed stage in: advance the clock, heat the model, re-profile.

        `seconds` and `cores_busy` are both measured, not assumed -- they come straight
        from the telemetry tracker that already wraps every stage. The model runs even
        when the governor is disabled; only the acting is gated.
        """
        sunlit = self.orbit.sunlit_at(self._elapsed_s)
        power = self.estimated_power_w(cores_busy)
        self.model.step(seconds, power, sunlit)
        self._elapsed_s += seconds
        return self.reassess(stage)

    def advance_idle(self, seconds: float) -> ThermalState:
        """Coast at idle power -- between passes, or while waiting for a downlink window."""
        sunlit = self.orbit.sunlit_at(self._elapsed_s)
        state = self.model.step(seconds, self.IDLE_POWER_W, sunlit)
        self._elapsed_s += seconds
        return state

    # --- reporting -------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Goes into `edge_telemetry.json`, next to the RAM and CPU numbers."""
        state = self.model.state
        degraded = [d for d in self.decisions if d.profile != "FULL"]
        return {
            "enabled": self.enabled,
            "pinned": self.pinned,
            "temperature_source": self.temperature_source,
            "validated_on_hardware": False,  # never true for us; see module docstring
            "final_profile": self.profile.name,
            "final_profile_detail": self.profile.as_dict(),
            "junction_c": round(self._current_junction_c(), 2),
            "chassis_c": round(state.chassis_c, 2),
            "throttle_c": self.model.parameters.throttle_c,
            "headroom_c": round(self.model.parameters.throttle_c - self._current_junction_c(), 2),
            "sunlit": self.orbit.sunlit_at(self._elapsed_s),
            "modelled_elapsed_s": round(self._elapsed_s, 2),
            "profiles_used": sorted({d.profile for d in self.decisions}),
            "degraded_stage_count": len(degraded),
            "decisions": [d.as_dict() for d in self.decisions],
        }


def simulate_orbit(governor: "EdgeGovernor", minutes: int,
                   stage_seconds: float = 60.0, cores_busy: float = 5.5) -> Dict[str, Any]:
    """
    Drive a governor through `minutes` of CONTINUOUS full-tilt processing and return the
    trace. The worst realistic duty cycle on purpose: a real pass images an AOI and then
    idles, so anything that stays under the throttle point here stays under it in flight.

    Lives here rather than in the demo script because the ground console shows the same
    curve, and two implementations of the same claim would eventually disagree.
    """
    trace = []
    for i in range(minutes):
        governor.observe(f"scene{i}", stage_seconds, cores_busy)
        state = governor.model.state
        trace.append({
            "minute": round(governor._elapsed_s / 60.0, 1),
            "junction_c": round(state.junction_c, 2),
            "chassis_c": round(state.chassis_c, 2),
            "sunlit": state.sunlit,
            "profile": governor.profile.name,
        })

    throttle = governor.model.parameters.throttle_c
    transitions = [b for a, b in zip(trace, trace[1:]) if a["profile"] != b["profile"]]
    return {
        "eclipse_fraction": governor.orbit.eclipse_fraction,
        "period_min": round(governor.orbit.period_s / 60.0, 1),
        "minutes": minutes,
        "final_profile": governor.profile.name,
        "peak_junction_c": round(max(p["junction_c"] for p in trace), 2) if trace else None,
        "throttle_c": throttle,
        "target_c": governor.model.parameters.target_c,
        "throttled": any(p["junction_c"] >= throttle for p in trace),
        "profiles_used": sorted({p["profile"] for p in trace}),
        "transitions": len(transitions),
        "trace": trace,
    }
