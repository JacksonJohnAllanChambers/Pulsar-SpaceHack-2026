"""
A MODEL of a passive RF geolocation cue, for the "what should the next spacecraft carry"
trade in docs/GALAXIA_ALIGNMENT.md section 6.

=======================================================================================
THIS MODULE IS A MODEL. NOTHING IN IT IS MEASURED BY US FROM AN RF RECEIVER.
=======================================================================================

There is no open dataset of spaceborne RF geolocation fixes on vessels: Unseenlabs and
HawkEye 360 are commercial, their products are marked "not for redistribution" and
"strictly confidential", and ESA's own assessment grades the data as not FAIR-compliant
because there is no searchable catalogue [3]. So the cue cannot be measured here; it can
only be simulated, and every number that leaves this module stays labelled a model.

What it CAN be simulated from is better than vendor marketing. ESA commissioned an
independent quality assessment that paired 14 real Unseenlabs products against AIS and
published the error statistics [2]. Those are third-party measurements of a flown system
and they are what the default distribution here is fitted to.

WHY THIS EXISTS
---------------
`scripts/cue_geometry.py` shows a cue is only useful while the target is still inside the
swath you point at it -- 14.7 minutes for a 15-knot vessel and a 70 %-usable 19.4 km
swath. That argues for a cue generated on the same bus, and marine navigation radar is
the emission a vessel under way cannot switch off and keep moving safely.

The useful question is not whether RF helps us SEE better. It would not: the CFAR already
finds the hulls. The useful question is whether RF helps us DISAMBIGUATE, because that is
where this project is weakest -- US precision 0.697, and 115 Arctic contacts in which
hand adjudication found zero real vessels.

An ice floe does not carry a navigation radar. That is the whole argument, and this
module exists so it can be quantified instead of asserted.

WHAT A CUE CAN AND CANNOT BE USED TO CLAIM
------------------------------------------
Read this before using anything here in a result.

A synthesised cue has to be placed on a position we believe holds an emitter. The only
vessel positions we hold ground truth for come from AIS or from a human reviewing pixels
-- COOPERATIVE or reviewer-confirmed vessels, which are by definition not dark.
Therefore:

  * LEGITIMATE: cue-to-detection geometry. The detector never looks at AIS, so how a
    contact responds to a cue placed on it does not depend on whether that ship
    broadcasts.

  * LEGITIMATE: clutter rejection and false-association rate. Measured from the real
    spatial distribution of real contacts; the emitter population is a model input, and
    in the Arctic case it is the EMPTY SET, which is the strongest form of the claim and
    does not depend on the error distribution at all.

  * NOT LEGITIMATE: any claim that an AIS-derived cue improves DARK-vessel detection.
    A dark vessel has no AIS, so no AIS-derived cue is ever placed on one. Any number
    computed that way is measuring cooperative ships and calling them dark.

`scripts/rf_cue_study.py` is written to stay on the right side of that line and says so
in its own output.

THE ERROR MODEL
---------------
Two families, both seeded and reproducible, selected by `tail_fraction`.

1. CIRCULAR RAYLEIGH (`tail_fraction = 0`, the textbook case). A 2-D Gaussian fix with
   equal variances has a Rayleigh radial error, and CEP -- the 50 % containment radius,
   which is how accuracy is quoted -- is exactly sigma * sqrt(2 ln 2) = 1.1774 sigma [6].
   Containment radius for probability p is sigma * sqrt(-2 ln(1-p)), so r95 = 2.079 CEP.
   That factor matters: a 1 km CEP does not buy a 1 km gate, it buys a 2.1 km gate if you
   want 95 % of your fixes inside it.

2. RAYLEIGH CORE PLUS HEAVY TAIL (`tail_fraction > 0`). The ESA assessment found the real
   error distribution is NOT Rayleigh: median 2.5 km but mean 5.4 km, standard deviation
   ~6.7 km, a non-negligible population past 10 km and a maximum expected error around
   30 km [2]. A mean more than twice the median cannot come from a Rayleigh (whose ratio
   is fixed at 1.064), so a pure Rayleigh materially understates the tail. We model it as
   a mixture: with probability `tail_fraction` the whole fix is drawn at
   `tail_sigma_multiplier` times the scale. `RFCueModel.esa_edap_2023()` returns the
   fitted parameters.

   The consequence is the single most important number in this module:

       for the ESA-measured distribution, r95 is about 20 km, not 5 km.

   That is an eightfold larger gate than the CEP suggests, and it is what makes the
   association problem hard. Two independent distribution families -- our fitted mixture
   and a lognormal matched to the same median and mean -- both land at 19-21 km, so the
   figure is a property of the published moments rather than of our choice of family.

GDOP AND THE ELLIPSE
--------------------
`gdop_axial_ratio` > 1 stretches the 1-sigma ellipse along one axis and shrinks it across,
holding the requested CEP fixed rather than the ellipse area -- see `sigmas_m` for why
those two cannot both be invariant and why CEP is the one that wins. Two honesty notes:

  * This is the HawkEye 360 failure mode, not the Unseenlabs one. HawkEye geolocates with
    TDOA/FDOA from three-satellite clusters, where GDOP applies in the textbook sense [5].
    Unseenlabs is a SINGLE-satellite direction-of-arrival system [1], so its error is
    driven by received signal quality, not constellation geometry -- ESA reports accuracy
    banded by a HIGH/MEDIUM/LOW signal-quality flag [2]. Do not attribute the Unseenlabs
    tail to satellite geometry; that would be wrong.
  * ESA also found the real error azimuths are NOT isotropic, and Unseenlabs offered no
    technical explanation [2]. We draw the ellipse orientation uniformly, so this model is
    isotropic in the pooled sense and does not reproduce that finding.

Not a flight stage. Nothing in `applet/pipelines` imports this module and nothing in
`run_pass` calls it, so a default pass is byte-for-byte what it was before this file
existed. It lives under `applet/core` because it is written to the flight contract --
pure, seeded, no I/O, no new dependencies -- and is the natural home for the cue input
described as item 4 of docs/GALAXIA_ALIGNMENT.md section 8, not because it runs today.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Geometry constants
# --------------------------------------------------------------------------------------

EARTH_RADIUS_M = 6371008.8  # IUGG mean radius; the study never spans more than a scene
M_PER_DEG_LAT = math.pi * EARTH_RADIUS_M / 180.0  # 111195 m

# CEP = sigma * sqrt(2 ln 2) for a circular bivariate normal. Exact, not a fit.
CEP_OVER_SIGMA = math.sqrt(2.0 * math.log(2.0))  # 1.177410...

# Marine navigation radar: the emission the proposed payload would intercept. These are
# the IMO/ITU maritime radionavigation allocations, and they are also, measurably, what a
# flown RF-geolocation cubesat actually collects -- the published collection windows for
# Unseenlabs passes are 9374.2-9424.8 MHz and 3024.0-3077.0 MHz [4].
X_BAND_GHZ = (9.3, 9.5)
S_BAND_GHZ = (2.9, 3.1)

# --------------------------------------------------------------------------------------
# ESA EDAP+ 2023: third-party error statistics for a flown system. See [2].
# These four numbers are MEASURED BY ESA, not by us, and not by the vendor.
# --------------------------------------------------------------------------------------
ESA_EDAP_MEDIAN_ERROR_M = 2500.0
ESA_EDAP_MEAN_ERROR_M = 5400.0
ESA_EDAP_STD_ERROR_M = 6700.0
ESA_EDAP_MAX_EXPECTED_ERROR_M = 30000.0

# Fitted to reproduce the three moments above with a two-component Rayleigh mixture.
# The fit pins the median exactly and lands mean 5399 m / std 6685 m against ESA's
# 5400 / 6700, and its 99th percentile is 28.9 km against ESA's ~30 km "max expected" --
# which was NOT fitted and is therefore a genuine out-of-sample check.
#
# What the fit does NOT reproduce: ESA's histogram shows a secondary local maximum near
# 5 km, and this mixture puts its second mode further out. The moments are right, the
# intermediate shape is not pinned, and any conclusion that depends on the 5-15 km region
# specifically should be treated as soft.
ESA_EDAP_TAIL_FRACTION = 0.278
ESA_EDAP_TAIL_SIGMA_MULTIPLIER = 6.8


def sigma_from_cep_m(cep_m: float) -> float:
    """1-sigma radial component of a CIRCULAR RAYLEIGH fix whose 50 % radius is `cep_m`."""
    if cep_m <= 0.0:
        raise ValueError("cep_m must be positive")
    return cep_m / CEP_OVER_SIGMA


def containment_radius_m(cep_m: float, probability: float = 0.95) -> float:
    """
    Radius containing the true emitter with `probability`, for a circular Rayleigh fix.

    r = sigma * sqrt(-2 ln(1 - p)), so r95 = 2.079 CEP and r99 = 2.578 CEP. The thing
    worth internalising is that this is NOT the CEP: an operator who quotes "1 km
    accuracy" and an analyst who draws a 1 km circle around the fix are describing a gate
    that misses the emitter half the time.

    For the heavy-tailed case use `RFCueModel.containment_radius_m`, which is four times
    larger at the ESA-measured parameters.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be in (0, 1)")
    return sigma_from_cep_m(cep_m) * math.sqrt(-2.0 * math.log(1.0 - probability))


def cep_from_sigmas_m(sigma_major_m: float, sigma_minor_m: float) -> float:
    """
    CEP of an elliptical fix: the standard 0.589 * (sigma_maj + sigma_min) approximation.

    Valid down to an axis ratio of about 0.3. At sigma_maj == sigma_min it returns
    1.178 * sigma, agreeing with the exact circular result to four digits -- asserted in
    the tests, because an approximation that disagrees with the exact case it contains is
    a bug. Torrieri's alternative, CEP ~ 0.75 * sqrt(s1^2 + s2^2) [6], gives 1.061 sigma
    in that same circular case, i.e. 10 % low, so we do not use it.
    """
    return 0.589 * (sigma_major_m + sigma_minor_m)


def haversine_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Great-circle distance in metres. Used for every association gate in the study."""
    phi_a, phi_b = math.radians(lat_a), math.radians(lat_b)
    d_phi = phi_b - phi_a
    d_lam = math.radians(lon_b - lon_a)
    h = math.sin(d_phi / 2.0) ** 2 + math.cos(phi_a) * math.cos(phi_b) * math.sin(d_lam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, h)))


def offset_position(latitude: float, longitude: float, east_m: float, north_m: float
                    ) -> Tuple[float, float]:
    """
    Displace a position by a local east/north offset in metres.

    Flat-earth local tangent plane. Offsets here are at most tens of km, where the
    approximation error is under a metre -- but it breaks down at the pole, so the
    longitude scaling is floored. The Arctic scenes sit at 70-71 N where a degree of
    longitude is 38 km, and getting that factor wrong would silently shrink every
    east-west gate by a factor of three.
    """
    lat = latitude + north_m / M_PER_DEG_LAT
    cos_lat = max(math.cos(math.radians(latitude)), 1e-6)
    lon = longitude + east_m / (M_PER_DEG_LAT * cos_lat)
    return lat, lon


# --------------------------------------------------------------------------------------
# The cue model
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RFFix:
    """One modelled RF geolocation fix. The truth position rides along so a study can score itself."""

    latitude: float
    longitude: float
    cep_m: float
    sigma_major_m: float
    sigma_minor_m: float
    ellipse_bearing_deg: float
    error_m: float           # realised distance from the truth position
    truth_latitude: float
    truth_longitude: float
    gate_m: float            # containment radius this fix was drawn to be scored against
    from_tail: bool = False  # drawn from the heavy-tail component
    emitter_id: Optional[str] = None
    band_ghz: Tuple[float, float] = X_BAND_GHZ

    def contains(self, latitude: float, longitude: float) -> bool:
        """Is a position inside this fix's containment circle? The gate used throughout."""
        return haversine_m(self.latitude, self.longitude, latitude, longitude) <= self.gate_m


@dataclass
class RFCueModel:
    """
    Draws modelled RF fixes for known emitter positions.

    Seeded and reproducible: the same seed and the same emitter list always produce the
    same fixes, on any platform, because it uses `random.Random` and never numpy's global
    state. That matters in this repo -- a bundle has to hash the same everywhere, and a
    study that cannot be re-run to the same numbers is not evidence.

    `cep_m` is the accuracy you are ASSUMING. It is an input, not a property of anything
    we own. Sweep it; do not pick one and call it the answer.
    """

    cep_m: float
    gdop_axial_ratio: float = 1.0      # sigma_major / sigma_minor; 1.0 = circular
    tail_fraction: float = 0.0         # probability a fix is a gross outlier
    tail_sigma_multiplier: float = 1.0
    gate_probability: float = 0.95
    seed: int = 20260919
    band_ghz: Tuple[float, float] = X_BAND_GHZ
    _rng: random.Random = field(init=False, repr=False)
    # Quantiles of the mixture need a bisection, and `fix` needs one on every single draw
    # to stamp the gate. Solving it per draw made a 150k-sample test take two minutes; the
    # value is constant for a given model, so it is solved once and kept.
    _quantile_cache: Dict[float, float] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.cep_m <= 0.0:
            raise ValueError("cep_m must be positive")
        if self.gdop_axial_ratio < 1.0:
            raise ValueError("gdop_axial_ratio is sigma_major/sigma_minor and cannot be < 1")
        if not 0.0 <= self.tail_fraction < 1.0:
            raise ValueError("tail_fraction must be in [0, 1)")
        if self.tail_fraction > 0.0 and self.tail_sigma_multiplier <= 1.0:
            raise ValueError("a tail component must be wider than the core")
        self._rng = random.Random(self.seed)

    # -- named parameterisations -------------------------------------------------------
    @classmethod
    def esa_edap_2023(cls, seed: int = 20260919, **kwargs: Any) -> "RFCueModel":
        """
        The only independently-measured spaceborne RF geolocation error we could find [2].

        ESA paired 14 real Unseenlabs SURMAR products against AIS: median 2.5 km, mean
        5.4 km, std ~6.7 km, max expected ~30 km. This is the row of the study that is
        anchored to reality rather than to a vendor claim, and it is the pessimistic one.
        """
        return cls(cep_m=ESA_EDAP_MEDIAN_ERROR_M, tail_fraction=ESA_EDAP_TAIL_FRACTION,
                   tail_sigma_multiplier=ESA_EDAP_TAIL_SIGMA_MULTIPLIER, seed=seed, **kwargs)

    # -- radial distribution -----------------------------------------------------------
    def _standardised_quantile(self, probability: float) -> float:
        """
        Quantile of the radial error in units of the core sigma.

        Closed form when there is no tail; bisection on the mixture CDF otherwise. The
        mixture CDF is monotone, so bisection is exact to machine precision and needs no
        solver dependency.
        """
        if not 0.0 < probability < 1.0:
            raise ValueError("probability must be in (0, 1)")
        if self.tail_fraction == 0.0:
            return math.sqrt(-2.0 * math.log(1.0 - probability))
        cached = self._quantile_cache.get(probability)
        if cached is not None:
            return cached
        q, m = self.tail_fraction, self.tail_sigma_multiplier

        def cdf(z: float) -> float:
            return ((1.0 - q) * (1.0 - math.exp(-z * z / 2.0))
                    + q * (1.0 - math.exp(-z * z / (2.0 * m * m))))

        lo, hi = 0.0, 40.0 * m  # cdf(hi) is 1 to machine precision
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if cdf(mid) < probability:
                lo = mid
            else:
                hi = mid
        self._quantile_cache[probability] = 0.5 * (lo + hi)
        return self._quantile_cache[probability]

    def core_sigma_m(self) -> float:
        """Core 1-sigma scale chosen so the realised radial MEDIAN equals `cep_m`."""
        return self.cep_m / self._standardised_quantile(0.5)

    def containment_radius_m(self, probability: Optional[float] = None) -> float:
        """
        Gate radius containing the true emitter with `probability`, for THIS model.

        With no tail this reduces to 2.079 * CEP at p = 0.95. With the ESA-measured tail
        it is about 8.3 * CEP, i.e. roughly 20 km -- which is the number that decides
        whether RF association is easy or hard, and it is four times what the headline
        accuracy figure would lead you to draw.
        """
        p = self.gate_probability if probability is None else probability
        return self.core_sigma_m() * self._standardised_quantile(p)

    # -- axes --------------------------------------------------------------------------
    def sigmas_m(self) -> Tuple[float, float]:
        """
        (major, minor) 1-sigma axes reproducing `cep_m` at the configured axial ratio.

        `gdop_axial_ratio` is sigma_major / sigma_minor directly, so 3 means a 3:1
        ellipse, not 9:1. An earlier version scaled the axes by k and 1/k, which made the
        realised ratio k-squared, pushed the minor axis below the 0.3 validity floor of
        the CEP approximation, and quietly returned fixes ~7 % worse than asked for. The
        tests pin the realised CEP against the requested one for exactly that reason.

        What is held fixed is the CEP, NOT the ellipse area. That is a choice, and the two
        are mutually exclusive: CEP depends on sigma_maj + sigma_min while area depends on
        sigma_maj * sigma_min, so only one of them can be invariant under elongation. An
        earlier docstring claimed both and a test caught it. CEP wins because CEP is the
        knob the caller set -- asking for a 1 km fix must give a 1 km fix whatever the
        geometry. The consequence, which is real rather than an artefact, is that an
        elongated fix of a given CEP encloses LESS area than a circular one: the error is
        concentrated into a narrower band while the median radial miss stays put.
        """
        sigma = self.core_sigma_m()
        r = math.sqrt(self.gdop_axial_ratio)
        scale = sigma * 2.0 / (r + 1.0 / r)  # keeps 0.589 * (maj + min) fixed
        return scale * r, scale / r

    # -- drawing -----------------------------------------------------------------------
    def fix(self, latitude: float, longitude: float, emitter_id: Optional[str] = None,
            bearing_deg: Optional[float] = None) -> RFFix:
        """One fix for an emitter at (latitude, longitude)."""
        sigma_major, sigma_minor = self.sigmas_m()
        from_tail = self.tail_fraction > 0.0 and self._rng.random() < self.tail_fraction
        if from_tail:
            sigma_major *= self.tail_sigma_multiplier
            sigma_minor *= self.tail_sigma_multiplier
        # Ellipse orientation. Real geometry sets this; we do not model the constellation,
        # so a uniform orientation is the honest stand-in and keeps the pooled error
        # isotropic, which stops it accidentally favouring a particular clutter geometry.
        bearing = self._rng.uniform(0.0, 360.0) if bearing_deg is None else float(bearing_deg)
        along = self._rng.gauss(0.0, sigma_major)
        across = self._rng.gauss(0.0, sigma_minor)
        theta = math.radians(bearing)
        east = along * math.sin(theta) + across * math.cos(theta)
        north = along * math.cos(theta) - across * math.sin(theta)
        fix_lat, fix_lon = offset_position(latitude, longitude, east, north)
        return RFFix(
            latitude=fix_lat, longitude=fix_lon, cep_m=self.cep_m,
            sigma_major_m=sigma_major, sigma_minor_m=sigma_minor,
            ellipse_bearing_deg=bearing, error_m=math.hypot(east, north),
            truth_latitude=latitude, truth_longitude=longitude,
            gate_m=self.containment_radius_m(), from_tail=from_tail,
            emitter_id=emitter_id, band_ghz=self.band_ghz,
        )

    def fixes(self, emitters: Iterable[Dict[str, Any]]) -> List[RFFix]:
        """
        Fixes for a list of {"latitude", "longitude", optional "id"} emitters.

        An empty emitter list yields an empty fix list. That is not a degenerate case to
        be tidied away -- it IS the Arctic experiment. Ice does not emit, so the correct
        modelled RF picture over a pack-ice scene is no fixes at all, and everything the
        optical detector found there is unsupported by RF at any cue accuracy whatsoever.
        """
        return [self.fix(float(e["latitude"]), float(e["longitude"]), e.get("id")) for e in emitters]


# --------------------------------------------------------------------------------------
# Association
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Association:
    """Outcome of gating one contact against a set of fixes."""

    index: int
    retained: bool
    nearest_fix_m: Optional[float]
    nearest_emitter_id: Optional[str]


def associate(contacts: Sequence[Dict[str, Any]], fixes: Sequence[RFFix],
              gate_m: Optional[float] = None) -> List[Association]:
    """
    Gate optical contacts against modelled RF fixes.

    A contact is RETAINED when some fix lies within the gate of it -- the RF evidence is
    consistent with that contact being an emitter. With no fixes at all nothing is
    retained, which is the point of the Arctic case.

    `gate_m` overrides each fix's own containment radius, so a study can hold the gate
    fixed while sweeping CEP.

    This is deliberately a SYMMETRIC PROXIMITY TEST, not a claim about causation. A
    contact inside a gate is consistent with the emitter; it is not proven to be it, and
    in a dense clutter field several contacts share one gate. The study reports how many.
    """
    out: List[Association] = []
    for i, contact in enumerate(contacts):
        coords = contact.get("world_coordinates") or contact
        lat, lon = float(coords["latitude"]), float(coords["longitude"])
        best_d: Optional[float] = None
        best_id: Optional[str] = None
        retained = False
        for fix in fixes:
            d = haversine_m(fix.latitude, fix.longitude, lat, lon)
            if best_d is None or d < best_d:
                best_d, best_id = d, fix.emitter_id
            if d <= (fix.gate_m if gate_m is None else gate_m):
                retained = True
        out.append(Association(i, retained, best_d, best_id))
    return out


def expected_contacts_in_gate(contacts_per_km2: float, gate_radius_m: float) -> float:
    """
    Analytic contamination: how many contacts a gate sweeps up from a UNIFORM field.

    E[n] = density * pi * r^2. The Monte Carlo must agree with this for uniformly-placed
    cues, and where it does NOT agree the disagreement is the interesting part -- real
    clutter is clustered along ice edges and coastlines, so a cue landing inside a cluster
    sees far more than the scene average.
    """
    return contacts_per_km2 * math.pi * (gate_radius_m / 1000.0) ** 2


# --------------------------------------------------------------------------------------
# Published figures this model is parameterised from -- INPUTS, not our measurements
# --------------------------------------------------------------------------------------

# The CEP sweep itself lives in `scripts/rf_cue_study.py:cue_specs()` and deliberately does
# NOT have a copy here. Two lists of operating points in two files drift apart, and the one
# that drifts is always the one a report was quoted from.

CITATIONS = """
Sources for every figure this model is parameterised from. None is a measurement of ours.

[1] Unseenlabs -- RF geolocation of maritime emitters. ESA classifies the sensor as a
    SINGLE-satellite direction-of-arrival detector, not a multi-satellite TDOA system;
    the company markets "mono-satellite technology". Published accuracy is best-case
    phrasing: ESA's mission description says geolocation "can reach 1 kilometre
    accuracy", the company says "Accuracy up to the Kilometer". Coverage is quoted as up
    to ~300,000 km2 per pass (550 x 550 km).
    https://earth.esa.int/eogateway/missions/unseenlabs/description
    https://unseenlabs.com/en/technology/
    CONFIDENCE: the 1 km figure is a vendor floor, not a typical value. Do not use it as
    an expected accuracy.

[2] ESA EDAP+ independent quality assessment (Diez-Garcia, EDAP+.REP.023, Issue 1.1,
    14 Nov 2023). 14 real Unseenlabs products paired against AIS ground truth:
    mean error ~5.4 km, median 2.5 km, std ~6.7 km, "a non-negligible number of emitters
    scoring above 10 km of error", max expected ~30 km. Errors are banded by a signal
    QUALITY flag, not by constellation geometry, and the error azimuths are explicitly
    NOT isotropic. ESA notes these are an upper bound, since bad AIS pairings inflate
    them. RF-to-AIS pairing rates across the 14 products ranged 28.0 % to 88.2 %, lowest
    in congested coastal water and highest in open sea.
    https://earth.esa.int/eogateway/documents/20142/37627/Technical-Note-on-Quality-Assessment-for-BRO.pdf
    CONFIDENCE: high -- ESA-commissioned, AIS-referenced, on real flown products. This is
    the best public evidence in existence and it is what the default model is fitted to.

[3] No open dataset. The ESA assessment records the products as "Strictly confidential",
    "Not for redistribution without Unseenlabs consent", with no DOI and no searchable
    catalogue, and grades them not FAIR-compliant. There is, however, a gated route for
    researchers: ESA runs an Announcement of Opportunity offering free Unseenlabs tasking
    and archive data against an evaluated proposal, open to ESA Member/Cooperating States
    and Canada until 31 December 2026.
    https://earth.esa.int/eogateway/announcement-of-opportunity/unseenlabs
    This is why this file is a model and not an evaluation -- and it is also the concrete
    next step if anyone wants to replace the model with measurements.

[4] Spaceborne intercept of MARINE NAVIGATION RADAR is demonstrated, not speculative.
    This corrects an assumption we started with. Unseenlabs' primary signal IS maritime
    radar: ESA states "data acquired are RF emissions from maritime radars", notes
    acquisitions "often appear in pairs: one carrier frequency corresponding to X-band,
    and other in S-band", and a product sample carries RF_Frequency_MHz = 3042.2. A
    peer-reviewed third-party study reports the measured collection windows across three
    Unseenlabs passes as 9374.2-9424.8 MHz and 3024.0-3077.0 MHz -- inside the 9.3-9.5
    GHz and 2.9-3.1 GHz marine navigation radar bands.
    https://www.kjrs.org/journal/view.html?pn=mostread&uid=970&vmd=Full
    CONFIDENCE: high for the fact of interception. What we could NOT find is any
    published link budget or detection-threshold analysis for spaceborne marine-radar
    intercept, so sensitivity as a function of radar power, antenna pattern and range is
    a genuine gap.

[5] HawkEye 360 -- TDOA/FDOA geolocation from three-satellite clusters; X- and S-band
    maritime radar are in the published signal catalogue, alongside VHF marine comms.
    Results are delivered with "a confidence ellipse representing a 95% probability".
    https://www.prnewswire.com/news-releases/new-hawkeye-360-radar-signals-deliver-comprehensive-maritime-awareness-301083209.html
    https://www.eoportal.org/satellite-missions/hawkeye
    CONFIDENCE: the architecture and band coverage are well documented. NO numeric
    accuracy figure is published by the company or in peer-reviewed work; figures
    circulating in secondary blogs are untraceable and are deliberately not used here.

[6] Error-distribution theory. D. J. Torrieri, "Statistical Theory of Passive Location
    Systems", IEEE Trans. Aerospace and Electronic Systems AES-20(2):183-198, 1984 --
    the standard reference relating the concentration ellipse, CEP and GDOP for
    hyperbolic and DF location systems. CEP(F) = sigma * sqrt(-2 ln(1-F)) for the
    circular case, so CEP(0.5) = 1.1774 sigma.
    https://ui.adsabs.harvard.edu/abs/1984ITAES..20..183T/abstract

[7] Carriage of marine radar is regulated, not universal. SOLAS Chapter V Regulation 19
    requires a 9 GHz (X-band) radar on ships of 300 GT and upwards and on passenger ships
    of any size, and a second, functionally independent radar -- 3 GHz S-band, or a second
    9 GHz set -- from 3,000 GT upwards. Most of the world fleet is below those thresholds:
    FAO SOFIA 2024 puts the global fishing fleet near 4.9 million vessels, of which 89 %
    of those with recorded length are under 12 m.
    https://openknowledge.fao.org/server/api/core/bitstreams/66538eba-9c85-4504-8438-c1cf0a0a3903/content/sofia/2024/fishing-fleet.html
    ESA observed this directly in the RF data: 43 % of AIS vessels in the East China Sea
    had no RF detection, concentrated in Chinese coastal waters, which it calls
    "compatible with small fishing fleets operating without navigation radars" [2].
    CONFIDENCE: high on substance; the regulation's paragraph numbering is from secondary
    sources and should be checked against an official IMO copy before being quoted.
    THIS IS THE BINDING LIMITATION on using RF as a NECESSARY condition: it would reject
    exactly the small non-broadcasting craft an IUU-fishing mission exists to find.
""".strip()
