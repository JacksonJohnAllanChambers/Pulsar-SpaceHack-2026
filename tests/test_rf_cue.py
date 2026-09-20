"""
The modelled RF geolocation cue.

These tests assert what SHOULD happen, because nothing here can be checked against an RF
receiver -- we have never had one, and no open dataset of spaceborne RF fixes on vessels
exists to score against. What they can and do pin down is:

  * the statistics are self-consistent (the closed forms agree with what the sampler
    actually draws, and the approximations agree with the exact cases they contain),
  * the model reproduces the one set of INDEPENDENT third-party measurements that does
    exist -- ESA's assessment of 14 real Unseenlabs products against AIS -- including a
    statistic that was NOT fitted,
  * the honesty properties hold: no emitters means no retained contacts, at any accuracy,
  * and the two ways an earlier version of this module was wrong. Both have a named
    regression test below; neither was hypothetical.

If a number in docs or a report disagrees with a test here, the test is the number.
"""

import math

import pytest

from applet.core.rf_cue import (
    CEP_OVER_SIGMA,
    ESA_EDAP_MAX_EXPECTED_ERROR_M,
    ESA_EDAP_MEAN_ERROR_M,
    ESA_EDAP_MEDIAN_ERROR_M,
    ESA_EDAP_STD_ERROR_M,
    RFCueModel,
    S_BAND_GHZ,
    X_BAND_GHZ,
    associate,
    cep_from_sigmas_m,
    containment_radius_m,
    expected_contacts_in_gate,
    haversine_m,
    offset_position,
    sigma_from_cep_m,
)


def radial_errors(model: RFCueModel, n: int, lat: float = 40.0, lon: float = -70.0):
    return sorted(model.fix(lat, lon).error_m for _ in range(n))


def quantile(sorted_values, p: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(p * len(sorted_values)))]


# --------------------------------------------------------------------------------------
# The statistics, against their closed forms
# --------------------------------------------------------------------------------------

def test_cep_is_exactly_sqrt_2_ln_2_sigma():
    """The definition everything else rests on. Not a fit, not an approximation."""
    assert CEP_OVER_SIGMA == pytest.approx(math.sqrt(2.0 * math.log(2.0)))
    assert CEP_OVER_SIGMA == pytest.approx(1.17741, abs=1e-5)
    assert sigma_from_cep_m(1177.41) == pytest.approx(1000.0, rel=1e-4)


def test_containment_radius_is_not_the_cep():
    """
    The mistake this function exists to prevent.

    An operator quotes "1 km accuracy" and an analyst draws a 1 km circle. That circle
    misses the emitter half the time -- that is what a 50 % radius means. Keeping 95 % of
    the fixes costs 2.079 times the quoted number.
    """
    assert containment_radius_m(1000.0, 0.50) == pytest.approx(1000.0, rel=1e-9)
    assert containment_radius_m(1000.0, 0.95) == pytest.approx(2078.9, abs=0.5)
    assert containment_radius_m(1000.0, 0.99) == pytest.approx(2577.6, abs=0.5)
    # Monotone in probability, which a gate had better be.
    radii = [containment_radius_m(1000.0, p) for p in (0.1, 0.5, 0.9, 0.99)]
    assert radii == sorted(radii)


def test_sampled_errors_match_the_closed_form():
    """The sampler has to actually draw the distribution the closed form describes."""
    model = RFCueModel(cep_m=1000.0, seed=11)
    errors = radial_errors(model, 40000)
    assert quantile(errors, 0.50) == pytest.approx(1000.0, rel=0.03)
    assert quantile(errors, 0.95) == pytest.approx(containment_radius_m(1000.0, 0.95), rel=0.03)
    # Rayleigh mean is sigma * sqrt(pi/2), i.e. 1.0645 * CEP. A ratio far from that is the
    # signature of a tail the circular model is not supposed to have.
    assert sum(errors) / len(errors) == pytest.approx(1000.0 * 1.0645, rel=0.03)


def test_the_elliptical_cep_approximation_agrees_with_the_exact_circular_case():
    """
    An approximation that disagrees with the special case it contains is a bug.

    0.589 * (s1 + s2) at s1 == s2 gives 1.178 * sigma, which is sqrt(2 ln 2) to four
    digits. Torrieri's alternative 0.75 * sqrt(s1^2 + s2^2) gives 1.061 -- 10 % low -- so
    we do not use it, and this test is why.
    """
    sigma = 849.0
    assert cep_from_sigmas_m(sigma, sigma) == pytest.approx(CEP_OVER_SIGMA * sigma, rel=1e-3)
    torrieri = 0.75 * math.sqrt(2.0) * sigma
    assert torrieri < 0.93 * CEP_OVER_SIGMA * sigma


# --------------------------------------------------------------------------------------
# REGRESSION: the axial ratio meant its own square
# --------------------------------------------------------------------------------------

def test_gdop_axial_ratio_is_the_ratio_and_not_its_square():
    """
    An earlier version scaled the axes by k and 1/k, so asking for 3:1 produced 9:1.

    That is not a cosmetic naming problem. It drove the minor axis below the 0.3 validity
    floor of the CEP approximation, and the model then quietly returned fixes about 7 %
    worse than the CEP that was requested -- an error in the pessimistic direction, which
    is the kind that survives review.
    """
    for ratio in (1.0, 2.0, 3.0, 5.0):
        model = RFCueModel(cep_m=1000.0, gdop_axial_ratio=ratio, seed=3)
        major, minor = model.sigmas_m()
        assert major / minor == pytest.approx(ratio, rel=1e-9)


def test_requested_cep_survives_an_elongated_ellipse():
    """The realised median error must still be the CEP that was asked for."""
    for ratio in (1.0, 2.0, 3.0):
        model = RFCueModel(cep_m=1000.0, gdop_axial_ratio=ratio, seed=5)
        assert quantile(radial_errors(model, 30000), 0.50) == pytest.approx(1000.0, rel=0.04)


def test_gdop_preserves_the_cep_and_not_the_ellipse_area():
    """
    Elongation redistributes error. What it holds fixed is the CEP, not the area.

    An earlier version of this test asserted the area was invariant, and an earlier
    docstring claimed both. They cannot both hold: CEP goes as (s_maj + s_min) and area
    goes as (s_maj * s_min), so a transform that fixes one moves the other. CEP is the
    quantity the caller set, so CEP is the one that is preserved -- asking for a 1 km fix
    has to give a 1 km fix whatever the geometry.

    The area therefore SHRINKS with elongation, which is not a bug: the same median radial
    miss packed into a narrower band covers less ground. Pinning the direction here so
    nobody "fixes" it back.
    """
    flat = RFCueModel(cep_m=1000.0, gdop_axial_ratio=1.0, seed=1)
    elongated = RFCueModel(cep_m=1000.0, gdop_axial_ratio=4.0, seed=1)
    a_major, a_minor = flat.sigmas_m()
    b_major, b_minor = elongated.sigmas_m()

    assert b_major > a_major and b_minor < a_minor
    # CEP invariant: the 0.589 * (maj + min) approximation returns the requested value.
    # The 1e-3 tolerance is not slack -- it is the gap between 0.589 and the exact
    # 1/(2 sqrt(2 ln 2)) = 0.58870 that the circular axes are built from, and the two
    # agreeing to five parts in ten thousand is itself the check.
    assert cep_from_sigmas_m(b_major, b_minor) == pytest.approx(1000.0, rel=1e-3)
    assert cep_from_sigmas_m(a_major, a_minor) == pytest.approx(1000.0, rel=1e-3)
    # Area is not, and moves downward.
    assert b_major * b_minor < 0.9 * a_major * a_minor


def test_an_axial_ratio_below_one_is_rejected():
    """The ratio is major/minor. Below 1 it is a different quantity, not a small one."""
    with pytest.raises(ValueError):
        RFCueModel(cep_m=1000.0, gdop_axial_ratio=0.5)


# --------------------------------------------------------------------------------------
# REGRESSION: a pure Rayleigh cannot be the real distribution
# --------------------------------------------------------------------------------------

def test_a_rayleigh_cannot_have_mean_twice_its_median():
    """
    Why the heavy-tail component exists at all.

    ESA measured median 2.5 km and mean 5.4 km on real products -- a ratio of 2.16. A
    Rayleigh has mean/median fixed at sqrt(pi/2)/sqrt(2 ln 2) = 1.0645, full stop, at any
    sigma. So the published figures are arithmetically incompatible with the textbook
    model, and an earlier version of this module that used a bare Rayleigh was understating
    the tail by a factor of four in the gate radius.
    """
    rayleigh_ratio = math.sqrt(math.pi / 2.0) / CEP_OVER_SIGMA
    assert rayleigh_ratio == pytest.approx(1.0645, abs=1e-4)
    measured_ratio = ESA_EDAP_MEAN_ERROR_M / ESA_EDAP_MEDIAN_ERROR_M
    assert measured_ratio > 2.0
    assert measured_ratio > 1.9 * rayleigh_ratio


def test_esa_model_reproduces_the_three_published_moments():
    """
    The fit, checked against the numbers it was fitted to.

    ESA EDAP+ 2023, 14 Unseenlabs products against AIS: median 2.5 km, mean 5.4 km,
    std ~6.7 km. These come from the sampler, not from the fitting code, so this also
    proves the drawn mixture is the mixture that was solved for.
    """
    errors = radial_errors(RFCueModel.esa_edap_2023(seed=17), 150000)
    mean = sum(errors) / len(errors)
    variance = sum((e - mean) ** 2 for e in errors) / len(errors)

    assert quantile(errors, 0.50) == pytest.approx(ESA_EDAP_MEDIAN_ERROR_M, rel=0.03)
    assert mean == pytest.approx(ESA_EDAP_MEAN_ERROR_M, rel=0.03)
    assert math.sqrt(variance) == pytest.approx(ESA_EDAP_STD_ERROR_M, rel=0.05)


def test_esa_model_lands_on_a_statistic_it_was_not_fitted_to():
    """
    The out-of-sample check, and the reason to believe the fit is not just curve-drawing.

    Only the median, mean and standard deviation were fitted. ESA separately reports a
    maximum expected error around 30 km; the fitted mixture puts its 99th percentile at
    about 29 km, which nothing forced it to do.
    """
    p99 = RFCueModel.esa_edap_2023(seed=23).containment_radius_m(0.99)
    assert p99 == pytest.approx(ESA_EDAP_MAX_EXPECTED_ERROR_M, rel=0.12)


def test_the_measured_distribution_needs_a_gate_four_times_the_naive_one():
    """
    The single number this module exists to produce.

    At the ESA-measured distribution the 95 % containment radius is about 20.8 km, against
    5.2 km for a Rayleigh of the same 2.5 km CEP. Anyone sizing an association gate from
    the headline accuracy figure is off by a factor of four -- and 20.8 km is wider than
    the 19.4 km swath the cue would be used to point.
    """
    esa = RFCueModel.esa_edap_2023(seed=29)
    naive = containment_radius_m(ESA_EDAP_MEDIAN_ERROR_M, 0.95)
    assert naive == pytest.approx(5197.0, abs=50.0)
    assert esa.containment_radius_m(0.95) == pytest.approx(20772.0, abs=300.0)
    assert esa.containment_radius_m(0.95) / naive == pytest.approx(4.0, abs=0.2)
    assert 2.0 * esa.containment_radius_m(0.95) > 19_400.0  # wider than the HyperScape100 swath


def test_containment_radius_is_consistent_with_what_the_tailed_model_draws():
    """The analytic mixture quantile and the sampler must be the same distribution."""
    model = RFCueModel.esa_edap_2023(seed=31)
    errors = radial_errors(model, 120000)
    for p in (0.5, 0.9, 0.95, 0.99):
        assert quantile(errors, p) == pytest.approx(model.containment_radius_m(p), rel=0.06)


def test_a_tail_component_must_be_wider_than_the_core():
    """A 'tail' narrower than the core is a misconfiguration, not a subtle modelling choice."""
    with pytest.raises(ValueError):
        RFCueModel(cep_m=1000.0, tail_fraction=0.2, tail_sigma_multiplier=0.5)
    with pytest.raises(ValueError):
        RFCueModel(cep_m=1000.0, tail_fraction=1.0, tail_sigma_multiplier=5.0)


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------

def test_offset_position_round_trips_through_haversine():
    """A gate that mis-scales longitude at high latitude is a silently narrower gate."""
    for lat in (0.0, 40.0, 70.5):  # the Arctic probe scenes sit at 70-71 N
        for east, north in ((5000.0, 0.0), (0.0, 5000.0), (3000.0, -4000.0)):
            lat2, lon2 = offset_position(lat, -148.5, east, north)
            assert haversine_m(lat, -148.5, lat2, lon2) == pytest.approx(math.hypot(east, north), rel=2e-3)


def test_longitude_scaling_is_not_forgotten_at_arctic_latitude():
    """
    At 70.5 N a degree of longitude is 37 km, not 111 km.

    Dropping the cos(lat) factor would shrink every east-west gate by a factor of three,
    and the Arctic result is the centrepiece of this study.
    """
    _, lon_equator = offset_position(0.0, 0.0, 10000.0, 0.0)
    _, lon_arctic = offset_position(70.5, 0.0, 10000.0, 0.0)
    assert lon_arctic / lon_equator == pytest.approx(1.0 / math.cos(math.radians(70.5)), rel=1e-3)


def test_expected_contacts_in_gate_is_density_times_area():
    """The analytic cross-check the Monte Carlo is measured against."""
    assert expected_contacts_in_gate(1.0, 1000.0) == pytest.approx(math.pi)
    assert expected_contacts_in_gate(0.0789, 2078.9) == pytest.approx(0.0789 * math.pi * 2.0789 ** 2)


# --------------------------------------------------------------------------------------
# The honesty properties
# --------------------------------------------------------------------------------------

def test_no_emitters_means_no_fixes_and_no_retained_contacts():
    """
    The Arctic experiment, in three lines.

    Ice floes do not carry navigation radar, so the emitter set over a pack-ice scene is
    empty, so the fix list is empty, so nothing is retained -- at every cue accuracy,
    because an empty fix list has no accuracy. If this ever stops holding, the model has
    started inventing emitters and every Arctic number in the study is void.
    """
    contacts = [{"world_coordinates": {"latitude": 70.5 + 0.001 * i, "longitude": -148.5}}
                for i in range(115)]
    for make in (lambda: RFCueModel(cep_m=1000.0, seed=2),
                 lambda: RFCueModel(cep_m=10000.0, seed=2),
                 lambda: RFCueModel.esa_edap_2023(seed=2)):
        model = make()
        assert model.fixes([]) == []
        assert not any(a.retained for a in associate(contacts, model.fixes([])))


def test_association_is_a_proximity_test_that_admits_neighbours():
    """
    A contact inside a gate is CONSISTENT with the emitter, never proven to be it.

    Three contacts 1 km apart and one cue with a 10 km gate retains all three. The study
    reports that count rather than picking a winner, because picking a winner would be
    inventing an identification the RF fix cannot support.
    """
    contacts = [{"world_coordinates": {"latitude": 40.0 + 0.009 * i, "longitude": -70.0}}
                for i in range(3)]
    fixes = RFCueModel(cep_m=5000.0, seed=4).fixes([{"latitude": 40.009, "longitude": -70.0,
                                                     "id": "EM1"}])
    retained = [a for a in associate(contacts, fixes, gate_m=10000.0) if a.retained]
    assert len(retained) == 3
    assert all(a.nearest_emitter_id == "EM1" for a in retained)


def test_a_tight_gate_rejects_a_distant_contact():
    """The gate has to be able to say no, or clutter rejection means nothing."""
    contacts = [{"world_coordinates": {"latitude": 40.5, "longitude": -70.0}}]  # ~55 km away
    fixes = RFCueModel(cep_m=1000.0, seed=6).fixes([{"latitude": 40.0, "longitude": -70.0}])
    assert not any(a.retained for a in associate(contacts, fixes))


def test_fixes_are_reproducible_across_model_instances():
    """
    Same seed, same emitters, same fixes -- or the study is not evidence.

    This repo guarantees a byte-identical downlink tarball for a given bundle. A study
    that cannot be re-run to the same numbers does not meet the same bar, so the model
    uses `random.Random` and never numpy's global state.
    """
    emitters = [{"latitude": 40.0 + 0.01 * i, "longitude": -70.0, "id": f"E{i}"} for i in range(25)]
    a = RFCueModel(cep_m=2000.0, seed=1234).fixes(emitters)
    b = RFCueModel(cep_m=2000.0, seed=1234).fixes(emitters)
    assert [(f.latitude, f.longitude) for f in a] == [(f.latitude, f.longitude) for f in b]
    c = RFCueModel(cep_m=2000.0, seed=1235).fixes(emitters)
    assert [(f.latitude, f.longitude) for f in a] != [(f.latitude, f.longitude) for f in c]


def test_the_fix_carries_its_own_truth_and_error():
    """A fix that cannot be scored against its truth cannot be studied."""
    fix = RFCueModel(cep_m=3000.0, seed=8).fix(70.5, -148.5, emitter_id="ICE_TRANSIT")
    assert (fix.truth_latitude, fix.truth_longitude) == (70.5, -148.5)
    assert fix.emitter_id == "ICE_TRANSIT"
    assert fix.error_m == pytest.approx(
        haversine_m(fix.truth_latitude, fix.truth_longitude, fix.latitude, fix.longitude), rel=2e-3)


def test_the_modelled_band_is_marine_navigation_radar():
    """
    Not an arbitrary choice of band.

    9.3-9.5 GHz and 2.9-3.1 GHz are the maritime radionavigation allocations, they are
    what SOLAS V/19 requires ships to carry, and they are what a flown RF-geolocation
    cubesat measurably collects -- published Unseenlabs collection windows are
    9374.2-9424.8 MHz and 3024.0-3077.0 MHz, inside both.
    """
    assert X_BAND_GHZ == (9.3, 9.5) and S_BAND_GHZ == (2.9, 3.1)
    assert X_BAND_GHZ[0] <= 9.3742 and 9.4248 <= X_BAND_GHZ[1]
    assert S_BAND_GHZ[0] <= 3.0240 and 3.0770 <= S_BAND_GHZ[1]
    assert RFCueModel(cep_m=1000.0, seed=9).fix(40.0, -70.0).band_ghz == X_BAND_GHZ


# --------------------------------------------------------------------------------------
# How the study gets its labels
# --------------------------------------------------------------------------------------

def test_label_transfer_carries_ghost_fix_verdicts():
    """
    MISS_<scene>_<mmsi> verdicts must survive being carried onto a new contact set.

    They mean "a reviewer looked at this AIS broadcast and there was no vessel under it".
    Drop them and every ghost fix is charged to the detector as a miss, which understates
    recall. That was a real bug in this repo, and the RF study re-introduced the risk by
    transferring labels itself -- so `load_us_labels` refuses to continue if the count
    changes, and this pins the behaviour it relies on.
    """
    from scripts.transfer_labels import transfer

    reference = [{"detection_id": "S_T1", "scene_id": "S", "apex_px": [100, 100]}]
    labels = {"S_T1": "not_vessel", "MISS_S_123456789": "not_vessel"}
    contacts = [{"detection_id": "S_T9", "scene_id": "S", "apex_px": [101, 100]}]

    out, stats = transfer(reference, labels, contacts, radius_px=4.0)

    assert out["MISS_S_123456789"] == "not_vessel"
    assert stats["miss_verdicts_carried"] == 1
    # And the contact verdict moved to the new id, by position rather than by name.
    assert out["S_T9"] == "not_vessel"


def test_label_transfer_refuses_to_guess_a_distant_contact():
    """
    A contact with no labelled neighbour stays unlabelled -- never assumed either way.

    This is what makes position-carried scoring trustworthy where id-matching is not: the
    unlabelled are excluded and counted, instead of silently inheriting a verdict a human
    gave to a different object.
    """
    from scripts.transfer_labels import transfer

    reference = [{"detection_id": "S_T1", "scene_id": "S", "apex_px": [100, 100]}]
    labels = {"S_T1": "vessel"}
    contacts = [{"detection_id": "S_T9", "scene_id": "S", "apex_px": [400, 400]}]

    out, stats = transfer(reference, labels, contacts, radius_px=4.0)

    assert out == {}
    assert stats["contacts_without_a_labelled_counterpart"] == 1
    assert stats["labelled_vessels_not_redetected"] == 1


def test_the_study_never_scores_by_detection_id():
    """
    The methodological guard, asserted against the source.

    Detection ids renumber whenever the contact set changes, so a verdict keyed to an id
    from another run can land on a different object. Two direct-match runs of this bundle
    disagreed on precision by 0.14. The study must therefore reach its labels only through
    the position transfer, and `load_us_labels` is the single door.
    """
    import inspect
    import scripts.rf_cue_study as study

    source = inspect.getsource(study.load_us_labels)
    assert "transfer_labels(" in source, "labels must be carried by position"
    assert "MISS_" in source, "ghost-fix verdicts must be checked"
    # The raw label file is opened exactly once, inside the transfer path.
    assert sum("labels_path" in line and "open(" in line
               for line in inspect.getsource(study).splitlines()) == 1


# --------------------------------------------------------------------------------------
# The module must not be able to change a pass
# --------------------------------------------------------------------------------------

def test_no_pipeline_stage_imports_the_cue_model():
    """
    This module is additive and inert.

    The repo guarantees that the same input bundle produces a byte-identical downlink
    tarball. An RF cue is a pipeline INPUT we do not have, so nothing in the flight path
    may import this file; the study applies the gate after `run_pass` has returned. If
    someone wires it in later it belongs behind an off-by-default switch, exactly like
    `thermal.governor_enabled`, and this test should be replaced by one that proves the
    switch defaults to off.
    """
    import os
    import applet

    root = os.path.dirname(os.path.abspath(applet.__file__))
    offenders = []
    for folder, _, files in os.walk(root):
        for name in files:
            if not name.endswith(".py") or name == "rf_cue.py":
                continue
            path = os.path.join(folder, name)
            with open(path, "r", encoding="utf-8") as f:
                if "rf_cue" in f.read():
                    offenders.append(os.path.relpath(path, root))
    assert offenders == [], f"rf_cue must stay out of the flight path, found in {offenders}"
