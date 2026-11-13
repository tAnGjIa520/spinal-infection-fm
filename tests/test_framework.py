import numpy as np
import pytest

from spinefm import AnatomyGuidedLDA, Seeds
from spinefm.anatomy import affected_vertebrae, annotation_regions, VERTEBRA_LABEL, DISC_LABEL
from spinefm.evaluation import cross_validate, make_splits, metric
from spinefm.framework import GLOBAL_GROUPS
from spinefm.framework import moments

DIMS = {"ct_rad": 333, "ct_gmean": 512, "ct_gmax": 512, "med_t1_mean": 2304,
        "med_t1_max": 2304, "med_t2_mean": 2304, "med_t2_max": 2304}


def synthetic_cohort(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n // 3), np.zeros(n - n // 3)].astype(int)
    cohort = {g: (rng.normal(size=(n, DIMS[g])) + .5 * y[:, None]).astype(np.float32) for g in GLOBAL_GROUPS}
    sizes = rng.integers(0, 4, size=n)
    cohort["ct_bags"] = [rng.normal(size=(k, 512)).astype(np.float32) + y[i] for i, k in enumerate(sizes)]
    for mod in ("t1", "t2"):
        cohort[f"{mod}_bags"] = [rng.normal(size=(k, 3, 768)).astype(np.float32) for k in sizes]
    cohort["y"], cohort["ids"] = y, np.array([f"P{i:03d}" for i in range(n)])
    return cohort


def test_annotation_parsing():
    assert affected_vertebrae("T11-L1;L3") == ["T11", "T12", "L1", "L3"]
    regions = annotation_regions("T11-T12;L3")
    assert regions[0] == tuple(sorted({VERTEBRA_LABEL["T11"], VERTEBRA_LABEL["T12"], DISC_LABEL[("T11", "T12")]}))
    assert set(regions[1]) == {VERTEBRA_LABEL["L3"], DISC_LABEL[("L2", "L3")], DISC_LABEL[("L3", "L4")]}
    with pytest.raises(ValueError):
        annotation_regions("X9")


def test_moments_shrinkage_variance():
    bag = np.array([[0.] * 8, [2.] * 8], np.float32)
    z = moments([bag, np.zeros((0, 8), np.float32)])
    np.testing.assert_allclose(z[0, :8], 1)
    np.testing.assert_allclose(z[0, 8:], 2 / 4)   # sum of squares 2, divided by n + 2
    np.testing.assert_array_equal(z[1], 0)        # empty bag


def test_feature_dimensions_and_threshold():
    cohort = synthetic_cohort()
    train = np.arange(0, 60, 2)
    model = AnatomyGuidedLDA().fit(cohort, train, Seeds.for_fold(0, 0))
    assert model.features(cohort, train).shape == (30, 112 + 48)
    assert AnatomyGuidedLDA(use_regional=False).fit(cohort, train).features(cohort, train).shape[1] == 112
    assert AnatomyGuidedLDA(use_global=False).fit(cohort, train).features(cohort, train).shape[1] == 48


def test_held_out_patients_do_not_affect_fit():
    cohort = synthetic_cohort()
    train, test = np.arange(45), np.arange(45, 60)
    a = AnatomyGuidedLDA().fit(cohort, train).decision_function(cohort, train)
    for g in GLOBAL_GROUPS:
        cohort[g][test] = 1e3
    for key in ("ct_bags", "t1_bags", "t2_bags"):
        for i in test:
            cohort[key][i] = cohort[key][i] * 0 + 1e3
    b = AnatomyGuidedLDA().fit(cohort, train).decision_function(cohort, train)
    np.testing.assert_allclose(a, b)


def test_cross_validation_runs():
    cohort = synthetic_cohort()
    splits = make_splits(cohort["ids"], cohort["y"], repeats=2, folds=3)
    scores = cross_validate(cohort, splits)
    assert scores.shape == (2, 60)
    assert metric(cohort["y"], scores[0])["auc"] > .6


def test_baseline_selection_uses_training_patients_only():
    from spinefm.baselines import RadiomicsStyleModel
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(1)
    y = np.r_[np.ones(30), np.zeros(50)].astype(int)
    x = rng.normal(size=(80, 300))
    x[:, :5] += 1.5 * y[:, None]
    train, test = np.arange(0, 80, 2), np.arange(1, 80, 2)
    inner = [(train[a], train[b]) for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(train, y[train])]
    model = RadiomicsStyleModel().fit([x], y, train, inner)
    assert len(model.selected) >= 1
    s = model.decision_function([x], test)
    x2 = x.copy()
    x2[test] = 1e3
    again = RadiomicsStyleModel().fit([x2], y, train, inner)
    np.testing.assert_array_equal(model.selected, again.selected)
    assert metric(y[test], s)["auc"] > .7
