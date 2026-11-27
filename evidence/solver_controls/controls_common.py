"""Tools shared by the paired solver and measurement controls."""

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FOLLOWUP = HERE.parent / "followup"
sys.path.insert(0, str(FOLLOWUP))
from recentered_fit import best_recentered
from fit_chain import initialize_from_data
from study_common import errors, matrices, serial_body


def fit_case(q, a, loads, observed, q_test, test_inertias, truth, unseen, starts=None, *, fixed_chart=False):
    if starts is None:
        starts, eligible = initialize_from_data(q, loads, a, observed)
    else:
        eligible = None
    result = dict(eligible=eligible, starts=len(starts))
    if not starts:
        return dict(result, stages=[], selected=None, final=None, initial=None)
    fitted, index, stages = best_recentered(q, loads, a, observed, starts, fixed_chart=fixed_chart)

    def evaluate(geometry):
        prediction = matrices(serial_body(q_test, *geometry), test_inertias)
        return errors(prediction, truth, unseen)

    assert all(1 <= s["nfev"] <= 1600 for s in stages)
    np.testing.assert_allclose(np.linalg.norm(fitted[0], axis=1), 1., atol=1e-12)
    np.testing.assert_allclose(np.sum(fitted[0]*fitted[1], axis=1), 0., atol=1e-12)
    return dict(result, stages=stages, selected=index, final=evaluate(fitted),
                initial=evaluate(starts[index]))
