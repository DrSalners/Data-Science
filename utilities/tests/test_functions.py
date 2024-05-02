import pytest
from utilities import functions


@pytest.mark.parametrize(
    ("test_time_series", "test_autocorrelation"),
    [
        (
            [0, 4, 3, 5, 3, 2, 4, 4, 6, 7, 6, 7, 2, 4, 5, 6],
            [0.2406947890818859, 0.17813765182186236, -0.14534883720930233],
        )
    ],
)
def test_autocorr(test_time_series, test_autocorrelation):
    """Testing autocorr."""
    assert functions.autocorr(test_time_series, 4)[1] == test_autocorrelation
