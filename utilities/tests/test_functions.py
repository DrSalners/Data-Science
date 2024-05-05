import pytest
from utilities import functions
import numpy as np


@pytest.mark.parametrize(
    ("test_time_series", "test_autocorrelation"),
    [
        (
            [0, 4, 3, 5, 3, 2, 4, 4, 6, 7, 6, 7, 2, 4, 5, 6],
            [0.2406947890818859, 0.17813765182186236, -0.14534883720930233],
        ),
        (
            [0.4,0.2,1.56,8.1,10,34.9,1.53,5.7],
            [0.0033519122555751182, -0.0606177638383771, -1.3345891263246648],
        ),
        (
            np.array([0, 4, 3, 5, 3, 2, 4, 4, 6, 7, 6, 7, 2, 4, 5, 6]),
            [0.2406947890818859, 0.17813765182186236, -0.14534883720930233],
        ),
                (
            np.array([0.4,0.2,1.56,8.1,10,34.9,1.53,5.7]),
            [0.0033519122555751182, -0.0606177638383771, -1.3345891263246648],
        ),
    ],
)
def test_autocorr(test_time_series, test_autocorrelation):
    """Testing autocorr."""
    assert functions.autocorr(test_time_series, 4)[1] == test_autocorrelation


@pytest.mark.parametrize(
    ("test_data", "test_return_index"),
    [
        (
            [4,5,2,3,24.5,6,4,56.7,-1,1,-45.6],
            4,
        ),
    ],
)
def test_find_nearest(test_data, test_return_index):
    """Testing find_nearest."""
    assert functions.find_nearest(test_data, 32) == test_return_index


@pytest.mark.parametrize(
    ("test_n","test_w","test_return_distribution"),
    (
        (10,
        0.1,
        [0.027777777777777776, -0.02777777777777778, -0.03888888888888889, -0.005555555555555557,
         -0.01666666666666667, 0.005555555555555557, -0.01666666666666667, -0.01666666666666667,
         -0.02777777777777778, -0.01666666666666667]),
    ),
)
def test_parabdist(test_n,test_w,test_return_distribution):
    """Testing parabdist."""
    assert functions.parabdist(test_n,test_w,for_test=True) == test_return_distribution
