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


@pytest.mark.parametrize(
    ("test_input_data","test_return_x","test_return_y"),
    (
        (np.array([0,1,3,4,np.nan]),
         [0.0, 1.0, 3.0, 4.0],
         [1.0, 0.75, 0.5, 0.25],
         ),
    ),
)
def test_get_cum_dist(test_input_data,test_return_x,test_return_y):
    """Testing get_pdf"""
    assert functions.get_cum_dist(test_input_data)[0].tolist()==test_return_x
    assert functions.get_cum_dist(test_input_data)[1].tolist()==test_return_y


@pytest.mark.parametrize(
    ("test_input_data","test_return_x","test_return_y"),
    (
        (np.array([0,1,3,4,8,34,5.6,2,67,8,3.4,6,np.nan]),
         [ 0.0,22.33333333,44.66666667],
         [0.03731343,0.00373134,0.00373134]),
    ),
)
def test_get_pdf(test_input_data,test_return_x,test_return_y):
    assert functions.get_pdf(test_input_data,4,'lo',True)[0].tolist() == pytest.approx(test_return_x, abs=1e-6, rel=1e-9)
    assert functions.get_pdf(test_input_data,4,'lo',True)[1].tolist() == pytest.approx(test_return_y, abs=1e-6, rel=1e-9)
