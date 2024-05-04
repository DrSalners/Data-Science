import numpy as np
import scipy.stats as sp
from scipy.optimize import minimize
import scipy as sc
from scipy import signal as sig
import scipy.special as sc
import powerlaw
import pickle
import glob, os.path
import networkx as nx
import pandas as pd


def autocorr(timeseries: list[float | int] | np.ndarray[float | int],
             stepnumber: int,
             ) -> list[np.ndarray[int], np.ndarray[float]]:
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float | int] | np.array[float | int]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber].
        acf (list[float]): multistep regression estimate (autocorrelation function).
    """
    steps = np.arange(1, stepnumber, 1)
    acf=[sp.linregress(timeseries[:-k], timeseries[k:])[0] for k in steps]
    return [steps, acf]


def find_nearest(data: list[float],
                 value: float | int,
                 ) -> int:
    """Finds sample from data set closes to a target value.

    Args:
        data (list[float]): data to search.
        value (float | int): number of regression steps to take.

    Returns:
        indnear (int): index where data is closest to target value.
    """
    near = [abs(i - value) for i in data]
    indnear = near.index(min(near))
    return indnear


def parabdist(n: int,
              w: float | int,
              for_test: bool=False
              ) -> list[float]:
    """Generates n samples from a parabolic distribution with width w.
    parabolic distribution -> f(x)=3/(2w^3)(w^2-4x^2)

    Args:
        n (int): number of samples to draw.
        w (float | int): width of distribution to sample.
        for_test (bool): if true, uses seed value "15" for unit-testing. Defaults to False.

    Returns:
        dis (list[float]): list of samples drawn from parabolic distribution.

    """
    if for_test:
        np.random.seed(15)
    else:
        np.random.seed()
        
    x = np.linspace(-w / 2.0, w / 2.0, n)
    f = (3.0 / (2.0 * w**3)) * (w**2 - 4 * x**2)
    g = (
        (3.0 / 2.0)
        * w ** (-3.0)
        * (((w ** (3.0) / 3.0) + (w ** (2.0) * x) - ((4.0 / 3.0) * x ** (3.0))))
    )
    dis = [x[find_nearest(g,np.random.uniform(0,1,1))] for i in range(n)]
    return dis


# Function 7: estimating branching ratio with MSR estimator
def mrestimate(fires, steps, isi):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    steps, rk = autocorr(fires, steps)
    m = np.exp(sp.stats.linregress(steps, np.log(rk))[0] * isi)  
    #TODO: fix this: I don't know why isi is here


# Function 8: DFA
def dfa(signal, min, max, num):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    signal = np.cumsum(signal)
    x = np.logspace(np.log10(min), np.log10(max), num)
    y = []
    for t in x:
        t = int(t)
        sigmafull = []
        for j in range(len(signal)):
            if j == 0:
                continue
            start = j * t - int(t / 2.0)
            detrended = signal[start : start + int(t)]
            if len(detrended) > 1:
                detrended = sig.detrend(detrended)
                sigmafull.append(np.std(detrended))
        y.append(np.average(sigmafull))
    slope = sp.linregress(np.log10(x), np.log10(y))
    alpha = slope[0]
    return alpha


# Function 10: Kappa (see Poil 2011)
def kappa(s):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    [x, y] = getcumdist(s)
    y = 1 - y
    yth = 1 - x ** (-0.7)
    logbins = np.logspace(np.log10(1), np.log10(np.max(s)), 10)
    logbins = np.round(logbins) + 0.5
    for ii in range(len(logbins)):
        logbins[ii] = np.int(find_nearest(x, logbins[ii]))
    logbins = logbins.astype(int)
    kappa = 1 + np.average(yth[logbins] - y[logbins])
    return kappa


# Function 13: Log-binning
def log_binning(bininput, quantityinput, numbins):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    m = 0
    binbounds = np.logspace(
        np.log10(min(bininput)), np.log10(max(bininput)), numbins + 1
    )
    print(binbounds)
    X = np.zeros(numbins)
    Y = np.zeros(numbins)
    E = np.zeros(numbins)
    for i in range(len(binbounds) - 1):
        mask = (bininput > (binbounds[i])) & (bininput < (binbounds[i + 1]))
        Y[i] = np.sum(mask * quantityinput) / np.sum(mask)
        count = np.sum(mask != 0) + 0.1
        if sum(mask) == 0:
            E[i] = float("nan")
        else:
            E[i] = 2.0 * (np.std(quantityinput[mask]) / count)  # confidence interval
        X[i] = np.sqrt(binbounds[i] * binbounds[i + 1])
    return (X, Y, E, m)


def log_binning_2(bininput, quantityinput, numbins, type):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    m = 0
    binbounds = np.logspace(
        np.log10(min(bininput)), np.log10(max(bininput)), numbins + 1
    )
    print(binbounds)
    X = np.zeros(numbins)
    Y = np.zeros(numbins)
    E = np.zeros(numbins)
    for i in range(len(binbounds) - 1):
        mask = (bininput > (binbounds[i])) & (bininput < (binbounds[i + 1]))
        Y[i] = np.sum(mask * quantityinput) / np.sum(mask)
        count = np.sum(mask != 0) + 0.1
        if sum(mask) == 0:
            E[i] = float("nan")
        else:
            if type == "SD":
                E[i] = np.std(quantityinput[mask])
            elif type == "SEM":
                E[i] = sp.sem(quantityinput[mask])
            else:
                E[i] = 1.96 * (
                    np.std(quantityinput[mask]) / np.sqrt(count)
                )  # confidence interval
        X[i] = np.sqrt(binbounds[i] * binbounds[i + 1])
    return (X, Y, E, m)


def log_binning_3(bininput, quantityinput, numbins, type):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    m = 0
    binbounds = np.linspace(min(bininput), max(bininput), numbins + 1)
    print(binbounds)
    X = np.zeros(numbins)
    Y = np.zeros(numbins)
    E = np.zeros(numbins)
    for i in range(len(binbounds) - 1):
        mask = (bininput > (binbounds[i])) & (bininput < (binbounds[i + 1]))
        Y[i] = np.sum(mask * quantityinput) / np.sum(mask)
        count = np.sum(mask != 0) + 0.1
        if sum(mask) == 0:
            E[i] = float("nan")
        else:
            if type == "SD":
                E[i] = np.std(quantityinput[mask])
            elif type == "SEM":
                E[i] = sp.sem(quantityinput[mask])
            else:
                E[i] = 1.96 * (
                    np.std(quantityinput[mask]) / np.sqrt(count)
                )  # confidence interval
        X[i] = binbounds[i] + ((binbounds[i + 1] - binbounds[i]) / 2.0)
    return (X, Y, E, m, binbounds)


def log_binning_nan(bininput, quantityinput, numbins):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    binbounds = np.logspace(np.log10(1), np.log10(max(bininput)), numbins + 1)
    X = np.zeros(numbins)
    Y = np.zeros(numbins)
    E = np.zeros(numbins)
    m = np.sqrt(quantityinput)
    n = np.ones(len(m))
    for i in range(len(binbounds) - 1):
        mask = (bininput > (binbounds[i])) & (bininput < (binbounds[i + 1]))
        count = np.sum(mask != 0) + 0.1
        Y[i] = np.sum(mask * quantityinput) / np.sum(mask)
        m[mask] = m[mask] / Y[i]
        E[i] = 2.0 * (np.std(quantityinput[mask]) / count)
        X[i] = np.sqrt(binbounds[i] * binbounds[i + 1])
    return (X, Y, E, m)


def binning(bininput, quantityinput, numbins):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    binbounds = np.linspace(min(bininput), max(bininput), numbins + 1)
    X = np.zeros(numbins)
    Y = np.zeros(numbins)
    E = np.zeros(numbins)
    m = np.sqrt(quantityinput)
    n = np.ones(len(m))
    for i in range(len(binbounds) - 1):
        mask = (bininput > (binbounds[i])) & (bininput < (binbounds[i + 1]))
        count = np.sum(mask != 0) + 0.1
        Y[i] = np.sum(mask * quantityinput) / np.sum(mask)
        m[mask] = m[mask] / Y[i]
        E[i] = 2.0 * (np.std(quantityinput[mask]) / count)
        X[i] = np.sqrt(binbounds[i] * binbounds[i + 1])
    return (X, Y, E, m)


# Function 14: Bitest
def bitest(starts):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    H = []
    for i in np.arange(2, len(starts) - 2, 1):
        t1 = starts[i + 1] - starts[i]
        t2 = starts[i] - starts[i - 1]
        if t1 < t2:

            t = t1
            tau = starts[i + 2] - starts[i + 1]

        else:
            t = t2
            tau = starts[i - 1] - starts[i - 2]

        if (~np.isnan(t / (t + (tau / 2.0)))) & (~np.isinf(t / (t + (tau / 2.0)))):
            H.append(t / (t + (tau / 2.0)))
    return np.array(H)


# Function 15: Create spiking rate signal from raw time stamps
def spikingrate(spikes, dt):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    spikes = np.unique(spikes)
    time = np.arange(np.min(spikes) - 1, np.max(spikes) + 1, dt)
    fires = np.zeros(len(time))
    binn = 0
    count = 0
    for _i in spikes[:-1]:
        if (_i >= time[binn]) & (_i < time[binn + 1]):
            fires[binn] += 1
        else:
            while _i > time[binn + 1]:
                binn += 1
            fires[binn] += 1
        count += 1
    return time, fires


# Function 16: Powerspec
def powerspec(dt, data):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    n = len(data)
    spec = np.fft.rfft(data, n)
    sp = spec * np.conj(spec) / n
    freq = (1 / (dt * n)) * np.arange(n)
    L = np.arange(
        1,
        np.floor(
            n / 2,
        ),
        dtype="int",
    )
    return [freq[L], sp[L]]


# Function 17: Histogram
def tyler_pdf(starts, binnum, minn, maxx, type, density):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    if type == "log":
        logbins = np.logspace(np.log10(minn), np.log10(maxx), binnum)
    else:
        logbins = np.linspace(minn, maxx, binnum)
    density = np.histogram(starts, bins=logbins, density=density)
    x = density[1][:-1]
    y = density[0]
    return [x, y]


# Function 18: MSD
def msd(shapes, sizes):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    vel = np.zeros(len(sizes))
    for i in range(len(sizes)):

        vel[i] = max(shapes[i])

    return [vel, sizes]


# Function 19: Surface plotter
def surface_plot(matrix, errmat, **kwargs):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    ax.scatter(x, y, matrix + errmat, color="black", marker="+")
    ax.scatter(x, y, matrix - errmat, color="black", marker="+")
    ax.set(facecolor="grey")
    return (fig, ax, surf)


# Function 20: Get cummulative distribution with error bars
def inc_Beta_Fit(x, a, b, target):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    return np.abs(special.betainc(a, b, x) - target)


# Function 21: SHAPES
# Tyler Salners 5/24/19
# This code returns the rate of change of stress over time for an avalanche.
import numpy as np
from matplotlib import pyplot as plt


# This function finds the nearest point in an array to a given value. It takes in array, an array, and value, a float.
# It outputs idx, an int, which is the index of the nearest point.
def find_nearest(array, value):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    near = [abs(i - value) for i in array]
    indnear = near.index(min(near))
    return indnear


# This function bins the shapes based on duration of size. It takes lists durs, avs, shapes, times which are the output of get_slips.
# bins is the centers of the bins to be sorted into, type is a str and either 'size' or 'duration' is what you are binning by, and
# width is the width of the bins. It outputs lists times_sorted, shapes_sorted, durs_sorted, avs_sorted which are those binned and sorted.
def shape_bins(durs, avs, shapes, times, bins, type, width):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    # first sort the arrays
    shapes = np.asarray(shapes)
    times = np.asarray(times)
    if type == "duration":
        ind = np.argsort(durs)
    else:
        ind = np.argsort(avs)
    avs = avs[ind]
    shapes = shapes[ind]
    times = times[ind]
    durs = durs[ind]
    # make return array
    shapes_sorted = []
    times_sorted = []
    durs_sorted = []
    avs_sorted = []
    save = 0
    for i in range(len(bins)):
        if type == "size":
            idxx = find_nearest(avs, bins[i])  # find closest event to bin center
            mask = range(idxx - width, idxx + width)  # take all events within ben width
            print("begin", idxx - width)
            print("end", idxx + width)
            if save > idxx - width:
                print("shit")
            save = idxx + width
        elif type == "duration":
            idxx = find_nearest(durs, bins[i])  # find closest event to bin center
            mask = range(idxx - width, idxx + width)  # take all events within ben width
            print("begin", idxx - width)
            print("end", idxx + width)

        shapes_sorted.append(shapes[mask])
        times_sorted.append(times[mask])
        durs_sorted.append(durs[mask])
        avs_sorted.append(avs[mask])

    return [
        np.array(times_sorted),
        np.array(shapes_sorted),
        np.array(durs_sorted),
        np.array(avs_sorted),
    ]


# This averages over sizes. It takes lists shapes,times,avs,durs the output of get_slips. It outputs lists times_final,
# shapes_final, and err_final the time, velocity, and error vectors respectively.


def size_avg(shapes, times, avs, durs):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    shapes_final = []
    err_final = []
    times_final = []
    for i in range(len(shapes)):  # for each bin
        span = len(shapes[i])  # number of shapes
        lenind = np.argmax([len(j) for j in shapes[i]])
        print("lkjsd", lenind)
        length = len(times[i][lenind])
        avg_shape = np.zeros(length)
        avg_err = np.zeros(length)
        sort_shapes = np.zeros((np.size(shapes[i]), length))
        for k in range(np.size(shapes[i])):
            sort_shapes[k][0 : np.size(shapes[i][k])] = shapes[i][
                k
            ]  # collect shapes, padded at end with 0 if no data
        for k in range(length):
            avg_shape[k] = sum(sort_shapes[:, k]) / span  # average of shapes
            avg_err[k] = np.std(sort_shapes[:, k]) / np.sqrt(span)  # error
        shapes_final.append(avg_shape)
        err_final.append(avg_err)
        times_final.append(times[i][lenind])
    return [np.array(times_final), np.array(shapes_final), np.array(err_final)]


# This averages over durations. It takes lists shapes,times,avs,durs the output of get_slips. It outputs lists times_final,
# shapes_final, and err_final the time, velocity, and error vectors respectively.
def duration_avg(shapes, times, avs, durs):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    shapes_final = []
    err_final = []
    times_final = []
    for i in range(len(shapes)):  # for each bin
        length = np.size(shapes[i][0])  # length of time trace
        for k in range(np.size(shapes[i])):
            [times[i][k], shapes[i][k]] = resize(
                shapes[i][k], times[i][k], length
            )  # conform to length
        avg_shape = np.zeros(length)
        avg_err = np.zeros(length)
        sort_shapes = np.zeros((np.size(shapes[i]), length))
        span = np.size(shapes[i])  # number of shapes

        for k in range(np.size(shapes[i])):

            sort_shapes[k] = shapes[i][k]

        for k in range(length):
            avg_shape[k] = np.true_divide(np.sum(sort_shapes[:, k]), span)  # average
            avg_err[k] = np.true_divide(
                np.std(sort_shapes[:, k]), np.sqrt(span)
            )  # error

        shapes_final.append(avg_shape)
        err_final.append(avg_err)
        times_final.append(times[i][0])
    return (times_final, shapes_final, err_final)


# This code takes a vector and resizes it to a desired length. It takes a list vector and an associate list time. length is and int and
# is the desired len of the result. It outputs two lists points, the normalized time vector, and new, the new vector.
def resize(vector, time, length):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    time = np.asarray(time)
    vector = np.asarray(vector)
    time = time - time[0]  # so you allways start at 0
    time = time.astype(float)
    time = np.true_divide(time, time[-1])  # normalize
    new = np.zeros(length)
    points = np.linspace(0, 1, num=length)
    width2 = 1.0 / length  # step of points

    for i in range(length):
        # if i == 0:
        #     continue
        mask = (time >= (points[i] - (width2 / 2))) & (
            time <= (points[i] + (width2 / 2))
        )  # all parts of time within the range of a point
        new[i] = np.mean(vector[mask])
        if np.isnan(new[i]):
            new[i] = new[i - 1]  # remove nans
    return [points, new]


def Extract(lst, i):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    return [item[i] for item in lst]


def Scaling_relation(smin, s, dmin, d):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    test_s = s[s > smin]
    resultss = powerlaw.Fit(test_s, xmin=smin)
    print("smin", resultss.xmin)
    test_d = d[d > dmin]
    resultsd = powerlaw.Fit(test_d, xmin=dmin)
    print("dmin", resultsd.xmin)
    test_s = s[(s > smin) & (d > dmin)]
    test_d = d[(s > smin) & (d > dmin)]
    [x, y, e, m] = log_binning(test_s, test_d, 40)
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    signuz, intercept, r_value, p_value, std_err = sp.linregress(
        np.log10(x), np.log10(y)
    )
    tau = resultss.power_law.alpha
    tau_e = resultss.power_law.sigma
    alpha = resultsd.power_law.alpha
    alpha_e = resultsd.power_law.sigma
    x = 1.0 / signuz
    xerr = std_err
    y = (alpha - 1) / (tau - 1)
    yerr = (alpha / tau) * np.sqrt((alpha_e / alpha) ** 2 + (tau_e / tau) ** 2)
    return [x, y, xerr, yerr, tau, alpha, signuz, tau_e, alpha_e, std_err]


import math


def get_S_vs_Ninh(gcc):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    S = []
    N = []
    D = []
    tstart = []
    for i in range(0, len(gcc)):
        neurons = Extract(list(dict.fromkeys(gcc[i])), 0)
        times = Extract(list(dict.fromkeys(gcc[i])), 1)
        S.append(len(neurons))
        N.append(len(np.unique(neurons)))
        D.append(np.subtract(np.max(times), np.min(times)))
        tstart.append(np.min(times))
    S = np.array(S)
    N = np.array(N)
    D = np.array(D)
    tstart = np.array(tstart)
    return S, N, D, tstart


def get_S_vs_Ninh(gcc):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    S = []
    N = []
    D = []
    firingtimes = []
    totalspikes = []
    totalspikepairs = []
    tstart = []
    print("um here?", len(gcc))
    for i in range(0, len(gcc)):
        S.append(gcc[i].number_of_nodes())
        # N.append(ncc[i].number_of_nodes())
        spikes = []
        times = []
        temp = list(gcc[i].edges)
        pairs = pd.unique(Extract(temp, 0) + Extract(temp, 1))
        spikes = np.array(Extract(pairs, 0))
        times = np.array(Extract(pairs, 1))
        # for j in range(len(temp)):
        #     times.append(temp[j][0][1])
        #     times.append(temp[j][1][1])
        #     spikes.append(temp[j][0])
        #     spikes.append(temp[j][1])
        D.append(np.max(times) - np.min(times))
        tstart.append(np.min(times))
        firingtimes.append(np.sort(times))
        N.append(len(pd.unique(spikes)))
        totalspikes.append([spikes])
        totalspikepairs.append(temp)
    S = np.array(S)
    N = np.array(N)
    D = np.array(D)
    tstart = np.array(tstart)
    firingtimes = np.array(firingtimes)
    return S, N, D, tstart, totalspikes, totalspikepairs, firingtimes


def get_S_vs_Ninh_2species(gcc, inspec, exspec, unspec):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    Sinh = np.zeros(len(gcc))
    Ninh = np.zeros(len(gcc))
    Sexc = np.zeros(len(gcc))
    Nexc = np.zeros(len(gcc))
    D = np.zeros(len(gcc))
    Finh = []
    Tinh = []
    Fexc = []
    Texc = []
    for i in range(0, len(gcc)):
        temp = list(gcc[i].edges)
        pairs = pd.unique(Extract(temp, 0) + Extract(temp, 1))
        neurons = np.array(Extract(pairs, 0))
        times = np.array(Extract(pairs, 1)) * 1e-3
        excneuron = neurons[np.isin(neurons, exspec)]
        inhneuron = neurons[np.isin(neurons, inspec)]
        exctimes = times[np.isin(neurons, exspec)]
        inhtimes = times[np.isin(neurons, inspec)]
        Sinh[i] = len(inhneuron)
        Ninh[i] = len(pd.unique(inhneuron))
        Sexc[i] = len(excneuron)
        Nexc[i] = len(pd.unique(excneuron))
        D[i] = np.max(times) - np.min(times)
        Tinh.append(inhtimes.tolist())
        Texc.append(exctimes.tolist())
        Finh.append(inhneuron.tolist())
        Fexc.append(excneuron.tolist())
    # Sinh = np.array(Sinh)
    # Ninh = np.array(Ninh)
    # Sexc = np.array(Sexc)
    # Nexc = np.array(Nexc)
    # Tinh = np.array(Tinh)
    # Texc = np.array(Texc)
    # Finh = np.array(Finh)
    # Fexc = np.array(Fexc)
    # D = np.array(D)
    return D, Sinh, Ninh, Sexc, Nexc, Finh, Fexc, Tinh, Texc


def get_S_vs_Ninh_lite(gcc):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    D = np.zeros(len(gcc))
    ft = []
    fn = []
    S = np.zeros(len(gcc))
    N = np.zeros(len(gcc))
    for i in range(0, len(gcc)):
        temp = list(gcc[i].edges)
        pairs = pd.unique(Extract(temp, 0) + Extract(temp, 1))
        tempneurons = np.array(Extract(pairs, 0))
        temptimes = np.array(Extract(pairs, 1))

        # temptimes=(np.multiply(Extract(Extract(temp,0),1)+Extract(Extract(temp,1),1),1E-3)).tolist()
        # tempneurons=Extract(Extract(temp,0),0)+Extract(Extract(temp,1),0)
        tempneurons = np.array(tempneurons)
        temptimes = np.array(temptimes)

        tempneurons = tempneurons[np.argsort(temptimes)]
        temptimes = temptimes[np.argsort(temptimes)]
        tempneurons.tolist()
        temptimes.tolist()

        S[i] = len(tempneurons)
        fn.append(tempneurons)
        tempneurons = np.unique(tempneurons)
        D[i] = np.max(temptimes) - np.min(temptimes)  # durations
        ft.append(temptimes)
        N[i] = len(tempneurons)
    D = np.array(D)
    ft = np.array(ft)
    fn = np.array(fn)
    S = np.array(S)
    N = np.array(N)
    return ft, fn, S, D, N


def T_exp_bootstrap(samplemean, df, N):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    bootstrapsample = np.random.exponential(scale=samplemean, size=(df, N))
    sample_means = np.mean(bootstrapsample, axis=0)
    mean = np.mean(sample_means)
    std = np.std(sample_means)
    return [mean, std]


def Find_x0_y0(nsamp, isif):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    ## calculating moment ratios and errors
    isi = (isif,)
    bootstrap_ci = sp.bootstrap(
        isi,
        np.average,
        confidence_level=0.95,
        random_state=1,
        method="percentile",
        n_resamples=nsamp,
    )
    mu1 = np.average(isi)
    sig1 = bootstrap_ci.standard_error
    sigg1 = np.std(isi)

    isi = (isif**2,)
    bootstrap_ci = sp.bootstrap(
        isi,
        np.average,
        confidence_level=0.95,
        random_state=1,
        method="percentile",
        n_resamples=nsamp,
    )
    mu2 = np.average(isi)
    sig2 = bootstrap_ci.standard_error
    sigg2 = np.std(isi)

    isi = (isif**3,)
    bootstrap_ci = sp.bootstrap(
        isi,
        np.average,
        confidence_level=0.95,
        random_state=1,
        method="percentile",
        n_resamples=nsamp,
    )
    mu3 = np.average(isi)
    sig3 = bootstrap_ci.standard_error
    sigg3 = np.std(isi)

    isi = (isif**4,)
    bootstrap_ci = sp.bootstrap(
        isi,
        np.average,
        confidence_level=0.95,
        random_state=1,
        method="percentile",
        n_resamples=nsamp,
    )
    mu4 = np.average(isi)
    sig4 = bootstrap_ci.standard_error
    sigg4 = np.std(isi)

    isi = isif
    mutop = mu3
    mubot = mu1**3
    sigbot = 3 * sig1 * mubot / mu1
    sigbott = 3 * sigg1 * mubot / mu1
    sigtop = sig3
    sigtopp = sigg3
    xerr = (mutop / mubot) * np.sqrt((sigtop / mutop) ** 2 + (sigbot / mubot) ** 2)
    xer = (mutop / mubot) * np.sqrt((sigtopp / mutop) ** 2 + (sigbott / mubot) ** 2)
    mutop = mu4
    mubot = mu2**2
    sigbot = 2 * sig2 * mubot / mu2
    sigbott = 2 * sigg2 * mubot / mu2
    sigtop = sig4
    sigtopp = sigg4
    yerr = (mutop / mubot) * np.sqrt((sigtop / mutop) ** 2 + (sigbot / mubot) ** 2)
    x0 = (np.average(isi**3) / (np.average(isi) ** 3)) - 6
    y0 = (np.average(isi**4) / (np.average(isi**2) ** 2)) - 6
    yer = (mutop / mubot) * np.sqrt((sigtopp / mutop) ** 2 + (sigbott / mubot) ** 2)

    # #### optimization to find r and g
    # rmin=0.001
    # rmax=0.99
    # gmin=0.0005
    # gmax=2
    # bounds = ((rmin,rmax), (gmin,gmax))
    # os.chdir('/Users/tylersalners/Desktop/RESEARCH/gelson/sims')
    # vals=np.load('vals_3.npy')
    # indfinder=np.zeros(len(vals))
    # for kk in range(len(vals)):
    #     indfinder[kk]=(x0-vals[kk][1][0])**2+(y0-vals[kk][1][1])**2
    # result = minimize(run_sim, [vals[np.argmin(indfinder)][0][0], vals[np.argmin(indfinder)][0][1]],args=(x0,y0), bounds=bounds,jac=gradient_respecting_bounds(bounds, run_sim))
    # rs=result.x[0]
    # gs=result.x[1]
    return [x0, y0, xerr, yerr, xer, yer]


def Find_r_g(nsamp, isif):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    ## calculating moment ratios and errors
    isi = (isif,)
    bootstrap_ci = sp.bootstrap(
        isi,
        np.average,
        confidence_level=0.95,
        random_state=1,
        method="percentile",
        n_resamples=nsamp,
    )
    print("lkj")

    mu1 = np.average(isi)
    sig1 = bootstrap_ci.standard_error

    isi = (isif**2,)
    bootstrap_ci = sp.bootstrap(
        isi,
        np.average,
        confidence_level=0.95,
        random_state=1,
        method="percentile",
        n_resamples=nsamp,
    )
    mu2 = np.average(isi)
    sig2 = bootstrap_ci.standard_error

    isi = (isif**3,)
    bootstrap_ci = sp.bootstrap(
        isi,
        np.average,
        confidence_level=0.95,
        random_state=1,
        method="percentile",
        n_resamples=nsamp,
    )
    mu3 = np.average(isi)
    sig3 = bootstrap_ci.standard_error

    isi = (isif**4,)
    bootstrap_ci = sp.bootstrap(
        isi,
        np.average,
        confidence_level=0.95,
        random_state=1,
        method="percentile",
        n_resamples=nsamp,
    )
    mu4 = np.average(isi)
    sig4 = bootstrap_ci.standard_error

    isi = isif
    mutop = mu3
    mubot = mu1**3
    sigbot = 3 * sig1 * mubot / mu1
    sigtop = sig3
    xerr = (mutop / mubot) * np.sqrt((sigtop / mutop) ** 2 + (sigbot / mubot) ** 2)
    mutop = mu4
    mubot = mu2**2
    sigbot = 2 * sig2 * mubot / mu2
    sigtop = sig4
    yerr = (mutop / mubot) * np.sqrt((sigtop / mutop) ** 2 + (sigbot / mubot) ** 2)
    x0 = (np.average(isi**3) / (np.average(isi) ** 3)) - 6
    y0 = (np.average(isi**4) / (np.average(isi**2) ** 2)) - 6
    #### optimization to find r and g
    rmin = 0.001
    rmax = 0.99
    gmin = 0.0005
    gmax = 2
    bounds = ((rmin, rmax), (gmin, gmax))
    os.chdir("/Users/tylersalners/Desktop/RESEARCH/gelson/sims")
    vals = np.load("vals_3.npy")
    indfinder = np.zeros(len(vals))
    for kk in range(len(vals)):
        indfinder[kk] = (x0 - vals[kk][1][0]) ** 2 + (y0 - vals[kk][1][1]) ** 2
    result = minimize(
        run_sim,
        [vals[np.argmin(indfinder)][0][0], vals[np.argmin(indfinder)][0][1]],
        args=(x0, y0),
        bounds=bounds,
        jac=gradient_respecting_bounds(bounds, run_sim),
    )
    rs = result.x[0]
    gs = result.x[1]
    return [x0, y0, xerr, yerr]


def Find_r_g_special(nsamp, isif):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    ## calculating moment ratios and errors
    isi = (isif,)
    bootstrap_ci = sp.bootstrap(
        isi,
        np.average,
        confidence_level=0.95,
        random_state=1,
        method="percentile",
        n_resamples=nsamp,
    )

    mu1 = np.average(isi)
    sig1 = bootstrap_ci.standard_error

    isi = (isif**2,)
    bootstrap_ci = sp.bootstrap(
        isi,
        np.average,
        confidence_level=0.95,
        random_state=1,
        method="percentile",
        n_resamples=nsamp,
    )
    mu2 = np.average(isi)
    sig2 = bootstrap_ci.standard_error

    isi = (isif**3,)
    bootstrap_ci = sp.bootstrap(
        isi,
        np.average,
        confidence_level=0.95,
        random_state=1,
        method="percentile",
        n_resamples=nsamp,
    )
    mu3 = np.average(isi)
    sig3 = bootstrap_ci.standard_error

    isi = (isif**4,)
    bootstrap_ci = sp.bootstrap(
        isi,
        np.average,
        confidence_level=0.95,
        random_state=1,
        method="percentile",
        n_resamples=nsamp,
    )
    mu4 = np.average(isi)
    sig4 = bootstrap_ci.standard_error

    isi = isif
    mutop = mu3
    mubot = mu1**3
    sigbot = 3 * sig1 * mubot / mu1
    sigtop = sig3
    xerr = (mutop / mubot) * np.sqrt((sigtop / mutop) ** 2 + (sigbot / mubot) ** 2)
    mutop = mu4
    mubot = mu2**2
    sigbot = 2 * sig2 * mubot / mu2
    sigtop = sig4
    yerr = (mutop / mubot) * np.sqrt((sigtop / mutop) ** 2 + (sigbot / mubot) ** 2)
    x0 = (np.average(isi**3) / (np.average(isi) ** 3)) - 6
    y0 = (np.average(isi**4) / (np.average(isi**2) ** 2)) - 6
    #### optimization to find r and g
    rmin = 0.001
    rmax = 0.99
    gmin = 0.0005
    gmax = 2
    bounds = ((rmin, rmax), (gmin, gmax))
    os.chdir("/Users/tylersalners/Desktop/RESEARCH/gelson/sims")
    vals = np.load("vals_3.npy")
    indfinder = np.zeros(len(vals))
    for kk in range(len(vals)):
        indfinder[kk] = (x0 - vals[kk][1][0]) ** 2 + (y0 - vals[kk][1][1]) ** 2
    result = minimize(
        run_sim,
        [vals[np.argmin(indfinder)][0][0], vals[np.argmin(indfinder)][0][1]],
        args=(x0, y0),
        bounds=bounds,
        jac=gradient_respecting_bounds(bounds, run_sim),
    )
    rs = result.x[0]
    gs = result.x[1]
    return [x0, y0, xerr, yerr, rs, gs, rerr, gerr]


def prune_beggs(experiment, td):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    os.chdir("/Users/tylersalners/Desktop/beggs/data/causal_web_pkl")
    filename = "c-pairs_2013-01-%s-000_d-lt-20_rt0.5.pkl" % (experiment)
    f = open(filename, "rb")
    web = pickle.load(f, encoding="bytes")
    # web=web[:10000]
    f.close()
    # iweb = [((n0[0], int(n0[1])), (n1[0], int(n1[1]))) for n0, n1 in web if int(n1[1])-int(n0[1])<td]#n0[0]/n1[0] is neuron i/j,n0[1]/n1[1] is time i,j
    gcc = load_causal_webs(web, td)
    [size, number, duration, tstart, total_spikes, total_spike_pairs] = get_S_vs_Ninh(
        list(gcc)
    )
    return total_spikes


def splitter(t, sig, n):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    l = np.linspace(np.min(t), np.max(t), n)
    full_sig = []
    full_t = []
    for i in range(len(l) - 1):
        full_sig.append(sig[(t >= l[i]) & (t < l[i + 1])])
        full_t.append(t[(t >= l[i]) & (t < l[i + 1])])
    return [full_t, full_sig, l]


def tyler_specgram(t, sig, n):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    f = []
    s = []
    [split_t, split_sig, t_centers] = splitter(t, sig, n)
    for i in range(len(split_t)):
        [freq, spec] = powerspec(t[1] - t[0], split_sig[i])
        f.append(freq)
        s.append(spec)
    return [f, s, t_centers]


def moving_average(a, n=3):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def surface_plot(matrix, **kwargs):
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.scatter(x, y, matrix, **kwargs)
    ax.set(facecolor="grey")
    return (fig, ax, surf)
