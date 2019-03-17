__author__ = 'antonio franco'

'''
Copyright (C) 2019  Antonio Franco (antonio_franco@live.it)
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from scipy import optimize
import random


class BivariateSampler(object):
    """
    Class to generate samples from a bivariate distribution
    """

    def __init__(self, my_seed=None):
        """
        :param my_seed (int): custom seed for reproducibility
        """
        random.seed(my_seed)

    def invert_fun(self, fun, a):
        """
        Calculates the inverted function fun in point a and returns the value in F.
        :param fun (lambda: x): function to invert (one argument)
        :param a (float): value in which the inverted function must be calculated
        :return (float): the value of the inverted function F in a
        """
        zfun = lambda t: np.abs(fun(t) - a)

        bnds = [(0, np.inf)]

        X0 = random.random()

        X = optimize.minimize(zfun, np.asarray(X0), bounds=bnds)

        return X.x[0]

    def generate_sample(self, fun):
        """
        Generates a sample from the distribution with CDF fun
        :param fun (lambda: x): CDF of the distribution to draw a sample from
        :return: a sample from the distribution with CDF fun
        """
        return self.invert_fun(fun, random.random())

    def generate_sample_bivariate(self, jointCDF, marginalCDF=None):
        """
        Generates a bivariate sample from the joint CDF jointCDF(x,y) and the marginal CDF  marginalCDF(x).
        :param jointCDF (lambda x,y:): joint CDF function for the rvs X and Y
        :param marginalCDF (lambda x:): marginal CDF for the rv X - the first argument of the joint CDF. If not provided lim y->inf jointCDF(x,y) is used.
        :return (x, y): a bidimensional sample from the joint distribution
        """
        marginal = lambda x: jointCDF(x, 1e5)
        if not (marginalCDF is None):
            marginal = marginalCDF

        X = self.generate_sample(marginal)

        conditioned = lambda y: jointCDF(X, y) / marginal(X)

        return (X, self.generate_sample(conditioned))


import numpy as np
from copy import deepcopy


class BivariateExponentialType3(object):
    """
    Represents a bivariate exponential of type 3 with parameter m.
    """

    def __init__(self, m, sampler=None):
        """
        :param m (positive int): parameter m
        :param sampler: optional sampler
        """
        assert(m > 0)
        self.m = float(m)
        if not (sampler is None):
            self.sampler = deepcopy(sampler)
        else:
            self.sampler = BivariateSampler()

    def joint_CDF(self, x, y):
        """
        Returns the value of the joint CDF in x,y
        :param x (float): x coordinate
        :param y (float): y coordinate
        :return (float): the value of the joint CDF in x,y
        """
        if x > 0 and y > 0:
            F = (1.0 - np.exp(-x) - np.exp(-y) + np.exp(-(x ** self.m + y ** self.m) ** (1.0 / self.m))) \
                * np.heaviside(x, 0) * np.heaviside(y, 0)
        else:
            F = 0.0
        return float(F)

    def joint_PDF(self, x, y):
        """
        Returns the value of the joint PDF in x,y
        :param x (float): x coordinate
        :param y (float): y coordinate
        :return (float): the value of the joint PDF in x,y
        """
        if x > 0 and y > 0:
            f = (x ** self.m + y ** self.m) ** (-2.0 + 1.0 / self.m) * x ** (self.m - 1.0) * y ** (self.m - 1.0) * \
                ((x ** self.m + y ** self.m) ** (1.0 / self.m) + self.m - 1.0) * np.exp(
                -(x ** self.m + y ** self.m) ** (1.0 / self.m)) \
                * np.heaviside(x, 0) * np.heaviside(y, 0)
        else:
            f = 0
        return float(f)

    def marginal_CDF_X(self, x):
        """
        Returns the value of the marginal CDF of X in x
        :param x (float): value in which to calculate
        :return (float): the value of the marginal CDF of X in x
        """
        F = (1.0 - np.exp(-x)) * np.heaviside(x, 0)
        return float(F)

    def marginal_CDF_Y(self, y):
        """
        Returns the value of the marginal CDF of Y in y
        :param y (float): value in which to calculate
        :return (float): the value of the marginal CDF of Y in y
        """
        F = (1.0 - np.exp(-y)) * np.heaviside(y, 0)
        return float(F)

    def generate_sample(self):
        """
        Generates a random sample from the bivariate exponential type 3
        :return (float): random sample from the bivariate exponential type 3
        """
        joint_fun = lambda x, y: self.joint_CDF(x, y)
        marg_fun = lambda x: self.marginal_CDF_X(x)
        return self.sampler.generate_sample_bivariate(joint_fun, marg_fun)


# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Test with a bivariate exponential type 3
    B = BivariateSampler(22)  # For reproducibility
    m = 2
    F = BivariateExponentialType3(m, B)

    n_samples = int(1e3)

    x = np.zeros(n_samples)
    y = np.zeros(n_samples)

    # Generating samples
    for i in range(0, n_samples):
        (tx, ty) = F.generate_sample()
        x[i] = tx
        y[i] = ty

    # Creating a normalized histogram and adjusting for xy coordinates
    hist, xedges, yedges = np.histogram2d(x, y, density=True)
    xx = xedges[0:-1] + np.diff(xedges) / 2.0
    yy = yedges[0:-1] + np.diff(yedges) / 2.0
    X, Y = np.meshgrid(xx, yy)

    # Calculating the true joint PDF with 100 points in the histogram range
    fun = np.vectorize(F.joint_PDF)
    n_points = 100
    fx = np.linspace(np.min(X.ravel()), np.max(X.ravel()), n_points)
    fy = np.linspace(np.min(Y.ravel()), np.max(Y.ravel()), n_points)
    fX, fY = np.meshgrid(fx, fy)
    f = fun(fX, fY)

    #Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X.ravel(), Y.ravel(), hist.ravel(), c='r', marker='x', label='sampling')
    ax.plot_wireframe(fX, fY, f, label='true')
    ax.legend()

    plt.show()
