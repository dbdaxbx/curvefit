#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:24:03 2018

@author: ducklin
"""

import numpy as np, bisect as bi
from collections import Iterable

class Curve(object):

    def idxt(self, x):
        idx = bi.bisect(self.x, x) - 1
        idx = max(idx, 0)
        idx = min(idx, len(self.x) - 2)
        t = 1.0 * (x - self.x[idx]) / (self.x[idx + 1] - self.x[idx])
        return (
         idx, t)

    def __call__(self, xs):
        return self.value(xs)

    def value(self, xs):
        if isinstance(xs, Iterable):
            return np.array([self.interp1(x) for x in xs])
        else:
            return self.interp1(xs)

    def integral(self, xs):
        if isinstance(xs, Iterable):
            return np.array([self.integral1(x) for x in xs])
        else:
            return self.integral1(xs)

    def deriv(self, xs):
        if isinstance(xs, Iterable):
            return np.array([self.deriv1(x) for x in xs])
        else:
            return self.deriv1(xs)

    def addKnot(self, x, y, dx=0.0001):
        eidx = np.abs(self.x - x).argmin()
        if abs(x - self.x[eidx]) < dx:
            self.y[eidx] = y
        else:
            eidx = bi.bisect(self.x, x)
            self.x = np.insert(self.x, eidx, x)
            self.y = np.insert(self.y, eidx, y)
        self.build(self.x, self.y)


class PiecewiseLinear(Curve):

    def build(self, x, y):
        self.x = np.copy(x)
        self.y = np.copy(y)
        dx = np.diff(x)
        intg = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
        self.intg = np.insert(intg, 0, 0)

    def interp1(self, x):
        idx, t = self.idxt(x)
        return self.y[idx] + t * (self.y[idx + 1] - self.y[idx])

    def integral1(self, x):
        idx, _ = self.idxt(x)
        return self.intg[idx] + 0.5 * (x - self.x[idx]) * (self.y[idx] + self.interp1(x))

    def deriv1(self, x):
        idx, _ = self.idxt(x)
        return (self.y[idx + 1] - self.y[idx]) / (self.x[idx + 1] - self.x[idx])


class PiecewiseFlat(Curve):

    def build(self, x, y):
        self.x = np.copy(x)
        self.y = np.copy(y)
        dx = np.diff(x)
        intg = np.cumsum(y[1:] * dx)
        self.intg = np.insert(intg, 0, 0)

    def interp1(self, x):
        eidx = min(bi.bisect_right(self.x, x), len(self.x) - 1)
        return self.y[eidx]

    def integral1(self, x):
        eidx = bi.bisect_left(self.x, x)
        return self.intg[eidx] + (x - self.x[eidx]) * self.y[eidx]


class Quadratic(Curve):

    def build(self, x, y):
        self.x = x
        self.y = y

    def interp1(self, x):
        idx, t = self.idxt(x)
        a = idx if idx % 2 == 0 else idx - 1
        if a > len(self.x) - 2:
            a -= 2
        m, b = a + 1, a + 2
        return self.y[a] * (x - self.x[m]) * (x - self.x[b]) / (self.x[a] - self.x[m]) / (self.x[a] - self.x[b]) + self.y[m] * (x - self.x[a]) * (x - self.x[b]) / (self.x[m] - self.x[a]) / (self.x[m] - self.x[b]) + self.y[b] * (x - self.x[a]) * (x - self.x[m]) / (self.x[b] - self.x[a]) / (self.x[b] - self.x[m])


class RationalTension(Curve):

    def __init__(self, lbd):
        self.lbd = float(lbd)

    def build(self, x, y):
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.a, self.b = RationalTension.buildMatrix(x, y, self.lbd)
        cs = np.linalg.solve(self.a, self.b)
        self.coefs = np.reshape(cs, [len(x) - 1, 4])
        dx = np.diff(x)
        intg = np.cumsum(dx * np.array([RationalTension.integrate(c, 1.0, self.lbd) for c in self.coefs]))
        self.intg = np.insert(intg, 0, 0)

    def interp1(self, x):
        idx, t = self.idxt(x)
        v = np.array([1.0, t, t * t, t * t * t])
        return self.coefs[idx].dot(v.T) / (self.lbd * t * (1.0 - t) + 1.0)

    def deriv1(self, x):
        idx, t = self.idxt(x)
        d = self.lbd * t * (t - 1.0) - 1.0
        t1 = self.lbd * (2.0 * t - 1) * self.coefs[idx].dot(np.array([1, t, t * t, t * t * t]))
        t2 = d * self.coefs[idx].dot(np.array([0, 1, 2 * t, 3 * t * t]))
        return (t1 - t2) / d / d / (self.x[idx + 1] - self.x[idx])

    def integral1(self, x):
        idx, t = self.idxt(x)
        return self.intg[idx] + RationalTension.integrate(self.coefs[idx], t, self.lbd) * (self.x[idx + 1] - self.x[idx])

    @staticmethod
    def __coefs(i):
        return np.arange(4) + (i - 1) * 4

    @staticmethod
    def t0(lbd, dx):
        lbd = float(lbd)
        a = np.array([[1, 0, 0, 0], [-lbd, 1, 0, 0], [2 * lbd * lbd + 2 * lbd, -2 * lbd, 2, 0]])
        a[1] /= dx
        a[2] /= dx * dx
        return a

    @staticmethod
    def t1(lbd, dx):
        lbd = float(lbd)
        a = np.array([[1, 1, 1, 1], [lbd, lbd + 1, lbd + 2, lbd + 3],
         [
          2 * lbd * lbd + 2 * lbd, 2 * lbd * lbd + 4 * lbd, 2 * lbd * lbd + 6 * lbd + 2, 2 * lbd * lbd + 8 * lbd + 6]])
        a[1] /= dx
        a[2] /= dx * dx
        return a

    @staticmethod
    def integrate(coefs, t, lbd):
        a, b, c, d = coefs
        lbd = float(lbd)
        if lbd < 1e-06:
            return a * t + 0.5 * b * t * t + 0.3333333333333333 * c * t ** 3 + 0.25 * d * t ** 4
        else:
            c0 = np.sqrt((lbd + 4.0) * lbd)
            s0 = -c0 * d * t * t - 2 * c0 * t * (c + d)
            s1 = np.log((np.sqrt(lbd + 4.0) + np.sqrt(lbd)) / (np.sqrt(lbd + 4.0) + np.sqrt(lbd) * (1.0 - 2.0 * t)))
            s2 = np.log((np.sqrt(lbd + 4.0) - np.sqrt(lbd)) / (np.sqrt(lbd + 4.0) - np.sqrt(lbd) * (1.0 - 2.0 * t)))
            c1 = d * c0 / lbd + c0 * (b + c + d)
            c2 = 2 * c + 3 * d + lbd * (2 * a + b + c + d)
            f = 0.5 / c0 / lbd
            return f * (s0 + s1 * (c1 + c2) + s2 * (c1 - c2))

    @staticmethod
    def buildMatrix(x, y, lbd):
        if not len(x) == len(y):
            raise AssertionError('input data size mismatch')
        lbd = float(lbd)
        n = len(y)
        a = np.zeros([4 * n - 4, 4 * n - 4])
        b = np.zeros(4 * n - 4)
        t0c = RationalTension.t0(lbd, x[1] - x[0])
        coef = RationalTension._RationalTension__coefs(1)
        a[(0, coef)] = t0c[0]
        b[0] = y[0]
        a[(1, coef)] = t0c[2]
        roff = 2
        for i in range(1, n - 1):
            coef0 = RationalTension._RationalTension__coefs(i)
            coef1 = RationalTension._RationalTension__coefs(i + 1)
            t0c = RationalTension.t0(lbd, x[i + 1] - x[i])
            t1c = RationalTension.t1(lbd, x[i] - x[i - 1])
            for j in range(3):
                a[(roff + j, coef0)] = t1c[j]
                a[(roff + j, coef1)] = -1.0 * t0c[j]

            a[(roff + 3, coef1)] = t0c[0]
            b[roff + 3] = y[i]
            roff += 4

        t1c = RationalTension.t1(lbd, x[n - 1] - x[n - 2])
        coef = RationalTension._RationalTension__coefs(n - 1)
        a[(roff, coef)] = t1c[0]
        b[roff] = y[n - 1]
        a[(roff + 1, coef)] = t1c[2]
        return (
         a, b)
