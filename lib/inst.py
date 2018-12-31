#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:24:04 2018

@author: ducklin
"""

import numpy as np, scipy.optimize as opt
from scipy.stats import norm
import lin

class FlatCurve(object):

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, t):
        return np.exp(-self.rate * t)


class ZeroCouponBond(object):
    """ A simple zero coupon bond"""

    def __init__(self, maturity):
        self.maturity = maturity

    def pv(self, disc, p):
        return disc(self.maturity)


class BlackScholes(object):
    """basic black scholes formulae"""

    @staticmethod
    def callPrice(r, S0, K, T, vol):
        f = 1.0 / np.sqrt(T) / vol
        m = np.log(S0 / K)
        d1 = f * (m + (r + 0.5 * vol ** 2) * T)
        d2 = f * (m + (r - 0.5 * vol ** 2) * T)
        disc = np.exp(-r * T)
        return norm.cdf(d1) * S0 - norm.cdf(d2) * K * disc

    @staticmethod
    def putPrice(r, S0, K, T, vol):
        f = 1.0 / np.sqrt(T) / vol
        m = np.log(S0 / K)
        d1 = f * (m + (r + 0.5 * vol ** 2) * T)
        d2 = f * (m + (r - 0.5 * vol ** 2) * T)
        disc = np.exp(-r * T)
        return -norm.cdf(-d1) * S0 + norm.cdf(-d2) * K * disc

    @staticmethod
    def callDelta(r, S0, K, T, vol):
        f = 1.0 / np.sqrt(T) / vol
        m = np.log(S0 / K)
        d1 = f * (m + (r + 0.5 * vol ** 2) * T)
        return norm.cdf(d1)

    @staticmethod
    def compImpliedVolFromCall(r, S0, K, T, c):
        try:
            return opt.brentq(lambda x: BlackScholes.callPrice(r, S0, K, T, x) - c, 0.0001, 5)
        except ValueError:
            return 0


class CDS(object):
    """ A simple CDS class, 
        we ignore the daycount convention and business day rules
    """

    def __init__(self, maturity, coupon, recovery=0.4, accruedtime=0.0, dt=0.25):
        if not np.allclose(4.0 * maturity, int(4.0 * maturity)):
            raise AssertionError('maturity has to be on even quarters')
        self.maturity = maturity
        self.coupon = coupon
        self.recovery = recovery
        self.accruedtime = accruedtime
        self.dt = dt

    def dates(self, inc0=False):
        fd = np.arange(self.dt, self.maturity + self.dt - 0.0001, self.dt) - self.accruedtime
        if inc0:
            return np.insert(fd, 0, 0)
        else:
            return fd

    def coupons(self):
        return np.diff(self.dates(True)) * self.coupon

    def pv01(self, disc, p):
        return np.sum(disc(self.dates()) * p(self.dates())) * self.dt

    def accrued(self):
        return self.accruedtime * self.coupon

    def protPV(self, disc, p):
        dp = -np.diff(p(self.dates(True)))
        df = disc(self.dates(True))
        return np.sum(dp * 0.5 * (df[1:] + df[:-1])) * (1 - self.recovery)

    def par(self, disc, p):
        return abs(self.protPV(disc, p) / self.pv01(disc, p))

    def pv(self, disc, p):
        """the clean PV, do not include the accrued coupon
           the PV is for a long position, i.e., protection seller
        """
        return self.pv01(disc, p) * self.coupon - self.protPV(disc, p)

    def __str__(self):
        return 'CDS: maturity %.2f, coupon %.2f, recovery %.2f, time accrued %.2f' % (
         self.maturity, self.coupon, self.recovery, self.accrued)


def fwd2disc(curve):
    return lambda ts: np.exp(-curve.integral(ts))


def zero2disc(curve):
    return lambda ts: np.exp(-curve(ts))


def cdspv(disc, spfunc):
    return lambda cds, curve: cds.pv(disc, spfunc(curve))


def bootstrap(insts, curve, pfunc, bds):
    """bootstrap a curve
    Args:
        insts: a dictionary of benchmark instruments and their PVs {inst : pv}
        curve: a curve object to be boostraped, must have a curve.addKnot(x, y) method
        pfunc: a function that computes the pv of an instrument pfunc(inst, curve)
        bds: the boundary of root search [lowerbound, upperbound]
    Returns:
        the curve built
    """
    dx = 0.0033333333333333335

    def objf(x, inst, pv):
        curve.addKnot(inst.maturity, x, dx)
        return pfunc(inst, curve) - pv

    for inst, pv in sorted(list(insts.items()), key=lambda x: x[0].maturity):
        try:
            opt_x = opt.brentq(objf, bds[0], bds[1], args=(inst, pv))
            curve.addKnot(inst.maturity, opt_x, dx)
        except ValueError:
            print('failed to find root for ', inst)

    return curve


def iterboot(bms, pfunc, x0, lbd=0.0, bds=[-1, 1], its=5, mixf=0.0, make_curve=lin.RationalTension):
    """bootstrap a tension spline curve with multiple iterations
    Args:
        bms: a dictionary of benchmark instruments and their PVs {inst : pv}
        pfunc: a function that computes the pv of an instrument pfunc(inst, curve)
        x0: the curve's start value at time 0
        lbd: the tension parameter 
        bds: the boundary of curve points [lowerbound, upperbound]
        its: number of iterations
        mixf: the mixing factor between the new and old values for the next iteration, 0: take the new value
    Returns:
        tsit: the curve built
        e: the errors in pv after each iteration
    """
    ts = np.sort([i.maturity for i in list(bms.keys())])
    tsit = make_curve(lbd)
    tsit.build(ts, np.ones(len(ts)) * 0.01)
    tsit.addKnot(0, x0)
    px = np.copy(tsit.x)
    py = np.copy(tsit.y)
    insts = sorted(list(bms.keys()), key=lambda i: i.maturity)
    es = []
    for it in range(0, its):
        tsit = bootstrap(bms, tsit, pfunc, bds)
        pve = np.array([pfunc(x, tsit) - bms[x] for x in insts])
        tsit.build(mixf * px + (1.0 - mixf) * tsit.x, mixf * py + (1.0 - mixf) * tsit.y)
        px = np.copy(tsit.x)
        py = np.copy(tsit.y)
        es.append(pve)

    return (
     tsit, np.array(es))


def pvfunc(ir, cds):
    pvf = cdspv(ir, zero2disc)
    return lambda trade: pvf(trade, cds)


def pert_bmk(bms, ir, lbd, its=3, pert=0.0001):
    """ return the base cds and a map of perturbed cds curves"""
    pvf = cdspv(ir, zero2disc)
    bms_ps = {CDS(k.maturity, k.coupon - pert, k.recovery):v for k, v in list(bms.items())}
    cds0, _ = iterboot(bms, pvf, x0=0, lbd=lbd, its=its)
    mkt0 = pvfunc(ir, cds0)
    ir1 = FlatCurve(ir.rate - pert)
    cds1, _ = iterboot(bms, cdspv(ir1, zero2disc), x0=0, lbd=lbd, its=its)
    mkts = [pvfunc(ir1, cds1)]
    pkeys = ['IR']
    for i in sorted(list(bms_ps.keys()), key=lambda i: i.maturity):
        pbm = {k:v for k, v in list(bms.items()) if k.maturity != i.maturity}
        pbm[i] = bms_ps[i]
        cds1, _ = iterboot(pbm, pvf, x0=0, lbd=lbd, its=its)
        mkts.append(pvfunc(ir, cds1))
        m = i.maturity
        pkeys.append('CDS @%.2fY' % m if m < 1.0 else 'CDS @%.fY' % m)

    return (
     mkt0, mkts, pkeys, cds0)


def pert_curve(ir, cds, pert=0.0001):
    """ return the base cds and a map of perturbed cds curves"""
    mkt0 = pvfunc(ir, cds)
    ir1 = FlatCurve(ir.rate - pert)
    mkts = [pvfunc(ir1, cds)]
    pkeys = ['IR']
    for x, y in zip(cds.x, cds.y):
        if x > 0:
            cds1 = lin.RationalTension(cds.lbd)
            cds1.build(cds.x, cds.y)
            cds1.addKnot(x, y - pert)
            mkts.append(pvfunc(ir, cds1))
            pkeys.append('H @%.2fY' % x if x < 1.0 else 'H @%.fY' % x)

    return (mkt0, mkts, pkeys, cds)


def pv_deltas(mkt0, mkts, trades):
    pv0 = np.array(list(map(mkt0, trades)))
    deltas = np.array([np.array(list(map(v, trades))) - pv0 for v in mkts])
    return (
     deltas.T, pv0)