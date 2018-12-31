#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 20:52:02 2018

@author: ducklin
"""
import pandas as pd
import numpy as np
import lib.lin as lin
import lib.fmt as fmt
import os
import matplotlib.pyplot as plt
import sympy as sp

plt.style.use('ggplot')

#%%
location = '/Users/ducklin/Desktop/CurveFit'
file = 'Daily_Treasury_Yield_Curve.xlsx'
treasuryYield = os.path.join(location, file)

tYield = pd.read_excel(treasuryYield, sheets = 'sheet1', index_col = 0)
tYield.columns = np.array([1/12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
date = '2017-11-01'
yield1d = tYield.loc[date]/100

#%%

""" 1. Cubic Spline

Spline interpolation is very smooth

However, it does not preserve monotonicity and convexity

it could overshoot
can result in arbitrages in many situations """

#%%

from scipy.interpolate import CubicSpline
y = yield1d
t = np.array(yield1d.index)
cs = CubicSpline(t, y, bc_type='not-a-knot')
x1d = np.arange(0, 32, .1)

plt.figure(figsize=(10, 5))
plt.plot(x1d, cs(x1d))
plt.plot(yield1d, 'o', color='b')

plt.legend(loc='lower left', ncol=2)
plt.title('Cubic Spline - Scipy');
plt.show()

coef = cs.c   
#%%
ts = lin.RationalTension(0.)
ts.build(t, np.array(y))

plt.figure(figsize=[12, 4])
plt.plot(x1d, ts.value(x1d))
plt.plot(t, y, 'o');
plt.title('Cubic Spline - Yadong');
plt.show()
      
#%%
def plotTension(x, y, lbds, xd) :
    plt.plot(x, y, 'o');
    plt.title('Tension Spline');

    for lbd in lbds:
        ts = lin.RationalTension(lbd)
        ts.build(x, y)
        plt.plot(xd, ts.value(xd))
    
    plt.legend(['Actual points'] + ['$\lambda = %.f$' % l for l in lbds], loc='best');

lbds = (0, 2, 10, 50)

plt.figure(figsize=[12, 4])
plotTension(t, np.array(y), lbds, x1d)

#%%
""" Fit forward curve with Tension Spline when lambda increse from 0 to inf """
swap = pd.read_excel("swap.xlsx")
swap.columns = ["Maturity", "Spread", "Bid_Ask"]
swap.drop(10, inplace = True)
t = np.arange(0, 30, 0.1)


ts = lin.RationalTension(0.)
ts.build(swap.Maturity, swap.Spread)

plotTension(swap['Maturity'], swap['Spread'].values, lbds, t)

#%%
x = swap['Maturity'].values
y = swap['Spread'].values
m = dict(zip(x, y))
m.update(dict(zip(x[:-1] + 1e-6, y[1:])))
k, v = zip(*sorted(m.items()))
plt.plot(k, v, 'o-', color = 'b')
plt.title('Piecewise Constant');

#%%
import sympy as sp
a, b, c, d, t = sp.symbols('a b c d t', real = True)
l = sp.symbols('lambda', positive = True)
f = sp.Function('f')
df = sp.Function('\dot{f}')
ddf = sp.Function('\ddot{f}')

r = (a + b*t + c*t**2 + d*t**3)/(1 + 1*t*(1-t))
s_x, xi, xii, xi_ = sp.symbols('x, x_i, x_{i+1}, x_[i-1]', real = True)


def lincollect(e, tms) :
    m = sp.collect(sp.expand(e), tms, evaluate=False)
    return [m[k] if k in m else 0 for k in tms]


#%%
def plotPerturb(x, y, yp, xd, lbds) :
    plt.plot(x, yp-y, 'o')
    for lbd in lbds:
        ts = lin.RationalTension(lbd)
        ts.build(x, y)
        tsp = lin.RationalTension(lbd)
        tsp.build(x, yp)

        plt.plot(xd, tsp.value(xd) - ts.value(xd))
        
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Changes in Spline')
    plt.legend(['knots'] + ['$\lambda = %.f$' % l for l in lbds], loc='best');

dy = .01
plt.figure(figsize=[12, 4])
plt.subplot(1, 2, 1)
idx = 3
yp = np.copy(swap.Spread)
yp[idx] *= 1. + dy

plotPerturb(swap.Maturity, swap.Spread, yp, x1d, lbds)

plt.subplot(1,2,2)
plotTension(swap['Maturity'], swap['Spread'].values, lbds, t)

#%%
coefs = sp.Matrix([a, b, c, d])
drt0 = sp.Matrix([lincollect(r.diff(t, i).subs({t:0}), coefs) for i in range(3)])
drt1 = sp.Matrix([lincollect(r.diff(t, i).subs({t:1}), coefs) for i in range(3)])

derivs = sp.Matrix([f(t), df(t), ddf(t)])
s_v = sp.Matrix([f(0), f(1), ddf(0), ddf(1)])
s_a = sp.Matrix([drt0[0, :], drt1[0, :], drt0[2, :], drt1[2, :]])