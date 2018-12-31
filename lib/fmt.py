#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:24:05 2018

@author: ducklin
"""

import sympy as sp, pandas as pd, numpy as np
from IPython.display import display, Math, HTML
import matplotlib as plt

def displayMath(*ms, **kargs):
    sep = '\\;,\\;\\;\\;'
    replace = {'[':'(',  ']':')'}
    pre = ''
    if 'sep' in kargs:
        sep = kargs['sep']
    if 'replace' in kargs:
        replace = kargs['replace']
    if 'pre' in kargs:
        pre = kargs['pre']
    lstr = sep.join([sp.latex(x) for x in ms])
    for a, b in list(replace.items()):
        lstr = lstr.replace(a, b)

    display(Math(pre + ' ' + lstr))


def joinMath(sep, *ms):
    return sep.join([sp.latex(x) for x in ms])


def displayHTMLs(*fontsize):
    headers = []
    if 'headers' in kargs:
        headers = kargs['headers']
    tbl = ('<center><font size={:d}><table style="border-style:hidden; border-collapse:collapse; text-align:center;">').format(fontsize)
    if headers:
        tbl += '<tr style="border:none">' + ('').join(['<th style="border:none; text-align:center">%s</th>' % h for h in headers]) + '</tr>'
    tbl += '<tr style="border:none">' + ('').join(['<td style="border:none; padding:5px; text-align:center">%s</td>' % t for t in ts]) + '</tr>'
    tbl += '</table></font></center>'
    display(HTML(tbl))


def displayDFs(*ts, fontsize=2, **kargs):
    fmtstr = '{:,.4g}'
    if 'fmt' in kargs:
        fmtstr = '{:,.%s}' % kargs['fmt']
    pd.options.display.float_format = fmtstr.format
    htmls = [t.to_html() for t in ts]
    displayHTMLs(fontsize=fontsize, **kargs)


def displayDF(df, fmt='4f', fontsize=2):
    fmtstr = '{:,.%s}' % fmt
    pd.options.display.float_format = fmtstr.format
    html = ('<center><font size={:d}>').format(fontsize) + df.to_html() + '</font></center>'
    display(HTML(html))


def math2df(df, sz=''):
    return pd.DataFrame(np.reshape(['$%s' % sz + sp.latex(tc) + '$' for tc in np.ravel(df)], np.shape(df)), index=df.index, columns=df.columns)


def plotTensionSpline(tsit, lbd, ax, tagsf, xs):
    lbd_tag = '$\\lambda=%.f$' % lbd
    df = pd.DataFrame({'$t$': xs}).set_index(['$t$'])
    for tag, f in list(tagsf.items()):
        df[tag] = f(tsit, xs)

    df.plot(ax=ax, secondary_y=[list(tagsf.keys())[0]], title='Tension Spline ' + lbd_tag)