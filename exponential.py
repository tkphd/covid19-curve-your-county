# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import strptime
from datetime import date
from string import Template
from scipy.optimize import curve_fit, least_squares
from scipy.stats import describe, t
from matplotlib import style
style.use("seaborn")

def model(t, a, b):
    return a * (1 + b) ** t

def jacobian(t, a, b):
    return np.array([(1 + b) ** t,
                     (a * t * (1 + b) ** t) / (1 + b)]).T


fig = plt.figure(figsize=(6, 4))
plt.suptitle("COVID-19 Cases: Montgomery County, MD", fontweight="bold")
plt.title("github.com/tkphd/covid19-curve-your-county", style="oblique")
plt.xlabel("Day of Record")
plt.ylabel("# Diagnosed Cases")

# Data

df = pd.read_csv("us_md_montgomery.csv")

y = np.array(df["diagnosed"])
start = strptime(df["date"].iloc[0], "%Y-%m-%d")
start = date(start.tm_year, start.tm_mon, start.tm_mday).toordinal()
today = strptime(df["date"].iloc[-1], "%Y-%m-%d")
today = date(today.tm_year, today.tm_mon, today.tm_mday).toordinal()

t = np.zeros_like(y)
for i in range(len(t)):
    day = strptime(df["date"].iloc[i], "%Y-%m-%d")
    t[i] = date(day.tm_year, day.tm_mon, day.tm_mday).toordinal() - start

plt.scatter(t, y, marker=".", s=10, color="k", zorder=10)

# Levenburg-Marquardt Least-Squares Fit
"""
Note that sigma represents the relative error associated with each data point. By default, curve_fit
will assume an array of ones (constant values imply no difference in error), which is probably
incorrect: given sparsity of testing, there's considerable uncertainty, and the earlier numbers may
be lower than the truth to a greater extent than the later numbers. Quantifying this error by some
non-trivial means for each datapoint would produce much more realistic uncertainty bands in the
final plots.
"""
popt, pcov = curve_fit(model, t, y, sigma=None, method="lm", jac=jacobian)
perr = np.sqrt(np.diag(pcov))
coef = describe(pcov)
a, b = popt

# Confidence Band: dfdp represents the partial derivatives of the model with respect to each parameter p (i.e., a and b)

that = np.linspace(0, t[-1] + 7, 100)
yhat = model(that, a, b)

upr_a = a + perr[0]
lwr_a = a - perr[0]

upr_b = b + perr[1]
lwr_b = b - perr[1]

upper = model(that, upr_a, upr_b)
lower = model(that, lwr_a, lwr_b)

it = np.argsort(that)
plt.plot(that[it], yhat[it], c="red", lw=1, zorder=5)
plt.fill_between(
    that[it], upper[it], yhat[it], edgecolor=None, facecolor="silver", zorder=1
)
plt.fill_between(
    that[it], lower[it], yhat[it], edgecolor=None, facecolor="silver", zorder=1
)

# Predictions

dx = 0.25
dt = 14

tomorrow = date.fromordinal(today + 1)
nextWeek = date.fromordinal(today + 7)

that = np.array([tomorrow.toordinal() - start, nextWeek.toordinal() - start])
yhat = model(that, a, b)

upper = model(that, upr_a, upr_b)
lower = model(that, lwr_a, lwr_b)


plt.text(0.5, (yhat[1] + 3 * upper[1]) / 4,
         r"$y = ({0:.4f} \pm {2:.4f}) \times [1 + ({1:.4f} \pm {3:.4f})]^t$".format(a, b, perr[0], perr[1]),
         zorder=5,
         bbox=dict(boxstyle="round", ec="black", fc="white", linewidth=2*dx)
)

plt.text(
    that[0] - dt,
    yhat[0],
    "{0}/{1}: ({2:.0f} < {3:.0f} < {4:.0f})".format(
        tomorrow.month, tomorrow.day, lower[0], yhat[0], upper[0]
    ),
    va="center",
    zorder=5,
    bbox=dict(boxstyle="round", ec="black", fc="white", linewidth=dx)
)
plt.text(
    that[1] - dt,
    yhat[1],
    "{0}/{1}: ({2:.0f} < {3:.0f} < {4:.0f})".format(
        nextWeek.month, nextWeek.day, lower[1], yhat[1], upper[1]
    ),
    va="center",
    zorder=5,
    bbox=dict(boxstyle="round", ec="black", fc="white", linewidth=dx)
)

hw = 12
hl = that[1] / 100

plt.arrow(
    that[0] - dt,
    yhat[0],
    dt - dx + 0.0625,
    0,
    fc="black",
    ec="black",
    head_width=hw,
    head_length=hl,
    overhang=dx,
    length_includes_head=True,
    linewidth=0.5,
    zorder=2,
)
plt.arrow(
    that[1] - dt,
    yhat[1],
    dt - dx + 0.0625,
    0,
    fc="black",
    ec="black",
    head_width=hw,
    head_length=hl,
    overhang=dx,
    length_includes_head=True,
    linewidth=0.5,
    zorder=2,
)

# Plot Boundaries

plt.xlim([0, that[-1]])
plt.ylim([0, upper[-1]])

# Save figure

plt.savefig("us_md_montgomery.png", dpi=400, bbox_inches="tight")
