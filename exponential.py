# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import strptime
from datetime import date
from scipy.optimize import curve_fit, least_squares
from scipy.stats import describe, t
from matplotlib import style
style.use("seaborn")

def model(x, a, b):
    return a * (1 + b) ** x

def jacobian(x, a, b):
    return np.array([(1 + b) ** x,
                     (a * x * (1 + b) ** x) / (1 + b)]).T


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

x = np.zeros_like(y)
for i in range(len(x)):
    day = strptime(df["date"].iloc[i], "%Y-%m-%d")
    x[i] = date(day.tm_year, day.tm_mon, day.tm_mday).toordinal() - start

plt.scatter(x, y, marker=".", s=10, color="k", zorder=10)

# Levenburg-Marquardt Least-Squares Fit
"""
Note that sigma represents the relative error associated with each data point. By default, curve_fit
will assume an array of opnes (constant values imply no difference in error), which is probably
incorrect: given sparsity of testing, there's considerable uncertainty, and the earlier numbers may
be lower than the truth to a greater extent than the later numbers. Quantifying this error by some
non-trivial means for each datapoint would produce much more realistic uncertainty bands in the
final plots.
"""
popt, pcov = curve_fit(model, x, y, sigma=None, method="lm", jac=jacobian)
perr = np.sqrt(np.diag(pcov))
coef = describe(pcov)
a, b = popt

print("cases ~ {0:.3g} * (1 + {1:.3g})^t".format(a, b))

# Confidence Band: dfdp represents the partial derivatives of the model with respect to each parameter p (i.e., a and b)

xhat = np.linspace(0, x[-1] + 7, 100)
yhat = model(xhat, a, b)

upr_a = a + perr[0]
upr_b = b + perr[1]
lwr_a = a - perr[0]
lwr_b = b - perr[1]

upper = model(xhat, upr_a, upr_b)
lower = model(xhat, lwr_a, lwr_b)

ix = np.argsort(xhat)
plt.plot(xhat[ix], yhat[ix], c="red", lw=1, zorder=5)
plt.fill_between(
    xhat[ix], upper[ix], yhat[ix], edgecolor=None, facecolor="silver", zorder=1
)
plt.fill_between(
    xhat[ix], lower[ix], yhat[ix], edgecolor=None, facecolor="silver", zorder=1
)

# Predictions

dx = 0.25
dt = 14

tomorrow = date.fromordinal(today + 1)
nextWeek = date.fromordinal(today + 7)

xhat = np.array([tomorrow.toordinal() - start, nextWeek.toordinal() - start])
yhat = model(xhat, a, b)

upper = model(xhat, upr_a, upr_b)
lower = model(xhat, lwr_a, lwr_b)

plt.text(
    xhat[0] - dt,
    yhat[0],
    "{0}/{1}: ({2:.0f} < {3:.0f} < {4:.0f})".format(
        tomorrow.month, tomorrow.day, lower[0], yhat[0], upper[0]
    ),
    va="center",
    zorder=5,
    bbox=dict(boxstyle="round", ec="black", fc="white", linewidth=dx),
)
plt.text(
    xhat[1] - dt,
    yhat[1],
    "{0}/{1}: ({2:.0f} < {3:.0f} < {4:.0f})".format(
        nextWeek.month, nextWeek.day, lower[1], yhat[1], upper[1]
    ),
    va="center",
    zorder=5,
    bbox=dict(boxstyle="round", ec="black", fc="white", linewidth=dx),
)

hw = 12
hl = xhat[1] / 100

plt.arrow(
    xhat[0] - dt,
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
    xhat[1] - dt,
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

plt.xlim([0, xhat[-1]])
plt.ylim([0, upper[-1]])

# Save figure

plt.savefig("us_md_montgomery.png", dpi=400, bbox_inches="tight")
