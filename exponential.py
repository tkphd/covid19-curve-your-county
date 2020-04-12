# -*- coding: utf-8 -*-

# Choose whether to plot cases, deaths, or both (as column titles)

columns = ["diagnosed", "killed"]

# Specify which model to use: "exp" for exponential, "log" for logistic

models = {"diagnosed": "log",
          "killed": "exp"}

equations = {"exp": "$f(t) = a (1 + b)^t$\n$a = {0:.4f} \pm {2:.4f}$\n$b = {1:.4f} \pm {3:.4f}$",
             "log": "$f(t) = c / (1 + \exp((b - t)/a))$\n$a = {0:.4f} \pm {3:.4f}$\n$b = {1:.4f} \mp {4:.4f}$\n$c = {2:.4f} \pm {5:.4f}$"}

# Set colors for the plot

colors = {"diagnosed": "red",
          "killed": "black"}

# Specify names for dataset (input) and plot (output)

dataname = "us_md_montgomery.csv"
imgname = "us_md_montgomery.png"

# Prepare dicts for stats

residuals = {"diagnosed": [],
             "killed": []}

chi_sq_red = {"diagnosed": 1.0,
              "killed": 1.0}

# Everything else is details

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import strptime
from datetime import date
from string import Template
from scipy.optimize import curve_fit
from scipy.stats import describe, chisquare, t
from matplotlib import style
style.use("seaborn")

# Define the model equation and its Jacobian

def f_exp(t, a, b):
    # Exponential growth law, $f(t) = a * (1 + b) ^ t$,
    # where $a$ is the number of cases at $t=0$ and $b$ is the growth rate.
    return a * (1 + b) ** t

def df_exp(t, a, b):
    # Jacobian: df/dp for p=(a, b)
    return np.array([(1 + b) ** t,
                     (a * t * (1 + b) ** t) / (1 + b)]).T

def f_log(t, a, b, c):
    # Logistic growth law, $f(t) = c / (exp((b - t)/a) + 1)$
    return c / (np.exp((b - t)/a) + 1)

def df_log(t, a, b, c):
    # Jacobian: df/dp for p=(a, b, c)
    return np.array([c*(b - t)*np.exp((b - t)/a)/(a**2*(np.exp((b - t)/a) + 1)**2),
                     -c*np.exp((b - t)/a)/(a*(np.exp((b - t)/a) + 1)**2),
                     1/(np.exp((b - t)/a) + 1)]).T

fig = plt.figure(figsize=(6, 4))
plt.suptitle("COVID-19 in Montgomery County, Maryland, USA", fontweight="bold")
plt.title("github.com/tkphd/covid19-curve-your-county", style="oblique")
plt.xlabel("Day of Record")
plt.ylabel("Number of People")

# Data

data = pd.read_csv(dataname)

start = strptime(data["date"].iloc[0], "%Y-%m-%d")
start = date(start.tm_year, start.tm_mon, start.tm_mday).toordinal()
today = strptime(data["date"].iloc[-1], "%Y-%m-%d")
today = date(today.tm_year, today.tm_mon, today.tm_mday).toordinal()

t_max = 0
y_max = 0

for key in columns:
    model = models[key]
    f  = f_exp
    df = df_exp

    if model == "log":
        f  = f_log
        df = df_log

    y = np.array(data[key])

    t = np.zeros_like(y)
    for i in range(len(y)):
        day = strptime(data["date"].iloc[i], "%Y-%m-%d")
        t[i] = date(day.tm_year, day.tm_mon, day.tm_mday).toordinal() - start

    plt.scatter(t, y, marker=".", s=10, color="white", edgecolors=colors[key], zorder=10)

    # Levenburg-Marquardt Least-Squares Fit
    """
    Note that sigma represents the relative error associated with each data point. By default, `curve_fit`
    will assume an array of ones (constant values imply no difference in error), which is probably
    incorrect: given sparsity of testing, there's considerable uncertainty, and the earlier numbers may
    be lower than the truth to a greater extent than the later numbers. Quantifying this error by some
    non-trivial means for each datapoint would produce much more realistic uncertainty bands in the
    final plots.
    """
    p, pcov = curve_fit(f, t, y, sigma=None, method="lm", jac=df, maxfev=1000)
    coef = describe(pcov)
    perr = np.sqrt(np.diag(pcov))

    # Reduced chi-square goodness of fit
    ## https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic

    chisq, chip = chisquare(y, f(t, *p))
    ndof = len(y) - len(p) - 1

    residuals[key] = f(t, *p) - y
    chi_sq_red[key] = chisq / ndof

    # Confidence Band: dfdp represents the partial derivatives of the model with respect to each parameter p (i.e., a and b)

    t_hat = np.linspace(0, t[-1] + 7, 100)
    y_hat = f(t_hat, *p)

    if (models[key] == "log"):
        perr[1] *= -1
    upr_p = p + perr
    lwr_p = p - perr
    if (models[key] == "log"):
        perr[1] *= -1

    upper = f(t_hat, *upr_p)
    lower = f(t_hat, *lwr_p)

    it = np.argsort(t_hat)
    plt.plot(t_hat[it], y_hat[it], c=colors[key], lw=1, zorder=5, label=key.capitalize())
    plt.fill_between(
        t_hat[it], upper[it], y_hat[it], edgecolor=None, facecolor="silver", zorder=1
    )
    plt.fill_between(
        t_hat[it], lower[it], y_hat[it], edgecolor=None, facecolor="silver", zorder=1
    )

    # Predictions

    dx = 0.25
    dt = 10

    tomorrow = date.fromordinal(today + 1)
    nextWeek = date.fromordinal(today + 7)

    t_hat = np.array([tomorrow.toordinal() - start, nextWeek.toordinal() - start])
    y_hat = f(t_hat, *p)

    upper = f(t_hat, *upr_p)
    lower = f(t_hat, *lwr_p)

    # Overlay model on plot
    plt.text(0.5, max(1000, 0.5 * (upper[1] + y_hat[1])),
             equations[models[key]].format(*p, *perr),
             color=colors[key],
             va="top",
             zorder=4,
             bbox=dict(boxstyle="round", ec="gray", fc="ghostwhite", linewidth=2.5*dx)
    )

    # Overlay projections on plot
    plt.text(
        t_hat[0] - dt,
        y_hat[0],
        "{0}/{1}: ({2:.0f} < {3:.0f} < {4:.0f})".format(
            tomorrow.month, tomorrow.day, lower[0], y_hat[0], upper[0]
        ),
        color=colors[key],
        va="center",
        ha="center",
        zorder=4,
        bbox=dict(boxstyle="round", ec="gray", fc="ghostwhite", linewidth=dx)
    )

    plt.text(
        t_hat[1] - dt,
        y_hat[1],
        "{0}/{1}: ({2:.0f} < {3:.0f} < {4:.0f})".format(
            nextWeek.month, nextWeek.day, lower[1], y_hat[1], upper[1]
        ),
        color=colors[key],
        va="center",
        ha="center",
        zorder=3,
        bbox=dict(boxstyle="round", ec="gray", fc="ghostwhite", linewidth=dx)
    )

    hw = 12
    hl = t_hat[1] / 100

    plt.arrow(
        t_hat[0] - dt,
        y_hat[0],
        dt - dx + 0.0625,
        0,
        fc="black",
        ec="black",
        head_width=hw,
        head_length=hl,
        overhang=dx,
        length_includes_head=True,
        linewidth=0.5,
        zorder=3,
    )

    plt.arrow(
        t_hat[1] - dt,
        y_hat[1],
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

    if t_hat[-1] > t_max:
        t_max = t_hat[-1]
    if upper[-1] > y_max:
        y_max = upper[-1]

# Plot Boundaries

plt.xlim([0, t_max])
plt.ylim([0, y_max])

# Save figure

plt.legend(loc="center left")
plt.savefig(imgname, dpi=400, bbox_inches="tight")
plt.close()


# Plot residuals

fig = plt.figure(figsize=(6, 4))
plt.title("Residuals: $y - f(t)$")
plt.xlabel("Day of Record")
plt.ylabel("Residual")

N = 5

for key in columns:
    y = residuals[key]
    x = np.arange(0, len(y))
    plt.bar(x, y, align="edge", color=colors[key],
            label="{0}: $\\chi^2_\\nu={1:.3g}$".format(key.capitalize(), chi_sq_red[key]))
    """
    mov_avg = np.convolve(y, np.ones((N,))/N, mode='valid')
    x = np.arange(N-3, len(y)-(N-3))
    plt.scatter(x, mov_avg, color=colors[key])
    """

plt.legend(loc="best")
plt.xlim([0, len(y)])
plt.savefig("residuals.png", dpi=400, bbox_inches="tight")
plt.close()
