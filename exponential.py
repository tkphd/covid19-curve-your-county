# -*- coding: utf-8 -*-

# Choose whether to plot cases, deaths, or both (as column titles)

columns = ["diagnosed", "killed"]

# Specify which model to use: "exp" for exponential, "log" for logistic

models = {"diagnosed": "log", "killed": "log"}
labels = {"diagnosed": "Cases", "killed": "Deaths"}

equations = {
    "exp": "$f(t) = a (1 + b)^t$\n$a = {0:.4f} \pm {2:.5f}$\n$b = {1:.4f} \pm {3:.6f}$",
    "log": "$f(t) = c / (1 + \exp((b - t)/a))$\n$a = {0:.3f} \pm {3:.4f}$\n$b = {1:.2f} \mp {4:.4f}$\n$c = {2:.0f}. \pm {5:.2f}$",
}

# Set colors for the plot

colors = {"diagnosed": "red", "killed": "black"}

# Specify names for dataset (input) and plot (output)

dataname = "us_md_montgomery.csv"
imgname = "us_md_montgomery.png"

# Prepare dicts for stats

residuals = {"diagnosed": [], "killed": []}

chi_sq_red = {"diagnosed": 1.0, "killed": 1.0}

# Everything else is details

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from calendar import month_name
from datetime import date
from time import strptime
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
    return np.array([(1 + b) ** t, (a * t * (1 + b) ** t) / (1 + b)]).T


def f_log(t, a, b, c):
    # Logistic growth law, $f(t) = c / (exp((b - t)/a) + 1)$
    return c / (np.exp((b - t) / a) + 1)


def df_log(t, a, b, c):
    # Jacobian: df/dp for p=(a, b, c)
    # SymPy-simplified quotients of exponentials
    return np.array(
        [
            c
            * (b - t)
            / (4 * a ** 2)
            * (
                1 / np.cosh((b - t) / (2 * a)) ** 2
            ),  # c*(b - t)*np.exp((b - t)/a)/(a**2*(np.exp((b - t)/a) + 1)**2),
            -c
            / (4 * a)
            * (
                1 / np.cosh((b - t) / (2 * a)) ** 2
            ),  # -c*np.exp((b - t)/a)/(a*(np.exp((b - t)/a) + 1)**2),
            1 / (np.exp((b - t) / a) + 1),
        ]
    ).T


def sigfig(x, n):
    # Round a float, x, to n significant figures.
    # Source: https://github.com/corriander/python-sigfig
    n = int(n)

    e = np.floor(np.log10(np.abs(x)) - n + 1)  # exponent, 10 ** e
    shifted_dp = x / (10 ** e)  # decimal place shifted n d.p.
    return np.around(shifted_dp) * (10 ** e)  # round and revert


fig = plt.figure(figsize=(6, 4))
plt.suptitle("COVID-19 in Montgomery County, Maryland, USA", fontweight="bold")
plt.title("github.com/tkphd/covid19-curve-your-county", style="oblique")
plt.xlabel("Days since 5 March 2020")
plt.ylabel("Number of People")

# Data

data = pd.read_csv(dataname)
dmax = 1
for key in columns:
    dmax = max(dmax, data[key].max())

start = strptime(data["date"].iloc[0], "%Y-%m-%d")
start = date(start.tm_year, start.tm_mon, start.tm_mday).toordinal()

# Before March 5, 2020, there were no COVID data available.
# Keep track of when successive months began, in terms of days since the "epoch", 2020-03-05.

months = [
    ["April", date(2020, 4, 1).toordinal() - start],
    ["May", date(2020, 5, 1).toordinal() - start],
    ["June", date(2020, 6, 1).toordinal() - start],
    ["July", date(2020, 7, 1).toordinal() - start],
    ["August", date(2020, 8, 1).toordinal() - start],
    ["September", date(2020, 9, 1).toordinal() - start],
    ["October", date(2020, 10, 1).toordinal() - start],
    ["November",  date(2020, 11, 1).toordinal() - start],
    # ["December",  date(2020, 12, 1).toordinal() - start]
]


today = strptime(data["date"].iloc[-1], "%Y-%m-%d")
today = date(today.tm_year, today.tm_mon, today.tm_mday).toordinal()

y_off = 1 + 0.35  # offset for equation boxes

for key in columns:
    model = models[key]
    f = f_exp
    df = df_exp

    if model == "log":
        f = f_log
        df = df_log

    y = np.array(data[key])

    t = np.zeros_like(y)
    for i in range(len(y)):
        day = strptime(data["date"].iloc[i], "%Y-%m-%d")
        t[i] = date(day.tm_year, day.tm_mon, day.tm_mday).toordinal() - start

    plt.scatter(
        t, y, marker=".", s=10, color="white", edgecolors=colors[key], zorder=10, label=labels[key]
    )

# Plot Boundaries
tmin, tmax = plt.xlim()
plt.xlim([-0.2, tmax - 7])

ymin, ymax = plt.ylim()
plt.ylim([-50, ymax])

# Label months
for month, day in months:
    plt.plot((day, day), (0, ymax), c="gray", alpha=0.5, zorder=1)
    plt.text(day + 1, 300, month, rotation=90, c="gray", alpha=0.5, zorder=1)

# Save figure
plt.legend(loc="center left")
plt.savefig(imgname, dpi=400, bbox_inches="tight")
plt.close()

# === Plot Increments ===

fig, ax1 = plt.subplots(figsize=(6, 4))
plt.suptitle("COVID-19 in Montgomery County, Maryland, USA", fontweight="bold")
plt.title("github.com/tkphd/covid19-curve-your-county", style="oblique")

key = "diagnosed"
x = np.array(data[key])
y = np.array(x)

for i in np.arange(len(y) - 1, 1, -1):
    y[i] -= y[i - 1]

ax1.set_xlabel("Number of Confirmed Cases")
ax1.set_xlim([-20, np.max(x) + 20])
ax1.set_ylim([0, 320])
ax1.set_ylabel("Increment of People {0}".format(key.capitalize()), color=colors[key])
ax1.scatter(x, y, s=2.5, color=colors[key], label=key.capitalize(), zorder=5)
ax1.plot(x, y, linewidth=0.4, color="maroon", label=None, zorder=1)

# Moving average
window_width = 5
cumsum_vec = np.cumsum(np.insert(y, 0, 0))
mavg = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
ax1.plot(
    x[window_width - 1 :],
    mavg,
    color=colors[key],
    linewidth=0.8,
    label="{0}-day avg".format(window_width),
)

key = "killed"
ax2 = ax1.twinx()
ax2.set_ylim([0, 32])
ax2.set_ylabel("Increment of People {0}".format(key.capitalize()), color=colors[key])
ax2.grid(b=False)

y = np.array(data[key])

for i in np.arange(len(y) - 1, 1, -1):
    y[i] -= y[i - 1]

ax2.scatter(x, y, s=2.5, color=colors[key], label=key.capitalize(), zorder=5)
ax2.plot(x, y, linewidth=0.4, color="gray", label=None, zorder=1)

cumsum_vec = np.cumsum(np.insert(y, 0, 0))
mavg = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
ax2.plot(
    x[window_width - 1 :],
    mavg,
    color=colors[key],
    linewidth=0.8,
    label="{0}-day avg".format(window_width),
)

# Label months
for month, day in months:
    cases = data.loc[day, "diagnosed"]
    plt.plot((cases, cases), (0, 32), c="gray", alpha=0.5, zorder=1)
    plt.text(cases + 1, 1, month, rotation=90, c="gray", alpha=0.5, zorder=1)

plt.savefig("increment.png", dpi=400, bbox_inches="tight")
plt.close()

# === Write the Tweet ===

today = strptime(data["date"].iloc[-1], "%Y-%m-%d")
today = "{} {} {}".format(today.tm_mday, month_name[today.tm_mon], today.tm_year)

nCases = data["diagnosed"].iloc[-1]
nKills = data["killed"].iloc[-1]

dCases = nCases - data["diagnosed"].iloc[-2]
dKills = nKills - data["killed"].iloc[-2]

print(
    "\nToday, {}, there were {} new cases and {} new death{} in @MontgomeryCoMD.".format(
        today, dCases, "no" if dKills == 0 else dKills, "" if dKills == 1 else "s"
    ),
    "Since March 5, 2020, we have seen {:,} confirmed cases of #COVID19 (cumulative),".format(
        nCases
    ),
    "and {:,} #MoCo residents have been killed. #WearAMask #StayHomeSaveLives\n".format(
        nKills
    ),
)
print(
    "Number of COVID-19 cases and deaths in Montgomery County, Maryland, every day since 5 March 2020.",
    "As of {}, there have been {:,} cases and {:,} deaths.\n".format(
        today, nCases, nKills
    ),
)
print(
    "Increment in number of cases, and deaths, due to COVID-19 as a function of the cumulative number of cases.",
    "Today, {}, there were {:} new cases and {:} new death{}.\n".format(
        today, dCases, "no" if dKills == 0 else dKills, "" if dKills == 1 else "s"
    ),
)
