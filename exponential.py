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

import locale
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from calendar import month_name
from datetime import date
from matplotlib import style
from scipy.optimize import curve_fit
from scipy.stats import describe, chisquare, t
from string import Template
from time import strptime

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

data = pd.read_csv(dataname)

fig = plt.figure(figsize=(6, 4))
plt.suptitle("COVID-19 in Montgomery County, Maryland, USA", fontweight="bold")
plt.title("github.com/tkphd/covid19-curve-your-county", style="oblique")
plt.xlabel("Days since 5 March 2020")
plt.ylabel("Number of People")


start = strptime(data["date"].iloc[0], "%Y-%m-%d")
start = date(start.tm_year, start.tm_mon, start.tm_mday).toordinal()

# Before March 5, 2020, there were no COVID data available.
# Keep track of when successive months began, in terms of days since the "epoch", 2020-03-05.

months = [
    # 2020
    ["April",     date(2020,  4, 1).toordinal() - start],
    ["May",       date(2020,  5, 1).toordinal() - start],
    ["June",      date(2020,  6, 1).toordinal() - start],
    ["July",      date(2020,  7, 1).toordinal() - start],
    ["August",    date(2020,  8, 1).toordinal() - start],
    ["September", date(2020,  9, 1).toordinal() - start],
    ["October",   date(2020, 10, 1).toordinal() - start],
    ["November",  date(2020, 11, 1).toordinal() - start],
    ["December",  date(2020, 12, 1).toordinal() - start],
    #
    # 2021
    ["January",   date(2021,  1, 1).toordinal() - start],
    ["February",  date(2021,  2, 1).toordinal() - start],
    ["March",     date(2021,  3, 1).toordinal() - start],
    ["April",     date(2021,  4, 1).toordinal() - start],
    ["May",       date(2021,  5, 1).toordinal() - start],
    ["June",      date(2021,  6, 1).toordinal() - start],
    ["July",      date(2021,  7, 1).toordinal() - start],
    ["August",    date(2021,  8, 1).toordinal() - start],
    ["September", date(2021,  9, 1).toordinal() - start],
    ["October",   date(2021, 10, 1).toordinal() - start],
    ["November",  date(2021, 11, 1).toordinal() - start],
    ["December",  date(2021, 12, 1).toordinal() - start],
]

holidays = [
    # 2020
    ["Memorial",      date(2020,  5, 25).toordinal() - start],
    ["Juneteenth",    date(2020,  6, 19).toordinal() - start],
    ["Independence",  date(2020,  7,  4).toordinal() - start],
    ["Labor",         date(2020,  9,  7).toordinal() - start],
    ["Halloween",     date(2020, 10, 31).toordinal() - start],
    ["Veterans",      date(2020, 11, 11).toordinal() - start],
    ["Thanksgiving",  date(2020, 11, 26).toordinal() - start],
    ["Christmas",     date(2020, 12, 25).toordinal() - start],
    #
    # 2021
    ["New Year",      date(2021,  1,  1).toordinal() - start],
    ["MLK, Jr",       date(2021,  1, 18).toordinal() - start],
    ["Inauguration",  date(2021,  1, 20).toordinal() - start],
    ["Presidents",    date(2021,  2, 15).toordinal() - start],
    ["Earth",         date(2021,  4, 22).toordinal() - start],
    ["Memorial",      date(2021,  5, 31).toordinal() - start],
    ["Juneteenth",    date(2021,  6, 19).toordinal() - start],
    ["Independence",  date(2021,  7,  4).toordinal() - start],
    ["Labor",         date(2021,  9,  6).toordinal() - start],
    ["Indigenous",    date(2021, 10, 11).toordinal() - start],
    ["Halloween",     date(2021, 10, 31).toordinal() - start],
    ["Veterans",      date(2021, 11, 11).toordinal() - start],
    ["Thanksgiving",  date(2021, 11, 25).toordinal() - start],
    ["Christmas",     date(2021, 12, 25).toordinal() - start],

]

today = strptime(data["date"].iloc[-1], "%Y-%m-%d")
today = date(today.tm_year, today.tm_mon, today.tm_mday).toordinal() - start

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
        t,
        y,
        marker=".",
        s=10,
        color="white",
        edgecolors=colors[key],
        zorder=10,
        label=labels[key],
    )

# Plot Boundaries
tmin, tmax = plt.xlim()
plt.xlim([-0.2, tmax - 7])

ymin, ymax = plt.ylim()
plt.ylim([-50, ymax])

# Label months
for month, day in months:
    if day <= today:
        plt.plot((day, day), (0, ymax), c="gray", alpha=0.5, zorder=1)
        plt.text(day + 1, 2000, month, rotation=90, c="gray", alpha=0.5, zorder=1)

# Label holidays
for holiday, day in holidays:
    if day <= today:
        plt.plot((day, day), (0, ymax - 6500), c="gray", linestyle='dashed', linewidth=0.5, alpha=0.5, zorder=1)
        plt.text(day - 2.5, ymax - 6000, holiday, rotation=90, c="gray", fontsize=6, alpha=0.5, zorder=1)

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

ymin, ymax = plt.ylim()
ax1.set_ylim([0, ymax])

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
    if day <= today:
        cases = data.loc[day, "diagnosed"]
        plt.plot((cases, cases), (0, 32), c="gray", alpha=0.5, zorder=1)
        plt.text(cases + 1, 1, month, rotation=90, c="gray", alpha=0.5, zorder=1)

# Label holidays
for holiday, day in holidays:
    if day <= today:
        cases = data.loc[day, "diagnosed"]
        plt.plot((cases, cases), (0, 26.25), c="gray", linestyle='dashed', linewidth=0.5, alpha=0.5, zorder=1)
        plt.text(cases - 300, 26.75, holiday, rotation=90, c="gray", fontsize=6, alpha=0.5, zorder=1)


plt.savefig("increment.png", dpi=400, bbox_inches="tight")
plt.close()

# === Write the Tweet ===

today = strptime(data["date"].iloc[-1], "%Y-%m-%d")
today = "{} {} {}".format(today.tm_mday, month_name[today.tm_mon], today.tm_year)

nCases = data["diagnosed"].iloc[-1]
nKills = data["killed"].iloc[-1]

dCases = nCases - data["diagnosed"].iloc[-2]
dKills = nKills - data["killed"].iloc[-2]

print()

print(
    "Today, {}, there were {} new cases and {} new death{} in @MontgomeryCoMD.".format(
        today, dCases, "no" if dKills == 0 else dKills, "" if dKills == 1 else "s"
    ),
    "Since March 5, 2020, we have seen {:,} confirmed cases of #COVID19 (cumulative),".format(
        nCases
    ),
    "and {:,} #MoCo residents have been killed. #WearAMask #StayHomeSaveLives".format(
        nKills
    ),
)

print()

print(
    "Cumulative cases and deaths due to COVID-19 in Montgomery County, Maryland, since 5 March 2020.",
    "Current total is {:,} infected and {:,} killed.".format(
        nCases, nKills
    ),
)

print()

print(
    "Increment in cases and deaths due to COVID-19 in Montgomery County, Maryland, as a function of cumulative cases.",
    "The increment for {}".format(
        today
    ),
    "is {:,} infections and {:,} deaths.".format(
        dCases, dKills
    ),
)

print()
