# -*- coding: utf-8 -*-
# Based on https://stackoverflow.com/questions/24633664/confidence-interval-for-exponential-curve-fit/37080916#37080916

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kapteyn import kmpfit
from time import strptime

def model(p, x):
    a, b, c = p
    return a * np.exp(b * x) + c


df = pd.read_csv("us_md_montgomery.csv")

y = np.array(df["diagnosed"])
start = strptime(df["date"].iloc[0], "%Y-%m-%d").tm_yday

x = np.zeros_like(y)
for i in range(len(x)):
    x[i] = strptime(df["date"].iloc[i], "%Y-%m-%d").tm_yday - start

# fit
f = kmpfit.simplefit(model, [1, 1, 1], x, y)
a, b, c = f.params
cov = f.covar
print("cases ~ {0:.2g} * exp({1:.2g} * (t - t0)) - {2:.2g}".format(a, b, -c))

# confidence band; dfdp represents the partial derivatives of the model with respect to each parameter p (i.e., a, b, and c)
dfdp = [np.exp(b * x), a * x * np.exp(b * x), 1]
yhat, upper, lower = f.confidence_band(x, dfdp, 0.99, model)

plt.scatter(x, y, marker=".", s=10, color="#0000ba")
ix = np.argsort(x)
for i, l in enumerate((upper, lower, yhat)):
    plt.plot(x[ix], l[ix], c="g" if i == 2 else "r", lw=2)


plt.savefig("exponential.png", dpi=400, bbox_inches="tight")

print("Prediction, 1 day  from now: {0:.0f}".format(a * np.exp(b * (x[-1]+1)) + c))
print("Prediction, 1 week from now: {0:.0f}".format(a * np.exp(b * (x[-1]+7)) + c))
