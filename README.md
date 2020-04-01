# Extrapolated COVID-19 Infections

Ported from @psteinb's excellent chart for [Dresden, Germany](https://github.com/psteinb/covid19-curve-your-city) 

## Montgomery County, Maryland, USA

![MoCo](us_md_montgomery.png)
*Data source: [Maryland Department of Health](https://coronavirus.maryland.gov/) and [@MontgomeryCoMD](https://twitter.com/MontgomeryCoMD)*

## About the Model

The figure above is a least-squares fit to a power-law using the available data:

```latex
f(t) = a * (1 + b)^x
```

The fitting process used in this analysis gives a covariance matrix for the model parameters (*a*
and *b*). From the covariance matrix, it's possible to compute the one-standard-deviation (sigma)
bounds on the parameters, assuming that the uncertainty on the number of COVID-19 cases for each day
is the same.

The gray bands are the plus-one-sigma (upper) and minus-one-sigma (lower) deviations from the
least-squares fit.

## Reproduce This!

1. Install [Python 3](https://www.anaconda.com/distribution/)
2. Install dependencies

   ```bash
   $ conda install matplotlib numpy pandas scipy
   ```

3. Run the `exponential.py` script against the included dataset

   ``` 
   $ python exponential.py
   ```

4. Create your own dataset and compare your location, *e.g.*,

   ```csv
   date,diagnosed,source
   2020-03-05,3,"https://www.montgomerycountymd.gov/HHS/RightNav/Coronavirus.html"
   2020-03-08,4,"https://www2.montgomerycountymd.gov/mcgportalapps/Press_Detail.aspx?Item_ID=23951"
   ```

   Gaps in the data are OK: just provide what you have. You will want to edit the script to set the
   proper place-name and URL in the title.

5. Share your findings to help others assess the spread of SARS-CoV-2, and to gauge the
   effectiveness of our collective response.
