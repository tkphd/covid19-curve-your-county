# Extrapolated COVID-19 Infections

Ported from @psteinb's excellent chart for [Dresden, Germany](https://github.com/psteinb/covid19-curve-your-city) 

## Montgomery County, Maryland, USA

![MoCo](us_md_montgomery.png)

Data source: https://coronavirus.maryland.gov/

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

4. Create your own dataset and compare your location! *E.g.*,

   ```csv
   date,diagnosed,source
   2020-03-05,3,"https://www.montgomerycountymd.gov/HHS/RightNav/Coronavirus.html"
   2020-03-08,4,"https://www2.montgomerycountymd.gov/mcgportalapps/Press_Detail.aspx?Item_ID=23951"
   ```

   Gaps in the data are OK, just provide what you have. You will want to edit the script to set the
   proper place-name and URL in the title.

5. Share your findings to help others assess the spread of SARS-CoV-2, and to gauge the
   effectiveness of our collective response.
