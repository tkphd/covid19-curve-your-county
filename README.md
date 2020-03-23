# Extrapolated COVID-19 Infections

Ported from @psteinb's excellent chart for [Dresden, Germany](https://github.com/psteinb/covid19-curve-your-city) 

## Montgomery County, Maryland, USA

![MoCo](us_md_montgomery.png)

Data source: https://coronavirus.maryland.gov/

## Reproduce This!

1. Install [Python 3](https://www.anaconda.com/distribution/)
2. Install dependencies

   ```bash
   $ conda install matplotlib numpy pandas pip scipy
   $ pip install https://www.astro.rug.nl/software/kapteyn/kapteyn-3.0.tar.gz
   ```

3. Run the `exponential.py` script against the included dataset

   ``` 
   $ python exponential.py
   ```

4. Create your own dataset and compare your location! *E.g.*,

   ```csv
   location,date,diagnosed
   Montgomery,2020-03-07,2
   ```

   Gaps in the data are OK, just provide what you have.
