# Extrapolated COVID-19 Infections

Ported from @psteinb's excellent chart for [Dresden, Germany](https://github.com/psteinb/covid19-curve-your-city) 

## Montgomery County, Maryland, USA

![MoCo](us_md_montgomery.png)

Data source: [washingtonpost.com](https://www.washingtonpost.com)

# Reproduce This!

1. Install [R](https://www.r-project.org)
2. Install ["tidyverse"](https://www.tidyverse.org/)

   ``` r
   > install.packages("tidyverse")
   > install.packages(c("ggplot2","dplyr","readr","optparse"))
   ```

3. Run the `exponential.R` script against the included dataset

   ``` 
   $ Rscript exponential.R -i us_md_montgomery.csv
   ```

4. Create your own dataset and compare your location! *E.g.*,

   ```csv
   location,date,diagnosed
   Dresden,2020-03-07,2
   ```
