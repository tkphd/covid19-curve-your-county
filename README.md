# Extrapolated COVID-19 Infections

## Dresden 

![](plus5.png)

Data source: [dresden.de](https://www.dresden.de/de/leben/gesundheit/hygiene/infektionsschutz/corona.php)

# Reproduce This!

1. Install [R](https://www.r-project.org)
2. Install ["tidyverse"](https://www.tidyverse.org/)

   ``` r
   > install.packages("tidyverse")
   > install.packages(c("ggplot2","dplyr","readr","optparse"))
   ```

3. Run the `exponential.R` script against the included dataset

   ``` 
   $ Rscript exponential.R -i de_dresden.csv
   ```

4. Create your own dataset and compare your location! E.g.,

   ```csv
   Location,Date,Diagnosed
   Dresden,2020-03-07,2
   ```
