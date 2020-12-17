# coding: utf-8

# # Covid Scraper
#
# Scrape the Maryland COVID dashboard for data
#
# ## Dependencies
#
# * Mozilla Firefox
# * Selenium
# * geckodriver
# * numpy
# * pandas
#
# ```bash
# $ conda install -c conda-forge geckodriver numpy pandas selenium
# ```

import datetime
import numpy as np
import pandas as pd
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

xpath = '//*[@id="ember116"]/div/table[1]'

opts = Options()
opts.headless = True
browser = Firefox(options=opts)
browser.get("https://coronavirus.maryland.gov/")
element = WebDriverWait(browser, 30).until(
    EC.presence_of_element_located((By.XPATH, xpath))
)

covid_table = browser.find_elements_by_xpath(xpath)[0]


data = [item.text for item in covid_table.find_elements_by_css_selector("th,td")]
headers = data[1:3]
headers.append('Related_Deaths')
headers.append('Total_Deaths')

county = data[3:-8:4]
cases = np.array([int(x.replace(',', '')) if x else 0 for x in data[4:-7:4]])
deaths = np.array([int(x.replace(',', '').strip('()')) if x else 0 for x in data[5:-6:4]])
related = np.array([int(x.strip('*')) if x else 0 for x in data[6:-5:4]])
total_deaths = deaths + related

df = pd.DataFrame(list(zip(cases, deaths, related, total_deaths)), index=county, columns=headers)

csv = pd.read_csv("us_md_montgomery.csv")

new_cases = df["Cases"]["Montgomery"].item()
new_deaths = df["Total_Deaths"]["Montgomery"].item()
new_confirmed = df["Deaths"]["Montgomery"].item()
new_related = df["Related_Deaths"]["Montgomery"].item()

old_cases = csv["diagnosed"][-1:].item()
old_deaths = csv["killed"][-1:].item()

the_same = (old_cases == new_cases and old_deaths == new_deaths)

if not the_same:
    todaysDate = datetime.date.today()
    print('{0},{1},{2},"https://coronavirus.maryland.gov/"'.format(todaysDate, new_deaths, new_cases))
