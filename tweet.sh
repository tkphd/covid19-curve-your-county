#!/bin/bash -e

# Sleep until the specified time

OLD=$(date +%s)
NEW=$(date --date='11:00 tomorrow' +%s)

DIF=$(( NEW - OLD ))

sleep ${DIF}

python scrape.py >> us_md_montgomery.csv && python exponential.py && git add -u && git cm "$(date +%Y-%m-%d) update" && git push
