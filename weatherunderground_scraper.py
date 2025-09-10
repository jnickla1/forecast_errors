import time

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import pandas as pd

opts = Options()
opts.add_argument("--headless")   # run without windows 


def get_day_data_uid(day: str, uid: str):
    driver = webdriver.Firefox(options=opts)
    print(f"WU scraping {day}")
    url = f"https://www.wunderground.com/dashboard/pws/{uid}/table/{day}/{day}/daily"
    driver.get(url)
    table = WebDriverWait(driver, 15).until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "table.history-table.desktop-table")
    ))

    # 3) Read headers
    headers = [th.text.strip() for th in table.find_elements(By.CSS_SELECTOR, "thead th")]

    rows = []
    for tr in table.find_elements(By.CSS_SELECTOR, "tbody tr"):
        tds = [td.text.strip() for td in tr.find_elements(By.CSS_SELECTOR, "td")]
        if tds:  # ignore empty/placeholder rows
            
            rows.append(tds)
            
    driver.close()
    out_df = pd.DataFrame(rows,columns=headers)
    out_df['date'] = day
    out_df['station_id'] = uid
    out_df = out_df.rename(columns={'Time': 'time_local'})
    # parse to datetime (no date part, so we just supply a dummy date)
    out_df['time_local'] = pd.to_datetime(out_df['time_local'], format="%I:%M %p").dt.strftime("%H:%M")
    return out_df


import re

def clean_row(row):
    """
    row: list like ['6:59 PM', '39.4 °F', '32.0 °F', '78 %', ...]
    returns: ['6:59 PM', '39.4', '78']
    """
    time = row[0]
    temp = re.sub(r"[^\d\.\-]", "", row[1])  # keep digits, dot, minus
    rh   = re.sub(r"[^\d\.\-]", "", row[3])
    return [time, temp, rh]

if __name__ == "__main__":
    rows = get_day_data_uid(day= "2023-11-10", uid = "KRIPROVI104").values.tolist()
    cleaned = [clean_row(r) for r in rows]
    print(cleaned)
    # Test the getting script
