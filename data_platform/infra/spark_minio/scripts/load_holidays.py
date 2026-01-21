import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from sqlalchemy import create_engine, text

def get_holidays_from_web(year, country_code=1):
    """Fetch US holidays from timeanddate.com."""
    try:
        url = f"https://www.timeanddate.com/calendar/custom.html?year={year}&country={country_code}&cols=3&df=1&hol=25"
        response = requests.get(url, timeout=10)
        dom = BeautifulSoup(response.content, "html.parser")
        trs = dom.select("table.cht.lpad tr")
        
        holidays = []
        for tr in trs:
            try:
                datestr = tr.select_one("td:nth-of-type(1)").text
                holiday_name = tr.select_one("td:nth-of-type(2)").text
                date = datetime.strptime(f"{year} {datestr}", "%Y %b %d")
                holidays.append({"date": date, "holiday": holiday_name})
            except:
                continue
        
        return pd.DataFrame(holidays)
    except Exception as e:
        print(f"Failed to fetch holidays for {year}: {e}")
        return pd.DataFrame(columns=["date", "holiday"])

def get_blackfriday_dates():
    """Return hardcoded Black Friday dates for 2012-2014."""
    dates = [
        "2012-11-23", "2012-11-24", "2012-11-25", "2012-11-26",
        "2013-11-29", "2013-11-30", "2013-12-01", "2013-12-02",
        "2014-11-28", "2014-11-29", "2014-11-30", "2014-12-01",
    ]
    return pd.to_datetime(dates)

engine = create_engine('postgresql://postgres:changeme@localhost:5432/postgres')

with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw"))
    conn.commit()

years = [2012, 2013, 2014]
all_holidays = []

for year in years:
    df_holidays = get_holidays_from_web(year, country_code=1)
    if not df_holidays.empty:
        all_holidays.append(df_holidays)

if all_holidays:
    df_all_holidays = pd.concat(all_holidays, ignore_index=True)
    df_all_holidays['date'] = pd.to_datetime(df_all_holidays['date'])
    df_all_holidays.to_sql('raw_holidays', engine, schema='raw', if_exists='replace', index=False)
    print(f"Loaded {len(df_all_holidays)} holidays → raw.raw_holidays")

blackfriday_dates = get_blackfriday_dates()
df_blackfriday = pd.DataFrame({
    'date': blackfriday_dates,
    'event_name': 'Black Friday Weekend'
})
df_blackfriday.to_sql('raw_blackfriday', engine, schema='raw', if_exists='replace', index=False)
print(f"Loaded {len(df_blackfriday)} Black Friday dates → raw.raw_blackfriday")
