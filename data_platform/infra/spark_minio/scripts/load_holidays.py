# === OLD CODE (PostgreSQL) ===
# import pandas as pd
# import requests
# from bs4 import BeautifulSoup
# from datetime import datetime
# from sqlalchemy import create_engine, text

# def get_holidays_from_web(year, country_code=1):
#     """Fetch US holidays from timeanddate.com."""
#     try:
#         url = f"https://www.timeanddate.com/calendar/custom.html?year={year}&country={country_code}&cols=3&df=1&hol=25"
#         response = requests.get(url, timeout=10)
#         dom = BeautifulSoup(response.content, "html.parser")
#         trs = dom.select("table.cht.lpad tr")
#         holidays = []
#         for tr in trs:
#             try:
#                 datestr = tr.select_one("td:nth-of-type(1)").text
#                 holiday_name = tr.select_one("td:nth-of-type(2)").text
#                 date = datetime.strptime(f"{year} {datestr}", "%Y %b %d")
#                 holidays.append({"date": date, "holiday": holiday_name})
#             except:
#                 continue
#         return pd.DataFrame(holidays)
#     except Exception as e:
#         print(f"Failed to fetch holidays for {year}: {e}")
#         return pd.DataFrame(columns=["date", "holiday"])

# def get_blackfriday_dates():
#     dates = [
#         "2012-11-23", "2012-11-24", "2012-11-25", "2012-11-26",
#         "2013-11-29", "2013-11-30", "2013-12-01", "2013-12-02",
#         "2014-11-28", "2014-11-29", "2014-11-30", "2014-12-01",
#     ]
#     return pd.to_datetime(dates)

# engine = create_engine('postgresql://postgres:changeme@localhost:5432/postgres')
# with engine.connect() as conn:
#     conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw"))
#     conn.commit()

# years = [2012, 2013, 2014]
# all_holidays = []
# for year in years:
#     df_holidays = get_holidays_from_web(year, country_code=1)
#     if not df_holidays.empty:
#         all_holidays.append(df_holidays)

# if all_holidays:
#     df_all_holidays = pd.concat(all_holidays, ignore_index=True)
#     df_all_holidays['date'] = pd.to_datetime(df_all_holidays['date'])
#     df_all_holidays.to_sql('raw_holidays', engine, schema='raw', if_exists='replace', index=False)
#     print(f"Loaded {len(df_all_holidays)} holidays → raw.raw_holidays")

# blackfriday_dates = get_blackfriday_dates()
# df_blackfriday = pd.DataFrame({'date': blackfriday_dates, 'event_name': 'Black Friday Weekend'})
# df_blackfriday.to_sql('raw_blackfriday', engine, schema='raw', if_exists='replace', index=False)
# print(f"Loaded {len(df_blackfriday)} Black Friday dates → raw.raw_blackfriday")
# === END OLD CODE ===


import os
import tempfile
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from minio import Minio
from minio.error import S3Error

ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
BUCKET = os.getenv("MINIO_BUCKET", "datalake")
PREFIX = os.getenv("PREFIX", "staging/raw/")


def get_holidays_pandas(start_year, end_year):
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=f'{start_year}-01-01', end=f'{end_year}-12-31')
    return pd.DataFrame({
        "date": holidays,
        "holiday": "US Federal Holiday"
    })


def get_blackfriday_dates():
    dates = [
        "2012-11-23", "2012-11-24", "2012-11-25", "2012-11-26",
        "2013-11-29", "2013-11-30", "2013-12-01", "2013-12-02",
        "2014-11-28", "2014-11-29", "2014-11-30", "2014-12-01",
    ]
    return pd.to_datetime(dates)


def upload_df_to_minio(client, df, filename):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name
    
    object_name = f"{PREFIX}{filename}"
    client.fput_object(BUCKET, object_name, temp_path, content_type="text/csv")
    os.unlink(temp_path)
    print(f"Uploaded {filename} ({len(df)} rows) -> s3://{BUCKET}/{object_name}")


def main():
    client = Minio(ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=SECURE)

    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)

    df_holidays = get_holidays_pandas(2012, 2014)
    upload_df_to_minio(client, df_holidays, "holidays.csv")

    df_blackfriday = pd.DataFrame({
        'date': get_blackfriday_dates(),
        'event_name': 'Black Friday Weekend'
    })
    upload_df_to_minio(client, df_blackfriday, "blackfriday.csv")


if __name__ == "__main__":
    try:
        main()
    except S3Error as e:
        raise SystemExit(e)
