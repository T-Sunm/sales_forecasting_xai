import os
import logging
import psycopg2
from psycopg2 import sql

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASS = os.environ.get("PG_PASS", "changeme")

TARGET_DB = os.environ.get("PG_DB", "sales_forecasting_lakehouse")
ADMIN_DB = os.environ.get("PG_ADMIN_DB", "postgres")
EXTRA_DATABASES = ["nessie"]


def create_database_if_not_exists(cur, dbname):
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
    if cur.fetchone() is None:
        logger.info("Creating database: %s", dbname)
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
    else:
        logger.info("Database already exists: %s", dbname)


def ensure_database():
    all_dbs = [TARGET_DB] + EXTRA_DATABASES
    logger.info("Ensuring databases exist: %s", all_dbs)

    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=ADMIN_DB,
        user=PG_USER, password=PG_PASS
    )
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            for dbname in all_dbs:
                create_database_if_not_exists(cur, dbname)
    finally:
        conn.close()


def ensure_schemas():
    logger.info("Ensure schemas exist. db=%s", TARGET_DB)

    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=TARGET_DB,
        user=PG_USER, password=PG_PASS
    )
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS raw;")
            cur.execute("CREATE SCHEMA IF NOT EXISTS intermediate;")
            cur.execute("CREATE SCHEMA IF NOT EXISTS marts;")
    finally:
        conn.close()


def main():
    try:
        ensure_database()
        ensure_schemas()
        logger.info("All databases and schemas are ready.")
    except Exception:
        logger.exception("Initialization failed.")


if __name__ == "__main__":
    main()
