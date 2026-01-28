FROM apache/airflow:3.1.6

USER root

# 1. Install System Dependencies (Git & Java for Spark)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    openjdk-17-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Environment Variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# 3. Download Spark/Postgres JARs (Driver-side requirements)
RUN mkdir -p /opt/airflow/jars && \
    curl -L -o /opt/airflow/jars/hadoop-aws-3.3.4.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    curl -L -o /opt/airflow/jars/aws-java-sdk-bundle-1.12.262.jar https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar && \
    curl -L -o /opt/airflow/jars/postgresql-42.7.0.jar https://repo1.maven.org/maven2/org/postgresql/postgresql/42.7.0/postgresql-42.7.0.jar && \
    chown -R airflow:root /opt/airflow/jars

USER airflow

# 4. Setup dbt in a virtual environment
RUN python -m venv /opt/airflow/dbt_venv && \
    /opt/airflow/dbt_venv/bin/pip install --no-cache-dir dbt-postgres

# 5. Install Python dependencies
# - Copy requirements.txt first
COPY requirements.txt /tmp/requirements.txt

# - Combine installs: main requirements + forced Spark/Data stack
#   We force install pyspark, pandas, pyarrow to bypass Airflow constraints and match Spark Cluster
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.1.6/constraints-3.12.txt" && \
    pip install --no-cache-dir "pyspark==3.5.3" "pandas" "pyarrow"
