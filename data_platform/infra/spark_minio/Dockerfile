FROM apache/spark:3.5.3-scala2.12-java17-python3-ubuntu
USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

# Cài pip cho Python 3.12 rồi cài deps (bao gồm setuptools/wheel để có distutils)
RUN python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --no-cache-dir -U pip setuptools wheel pandas pyarrow

# Jars
RUN curl -L -o /opt/spark/jars/hadoop-aws-3.3.4.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    curl -L -o /opt/spark/jars/aws-java-sdk-bundle-1.12.262.jar https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar && \
    curl -L -o /opt/spark/jars/postgresql-42.7.0.jar https://repo1.maven.org/maven2/org/postgresql/postgresql/42.7.0/postgresql-42.7.0.jar && \
    curl -L -o /opt/spark/jars/iceberg-spark-runtime-3.5_2.12-1.10.1.jar https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/1.10.1/iceberg-spark-runtime-3.5_2.12-1.10.1.jar && \
    curl -L -o /opt/spark/jars/nessie-spark-extensions-3.5_2.12-0.107.2.jar https://repo1.maven.org/maven2/org/projectnessie/nessie-integrations/nessie-spark-extensions-3.5_2.12/0.107.2/nessie-spark-extensions-3.5_2.12-0.107.2.jar

USER 185
