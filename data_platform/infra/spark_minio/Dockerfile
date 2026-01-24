FROM apache/spark:3.5.3

USER root

RUN apt-get update && apt-get install -y curl python3-pip && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir pandas pyarrow

RUN curl -o /opt/spark/jars/hadoop-aws-3.3.4.jar https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    curl -o /opt/spark/jars/aws-java-sdk-bundle-1.12.262.jar https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar

USER 185