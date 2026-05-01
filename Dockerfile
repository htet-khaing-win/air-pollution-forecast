ARG AIRFLOW_VERSION=2.9.3
ARG PYTHON_VERSION=3.11

FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

USER root

# System deps (needed for MLflow + healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python deps
COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt && \
    pip install --no-cache-dir mlflow

# Copy project
COPY src/       /opt/airflow/src/
COPY config/    /opt/airflow/config/
COPY dags/      /opt/airflow/dags/

# Ensure MLflow dirs exist
RUN mkdir -p /opt/airflow/mlruns/artifacts

ENV PYTHONPATH="/opt/airflow:${PYTHONPATH}"

