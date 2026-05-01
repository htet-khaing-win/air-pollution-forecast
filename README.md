# PM2.5 Air Pollution Forecasting System — Yangon, Myanmar

A full end-to-end MLOps pipeline that forecasts PM2.5 air pollution 24 hours ahead for Yangon, Myanmar. The system ingests live sensor and weather data, trains a LightGBM model with automated weekly retraining via Apache Airflow, tracks experiments and model versions in MLflow, and serves predictions through a FastAPI REST API — all containerised with Docker Compose for local development and deployed to AWS ECS Fargate for cloud validation.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Sources & Feature Engineering](#2-data-sources--feature-engineering)
3. [ML Model & Registry](#3-ml-model--registry)
4. [API Endpoints](#4-api-endpoints)
5. [Local Setup & Quickstart](#5-local-setup--quickstart)
6. [AWS Deployment](#6-aws-deployment)
7. [Phase 7: Load Testing & Observability](#7-phase-7-load-testing--observability)
8. [Environment Variables Reference](#8-environment-variables-reference)
9. [Troubleshooting](#9-troubleshooting)
10. [Project Structure](#10-project-structure)
11. [Roadmap](#11-roadmap)

---

## 1. Architecture Overview

### Local / Docker Compose Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                  │
│                   Network: airflow_net                   │
│                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │  PostgreSQL  │   │   Airflow    │   │   FastAPI   │  │
│  │  (metadata)  │   │  Webserver   │   │  Inference  │  │
│  │  port: 5432  │   │  port: 8080  │   │  port: 8000 │  │
│  └──────┬───────┘   └──────────────┘   └─────────────┘  │
│         │                                                │
│  ┌──────┴───────────────────────────────────────────┐   │
│  │           Airflow Scheduler + MLflow             │   │
│  │     MLflow co-located on port 5000               │   │
│  │     Airflow Scheduler manages DAG execution      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
              │                        │
         port 8080                port 5000
         (Airflow UI)             (MLflow UI)
```

### Key Design Decision: Co-located MLflow

MLflow runs **inside the Airflow Scheduler container** as a background process rather than as a separate service. This eliminates cross-container DNS resolution entirely — Airflow tasks connect to `http://localhost:5000`, which never requires a DNS lookup. This was the root-cause fix for recurring `NameResolutionError` failures on Docker Desktop for Windows where the userspace DNS resolver intermittently fails in task subprocess contexts (see [Troubleshooting](#9-troubleshooting)).

### AWS Deployment Architecture

```
Internet
    │
    ├── {API_PUBLIC_IP}:8000       FastAPI inference (ECS Fargate, 1 vCPU / 2 GB)
    ├── {SCHEDULER_PUBLIC_IP}:5000 MLflow UI         (ECS Fargate, 2 vCPU / 4 GB)
    └── {SCHEDULER_PUBLIC_IP}:8080 Airflow UI
                │
                ├── Amazon EFS   (/opt/airflow/data — master CSV shared across tasks)
                ├── Amazon S3    (MLflow model artifacts, persistent across restarts)
                └── Amazon ECR   (pm25-scheduler + pm25-api Docker images)
```

### Volume Strategy (Local)

| Volume | Type | Purpose |
|---|---|---|
| `mlruns_data` | Named (Docker-managed) | MLflow artifacts + SQLite DB — avoids NTFS permission errors |
| `postgres_data` | Named (Docker-managed) | Airflow metadata persistence |
| `./src` | Bind mount | Live code — changes reflected without rebuild |
| `./config` | Bind mount | `settings.py` — centralised config |
| `./dags` | Bind mount | `retrain_dag.py` — hot-reload by scheduler |
| `./data` | Bind mount | Raw + processed CSVs |

> **Note on `mlruns_data`:** This is a named volume, not a bind mount. Bind-mounting `./mlruns` from Windows causes SQLite `attempt to write a readonly database` errors due to NTFS uid/gid mismatch with the `airflow` user (uid 50000) inside the container.

---

## 2. Data Sources & Feature Engineering

### Data Sources

| Source | Type | Coverage | Auth |
|---|---|---|---|
| OpenAQ v3 API | Ground-truth PM2.5 sensor readings | 90 days (free tier) | `OPENAQ_API_KEY` |
| Open-Meteo Archive API | Weather (temp, humidity, wind, pressure) | ~4 years historical | None (free) |
| OWM Air Pollution API | Model-driven AQI supplement (`owm_*` prefix) | 365 days | `OWM_API_KEY` |
| WAQI Historical CSV | PM2.5 daily AQI Dec 2020–Jan 2025 | Static one-time import | Manual download |

### WAQI Data Processing

WAQI exports AQI index values rather than µg/m³ concentrations. A piecewise EPA conversion is applied at ingestion (e.g. AQI 101–150 maps to 35.5–55.4 µg/m³). Daily values are then disaggregated to hourly resolution using a Yangon-specific diurnal weighting pattern that reflects morning and evening traffic peaks and a midday thermal dispersal trough.

### Weekly Ingestion Pipeline

The Airflow DAG runs every Sunday at midnight UTC and fetches the latest 7 days from all live sources, appending to the master CSV with timestamp-based deduplication (last value wins on conflict).

### Feature Engineering

All features are computed in `src/preprocessing/features.py`:

| Feature Group | Columns | Notes |
|---|---|---|
| Lag features | `pm25_lag_1h/3h/6h/12h/24h/48h/72h` | Past observations — safe at inference time |
| Rolling stats | `pm25_roll_6h/12h/24h_mean/std/max` | `closed='left'` — current row excluded to prevent data leakage |
| Time features | `hour_of_day`, `day_of_week`, `month`, `is_weekend` | Raw integers for tree splits |
| Cyclical encoding | `sin_hour`, `cos_hour`, `sin_month`, `cos_month` | Maps temporal cycles onto unit circle |
| Interaction | `humidity_x_temp` | Physical dispersion proxy |
| OWM delta | `pm25_owm_delta = pm25 - owm_pm25` | Excluded from training — algebraically contains `pm25` and leaks the target |
| Target | `pm25.shift(-24)` | 24-hour forecast horizon |

---

## 3. ML Model & Registry

### Model

**Algorithm:** LightGBM Regressor  
**Target:** PM2.5 µg/m³ at T+24h  
**Training data:** ~37,000 rows after WAQI backfill + Open-Meteo 4-year weather history  
**Test split:** Time-based 80/20 — last 20% chronologically, no shuffle

**Hyperparameters:**

```python
{
    "n_estimators": 200,     "learning_rate": 0.05,
    "max_depth": 4,          "num_leaves": 15,
    "subsample": 0.7,        "colsample_bytree": 0.7,
    "min_child_samples": 20, "reg_alpha": 0.5,
    "reg_lambda": 2.0,       "n_jobs": 2,
    "random_state": 42,
}
```

> `n_jobs=2` rather than `-1` — using all cores hangs on Docker Desktop for Windows.

**Champion model metrics (version 8):**

| Metric | Value |
|---|---|
| Test RMSE | 10.53 µg/m³ |
| Test MAE | 7.45 µg/m³ |
| Test R² | 0.80 |
| CV mean RMSE | 15.24 ± 1.42 |

The gap between CV RMSE (15.24) and test RMSE (10.53) exists because CV folds span multiple seasonal windows while the holdout set is recent data. This gap will narrow once monsoon season data (May–October) accumulates via weekly DAG runs.

### MLflow Model Registry

All models are registered under the name `pm25_lightgbm` using the MLflow aliases API (MLflow ≥ 2.9 — stages API is deprecated):

| Alias | Meaning | Set by |
|---|---|---|
| `@champion` | Current production model served by FastAPI | `evaluate.py` on promotion |
| `@challenger` | Previous champion, retained for rollback | `evaluate.py` on promotion |

### Promotion Logic

A new model is promoted to `@champion` only if its test RMSE improves by at least **4%** over the current champion. If no champion exists (first deployment), the latest version is promoted unconditionally. The threshold was set at 4% rather than 5% because early champion versions were trained on limited data without weather features, making strict comparison against them counterproductive.

### Airflow DAG: `pm25_retrain_pipeline`

**Schedule:** Every Sunday at midnight UTC  
**Executor:** LocalExecutor (local) / SequentialExecutor (AWS, SQLite-backed)

```
fetch_data → preprocess → train → evaluate → promote_if_better
```

| Task | What it does |
|---|---|
| `fetch_data` | Pulls 7 days from OpenAQ + OWM + Open-Meteo, appends to master CSV |
| `preprocess` | Drops impossible values, forward-fills weather gaps ≤3h, rebuilds all features |
| `train` | Walk-forward CV (5 folds) + final LightGBM, logs run + artifact to MLflow |
| `evaluate` | Compares new model RMSE vs `@champion`, promotes if improvement ≥ 4% |
| `promote_if_better` | Logs structured weekly summary — placeholder hook for Slack/email alerts |

Each task has `retries=1`, `retry_delay=5min`, `execution_timeout=2h`. Failure at any stage skips all downstream tasks.

---

## 4. API Endpoints

The FastAPI service loads the `@champion` model from MLflow once at startup and caches it. All endpoints return structured JSON.

### `GET /health`

Liveness and readiness check. Returns `status: "ok"` when the model is loaded, `status: "degraded"` when not.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "champion",
  "loaded_at": "2026-03-30T19:00:00+00:00",
  "city": "Yangon",
  "mlflow_uri": "http://localhost:5000"
}
```

### `GET /predict`

Forecast PM2.5 24 hours ahead from the current UTC hour. No request body needed.

```bash
curl http://localhost:8000/predict
```

### `POST /predict`

Forecast from a specific timezone-aware timestamp. Rejects naive datetimes and timestamps more than 7 days in the future or before the training data cutoff (2020-12-01).

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"timestamp": "2026-03-30T14:00:00+00:00"}'
```

```json
{
  "city": "Yangon",
  "prediction_time": "2026-03-30T14:00:00+00:00",
  "forecast_time": "2026-03-31T14:00:00+00:00",
  "forecast_horizon_h": 24,
  "pm25_predicted": 38.67,
  "aqi_category": "Unhealthy for Sensitive Groups",
  "model_version": "champion",
  "feature_warnings": []
}
```

Interactive API docs available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.

---

## 5. Local Setup & Quickstart

### Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine + Compose (Linux)
- Python 3.10+ (for running scripts outside Docker)
- OpenAQ API key — register at [explore.openaq.org/register](https://explore.openaq.org/register)
- OpenWeatherMap API key — register at [openweathermap.org](https://openweathermap.org)

### 1. Clone and configure

```bash
git clone https://github.com/htet-khaing-win/air-pollution-forecast.git
cd air-pollution-forecast
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```
OPENAQ_API_KEY=your_key_here
OWM_API_KEY=your_key_here
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow
AIRFLOW_ADMIN_USERNAME=admin
AIRFLOW_ADMIN_PASSWORD=admin
AIRFLOW_WEBSERVER_SECRET_KEY=your_secret_key
MLFLOW_MODEL_NAME=pm25_lightgbm
```

### 2. Start all services

```bash
docker compose up -d --build
```

Wait approximately 45 seconds for all services to initialise (Airflow performs a DB migration and creates the admin user on first start).

### 3. Verify everything is running

```bash
bash e2e_test.sh
```

This checks container health, MLflow reachability from both host and API container, the `/health` endpoint, and a live `/predict` call.

### 4. Access the UIs

| Service | URL | Credentials |
|---|---|---|
| Airflow | `http://localhost:8080` | admin / admin |
| MLflow | `http://localhost:5000` | — |
| FastAPI docs | `http://localhost:8000/docs` | — |

### 5. Run a manual ingestion (optional)

```bash
docker compose exec scheduler python -m src.ingestion.ingest
```

### 6. Trigger the retraining DAG manually

Enable the `pm25_retrain_pipeline` DAG in the Airflow UI and trigger it, or run:

```bash
docker compose exec scheduler airflow dags trigger pm25_retrain_pipeline
```

### 7. Smoke test

```bash
docker compose exec scheduler python smoke_test.py
```

### Stopping and cleaning up

```bash
docker compose down          # stop containers, keep volumes
docker compose down -v       # stop containers and delete all volumes
```

---

## 6. AWS Deployment

The full system was deployed to AWS ECS Fargate (ap-southeast-2 / Sydney) and verified live. All resources have since been torn down. This section documents the architecture and procedure for reproducibility.

### Resource Inventory

| Resource | Name | Purpose |
|---|---|---|
| ECS Cluster | `pm25-cluster` | Hosts both Fargate services |
| ECS Service | `pm25-scheduler-svc` | Scheduler image, desired count 1 |
| ECS Service | `pm25-api-svc` | API image, desired count 1 |
| ECR Repository | `pm25-scheduler` | Scheduler Docker image |
| ECR Repository | `pm25-api` | API Docker image |
| S3 Bucket | `pm25-mlflow-artifacts-{ACCOUNT_ID}` | MLflow model artifact store |
| EFS Filesystem | `pm25-efs` | Shared `data/` CSV files across task restarts |
| Security Group | `pm25-ecs-sg` | Attached to both ECS tasks |
| Security Group | `pm25-efs-sg` | Attached to EFS mount targets |
| IAM Role | `pm25EcsExecutionRole` | ECS execution + task role (combined) |
| CloudWatch Log Group | `/ecs/pm25-scheduler` | Scheduler container logs |
| CloudWatch Log Group | `/ecs/pm25-api` | API container logs |

**Estimated cost for a 4-hour demo session: ~$0.76 USD total**

| Resource | Cost |
|---|---|
| ECS Fargate scheduler (2 vCPU, 4 GB, 4h) | ~$0.35 |
| ECS Fargate API (1 vCPU, 2 GB, 4h) | ~$0.09 |
| ECR storage (~3 GB images) | ~$0.30 |
| EFS, S3, CloudWatch | < $0.02 |

### Two Docker Images

**Image 1 — Scheduler (`pm25-scheduler`):** Runs MLflow tracking server (port 5000), Airflow webserver (port 8080), and Airflow scheduler in a single container. MLflow is co-located to eliminate cross-container DNS resolution — Airflow tasks connect to `http://localhost:5000`.

**Image 2 — API (`pm25-api`):** Runs the FastAPI uvicorn server (port 8000). Connects to MLflow via the scheduler task's **private IP** at port 5000, since Fargate tasks have no stable internal DNS without a load balancer or AWS Cloud Map.

### Storage Strategy on AWS

EFS is mounted at `/opt/airflow/data` in both containers for sharing master CSV files across task restarts. SQLite databases for MLflow and Airflow are stored in `/tmp` (local ephemeral container storage) because EFS (NFS-backed) does not support the POSIX file locks that SQLite requires — attempting to run SQLite on EFS causes an `OperationalError` on table creation.

MLflow model artifacts are stored in S3 (`s3://pm25-mlflow-artifacts-{ACCOUNT_ID}/mlruns/`), which persists across task restarts even though the MLflow run history DB does not.

### Scheduler Startup Sequence

Because `bash -c` string parsing treats `&` and `&&` with ambiguous precedence in JSON-encoded ECS command strings, the startup logic is written to `/tmp/start.sh` at runtime and executed as a proper shell script. The sequence is:

1. Create required directories (`/tmp/mlruns`, `/tmp/airflow/dags`, `/opt/airflow/data/raw`, `/opt/airflow/data/processed`)
2. Start MLflow server as a background process, pointing to `/tmp/mlruns/mlflow.db` and the S3 artifact bucket
3. Sleep 25 seconds to ensure MLflow finishes table creation before Airflow starts
4. Run `airflow db init` (not `db migrate` — Airflow 2.9.3 requires `db init` for fresh databases)
5. Create the admin user
6. Start Airflow webserver as a background process
7. Start Airflow scheduler in the foreground to keep the container alive

### Networking

Both ECS services are launched with `assignPublicIp=ENABLED`. Subnets must also have `MapPublicIpOnLaunch=true`. The security groups allow inbound traffic on ports 5000 (MLflow), 8080 (Airflow), and 8000 (FastAPI) from `0.0.0.0/0`, plus all traffic within the `pm25-ecs-sg` security group for internal task-to-task communication.

### IAM Role

`pm25EcsExecutionRole` is used as both `executionRoleArn` and `taskRoleArn`. It carries the following policies:
- `AmazonECSTaskExecutionRolePolicy`
- `AmazonEC2ContainerRegistryReadOnly`
- `AmazonS3FullAccess` (MLflow artifact read/write)
- `CloudWatchLogsFullAccess` (log streaming)

### API Task Definition Dependency

The API task definition must be registered **after** the scheduler service is running, because `MLFLOW_TRACKING_URI` in the API container must contain the scheduler task's actual private IP address. Without a load balancer or AWS Cloud Map service discovery, there is no stable hostname for Fargate task IPs.

**Deployment workflow:** launch scheduler → retrieve private IP from ECS task description → register API task definition with that IP → launch API service.

### Champion Model Upload Procedure

Because the MLflow database is ephemeral (`/tmp`), the champion model must be re-registered after every scheduler task restart:

1. Sync local `mlruns` artifacts to S3: `aws s3 sync mlruns s3://pm25-mlflow-artifacts-{ACCOUNT_ID}/mlruns`
2. Run a one-off ECS task against the scheduler task definition with a Python command override that calls `mlflow.register_model()` and `client.set_registered_model_alias()` to set the `@champion` alias
3. Verify in the MLflow UI at `http://{SCHEDULER_PUBLIC_IP}:5000`

### Verified Endpoints (Post-Deployment)

All four endpoints were confirmed live after full deployment:

| Endpoint | URL |
|---|---|
| FastAPI health | `http://{API_PUBLIC_IP}:8000/health` |
| FastAPI predict | `http://{API_PUBLIC_IP}:8000/predict` |
| FastAPI docs | `http://{API_PUBLIC_IP}:8000/docs` |
| MLflow UI | `http://{SCHEDULER_PUBLIC_IP}:5000` |
| Airflow UI | `http://{SCHEDULER_PUBLIC_IP}:8080` (admin / admin) |

### Teardown Order

Dependencies must be respected — deleting in the wrong order causes dependency errors:

1. Set desired count to 0 on both ECS services (stops billing immediately)
2. Delete `pm25-api-svc`, then `pm25-scheduler-svc`, then `pm25-cluster`
3. Delete ECR images then repositories for both `pm25-api` and `pm25-scheduler`
4. Empty the S3 bucket (`aws s3 rm --recursive`), then delete it
5. Delete EFS mount targets (one per subnet) — wait ~60 seconds for deletion to complete
6. Delete the EFS filesystem
7. Delete `pm25-ecs-sg`, then `pm25-efs-sg` (this order matters — `pm25-efs-sg` has an inbound rule referencing `pm25-ecs-sg` and will fail if `pm25-ecs-sg` still exists)
8. Detach all policies from `pm25EcsExecutionRole`, then delete the role
9. Delete CloudWatch log groups `/ecs/pm25-scheduler` and `/ecs/pm25-api`

---

## 7. Phase 7: Load Testing & Observability

Phase 7 added production-grade observability and validation to the `/predict` endpoint.

- **Load testing** with Locust targeting 50 RPS with p99 latency under 500ms
- **Prometheus metrics** instrumented on all endpoints: `predictions_total`, `prediction_latency_seconds`, `model_version` exposed via `/metrics`
- **Prediction drift detection** — rolling 7-day RMSE is monitored against the champion baseline; an alert triggers if degradation exceeds 15%
- **Data quality checks** in `clean.py` — alerts when more than 20% of a day's readings are missing

---

## 8. Environment Variables Reference

### Shared (both containers)

| Variable | Required | Description |
|---|---|---|
| `OPENAQ_API_KEY` | Yes | OpenAQ v3 API key |
| `OWM_API_KEY` | Yes | OpenWeatherMap API key |
| `MLFLOW_MODEL_NAME` | Yes | Registry name — must be `pm25_lightgbm` |

### Scheduler container

| Variable | Value | Notes |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | Co-located — no DNS needed |
| `MLFLOW_SERVER_ALLOWED_HOSTS` | `*` | Required to allow cross-origin MLflow UI access |
| `AIRFLOW__CORE__EXECUTOR` | `LocalExecutor` (local) / `SequentialExecutor` (AWS) | SQLite requires SequentialExecutor |
| `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` | PostgreSQL URI (local) / SQLite `/tmp` (AWS) | |
| `AIRFLOW__WEBSERVER__SECRET_KEY` | Set via env | Do not hardcode |
| `AIRFLOW_HOME` | `/tmp/airflow` | AWS only — keeps DB on local ephemeral storage |
| `PYTHONPATH` | `/opt/airflow` | |

### API container

| Variable | Value | Notes |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://scheduler:5000` (local) / `http://{PRIVATE_IP}:5000` (AWS) | |
| `MLFLOW_ARTIFACT_URI` | `http://scheduler:5000` | Enables HTTP artifact proxy |
| `no_proxy` / `NO_PROXY` | `scheduler,postgres,localhost,127.0.0.1` | Bypasses system proxies for internal networking |
| `PYTHONPATH` | `/app` | |

---

## 9. Troubleshooting

### `NameResolutionError` — MLflow hostname not resolving in Airflow tasks

**Symptom:** Airflow train or evaluate task fails with alternating `[Errno -5] No address associated with hostname` and `[Errno -2] Name or service not known` when resolving the `mlflow` service name. The alternating errno codes are the fingerprint of DNS flapping.

**Root cause:** Docker Desktop for Windows uses a userspace DNS resolver at `127.0.0.11`. Airflow's `LocalExecutor` runs each task as a subprocess — not in-process — and that subprocess performs a fresh DNS lookup for the `mlflow` service name. On Windows, the userspace resolver intermittently fails in subprocess contexts even when the network configuration is correct.

**Resolution:** MLflow is co-located inside the scheduler container. Tasks connect to `http://localhost:5000` — `localhost` never requires a DNS lookup.

### `WinError 10061` — Connection actively refused

**Symptom:** `[WinError 10061] No connection could be made because the target machine actively refused it` when running training or inference scripts locally.

**Resolution:** MLflow must be running before any script that calls it. Start it manually for local development: `mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns/artifacts`

### `OSError` — Linux artifact paths on Windows

**Symptom:** `No such file or directory: '\opt\airflow\mlruns\artifacts\...'` when the FastAPI service tries to load the `@champion` model outside Docker.

**Root cause:** MLflow records artifact URIs as absolute paths at training time. Models trained inside Docker record Linux paths (`/opt/airflow/...`), which Windows cannot resolve.

**Resolution:** The FastAPI service is containerised and accesses `mlruns_data` named volume directly — Linux paths resolve correctly inside the container. For local development outside Docker, retrain locally so artifacts are recorded as native paths. Long-term: migrate artifact store to S3.

### SQLite `attempt to write a readonly database`

**Symptom:** MLflow crashes on startup with `sqlite3.OperationalError: attempt to write a readonly database`.

**Root cause:** `./mlruns` was bind-mounted from Windows. NTFS assigns ownership to the Windows user; the `airflow` user (uid 50000) inside the container has no write access.

**Resolution:** Use a Docker-managed named volume (`mlruns_data`) instead of a bind mount. Named volumes are stored on Docker Desktop's Linux VM (ext4) and are owned by the `airflow` user by default.

### `AirflowConfigException` — LocalExecutor incompatible with SQLite (AWS)

**Symptom:** `airflow.exceptions.AirflowConfigException: error: cannot use SQLite with the LocalExecutor`

**Root cause:** Airflow 2.9.3 added a hard compatibility check. `LocalExecutor` requires PostgreSQL or MySQL because it runs tasks as concurrent subprocesses with simultaneous DB access.

**Resolution:** Set `AIRFLOW__CORE__EXECUTOR=SequentialExecutor`. It runs one task at a time in-process and accepts SQLite as the metadata backend. Sufficient for this DAG (linear 5-task chain, `max_active_runs=1`, weekly schedule).

### `SQLite OperationalError` on EFS (AWS)

**Symptom:** `unable to open database file` crash at `CREATE TABLE logged_models` during MLflow init, or at `airflow db init`.

**Root cause:** EFS is NFS-backed. SQLite requires POSIX advisory file locks (`fcntl`/`flock`), which NFS does not implement atomically. AWS documentation explicitly states EFS is incompatible with SQLite.

**Resolution:** Move all SQLite databases to `/tmp` (local ephemeral container storage). EFS is only used for `data/` CSV files, which use sequential append writes with no locking.

### ECS tasks launched with no public IP (AWS)

**Symptom:** `InvalidNetworkInterfaceID.NotFound` immediately after a task reaches `RUNNING`. The ENI was deallocated before it could be queried because the task stopped instantly.

**Root cause:** Two issues compounding — the subnet did not have `MapPublicIpOnLaunch` enabled, and the ECS service was created without `assignPublicIp=ENABLED`.

**Resolution:** Run `modify-subnet-attribute --map-public-ip-on-launch` on all subnets before creating any ECS service, and always pass `assignPublicIp=ENABLED` explicitly in `create-service` and `run-task` network configuration.

### `You need to initialize the database` despite running `db migrate` (AWS)

**Symptom:** Airflow crashes with `ERROR: You need to initialize the database. Please run airflow db init.`

**Root cause:** `airflow db migrate` upgrades an existing schema. For a brand-new empty SQLite file, Airflow 2.9.3 requires `db init`. Additionally, the startup sleep was not executing before `db init` because `bash -c` string parsing treated `&` after `mlflow server` with lower precedence than `&&`.

**Resolution:** Use `airflow db init` (not `db migrate`) for fresh databases. Write startup logic to `/tmp/start.sh` via `printf` and execute the script file — in a real script, line-by-line execution is unambiguous.

---

## 10. Project Structure

```
air_pollution_forecast/
├── Dockerfile                     # Scheduler image: Airflow + MLflow + all ML deps
├── Dockerfile.api                 # API image: FastAPI + inference deps
├── docker-compose.yaml            # All services + airflow_net bridge
├── requirements.txt               # lightgbm, mlflow, fastapi, optuna, boto3...
├── .env.example                   # Template — copy to .env and fill in secrets
├── e2e_test.sh                    # End-to-end local verification script
├── smoke_test.py                  # Quick import + config verification
│
├── dags/
│   └── retrain_dag.py             # Weekly retraining DAG (5-task pipeline)
│
├── config/
│   └── settings.py                # Centralised config — reads from environment
│
├── src/
│   ├── ingestion/
│   │   ├── ingest.py              # 4-source orchestrator: fetch → merge → save
│   │   ├── openaq.py              # OpenAQ v3 client (sensor locations + measurements)
│   │   ├── openmeteo.py           # Open-Meteo archive client (weather history)
│   │   ├── openweather.py         # OWM air pollution history + weather forecast
│   │   ├── waqi_historical.py     # WAQI CSV → hourly AQI → µg/m³ conversion
│   │   └── backfill.py            # One-time 4-year historical pull
│   │
│   ├── preprocessing/
│   │   ├── clean.py               # Validation, forward-fill, outlier removal
│   │   └── features.py            # Lag, rolling, time, and interaction features
│   │
│   ├── training/
│   │   ├── train.py               # LightGBM + walk-forward CV + MLflow logging
│   │   ├── evaluate.py            # Champion comparison + alias promotion
│   │   ├── tune.py                # Optuna hyperparameter search
│   │   └── ab_test.py             # 5-model A/B comparison runner
│   │
│   ├── inference/
│   │   └── predict.py             # Live feature construction + champion model inference
│   │
│   └── api/
│       └── app.py                 # FastAPI: /health, GET /predict, POST /predict
│
└── data/
    ├── raw/
    │   ├── yangon_pollution.csv   # Master CSV (all sources merged, deduplicated)
    │   └── waqi_yangon.csv        # Static WAQI historical download
    └── processed/
        ├── cleaned.csv
        ├── features.csv
        └── feature_importance.csv
```

---

## 11. Roadmap

The following improvements were identified during development and AWS deployment:

- **Persistent MLflow backend:** Migrate from SQLite/`/tmp` to Amazon RDS PostgreSQL so that MLflow run history survives task restarts
- **AWS Cloud Map service discovery:** Give Fargate tasks stable DNS names, eliminating the manual private-IP lookup step when registering the API task definition
- **Application Load Balancer:** Place ALB in front of both services for stable public URLs that survive task replacement and enable HTTPS
- **Infrastructure as Code:** Replace manual AWS CLI deployment steps with Terraform
- **Prometheus + CloudWatch alarms:** Add a `/metrics` endpoint and wire prediction latency and model drift detection to CloudWatch alarms
- **Amazon MWAA:** Replace the co-located Airflow + SQLite setup with Amazon Managed Workflows for Apache Airflow for a managed, persistent, production-grade scheduler
- **Prediction intervals:** Add quantile regression or conformal prediction to return uncertainty bounds alongside point estimates
- **Monsoon coverage:** The current training data has a gap for May–October monsoon months (May–Oct 2025 is absent). Model accuracy during monsoon season will improve as weekly DAG runs accumulate data through 2026

---

## Known Limitations

- **Monsoon data gap:** Training covers Dec 2020–Jan 2025 (WAQI) + Dec 2025–Mar 2026 (OpenAQ). May–October monsoon season is underrepresented — model accuracy will be lower during monsoon months until data accumulates
- **OpenAQ 90-day cap:** Free tier limits historical pulls to 90 days; ground-truth data grows by 7 days per week via the Airflow DAG
- **Artifact path portability:** Models trained inside Docker cannot be loaded outside Docker without the `mlruns_data` volume, until artifacts are migrated to S3
- **No prediction intervals:** The model returns point estimates only — uncertainty quantification is not yet implemented
- **Ephemeral MLflow history (AWS):** MLflow run history does not persist across scheduler task restarts in the AWS deployment; model artifacts persist because they live in S3

---

*City: Yangon, Myanmar | Model: LightGBM PM2.5 regressor | Forecast horizon: 24 hours | MLflow experiment: `air_pollution_forecast`*
