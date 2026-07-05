---
title: F1 Race Intelligence
emoji: 🏎️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---
# 🏎️ F1 Race Intelligence System

> Production-grade Formula 1 analytics platform — FastF1 telemetry + XGBoost + Multi-Agent LLM

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange)](https://xgboost.readthedocs.io)
[![FastF1](https://img.shields.io/badge/FastF1-3.8+-blue)](https://theoehrly.github.io/Fast-F1/)
[![Python](https://img.shields.io/badge/Python-3.13-yellow)](https://python.org)

---

## 🎯 What This Is

An end-to-end F1 analytics platform that ingests raw FIA telemetry, applies multiple ML models, and surfaces insights through an interactive web dashboard. Built as a portfolio project demonstrating production-grade data science engineering.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  FastF1 API  │  │  ML Models   │  │  Claude AI   │  │
│  │  (FIA Data)  │  │  (XGBoost +  │  │  Multi-Agent │  │
│  │  2018-2026   │  │   sklearn)   │  │   Analyst    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         └─────────────────┴─────────────────┘           │
│                            │                             │
│                    ┌───────▼───────┐                     │
│                    │   Jinja2 +    │                     │
│                    │   Plotly.js   │                     │
│                    │   Frontend    │                     │
│                    └───────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
FIA Timing Stream (FastF1)
        │
        ▼
Session Cache (in-memory + 10.5 GB disk)
        │
        ├──► Tyre Degradation Model (XGBoost)      →  R²=0.962, MAE=1.229s
        ├──► Quali→Race Predictor (Delta Lookup)   →  1,105 historical samples
        ├──► Incident Detector (IsolationForest ×2) →  ~5% contamination threshold
        ├──► Driver Embeddings (PCA, 8 features)   →  44 drivers indexed
        ├──► Monte Carlo Simulator                 →  500+ iterations
        └──► Multi-Agent LLM (Claude Sonnet 4.6)   →  3 agents + coordinator
```

---

## 📊 Validated Model Performance

| Model | Metric | Value | Training Data |
|-------|--------|-------|---------------|
| Tyre Degradation | R² | 0.962 | 170,351 race laps |
| Tyre Degradation | MAE | 1.229s | 2018-2026, 31 circuits |
| Quali→Race Prediction | MAE | ~2.3 positions | 1,105 historical samples |
| Driver Embeddings | PCA Variance | 53.7% | 44 drivers, 8 features |

---

## 🧠 ML Models

### 1. Tyre Degradation (XGBoost Regressor)
Predicts lap time as a function of tyre age, compound, circuit, and fuel load.

**Features:** `TyreLife`, `TyreLifeSquared`, `Compound`, `Driver`, `Circuit`, `LapNumber`, `RegEra`

**Why XGBoost:** Tabular data with 170k rows + heterogeneous features. Gradient boosting consistently outperforms neural approaches at this scale on tabular data.

### 2. Qualifying → Race Position Predictor
Predicts race finishing order from qualifying data using historical position delta patterns.

**Approach:** Per-grid-slot historical mean delta lookup across 1,105 driver-race entries. P1 starters historically finish ~P3 on average; P20 starters historically finish ~P16. This domain-knowledge approach outperforms regression at this sample size.

### 3. Incident Detection (Dual IsolationForest)
Two separate models — race/sprint and practice/qualifying — because lap time distributions differ significantly between session types.

**Why unsupervised:** No reliable ground-truth incident labels exist across 8 seasons of FIA data.

### 4. Driver Style Embeddings (PCA)
8 telemetry-derived features: `brake_aggression`, `throttle_smoothness`, `throttle_attack`, `top_speed`, `coasting_ratio`, `gear_aggression`, `consistency`, `tyre_management`.

**Production pattern:** Saved scaler + PCA allows projecting new drivers without retraining.

### 5. Multi-Agent LLM Analyst (Claude Sonnet 4.6)
Three specialist agents run in parallel (Pace, Strategy, Race Craft), then a coordinator synthesizes their outputs.

---

## 🗂️ Project Structure

```
f1-race-intelligence/
├── src/
│   ├── main.py                        # FastAPI app — 3,500+ lines
│   ├── tabnet_model.py                # ML predictors
│   ├── race_context.py                # LLM context builder
│   ├── race_schedule.py               # Schedule helpers
│   └── templates/                     # 17 Jinja2 HTML templates
├── data/
│   ├── all_races_2018_2026.parquet    # 325,983 laps, 4.1 MB
│   └── driver_embeddings_alltime.json # 44 drivers
├── models/
│   ├── tire_degradation_model.pkl     # XGBoost, 2.75 MB
│   ├── incident_isolation_forest.pkl  # Race/sprint
│   └── ...
├── notebooks/                         # 5 EDA + training notebooks
├── docs/
│   └── F1_Race_Intelligence_Methodology.pdf
├── Dockerfile
└── requirements.txt
```

---

## 🚀 Platform Features

| Feature | Description |
|---------|-------------|
| **Race Analysis** | Telemetry, incidents, strategy, pace heatmaps |
| **Predictions** | Auto-detects mode: FP pace ranking → Quali→Race → Post-race |
| **Multi-Agent Analyst** | Quick (single Claude) or Deep (3 agents + coordinator) |
| **Monte Carlo Simulation** | 500+ iteration race strategy optimizer |
| **Counterfactual Analysis** | What-if grid position, tyre, pit stop scenarios |
| **Driver Embeddings** | Telemetry-based style fingerprints, cosine similarity |
| **Circuit & Constructor Intel** | Historical analysis across 31 circuits, 2018-2026 |
| **Pipeline Monitor** | Data freshness, model health, cache status |

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Jinja2 |
| ML | XGBoost, scikit-learn (IsolationForest, PCA, Ridge) |
| Data | FastF1, pandas, numpy |
| Visualization | Plotly.js |
| LLM | Anthropic Claude Sonnet 4.6 |
| Frontend | Tailwind CSS |
| Deployment | Docker + HuggingFace Spaces |

---

## 🏃 Running Locally

```bash
git clone https://github.com/sanjay-099/f1-race-intelligence.git
cd f1-race-intelligence

python3 -m venv f1-env
source f1-env/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add ANTHROPIC_API_KEY to .env

python3 -m uvicorn src.main:app --reload --port 7860
open http://127.0.0.1:7860
```

---

## 🌐 API

Full Swagger docs at `/docs` when running locally.

```
GET  /api/predictions/{year}/{round}     Auto-detect race prediction mode
POST /analyst/ask                        Multi-agent LLM analysis
GET  /api/circuit-intelligence/{circuit} Historical circuit analysis
GET  /api/driver-embeddings/{year}       Driver style embeddings
GET  /api/pipeline-status               System health check
POST /api/montecarlo                     Race strategy simulation
POST /api/counterfactual                 What-if scenario analysis
```

---

## 📋 Known Limitations

- **Race prediction noise floor:** F1 outcomes are dominated by safety cars/DNFs — MAE ~2.3 positions is a fundamental floor, not a model failure
- **Tyre model in-sample validated:** R²=0.962 on full training set; out-of-sample will be lower
- **2026 regulation era:** Models trained on 2018-2025 data; performance improves as 2026 data accumulates
- **Ergast API deprecated:** jolpi.ca mirror unreliable; system falls back to FastF1 cache gracefully

---

## 👨‍💻 About

Built by **Sanjay Chowdary** — MS Data Science, University of Alabama at Birmingham

📄 [Technical Methodology](docs/F1_Race_Intelligence_Methodology.pdf)

*FastAPI · XGBoost · FastF1 · Claude AI · Docker*
