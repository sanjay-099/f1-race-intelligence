# AGENTS.md — F1 Race Intelligence System

## Environment
- ALWAYS use the f1-env virtual environment for all Python commands
- Activate with: source f1-env/bin/activate
- The env is located at: ./f1-env/ (inside the project folder)
- NEVER install packages in base/system Python
- NEVER use the DS environment for this project
- Verify env is active (look for (f1-env) in terminal) before any pip install
- Python 3.10+ required

## Project Type
- Data Science / ML / Streamlit application
- Uses FastF1 for Formula 1 telemetry data
- Build target: deployable Docker container on AWS

## Code Standards
- Type hints required on all function signatures
- Docstrings required on all public methods
- Follow PEP 8 strictly
- All ML experiments tracked locally (and later MLflow)

## Workflow
- Ask before running pip install commands
- Show plan before making multi-file changes
- Commit after every working feature

## Data
- Never commit large datasets to git
- FastF1 cache goes in data/cache/ (gitignored)
- Models saved to models/ (gitignored) 