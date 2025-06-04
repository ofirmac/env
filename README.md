# Guest Environment RL Experiments

This repository contains a simple reinforcement learning environment along with
training scripts and various experiments. It was reorganised to follow a more
standard project layout.

## Project Structure

- `guest_env/` – Python package containing the environment implementation.
- `scripts/` – Stand‑alone training script using the environment.
- `examples/` – Example scripts used to solve or visualise the environment.
- `plots/` – Generated figures demonstrating different heuristics.

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the training script:

```bash
python scripts/train.py
```

See the files in `examples/` for additional usage examples.
