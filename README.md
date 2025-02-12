# Clock Sync via HOM Interference Simulation Project

This project provides a simulation toolkit for quantum optical experiments focused on Hong-Ou-Mandel (HOM) interference in fiber-optic networks. The simulation allows users to study and optimize configurations by adjusting source parameters, MDL distances, and polarization states, leveraging optimization algorithms tailored for noisy objective functions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Simulation](#running-the-simulation)
  - [Standard Run](#standard-run)
  - [High-Performance Cluster Setup](#high-performance-cluster-setup)
- [Project Structure](#project-structure)
- [Examples and Usage](#examples-and-usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The simulation models a phase-based experiment where:
- **Phase 1:** Alice sends signals to Bob.
- **Phase 2:** The roles are reversed (Bob sends to Alice) to capture any asymmetries in the system.

The optimization routines (e.g., Golden Section Search, Gradient Descent, SPSA) are integrated to find the optimal measurement device length (MDL) that minimizes the coincidence probability, an indicator of interference quality.

## Features

- **Configurable Experiment Parameters:** Set up source type, repetition rates, delays, and offsets for both Alice and Bob.
- **Optimization Algorithms:** 
  - Golden Section Search for fine-tuning the MDL distance.
  - Gradient Descent and SPSA methods suited for handling noise.
- **Parallel Processing:** Uses Joblib for parallel evaluations, speeding up objective function evaluations.
- **Visualization:** Provides plotting tools to visualize convergence and simulation results.
- **HPC Compatibility:** Easily adapted to run on high-performance computing clusters with parallelized routines and an egr parser for YAML configurations.
- **Phase-based Simulations:** Supports running sequential simulations for both phase 1 and phase 2 (swapping roles between Alice and Bob).

## Installation

### Prerequisites

- Python 3.8 or higher
- `pip` package manager

### Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/clock_sync_sim.git
   cd clock_sync_sim
```

### Run Code
```bash
python scripts/run_simulation.py --config ./config/config.yaml
```
