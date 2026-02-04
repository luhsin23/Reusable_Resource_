# Reusable_Resource_

# Multi-Policy Network Simulator

This repository contains a discrete-event simulation framework designed to evaluate job admission control policies across different network topologies. The system compares three primary strategies: **PTP** (Decentralized Policy), **CHP** (Centralized Policy), and **SP** (Strict Policy).

## 🏗 Project Architecture

The simulation is modularized into four main components:

### 1. Configuration & Parameters (`parameters.py`)
This file defines the mathematical properties of the networks:
* **Linear**: A sequential resource dependency model.
* **2H (2-Tier Hierarchical)**: A hierarchical structure where resources are shared across tiers.
* **Non-Hierarchical**: A specific 4-node network configuration.
* **Constants**: Defines arrival rates ($\Lambda$), service rates ($\mu^{-1}$), resource thresholds ($q$), and rewards ($r$).

### 2. Data Generation (`gen_data.py`)
To ensure a fair comparison, job traces are pre-generated before the simulation begins:
* Generates synthetic arrival and service times using exponential distributions.
* Utilizes `ProcessPoolExecutor` for high-performance parallel generation.
* Organizes data into folders by network type, scaling factor ($N$), and dimension ($d$).

### 3. Simulation Engine (`sim.py`)
The core engine that processes events and applies policy logic:
* **Event Handling**: Uses `heapq` (priority queues) to manage job departures and update system state ($X$) in real-time.
* **Policy Evaluation**: For every job arrival, it queries the decision logic (PTP, CHP, or SP) to determine if the job should be admitted.
* **Metrics**: Tracks cumulative average rewards and system headcounts (occupancy) across multiple epochs.

### 4. Driver Scripts (`driver_*.py`)
These are the entry points for running the experiment pipeline:
* **`driver_linear.py`**: Executes the pipeline for linear networks with specific $N$ ranges and epoch counts.
* **`driver_2H.py`**: Executes the pipeline for hierarchical (2H) networks.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.x
* NumPy (for numerical operations)
* Matplotlib (for visualization)

### Usage
To run a complete simulation (data generation + policy evaluation) for a specific network type, execute the corresponding driver:

**For Linear Networks:**
```bash
python driver_linear.py
