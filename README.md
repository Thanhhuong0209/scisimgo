# SciSimGo: Scientific Simulation Engine with Data Science Integration

[![Go Version](https://img.shields.io/badge/Go-1.21+-blue.svg)](https://golang.org)
[![Python Version](https://img.shields.io/badge/Python-3.9+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Go Build](https://img.shields.io/badge/Go-Build%20Passing-brightgreen.svg)](https://github.com/Thanhhuong0209/scisimgo/actions)
[![CI/CD Status](https://github.com/Thanhhuong0209/scisimgo/workflows/SciSimGo%20CI%2FCD%20Pipeline/badge.svg)](https://github.com/Thanhhuong0209/scisimgo/actions)

> A comprehensive simulation framework combining Go-based computational engines with advanced data analysis pipelines for scientific research and educational applications.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Thanhhuong0209/scisimgo.git
cd scisimgo

# Run simulations
go run cmd/sir-simulator/main.go --population 10000 --initial-infected 100 --duration 100s
go run cmd/predator-prey/main.go --initial-prey 1000 --initial-predator 100 --duration 200s
go run cmd/orbital-sim/main.go --time-step 1000 --duration 300s

# Analyze data
python notebooks/sir-analysis/sir_analysis.py
```

## Features

- **High-Performance Go Engine**: Concurrent simulation with goroutines and channels
- **Scientific Models**: SIR epidemiology, Lotka-Volterra ecology, Orbital mechanics
- **Data Science Pipeline**: EDA, statistics, machine learning, visualization
- **Real-time Export**: CSV/JSON data export for analysis
- **Containerized**: Docker support for easy deployment
- **CI/CD Ready**: Automated testing and deployment workflows

## Abstract

SciSimGo represents an integrated approach to scientific simulation and data analysis, designed to bridge the gap between computational modeling and empirical data science. The framework provides a robust foundation for studying complex systems through three primary simulation domains: epidemiological dynamics (SIR models), ecological interactions (Lotka-Volterra predator-prey systems), and celestial mechanics (orbital dynamics). This project demonstrates the synergy between high-performance Go concurrency and Python-based analytical methodologies.

## Research Objectives

The primary research objectives of SciSimGo include:

- **Computational Modeling**: Development of accurate mathematical models for complex biological and physical systems
- **Data Generation**: Systematic production of synthetic datasets for algorithm validation and educational purposes
- **Analytical Integration**: Implementation of comprehensive data science workflows including exploratory data analysis, statistical modeling, and machine learning applications
- **Educational Framework**: Creation of an accessible platform for computational science education and research methodology training

## Technical Architecture

### Core Simulation Engine (Go)

The simulation engine is built upon Go's concurrency primitives, providing:

- **Goroutine-based Parallelization**: Efficient concurrent execution of multiple simulation agents
- **Channel-based Communication**: Robust inter-process communication for complex system interactions
- **Real-time Data Export**: Continuous output generation in standardized formats (CSV/JSON)
- **Modular Design**: Extensible architecture supporting multiple simulation paradigms

### Data Science Pipeline (Python)

The analytical framework incorporates:

- **Statistical Analysis**: Comprehensive descriptive and inferential statistical methodologies
- **Machine Learning Integration**: Implementation of regression, classification, and time series analysis algorithms
- **Visualization Framework**: Advanced plotting capabilities using Matplotlib, Seaborn, and Plotly
- **Reproducible Research**: Jupyter notebook integration for transparent and reproducible analysis

## Methodology

### Simulation Framework

The simulation methodology follows established computational science principles:

1. **Model Initialization**: Parameter specification and initial condition establishment
2. **Temporal Evolution**: Iterative computation using appropriate numerical integration methods
3. **Data Collection**: Systematic sampling of system states at specified intervals
4. **Output Generation**: Standardized data export for subsequent analysis

### Analytical Approach

The data analysis methodology encompasses:

1. **Exploratory Data Analysis (EDA)**: Initial data exploration and pattern identification
2. **Statistical Modeling**: Application of appropriate statistical frameworks for hypothesis testing
3. **Machine Learning**: Implementation of supervised and unsupervised learning algorithms
4. **Model Validation**: Cross-validation and performance assessment using multiple metrics

## Implementation Details

### Project Structure

```
SciSimGo/
├── cmd/                    # Executable applications
│   ├── sir-simulator/     # Epidemiological simulation engine
│   ├── predator-prey/     # Ecological dynamics simulator
│   └── orbital-sim/       # Celestial mechanics engine
├── internal/              # Core implementation packages
│   ├── models/            # Mathematical model implementations
│   ├── engine/            # Simulation framework core
│   └── export/            # Data export utilities
├── data/                  # Generated simulation datasets
├── notebooks/             # Analytical workflows
│   ├── sir-analysis/      # Epidemiological data analysis
│   ├── predator-prey/     # Ecological data analysis
│   └── orbital-analysis/  # Orbital mechanics analysis
├── scripts/               # Automation and utility scripts
├── docs/                  # Documentation and research notes
├── go.mod                 # Go module dependencies
├── requirements.txt       # Python package specifications
└── README.md             # Project documentation
```

### Simulation Models

#### 1. SIR Epidemiological Model

The Susceptible-Infected-Recovered (SIR) model implements classical epidemiological dynamics:

- **State Variables**: S (susceptible), I (infected), R (recovered) populations
- **Parameters**: β (infection rate), γ (recovery rate), N (total population)
- **Differential Equations**: Standard SIR formulation with discrete time approximation
- **Applications**: Disease spread modeling, intervention effectiveness assessment

#### 2. Lotka-Volterra Predator-Prey Dynamics

Implementation of classical ecological interaction models:

- **State Variables**: Prey and predator population densities
- **Parameters**: Growth rates, predation efficiency, mortality rates
- **Dynamics**: Coupled differential equations describing population oscillations
- **Applications**: Ecosystem stability analysis, conservation biology

#### 3. Orbital Mechanics Simulation

Celestial body dynamics under gravitational interactions:

- **Physical Framework**: Newtonian gravitational mechanics
- **State Variables**: Position, velocity, and mass of celestial bodies
- **Numerical Methods**: Verlet integration for orbital trajectory computation
- **Applications**: Solar system modeling, orbital dynamics research

## Installation and Usage

### System Requirements

- **Go**: Version 1.21 or higher
- **Python**: Version 3.9 or higher
- **Dependencies**: See requirements.txt for complete Python package specifications

### Installation Procedure

```bash
# Clone repository
git clone <repository-url>
cd SciSimGo

# Install Go dependencies
go mod tidy

# Install Python dependencies
pip install -r requirements.txt
```

### Execution Examples

```bash
# Epidemiological simulation
go run cmd/sir-simulator/main.go --population 10000 --initial-infected 100 --infection-rate 0.3 --recovery-rate 0.1 --duration 100s --output data/sir

# Ecological dynamics simulation
go run cmd/predator-prey/main.go --initial-prey 1000 --initial-predator 100 --prey-growth-rate 0.1 --predation-rate 0.01 --predator-death-rate 0.1 --conversion-efficiency 0.1 --duration 200s --output data/predator-prey

# Orbital mechanics simulation
go run cmd/orbital-sim/main.go --time-step 1000 --enable-3d false --duration 300s --output data/orbital
```

## Data Science Workflow

### Analytical Pipeline

The comprehensive analytical workflow includes:

1. **Data Preprocessing**: Data cleaning, normalization, and feature engineering
2. **Statistical Analysis**: Descriptive statistics, correlation analysis, and hypothesis testing
3. **Machine Learning**: Supervised learning algorithms for prediction and classification
4. **Model Validation**: Cross-validation, performance metrics, and error analysis
5. **Visualization**: Publication-quality figures and interactive dashboards

### Available Analyses

- **Epidemiological Analysis**: Disease progression modeling, intervention impact assessment
- **Ecological Analysis**: Population dynamics, stability analysis, bifurcation studies
- **Orbital Analysis**: Trajectory analysis, energy conservation, orbital parameter estimation

## Research Applications

### Educational Applications

- **Computational Science Education**: Introduction to simulation and modeling
- **Data Science Training**: Practical application of statistical and machine learning methods
- **Research Methodology**: Demonstration of reproducible research practices

### Research Applications

- **Model Validation**: Testing theoretical predictions against computational results
- **Parameter Sensitivity**: Analysis of model robustness and uncertainty quantification
- **Algorithm Development**: Framework for testing new computational methods
- **Data Generation**: Production of synthetic datasets for algorithm benchmarking

## Future Directions

### Planned Enhancements

- **Advanced Numerical Methods**: Implementation of adaptive time-stepping and higher-order integration schemes
- **Parallel Computing**: GPU acceleration and distributed computing capabilities
- **Real-time Visualization**: Interactive simulation monitoring and control interfaces
- **Model Calibration**: Automated parameter estimation from empirical data
- **Uncertainty Quantification**: Comprehensive error analysis and confidence interval estimation

### Research Extensions

- **Multi-scale Modeling**: Integration of microscopic and macroscopic simulation levels
- **Machine Learning Integration**: Neural network-based model discovery and parameter estimation
- **High-dimensional Systems**: Extension to complex systems with many interacting components

## Contributing

This project welcomes contributions from the research and educational communities. Areas of particular interest include:

- **Model Development**: Implementation of additional mathematical models
- **Algorithm Optimization**: Performance improvements and numerical method enhancements
- **Documentation**: Research methodology documentation and educational materials
- **Testing and Validation**: Comprehensive testing frameworks and validation studies

## License

This project is licensed under the MIT License. See the LICENSE file for complete details.

## Citation

If you use this software in your research, please cite:

```
SciSimGo: Scientific Simulation Engine with Data Science Integration
[Your Name/Institution], [Year]
```

---

**SciSimGo** - Advancing computational science through integrated simulation and analysis.
