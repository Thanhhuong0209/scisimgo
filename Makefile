# SciSimGo - Scientific Simulation & Data Science Playground
# Makefile for easy project management

.PHONY: help install build test run clean analysis all

# Default target
help:
	@echo "SciSimGo - Scientific Simulation & Data Science Playground"
	@echo "=================================================="
	@echo ""
	@echo "Available targets:"
	@echo "  help        - Show this help message"
	@echo "  install     - Install Go and Python dependencies"
	@echo "  build       - Build all Go executables"
	@echo "  test        - Run Go tests"
	@echo "  run         - Run all simulations"
	@echo "  analysis    - Run data analysis"
	@echo "  clean       - Clean build artifacts and data"
	@echo "  all         - Install, build, test, and run everything"
	@echo ""
	@echo "Individual simulation targets:"
	@echo "  sir         - Run SIR disease model simulation"
	@echo "  predator    - Run Predator-Prey model simulation"
	@echo "  orbital     - Run Orbital mechanics simulation"
	@echo ""

# Install dependencies
install:
	@echo "Installing dependencies..."
	@echo "Installing Go dependencies..."
	go mod tidy
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

# Build Go executables
build:
	@echo "Building Go executables..."
	@echo "Building SIR simulator..."
	go build -o bin/sir-simulator cmd/sir-simulator/main.go
	@echo "Building Predator-Prey simulator..."
	go build -o bin/predator-prey cmd/predator-prey/main.go
	@echo "Building Orbital simulator..."
	go build -o bin/orbital-sim cmd/orbital-sim/main.go
	@echo "All executables built successfully!"

# Run Go tests
test:
	@echo "Running Go tests..."
	go test -v ./...
	@echo "Tests completed!"

# Run tests with coverage
test-coverage:
	@echo "Running Go tests with coverage..."
	go test -v -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

# Run tests with race detection
test-race:
	@echo "Running Go tests with race detection..."
	go test -race -v ./...
	@echo "Race detection tests completed!"

# Run tests with benchmarks
test-bench:
	@echo "Running Go tests with benchmarks..."
	go test -bench=. -benchmem ./...
	@echo "Benchmark tests completed!"

# Run all simulations
run:
	@echo "Running all simulations..."
	python scripts/run_all_simulations.py
	@echo "All simulations completed!"

# Run data analysis
analysis:
	@echo "Running data analysis..."
	python notebooks/sir-analysis/sir_analysis.py
	@echo "Data analysis completed!"

# Clean build artifacts and data
clean:
	@echo "Cleaning project..."
	rm -rf bin/
	rm -rf data/
	rm -f *.png
	rm -f simulation_report.md
	go clean
	@echo "Project cleaned!"

# Run SIR simulation
sir:
	@echo "Running SIR disease model simulation..."
	go run cmd/sir-simulator/main.go --population 10000 --initial-infected 100 --infection-rate 0.3 --recovery-rate 0.1 --duration 100s --output data/sir
	@echo "SIR simulation completed!"

# Run Predator-Prey simulation
predator:
	@echo "Running Predator-Prey model simulation..."
	go run cmd/predator-prey/main.go --initial-prey 1000 --initial-predator 100 --prey-growth-rate 0.1 --predation-rate 0.01 --predator-death-rate 0.1 --conversion-efficiency 0.1 --duration 200s --output data/predator-prey
	@echo "Predator-Prey simulation completed!"

# Run Orbital simulation
orbital:
	@echo "Running Orbital mechanics simulation..."
	go run cmd/orbital-sim/main.go --time-step 1000 --enable-3d false --duration 300s --output data/orbital
	@echo "Orbital simulation completed!"

# Run everything
all: install build test run analysis
	@echo "SciSimGo project completed successfully!"
	@echo ""
	@echo "Check the following for results:"
	@echo "  - data/ directory for simulation data"
	@echo "  - *.png files for visualizations"
	@echo "  - simulation_report.md for summary"

# Quick start (minimal setup)
quick: build
	@echo "Quick start - running simulations..."
	@mkdir -p data
	$(MAKE) sir
	$(MAKE) predator
	$(MAKE) orbital
	@echo "Quick simulations completed!"

# Development mode
dev: build
	@echo "Development mode - watching for changes..."
	@echo "Press Ctrl+C to stop..."
	@while true; do \
		echo "Running simulations..."; \
		$(MAKE) quick; \
		echo "Waiting 30 seconds before next run..."; \
		sleep 30; \
	done

# Profile mode (with detailed logging)
profile: build
	@echo "Profile mode - running with detailed logging..."
	go run cmd/sir-simulator/main.go --population 10000 --initial-infected 100 --infection-rate 0.3 --recovery-rate 0.1 --duration 100s --output data/sir --logging --export-interval 100ms
	go run cmd/predator-prey/main.go --initial-prey 1000 --initial-predator 100 --prey-growth-rate 0.1 --predation-rate 0.01 --predator-death-rate 0.1 --conversion-efficiency 0.1 --duration 200s --output data/predator-prey --logging --export-interval 100ms
	go run cmd/orbital-sim/main.go --time-step 1000 --enable-3d false --duration 300s --output data/orbital --logging --export-interval 100ms
	@echo "Profile simulations completed!"

# Benchmark mode
benchmark: build
	@echo "Benchmark mode - running performance tests..."
	@echo "Benchmarking SIR model..."
	go test -bench=. ./internal/models/ -run=^$
	@echo "Benchmarking CSV export..."
	go test -bench=. ./internal/export/ -run=^$
	@echo "Benchmarks completed!"

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "Go documentation..."
	godoc -http=:6060 &
	@echo "Documentation server started at http://localhost:6060"
	@echo "Press Ctrl+C to stop..."

# Docker support
docker-build:
	@echo "Building Docker image..."
	docker build -t scisimgo .
	@echo "Docker image built!"

docker-run:
	@echo "Running in Docker..."
	docker run -it --rm -v $(PWD)/data:/app/data scisimgo make all
	@echo "Docker run completed!"

# Helpers
check-go:
	@echo "Checking Go installation..."
	@go version || (echo "Go not installed. Please install Go first." && exit 1)
	@echo "Go installation verified!"

check-python:
	@echo "Checking Python installation..."
	@python --version || (echo "Python not installed. Please install Python first." && exit 1)
	@echo "Python installation verified!"

setup: check-go check-python
	@echo "Setting up project..."
	@mkdir -p bin data notebooks scripts
	@echo "Project setup completed!"

# Default target
.DEFAULT_GOAL := help
