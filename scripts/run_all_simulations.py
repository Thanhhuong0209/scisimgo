#!/usr/bin/env python3
"""
Run All Simulations Script
SciSimGo - Scientific Simulation & Data Science Playground

This script runs all three simulation models:
1. SIR Disease Model
2. Predator-Prey Model  
3. Orbital Mechanics Model

Then runs the data analysis scripts.
"""

import subprocess
import sys
import os
import time
import glob
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n {description}")
    print(f"Command: {command}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(" Command completed successfully")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f" Command failed with exit code {e.returncode}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return False

def check_go_installation():
    """Check if Go is installed and accessible"""
    try:
        result = subprocess.run(["go", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f" Go installed: {result.stdout.strip()}")
            return True
        else:
            print(" Go not accessible")
            return False
    except FileNotFoundError:
        print(" Go not found in PATH")
        return False

def check_python_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'scikit-learn', 'scipy', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f" {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f" {package} not installed")
    
    if missing_packages:
        print(f"\n  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "data/sir",
        "data/predator-prey", 
        "data/orbital",
        "notebooks/sir-analysis",
        "notebooks/predator-prey-analysis",
        "notebooks/orbital-analysis"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f" Created directory: {directory}")

def run_sir_simulation():
    """Run SIR disease model simulation"""
    print("\n Running SIR Disease Model Simulation")
    print("=" * 60)
    
    # Run with different parameters
    scenarios = [
        {
            "name": "Standard Epidemic",
            "params": "--population 10000 --initial-infected 100 --infection-rate 0.3 --recovery-rate 0.1 --duration 100s"
        },
        {
            "name": "Fast Spreading",
            "params": "--population 10000 --initial-infected 50 --infection-rate 0.5 --recovery-rate 0.05 --duration 80s"
        },
        {
            "name": "Slow Recovery",
            "params": "--population 10000 --initial-infected 200 --infection-rate 0.2 --recovery-rate 0.02 --duration 150s"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n Running: {scenario['name']}")
        command = f"go run cmd/sir-simulator/main.go {scenario['params']} --output data/sir"
        success = run_command(command, f"SIR Simulation - {scenario['name']}")
        
        if not success:
            print(f"Warning: SIR simulation '{scenario['name']}' failed, continuing...")
    
    return True

def run_predator_prey_simulation():
    """Run Predator-Prey model simulation"""
    print("\n Running Predator-Prey Model Simulation")
    print("=" * 60)
    
    # Run with different parameters
    scenarios = [
        {
            "name": "Balanced Ecosystem",
            "params": "--initial-prey 1000 --initial-predator 100 --prey-growth-rate 0.1 --predation-rate 0.01 --predator-death-rate 0.1 --conversion-efficiency 0.1 --duration 200s"
        },
        {
            "name": "Prey Dominant",
            "params": "--initial-prey 2000 --initial-predator 50 --prey-growth-rate 0.15 --predation-rate 0.005 --predator-death-rate 0.15 --conversion-efficiency 0.08 --duration 200s"
        },
        {
            "name": "Predator Dominant",
            "params": "--initial-prey 500 --initial-predator 200 --prey-growth-rate 0.08 --predation-rate 0.02 --predator-death-rate 0.08 --conversion-efficiency 0.12 --duration 200s"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n Running: {scenario['name']}")
        command = f"go run cmd/predator-prey/main.go {scenario['params']} --output data/predator-prey"
        success = run_command(command, f"Predator-Prey Simulation - {scenario['name']}")
        
        if not success:
            print(f"Warning: Predator-Prey simulation '{scenario['name']}' failed, continuing...")
    
    return True

def run_orbital_simulation():
    """Run Orbital mechanics simulation"""
    print("\n Running Orbital Mechanics Simulation")
    print("=" * 60)
    
    # Run with different parameters
    scenarios = [
        {
            "name": "Solar System (2D)",
            "params": "--time-step 1000 --enable-3d false --duration 300s"
        },
        {
            "name": "Solar System (3D)",
            "params": "--time-step 1000 --enable-3d true --duration 300s"
        },
        {
            "name": "High Resolution",
            "params": "--time-step 100 --enable-3d false --duration 200s"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n Running: {scenario['name']}")
        command = f"go run cmd/orbital-sim/main.go {scenario['params']} --output data/orbital"
        success = run_command(command, f"Orbital Simulation - {scenario['name']}")
        
        if not success:
            print(f"Warning: Orbital simulation '{scenario['name']}' failed, continuing...")
    
    return True

def run_data_analysis():
    """Run data analysis scripts"""
    print("\n Running Data Analysis")
    print("=" * 60)
    
    analysis_scripts = [
        {
            "name": "SIR Analysis",
            "script": "notebooks/sir-analysis/sir_analysis.py",
            "description": "Analyzing SIR disease model data"
        }
    ]
    
    for analysis in analysis_scripts:
        if os.path.exists(analysis["script"]):
            print(f"\n Running: {analysis['name']}")
            command = f"python {analysis['script']}"
            success = run_command(command, analysis["description"])
            
            if not success:
                print(f"Warning: {analysis['name']} analysis failed")
        else:
            print(f"Warning: Analysis script not found: {analysis['script']}")
    
    return True

def generate_report():
    """Generate a summary report of all simulations"""
    print("\n Generating Simulation Report")
    print("=" * 60)
    
    report = []
    report.append("# SciSimGo Simulation Report")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Check data files
    data_dirs = ["data/sir", "data/predator-prey", "data/orbital"]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            csv_files = glob.glob(f"{data_dir}/*.csv")
            report.append(f"## {data_dir.replace('data/', '').replace('-', ' ').title()}")
            report.append(f"Found {len(csv_files)} CSV files:")
            for csv_file in csv_files:
                file_size = os.path.getsize(csv_file)
                report.append(f"- {os.path.basename(csv_file)} ({file_size:,} bytes)")
            report.append("")
        else:
            report.append(f"## {data_dir.replace('data/', '').replace('-', ' ').title()}")
            report.append("No data directory found")
            report.append("")
    
    # Write report
    with open("simulation_report.md", "w") as f:
        f.write("\n".join(report))
    
    print(" Report generated: simulation_report.md")
    return True

def main():
    """Main function to run all simulations"""
    print(" SciSimGo - Running All Simulations")
    print("=" * 60)
    
    # Check prerequisites
    print(" Checking prerequisites...")
    
    if not check_go_installation():
        print("\ Go installation check failed. Please install Go first.")
        sys.exit(1)
    
    if not check_python_dependencies():
        print(" Python dependencies check failed. Please install required packages.")
        sys.exit(1)
    
    print(" All prerequisites met!")
    
    # Setup directories
    print("\n Setting up directories...")
    setup_directories()
    
    # Run simulations
    print("\n Starting simulations...")
    
    simulations = [
        ("SIR Disease Model", run_sir_simulation),
        ("Predator-Prey Model", run_predator_prey_simulation),
        ("Orbital Mechanics", run_orbital_simulation)
    ]
    
    for name, simulation_func in simulations:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            success = simulation_func()
            if success:
                print(f" {name} completed successfully")
            else:
                print(f" {name} had some issues")
        except Exception as e:
            print(f" {name} failed with error: {e}")
    
    # Run data analysis
    print("\nRunning data analysis...")
    try:
        run_data_analysis()
    except Exception as e:
        print(f" Data analysis failed with error: {e}")
    
    # Generate report
    print("\nGenerating final report...")
    try:
        generate_report()
    except Exception as e:
        print(f" Report generation failed with error: {e}")
    
    # Final summary
    print("\n" + "="*60)
    print(" All simulations completed!")
    print("="*60)
    print(" Check the following directories for results:")
    print("   - data/sir/ - SIR simulation data")
    print("   - data/predator-prey/ - Predator-Prey simulation data")
    print("   - data/orbital/ - Orbital mechanics data")
    print("   - simulation_report.md - Summary report")
    print("\n To analyze the data, run:")
    print("   python notebooks/sir-analysis/sir_analysis.py")
    print("\n SciSimGo project completed successfully!")

if __name__ == "__main__":
    main()
