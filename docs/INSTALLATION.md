# SciSimGo Installation Guide

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Go**: Version 1.21 or higher
- **Python**: Version 3.9 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free disk space

### Recommended Requirements
- **Operating System**: Windows 11, macOS 12+, or Ubuntu 20.04+
- **Go**: Version 1.22 or higher
- **Python**: Version 3.11 or higher
- **Memory**: 16GB RAM
- **Storage**: 10GB free disk space
- **CPU**: Multi-core processor (4+ cores recommended)

## Installation Methods

### Method 1: Clone from GitHub (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Thanhhuong0209/scisimgo.git
   cd scisimgo
   ```

2. **Install Go dependencies**:
   ```bash
   go mod download
   go mod tidy
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   go run cmd/sir-simulator/main.go --help
   ```

### Method 2: Docker Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Thanhhuong0209/scisimgo.git
   cd scisimgo
   ```

2. **Build Docker image**:
   ```bash
   docker build -t scisimgo .
   ```

3. **Run with Docker Compose**:
   ```bash
   docker-compose up
   ```

### Method 3: Manual Installation

1. **Install Go**:
   - Download from [golang.org](https://golang.org/dl/)
   - Follow platform-specific installation instructions
   - Verify: `go version`

2. **Install Python**:
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure pip is installed
   - Verify: `python --version`

3. **Download SciSimGo**:
   - Download the latest release from GitHub
   - Extract to your desired directory

4. **Install dependencies**:
   ```bash
   # Go dependencies
   go mod download
   
   # Python dependencies
   pip install -r requirements.txt
   ```

## Platform-Specific Instructions

### Windows

1. **Install Go**:
   - Download the Windows installer from golang.org
   - Run the installer and follow the prompts
   - Add Go to your PATH environment variable

2. **Install Python**:
   - Download Python from python.org
   - Check "Add Python to PATH" during installation
   - Install pip if not included

3. **Install Git** (if not already installed):
   - Download from [git-scm.com](https://git-scm.com/download/win)
   - Use Git Bash for command-line operations

4. **Clone and setup**:
   ```bash
   git clone https://github.com/Thanhhuong0209/scisimgo.git
   cd scisimgo
   go mod download
   pip install -r requirements.txt
   ```

### macOS

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Go**:
   ```bash
   brew install go
   ```

3. **Install Python**:
   ```bash
   brew install python@3.11
   ```

4. **Clone and setup**:
   ```bash
   git clone https://github.com/Thanhhuong0209/scisimgo.git
   cd scisimgo
   go mod download
   pip3 install -r requirements.txt
   ```

### Linux (Ubuntu/Debian)

1. **Update package list**:
   ```bash
   sudo apt update
   ```

2. **Install Go**:
   ```bash
   sudo apt install golang-go
   ```

3. **Install Python**:
   ```bash
   sudo apt install python3 python3-pip
   ```

4. **Install Git**:
   ```bash
   sudo apt install git
   ```

5. **Clone and setup**:
   ```bash
   git clone https://github.com/Thanhhuong0209/scisimgo.git
   cd scisimgo
   go mod download
   pip3 install -r requirements.txt
   ```

## Verification

### Test Go Installation

```bash
# Check Go version
go version

# Test compilation
go build ./...

# Run a simple simulation
go run cmd/sir-simulator/main.go -population=1000 -duration=5s
```

### Test Python Installation

```bash
# Check Python version
python --version

# Test Python packages
python -c "import pandas, numpy, matplotlib; print('All packages imported successfully')"

# Run analysis script
python notebooks/sir-analysis/sir_analysis.py
```

### Test Docker Installation

```bash
# Check Docker version
docker --version

# Build and test
docker build -t scisimgo .
docker run --rm scisimgo go run cmd/sir-simulator/main.go --help
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Simulation settings
SIMULATION_DURATION=100s
SIMULATION_TICK_RATE=100ms
ENABLE_LOGGING=true

# Export settings
EXPORT_FORMAT=csv
EXPORT_INTERVAL=10s
OUTPUT_DIRECTORY=./data

# Python settings
PYTHONPATH=./notebooks
```

### Go Configuration

Set Go environment variables:

```bash
# Windows
set GOPATH=C:\Users\%USERNAME%\go
set GOBIN=%GOPATH%\bin

# macOS/Linux
export GOPATH=$HOME/go
export GOBIN=$GOPATH/bin
export PATH=$PATH:$GOBIN
```

### Python Configuration

Create a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

1. **Go module errors**:
   ```bash
   go clean -modcache
   go mod download
   ```

2. **Python package conflicts**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

3. **Permission errors**:
   ```bash
   # Linux/macOS
   sudo chown -R $USER:$USER ~/.go
   
   # Windows - Run as Administrator
   ```

4. **Docker build failures**:
   ```bash
   docker system prune -a
   docker build --no-cache -t scisimgo .
   ```

### Performance Issues

1. **Slow simulations**:
   - Reduce tick rate
   - Use shorter durations for testing
   - Check system resources

2. **Memory issues**:
   - Increase export frequency
   - Use smaller population sizes
   - Monitor system memory usage

3. **Python analysis slow**:
   - Use smaller datasets for testing
   - Install optimized NumPy (Intel MKL)
   - Consider using Jupyter for interactive analysis

### Getting Help

1. **Check logs**:
   ```bash
   # Enable debug logging
   go run cmd/sir-simulator/main.go -duration=10s -enable-logging
   ```

2. **Run tests**:
   ```bash
   go test ./...
   python -m pytest notebooks/
   ```

3. **Check system resources**:
   ```bash
   # Linux/macOS
   top
   htop
   
   # Windows
   taskmgr
   ```

## Uninstallation

### Remove Go Installation

```bash
# Remove Go directory
rm -rf /usr/local/go  # Linux/macOS
# Or uninstall via package manager
```

### Remove Python Packages

```bash
# Remove virtual environment
rm -rf venv

# Remove global packages
pip uninstall -r requirements.txt -y
```

### Remove Docker Images

```bash
docker rmi scisimgo
docker system prune -a
```

### Remove Project Files

```bash
rm -rf scisimgo/
```

## Next Steps

After successful installation:

1. **Read the documentation**:
   - [API Reference](API_REFERENCE.md)
   - [User Guide](USER_GUIDE.md)
   - [Examples](EXAMPLES.md)

2. **Run your first simulation**:
   ```bash
   go run cmd/sir-simulator/main.go -population=10000 -duration=60s
   ```

3. **Explore the analysis notebooks**:
   ```bash
   jupyter notebook notebooks/
   ```

4. **Check out the examples**:
   ```bash
   cd examples/
   go run basic_sir.go
   ```

## Support

For installation issues:

1. Check the [GitHub Issues](https://github.com/Thanhhuong0209/scisimgo/issues)
2. Review the troubleshooting section above
3. Create a new issue with:
   - Operating system and version
   - Go and Python versions
   - Complete error messages
   - Steps to reproduce the issue
