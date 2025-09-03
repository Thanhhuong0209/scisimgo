# SciSimGo API Reference

## Overview

This document provides comprehensive API documentation for the SciSimGo simulation engine, including all interfaces, methods, and data structures.

## Core Interfaces

### SimulationEngine Interface

The `SimulationEngine` interface defines the contract for all simulation implementations.

```go
type SimulationEngine interface {
    // Core methods
    Run(ctx context.Context) error
    Pause() error
    Resume() error
    Stop() error
    GetState() SimulationState

    // Data methods
    GetResults() []SimulationResult
    ExportData(format string) error
    GenerateTickData() map[string]interface{}

    // Configuration
    GetConfig() SimulationConfig
    UpdateConfig(config SimulationConfig) error
}
```

### BaseSimulation

The `BaseSimulation` struct provides common functionality for all simulations.

```go
type BaseSimulation struct {
    mu         sync.RWMutex
    state      SimulationState
    config     SimulationConfig
    results    []SimulationResult
    startTime  time.Time
    iteration  int
    onTick     func(SimulationResult)
    onComplete func()
    simulation SimulationEngine
}
```

## Simulation Models

### SIR Model

The SIR (Susceptible-Infected-Recovered) epidemiological model.

#### Configuration

```go
type SIRConfig struct {
    engine.SimulationConfig
    PopulationSize    int     `json:"population_size"`
    InitialInfected   int     `json:"initial_infected"`
    InfectionRate     float64 `json:"infection_rate"`
    RecoveryRate      float64 `json:"recovery_rate"`
    EnableStochastic  bool    `json:"enable_stochastic"`
}
```

#### Methods

- `NewSIRModel() *SIRModel` - Creates a new SIR model instance
- `Initialize(config SIRConfig) error` - Initializes the model with configuration
- `GetCurrentState() map[string]interface{}` - Returns current model state
- `GetStatistics() map[string]interface{}` - Returns simulation statistics

### Predator-Prey Model

The Lotka-Volterra predator-prey ecological model.

#### Configuration

```go
type PredatorPreyConfig struct {
    engine.SimulationConfig
    InitialPrey          int     `json:"initial_prey"`
    InitialPredator      int     `json:"initial_predator"`
    PreyGrowthRate       float64 `json:"prey_growth_rate"`
    PredationRate        float64 `json:"predation_rate"`
    PredatorDeathRate    float64 `json:"predator_death_rate"`
    ConversionEfficiency float64 `json:"conversion_efficiency"`
    EnableStochastic     bool    `json:"enable_stochastic"`
}
```

#### Methods

- `NewPredatorPreyModel() *PredatorPreyModel` - Creates a new predator-prey model
- `Initialize(config PredatorPreyConfig) error` - Initializes the model
- `GetCurrentState() map[string]interface{}` - Returns current state
- `GetStatistics() map[string]interface{}` - Returns statistics

### Orbital Model

The orbital mechanics model for celestial body dynamics.

#### Configuration

```go
type OrbitalConfig struct {
    engine.SimulationConfig
    GravitationalConstant float64        `json:"gravitational_constant"`
    TimeStep             float64        `json:"time_step"`
    Enable3D             bool           `json:"enable_3d"`
    CelestialBodies      []CelestialBody `json:"celestial_bodies"`
}
```

#### Methods

- `NewOrbitalModel() *OrbitalModel` - Creates a new orbital model
- `Initialize(config OrbitalConfig) error` - Initializes the model
- `GetCurrentState() map[string]interface{}` - Returns current state
- `GetStatistics() map[string]interface{}` - Returns statistics

## Data Structures

### SimulationResult

Represents a single simulation result at a specific time point.

```go
type SimulationResult struct {
    Timestamp time.Time              `json:"timestamp"`
    Iteration int                    `json:"iteration"`
    Data      map[string]interface{} `json:"data"`
    Metadata  map[string]interface{} `json:"metadata"`
    Error     error                  `json:"error,omitempty"`
}
```

### SimulationConfig

Base configuration for all simulations.

```go
type SimulationConfig struct {
    Duration       time.Duration `json:"duration"`
    TickRate       time.Duration `json:"tick_rate"`
    MaxIterations  int           `json:"max_iterations"`
    EnableLogging  bool          `json:"enable_logging"`
    ExportInterval time.Duration `json:"export_interval"`
}
```

### SimulationState

Represents the current state of a simulation.

```go
type SimulationState int

const (
    StateInitialized SimulationState = iota
    StateRunning
    StatePaused
    StateCompleted
    StateError
)
```

## Export Utilities

### CSV Export

The CSV export utility provides functionality to export simulation results to CSV format.

```go
type CSVExporter struct {
    filePath string
    headers  []string
}

func NewCSVExporter(filePath string, headers []string) *CSVExporter
func (e *CSVExporter) ExportResults(results []SimulationResult) error
func (e *CSVExporter) ExportWithCustomHeaders(results []SimulationResult, headers []string) error
```

### JSON Export

The JSON export utility provides functionality to export simulation results to JSON format.

```go
type JSONExporter struct {
    filePath string
}

func NewJSONExporter(filePath string) *JSONExporter
func (e *JSONExporter) ExportResults(results []SimulationResult) error
```

## Error Handling

### Custom Error Types

```go
type ValidationError struct {
    Code    string
    Message string
    Field   string
}

func (e *ValidationError) Error() string
func NewValidationError(message string) *ValidationError
```

### Error Codes

- `ErrCodeInvalidParameter` - Invalid parameter value
- `ErrCodeMissingRequired` - Missing required parameter
- `ErrCodeOutOfRange` - Parameter out of valid range
- `ErrCodeInvalidConfig` - Invalid configuration
- `ErrCodeSimulationFailed` - Simulation execution failed
- `ErrCodeExportFailed` - Data export failed

## Logging

### Logger Interface

```go
type Logger interface {
    Debug(message string, fields ...map[string]interface{})
    Info(message string, fields ...map[string]interface{})
    Warn(message string, fields ...map[string]interface{})
    Error(message string, fields ...map[string]interface{})
    Fatal(message string, fields ...map[string]interface{})
}
```

### Log Levels

- `LogLevelDebug` - Debug information
- `LogLevelInfo` - General information
- `LogLevelWarn` - Warning messages
- `LogLevelError` - Error messages
- `LogLevelFatal` - Fatal errors

## Usage Examples

### Basic SIR Simulation

```go
package main

import (
    "context"
    "time"
    "github.com/scisimgo/internal/models"
)

func main() {
    // Create SIR model
    sir := models.NewSIRModel()
    
    // Configure simulation
    config := models.SIRConfig{
        SimulationConfig: engine.SimulationConfig{
            Duration: 100 * time.Second,
            TickRate: 100 * time.Millisecond,
        },
        PopulationSize:  10000,
        InitialInfected: 100,
        InfectionRate:   0.3,
        RecoveryRate:    0.1,
    }
    
    // Initialize and run
    if err := sir.Initialize(config); err != nil {
        panic(err)
    }
    
    ctx := context.Background()
    if err := sir.Run(ctx); err != nil {
        panic(err)
    }
    
    // Export results
    sir.ExportData("csv")
}
```

### Predator-Prey Simulation

```go
package main

import (
    "context"
    "time"
    "github.com/scisimgo/internal/models"
)

func main() {
    // Create predator-prey model
    pp := models.NewPredatorPreyModel()
    
    // Configure simulation
    config := models.PredatorPreyConfig{
        SimulationConfig: engine.SimulationConfig{
            Duration: 200 * time.Second,
            TickRate: 100 * time.Millisecond,
        },
        InitialPrey:          1000,
        InitialPredator:      100,
        PreyGrowthRate:       0.1,
        PredationRate:        0.01,
        PredatorDeathRate:    0.1,
        ConversionEfficiency: 0.1,
    }
    
    // Initialize and run
    if err := pp.Initialize(config); err != nil {
        panic(err)
    }
    
    ctx := context.Background()
    if err := pp.Run(ctx); err != nil {
        panic(err)
    }
    
    // Export results
    pp.ExportData("csv")
}
```

### Orbital Simulation

```go
package main

import (
    "context"
    "time"
    "github.com/scisimgo/internal/models"
)

func main() {
    // Create orbital model
    orbital := models.NewOrbitalModel()
    
    // Configure celestial bodies
    bodies := []models.CelestialBody{
        {Name: "Sun", Mass: 1.99e30, Position: [3]float64{0, 0, 0}, Velocity: [3]float64{0, 0, 0}},
        {Name: "Earth", Mass: 5.97e24, Position: [3]float64{1.496e11, 0, 0}, Velocity: [3]float64{0, 29780, 0}},
        {Name: "Mars", Mass: 6.39e23, Position: [3]float64{2.28e11, 0, 0}, Velocity: [3]float64{0, 24070, 0}},
    }
    
    // Configure simulation
    config := models.OrbitalConfig{
        SimulationConfig: engine.SimulationConfig{
            Duration: 300 * time.Second,
            TickRate: 100 * time.Millisecond,
        },
        GravitationalConstant: 6.67e-11,
        TimeStep:             1000,
        Enable3D:             false,
        CelestialBodies:      bodies,
    }
    
    // Initialize and run
    if err := orbital.Initialize(config); err != nil {
        panic(err)
    }
    
    ctx := context.Background()
    if err := orbital.Run(ctx); err != nil {
        panic(err)
    }
    
    // Export results
    orbital.ExportData("csv")
}
```

## Performance Considerations

### Concurrency

- All simulations use goroutines for concurrent execution
- Thread-safe operations with mutex protection
- Channel-based communication for data flow

### Memory Management

- Results are stored in memory during simulation
- Export functionality for large datasets
- Configurable export intervals to manage memory usage

### Optimization Tips

1. Use appropriate tick rates for your use case
2. Set reasonable export intervals to balance memory and disk usage
3. Consider using shorter durations for testing
4. Monitor memory usage for long-running simulations

## Troubleshooting

### Common Issues

1. **Deadlock**: Ensure callbacks don't call simulation methods
2. **Memory Issues**: Use shorter durations or more frequent exports
3. **Invalid Parameters**: Check parameter ranges and validation
4. **Export Failures**: Ensure output directory exists and is writable

### Debug Mode

Enable debug logging for detailed simulation information:

```go
config := engine.SimulationConfig{
    EnableLogging: true,
    // ... other config
}
```

## Version History

- **v1.0.0** - Initial release with SIR, Predator-Prey, and Orbital models
- **v1.1.0** - Added comprehensive data science analysis notebooks
- **v1.2.0** - Enhanced error handling and logging
- **v1.3.0** - Added CI/CD pipelines and documentation
