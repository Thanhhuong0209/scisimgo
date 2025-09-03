package models

import (
	"math"
	"math/rand"
	"time"

	"github.com/scisimgo/internal/engine"
)

// SIRModel represents the SIR (Susceptible-Infected-Recovered) disease model
type SIRModel struct {
	*engine.BaseSimulation

	// Model parameters
	PopulationSize  int     `json:"population_size"`
	InitialInfected int     `json:"initial_infected"`
	InfectionRate   float64 `json:"infection_rate"` // β (beta) - rate of infection
	RecoveryRate    float64 `json:"recovery_rate"`  // γ (gamma) - rate of recovery

	// Current state
	Susceptible int `json:"susceptible"`
	Infected    int `json:"infected"`
	Recovered   int `json:"recovered"`

	// Statistical tracking
	PeakInfected     int           `json:"peak_infected"`
	PeakTime         time.Time     `json:"peak_time"`
	TotalCases       int           `json:"total_cases"`
	EpidemicDuration time.Duration `json:"epidemic_duration"`

	// Random number generator
	rng *rand.Rand
}

// SIRConfig extends base simulation config with SIR-specific parameters
type SIRConfig struct {
	engine.SimulationConfig
	PopulationSize   int     `json:"population_size"`
	InitialInfected  int     `json:"initial_infected"`
	InfectionRate    float64 `json:"infection_rate"`
	RecoveryRate     float64 `json:"recovery_rate"`
	EnableStochastic bool    `json:"enable_stochastic"`
}

// NewSIRModel creates a new SIR model
func NewSIRModel() *SIRModel {
	sir := &SIRModel{
		BaseSimulation: engine.NewBaseSimulation(),
		rng:            rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	// Override the callbacks
	sir.BaseSimulation.SetOnTick(sir.onTick)
	sir.BaseSimulation.SetOnComplete(sir.onComplete)

	// Set simulation reference for method calls
	sir.BaseSimulation.SetSimulation(sir)

	return sir
}

// Initialize sets up the SIR model with configuration
func (sir *SIRModel) Initialize(config SIRConfig) error {
	// Initialize base simulation
	if err := sir.BaseSimulation.Initialize(config.SimulationConfig); err != nil {
		return err
	}

	// Validate SIR-specific parameters
	if config.PopulationSize <= 0 {
		return NewValidationError("population size must be positive")
	}

	if config.InitialInfected < 0 || config.InitialInfected > config.PopulationSize {
		return NewValidationError("initial infected must be between 0 and population size")
	}

	if config.InfectionRate < 0 {
		return NewValidationError("infection rate must be non-negative")
	}

	if config.RecoveryRate < 0 {
		return NewValidationError("recovery rate must be non-negative")
	}

	// Set model parameters
	sir.PopulationSize = config.PopulationSize
	sir.InitialInfected = config.InitialInfected
	sir.InfectionRate = config.InfectionRate
	sir.RecoveryRate = config.RecoveryRate

	// Initialize state
	sir.Susceptible = config.PopulationSize - config.InitialInfected
	sir.Infected = config.InitialInfected
	sir.Recovered = 0

	// Initialize statistics
	sir.PeakInfected = sir.Infected
	sir.PeakTime = time.Now()
	sir.TotalCases = sir.Infected
	sir.EpidemicDuration = 0

	return nil
}

// calculateNewInfections calculates new infections based on SIR equations
func (sir *SIRModel) calculateNewInfections() int {
	if sir.Susceptible == 0 || sir.Infected == 0 {
		return 0
	}

	// SIR differential equation: dS/dt = -β * S * I / N
	// For discrete time steps: ΔS = -β * S * I / N * Δt
	infectionProbability := sir.InfectionRate * float64(sir.Susceptible) * float64(sir.Infected) / float64(sir.PopulationSize)

	// Convert to integer infections
	newInfections := int(math.Round(infectionProbability))

	// Ensure we don't infect more than available susceptible
	newInfections = min(newInfections, sir.Susceptible)

	return newInfections
}

// calculateNewRecoveries calculates new recoveries based on SIR equations
func (sir *SIRModel) calculateNewRecoveries() int {
	if sir.Infected == 0 {
		return 0
	}

	// SIR differential equation: dR/dt = γ * I
	// For discrete time steps: ΔR = γ * I * Δt
	recoveryProbability := sir.RecoveryRate * float64(sir.Infected)

	// Convert to integer recoveries
	newRecoveries := int(math.Round(recoveryProbability))

	// Ensure we don't recover more than available infected
	newRecoveries = min(newRecoveries, sir.Infected)

	return newRecoveries
}

// onComplete handles simulation completion
func (sir *SIRModel) onComplete(results []engine.SimulationResult) error {
	// Calculate final statistics
	if sir.EpidemicDuration == 0 && sir.Infected == 0 {
		sir.EpidemicDuration = time.Since(sir.BaseSimulation.GetStartTime())
	}

	return nil
}

// GetCurrentState returns the current SIR state
func (sir *SIRModel) GetCurrentState() map[string]interface{} {
	return map[string]interface{}{
		"susceptible":       sir.Susceptible,
		"infected":          sir.Infected,
		"recovered":         sir.Recovered,
		"total_population":  sir.PopulationSize,
		"peak_infected":     sir.PeakInfected,
		"peak_time":         sir.PeakTime,
		"total_cases":       sir.TotalCases,
		"epidemic_duration": sir.EpidemicDuration,
		"infection_rate":    sir.InfectionRate,
		"recovery_rate":     sir.RecoveryRate,
	}
}

// GetStatistics returns comprehensive SIR statistics
func (sir *SIRModel) GetStatistics() map[string]interface{} {
	results := sir.BaseSimulation.GetResults()

	if len(results) == 0 {
		return map[string]interface{}{}
	}

	// Calculate time series statistics
	var susceptibleValues, infectedValues, recoveredValues []int
	var timestamps []time.Time

	for _, result := range results {
		if data, ok := result.Data["susceptible"]; ok {
			if val, ok := data.(int); ok {
				susceptibleValues = append(susceptibleValues, val)
			}
		}

		if data, ok := result.Data["infected"]; ok {
			if val, ok := data.(int); ok {
				infectedValues = append(infectedValues, val)
			}
		}

		if data, ok := result.Data["recovered"]; ok {
			if val, ok := data.(int); ok {
				recoveredValues = append(recoveredValues, val)
			}
		}

		timestamps = append(timestamps, result.Timestamp)
	}

	// Calculate growth rates
	var growthRates []float64
	for i := 1; i < len(infectedValues); i++ {
		if infectedValues[i-1] > 0 {
			growthRate := float64(infectedValues[i]-infectedValues[i-1]) / float64(infectedValues[i-1])
			growthRates = append(growthRates, growthRate)
		}
	}

	// Calculate doubling time (if applicable)
	var doublingTime time.Duration
	if len(growthRates) > 0 {
		avgGrowthRate := calculateAverage(growthRates)
		if avgGrowthRate > 0 {
			doublingTime = time.Duration(math.Log(2) / avgGrowthRate * float64(sir.BaseSimulation.GetConfig().TickRate))
		}
	}

	return map[string]interface{}{
		"total_iterations":    len(results),
		"simulation_duration": time.Since(results[0].Timestamp),
		"peak_infected":       sir.PeakInfected,
		"peak_time":           sir.PeakTime,
		"total_cases":         sir.TotalCases,
		"epidemic_duration":   sir.EpidemicDuration,
		"final_susceptible":   sir.Susceptible,
		"final_infected":      sir.Infected,
		"final_recovered":     sir.Recovered,
		"attack_rate":         float64(sir.TotalCases) / float64(sir.PopulationSize),
		"case_fatality_rate":  0.0, // SIR model doesn't include death
		"doubling_time":       doublingTime,
		"avg_growth_rate":     calculateAverage(growthRates),
		"max_growth_rate":     calculateMax(growthRates),
		"min_growth_rate":     calculateMin(growthRates),
	}
}

// onTick handles each simulation tick
func (sir *SIRModel) onTick(iteration int, data map[string]interface{}) error {
	// Update SIR state
	newInfections := sir.calculateNewInfections()
	newRecoveries := sir.calculateNewRecoveries()

	sir.Susceptible -= newInfections
	sir.Infected += newInfections - newRecoveries
	sir.Recovered += newRecoveries

	// Update peak tracking
	if sir.Infected > sir.PeakInfected {
		sir.PeakInfected = sir.Infected
		sir.PeakTime = time.Now()
	}

	sir.TotalCases = sir.Infected + sir.Recovered

	// Update data map
	data["susceptible"] = sir.Susceptible
	data["infected"] = sir.Infected
	data["recovered"] = sir.Recovered
	data["total_population"] = sir.PopulationSize
	data["infection_rate"] = sir.InfectionRate
	data["recovery_rate"] = sir.RecoveryRate
	data["peak_infected"] = sir.PeakInfected
	data["total_cases"] = sir.TotalCases

	return nil
}

// GenerateTickData overrides the base method to provide SIR-specific data
func (sir *SIRModel) GenerateTickData() map[string]interface{} {
	return map[string]interface{}{
		"susceptible":      sir.Susceptible,
		"infected":         sir.Infected,
		"recovered":        sir.Recovered,
		"total_population": sir.PopulationSize,
		"infection_rate":   sir.InfectionRate,
		"recovery_rate":    sir.RecoveryRate,
		"peak_infected":    sir.PeakInfected,
		"total_cases":      sir.TotalCases,
	}
}

// ExportData implements the SimulationEngine interface
func (sir *SIRModel) ExportData(format string) error {
	// Base implementation - can be overridden by subclasses
	return nil
}

// Helper functions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func calculateAverage(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func calculateMax(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	max := values[0]
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	return max
}

func calculateMin(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	min := values[0]
	for _, v := range values {
		if v < min {
			min = v
		}
	}
	return min
}

// ValidationError represents a validation error
type ValidationError struct {
	Message string
}

func (e ValidationError) Error() string {
	return e.Message
}

func NewValidationError(message string) error {
	return ValidationError{Message: message}
}
