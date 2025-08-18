package models

import (
	"math"
	"math/rand"
	"time"

	"github.com/scisimgo/internal/engine"
)

// PredatorPreyModel represents the Lotka-Volterra predator-prey model
type PredatorPreyModel struct {
	*engine.BaseSimulation
	
	// Model parameters
	InitialPrey     int     `json:"initial_prey"`
	InitialPredator int     `json:"initial_predator"`
	PreyGrowthRate  float64 `json:"prey_growth_rate"`   // r - intrinsic growth rate of prey
	PredationRate   float64 `json:"predation_rate"`     // a - predation efficiency
	PredatorDeathRate float64 `json:"predator_death_rate"` // m - natural death rate of predators
	ConversionEfficiency float64 `json:"conversion_efficiency"` // b - conversion efficiency of prey to predator biomass
	
	// Current state
	Prey     int `json:"prey"`
	Predator int `json:"predator"`
	
	// Statistical tracking
	PeakPrey       int       `json:"peak_prey"`
	PeakPredator   int       `json:"peak_predator"`
	PeakTime       time.Time `json:"peak_time"`
	MinPrey        int       `json:"min_prey"`
	MinPredator    int       `json:"min_predator"`
	CycleCount     int       `json:"cycle_count"`
	LastPeakTime   time.Time `json:"last_peak_time"`
	
	// Random number generator
	rng *rand.Rand
}

// PredatorPreyConfig extends base simulation config with model-specific parameters
type PredatorPreyConfig struct {
	engine.SimulationConfig
	InitialPrey     int     `json:"initial_prey"`
	InitialPredator int     `json:"initial_predator"`
	PreyGrowthRate  float64 `json:"prey_growth_rate"`
	PredationRate   float64 `json:"predation_rate"`
	PredatorDeathRate float64 `json:"predator_death_rate"`
	ConversionEfficiency float64 `json:"conversion_efficiency"`
	EnableStochastic bool   `json:"enable_stochastic"`
}

// NewPredatorPreyModel creates a new predator-prey model
func NewPredatorPreyModel() *PredatorPreyModel {
	pp := &PredatorPreyModel{
		BaseSimulation: engine.NewBaseSimulation(),
		rng:            rand.New(rand.NewSource(time.Now().UnixNano())),
	}
	
	// Override the callbacks
	pp.BaseSimulation.SetOnTick(pp.onTick)
	pp.BaseSimulation.SetOnComplete(pp.onComplete)
	
	return pp
}

// Initialize sets up the predator-prey model with configuration
func (pp *PredatorPreyModel) Initialize(config PredatorPreyConfig) error {
	// Initialize base simulation
	if err := pp.BaseSimulation.Initialize(config.SimulationConfig); err != nil {
		return err
	}
	
	// Validate model-specific parameters
	if config.InitialPrey <= 0 {
		return NewValidationError("initial prey population must be positive")
	}
	
	if config.InitialPredator <= 0 {
		return NewValidationError("initial predator population must be positive")
	}
	
	if config.PreyGrowthRate < 0 {
		return NewValidationError("prey growth rate must be non-negative")
	}
	
	if config.PredationRate < 0 {
		return NewValidationError("predation rate must be non-negative")
	}
	
	if config.PredatorDeathRate < 0 {
		return NewValidationError("predator death rate must be non-negative")
	}
	
	if config.ConversionEfficiency < 0 {
		return NewValidationError("conversion efficiency must be non-negative")
	}
	
	// Set model parameters
	pp.InitialPrey = config.InitialPrey
	pp.InitialPredator = config.InitialPredator
	pp.PreyGrowthRate = config.PreyGrowthRate
	pp.PredationRate = config.PredationRate
	pp.PredatorDeathRate = config.PredatorDeathRate
	pp.ConversionEfficiency = config.ConversionEfficiency
	
	// Initialize state
	pp.Prey = config.InitialPrey
	pp.Predator = config.InitialPredator
	
	// Initialize statistics
	pp.PeakPrey = pp.Prey
	pp.PeakPredator = pp.Predator
	pp.PeakTime = time.Now()
	pp.MinPrey = pp.Prey
	pp.MinPredator = pp.Predator
	pp.CycleCount = 0
	pp.LastPeakTime = time.Now()
	
	return nil
}

// onTick handles each simulation tick for predator-prey model
func (pp *PredatorPreyModel) onTick(iteration int, data map[string]interface{}) error {
	// Calculate population changes
	preyChange := pp.calculatePreyChange()
	predatorChange := pp.calculatePredatorChange()
	
	// Update populations
	pp.Prey += preyChange
	pp.Predator += predatorChange
	
	// Ensure non-negative values
	pp.Prey = max(0, pp.Prey)
	pp.Predator = max(0, pp.Predator)
	
	// Update statistics
	pp.updateStatistics()
	
	return nil
}

// calculatePreyChange calculates the change in prey population
func (pp *PredatorPreyModel) calculatePreyChange() int {
	// Lotka-Volterra equation: dN/dt = r*N - a*N*P
	// For discrete time steps: ΔN = (r*N - a*N*P) * Δt
	
	// Growth term: r*N
	growthTerm := pp.PreyGrowthRate * float64(pp.Prey)
	
	// Predation term: a*N*P
	predationTerm := pp.PredationRate * float64(pp.Prey) * float64(pp.Predator)
	
	// Net change
	netChange := growthTerm - predationTerm
	
	// Convert to integer change
	change := int(math.Round(netChange))
	
	// Ensure we don't lose more prey than available
	if change < 0 {
		change = max(change, -pp.Prey)
	}
	
	return change
}

// calculatePredatorChange calculates the change in predator population
func (pp *PredatorPreyModel) calculatePredatorChange() int {
	// Lotka-Volterra equation: dP/dt = b*a*N*P - m*P
	// For discrete time steps: ΔP = (b*a*N*P - m*P) * Δt
	
	// Predation benefit term: b*a*N*P
	benefitTerm := pp.ConversionEfficiency * pp.PredationRate * float64(pp.Prey) * float64(pp.Predator)
	
	// Death term: m*P
	deathTerm := pp.PredatorDeathRate * float64(pp.Predator)
	
	// Net change
	netChange := benefitTerm - deathTerm
	
	// Convert to integer change
	change := int(math.Round(netChange))
	
	// Ensure we don't lose more predators than available
	if change < 0 {
		change = max(change, -pp.Predator)
	}
	
	return change
}

// updateStatistics updates the statistical tracking
func (pp *PredatorPreyModel) updateStatistics() {
	// Update peaks and minimums
	if pp.Prey > pp.PeakPrey {
		pp.PeakPrey = pp.Prey
	}
	
	if pp.Predator > pp.PeakPredator {
		pp.PeakPredator = pp.Predator
	}
	
	if pp.Prey < pp.MinPrey {
		pp.MinPrey = pp.Prey
	}
	
	if pp.Predator < pp.MinPredator {
		pp.MinPredator = pp.Predator
	}
	
	// Detect cycles (when both populations are increasing)
	if pp.Prey > 0 && pp.Predator > 0 {
		// Simple cycle detection: when both populations are above their initial values
		if pp.Prey > pp.InitialPrey && pp.Predator > pp.InitialPredator {
			timeSinceLastPeak := time.Since(pp.LastPeakTime)
			if timeSinceLastPeak > 5*time.Second { // Minimum cycle duration
				pp.CycleCount++
				pp.LastPeakTime = time.Now()
			}
		}
	}
}

// onComplete handles simulation completion
func (pp *PredatorPreyModel) onComplete(results []engine.SimulationResult) error {
	// Final statistics update
	pp.updateStatistics()
	return nil
}

// generateTickData overrides the base method to provide predator-prey specific data
func (pp *PredatorPreyModel) generateTickData() map[string]interface{} {
	return map[string]interface{}{
		"prey":                pp.Prey,
		"predator":            pp.Predator,
		"prey_growth_rate":   pp.PreyGrowthRate,
		"predation_rate":      pp.PredationRate,
		"predator_death_rate": pp.PredatorDeathRate,
		"conversion_efficiency": pp.ConversionEfficiency,
		"peak_prey":           pp.PeakPrey,
		"peak_predator":       pp.PeakPredator,
		"min_prey":            pp.MinPrey,
		"min_predator":        pp.MinPredator,
		"cycle_count":         pp.CycleCount,
	}
}

// GetCurrentState returns the current predator-prey state
func (pp *PredatorPreyModel) GetCurrentState() map[string]interface{} {
	return map[string]interface{}{
		"prey":                pp.Prey,
		"predator":            pp.Predator,
		"initial_prey":        pp.InitialPrey,
		"initial_predator":    pp.InitialPredator,
		"peak_prey":           pp.PeakPrey,
		"peak_predator":       pp.PeakPredator,
		"min_prey":            pp.MinPrey,
		"min_predator":        pp.MinPredator,
		"cycle_count":         pp.CycleCount,
		"prey_growth_rate":   pp.PreyGrowthRate,
		"predation_rate":      pp.PredationRate,
		"predator_death_rate": pp.PredatorDeathRate,
		"conversion_efficiency": pp.ConversionEfficiency,
	}
}

// GetStatistics returns comprehensive predator-prey statistics
func (pp *PredatorPreyModel) GetStatistics() map[string]interface{} {
	results := pp.BaseSimulation.GetResults()
	
	if len(results) == 0 {
		return map[string]interface{}{}
	}
	
	// Calculate time series statistics
	var preyValues, predatorValues []int
	var timestamps []time.Time
	
	for _, result := range results {
		if data, ok := result.Data["prey"]; ok {
			if val, ok := data.(int); ok {
				preyValues = append(preyValues, val)
			}
		}
		
		if data, ok := result.Data["predator"]; ok {
			if val, ok := data.(int); ok {
				predatorValues = append(predatorValues, val)
			}
		}
		
		timestamps = append(timestamps, result.Timestamp)
	}
	
	// Calculate population dynamics
	var preyGrowthRates, predatorGrowthRates []float64
	var populationRatios []float64
	
	for i := 1; i < len(preyValues); i++ {
		// Prey growth rate
		if preyValues[i-1] > 0 {
			preyGrowthRate := float64(preyValues[i]-preyValues[i-1]) / float64(preyValues[i-1])
			preyGrowthRates = append(preyGrowthRates, preyGrowthRate)
		}
		
		// Predator growth rate
		if predatorValues[i-1] > 0 {
			predatorGrowthRate := float64(predatorValues[i]-predatorValues[i-1]) / float64(predatorValues[i-1])
			predatorGrowthRates = append(predatorGrowthRates, predatorGrowthRate)
		}
		
		// Population ratio (prey/predator)
		if predatorValues[i] > 0 {
			ratio := float64(preyValues[i]) / float64(predatorValues[i])
			populationRatios = append(populationRatios, ratio)
		}
	}
	
	// Calculate stability metrics
	stabilityIndex := calculateStabilityIndex(preyValues, predatorValues)
	
	return map[string]interface{}{
		"total_iterations":      len(results),
		"simulation_duration":   time.Since(results[0].Timestamp),
		"peak_prey":             pp.PeakPrey,
		"peak_predator":         pp.PeakPredator,
		"min_prey":              pp.MinPrey,
		"min_predator":          pp.MinPredator,
		"cycle_count":           pp.CycleCount,
		"final_prey":            pp.Prey,
		"final_predator":        pp.Predator,
		"avg_prey":              calculateAverageInt(preyValues),
		"avg_predator":          calculateAverageInt(predatorValues),
		"prey_variance":         calculateVarianceInt(preyValues),
		"predator_variance":     calculateVarianceInt(predatorValues),
		"avg_prey_growth_rate":  calculateAverage(preyGrowthRates),
		"avg_predator_growth_rate": calculateAverage(predatorGrowthRates),
		"stability_index":       stabilityIndex,
		"population_ratio_range": map[string]interface{}{
			"min": calculateMin(populationRatios),
			"max": calculateMax(populationRatios),
			"avg": calculateAverage(populationRatios),
		},
	}
}

// calculateStabilityIndex calculates a stability index for the system
func calculateStabilityIndex(preyValues, predatorValues []int) float64 {
	if len(preyValues) < 2 || len(predatorValues) < 2 {
		return 0
	}
	
	// Calculate coefficient of variation for both populations
	preyCV := calculateCoefficientOfVariation(preyValues)
	predatorCV := calculateCoefficientOfVariation(predatorValues)
	
	// Lower CV indicates more stability
	// Return average CV (lower is more stable)
	return (preyCV + predatorCV) / 2
}

// calculateCoefficientOfVariation calculates CV = std_dev / mean
func calculateCoefficientOfVariation(values []int) float64 {
	if len(values) == 0 {
		return 0
	}
	
	mean := float64(calculateAverageInt(values))
	if mean == 0 {
		return 0
	}
	
	variance := calculateVarianceInt(values)
	stdDev := math.Sqrt(variance)
	
	return stdDev / mean
}

// calculateAverageInt calculates average of integer slice
func calculateAverageInt(values []int) float64 {
	if len(values) == 0 {
		return 0
	}
	
	sum := 0
	for _, v := range values {
		sum += v
	}
	return float64(sum) / float64(len(values))
}

// calculateVarianceInt calculates variance of integer slice
func calculateVarianceInt(values []int) float64 {
	if len(values) < 2 {
		return 0
	}
	
	mean := calculateAverageInt(values)
	sumSquaredDiff := 0.0
	
	for _, v := range values {
		diff := float64(v) - mean
		sumSquaredDiff += diff * diff
	}
	
	return sumSquaredDiff / float64(len(values)-1)
}
