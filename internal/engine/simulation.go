package engine

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// SimulationState represents the current state of a simulation
type SimulationState int

const (
	StateInitialized SimulationState = iota
	StateRunning
	StatePaused
	StateCompleted
	StateError
)

// SimulationConfig holds configuration for simulations
type SimulationConfig struct {
	Duration       time.Duration `json:"duration"`
	TickRate       time.Duration `json:"tick_rate"`
	MaxIterations  int           `json:"max_iterations"`
	EnableLogging  bool          `json:"enable_logging"`
	ExportInterval time.Duration `json:"export_interval"`
}

// SimulationResult represents the output of a simulation
type SimulationResult struct {
	Timestamp time.Time              `json:"timestamp"`
	Iteration int                    `json:"iteration"`
	Data      map[string]interface{} `json:"data"`
	Metadata  map[string]interface{} `json:"metadata"`
	Error     error                  `json:"error,omitempty"`
}

// SimulationEngine defines the interface for all simulation engines
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

// BaseSimulation provides common functionality for all simulations
type BaseSimulation struct {
	mu         sync.RWMutex
	config     SimulationConfig
	state      SimulationState
	results    []SimulationResult
	ctx        context.Context
	cancel     context.CancelFunc
	ticker     *time.Ticker
	iteration  int
	startTime  time.Time
	lastExport time.Time

	// Callbacks
	onTick     func(iteration int, data map[string]interface{}) error
	onComplete func(results []SimulationResult) error
	onError    func(error) error

	// Reference to the concrete simulation for method calls
	simulation SimulationEngine
}

// NewBaseSimulation creates a new base simulation
func NewBaseSimulation() *BaseSimulation {
	return &BaseSimulation{
		state:   StateInitialized,
		results: make([]SimulationResult, 0),
	}
}

// Initialize sets up the simulation with configuration
func (bs *BaseSimulation) Initialize(config SimulationConfig) error {
	bs.mu.Lock()
	defer bs.mu.Unlock()

	if bs.state != StateInitialized {
		return fmt.Errorf("simulation already initialized")
	}

	// Validate configuration
	if config.TickRate <= 0 {
		return fmt.Errorf("tick rate must be positive")
	}

	if config.Duration <= 0 && config.MaxIterations <= 0 {
		return fmt.Errorf("must specify either duration or max iterations")
	}

	bs.config = config
	bs.state = StateInitialized
	bs.iteration = 0
	bs.startTime = time.Time{}
	bs.lastExport = time.Time{}

	return nil
}

// Run starts the simulation
func (bs *BaseSimulation) Run(ctx context.Context) error {
	bs.mu.Lock()
	if bs.state != StateInitialized && bs.state != StatePaused {
		bs.mu.Unlock()
		return fmt.Errorf("simulation not ready to run (state: %v)", bs.state)
	}

	bs.ctx, bs.cancel = context.WithCancel(ctx)
	bs.state = StateRunning
	bs.startTime = time.Now()
	bs.ticker = time.NewTicker(bs.config.TickRate)
	bs.mu.Unlock()

	defer bs.cleanup()

	// Main simulation loop
	for {
		select {
		case <-bs.ctx.Done():
			return bs.ctx.Err()

		case <-bs.ticker.C:
			if err := bs.processTick(); err != nil {
				bs.handleError(err)
				return err
			}

			// Check completion conditions
			if bs.shouldComplete() {
				bs.complete()
				return nil
			}
		}
	}
}

// processTick handles a single simulation tick
func (bs *BaseSimulation) processTick() error {
	bs.mu.Lock()
	defer bs.mu.Unlock()

	bs.iteration++

	// Generate data for this tick
	var data map[string]interface{}
	if bs.simulation != nil {
		data = bs.simulation.GenerateTickData()
	} else {
		data = bs.GenerateTickData()
	}

	// Create result
	result := SimulationResult{
		Timestamp: time.Now(),
		Iteration: bs.iteration,
		Data:      data,
		Metadata:  bs.getMetadata(),
	}

	bs.results = append(bs.results, result)

	// Call onTick callback if set
	if bs.onTick != nil {
		if err := bs.onTick(bs.iteration, data); err != nil {
			return fmt.Errorf("onTick callback error: %w", err)
		}
	}

	// Export data if interval reached
	if bs.shouldExport() {
		if err := bs.exportData(); err != nil {
			return fmt.Errorf("export error: %w", err)
		}
		bs.lastExport = time.Now()
	}

	return nil
}

// GenerateTickData generates data for the current tick
func (bs *BaseSimulation) GenerateTickData() map[string]interface{} {
	// Base implementation - subclasses should override
	return map[string]interface{}{
		"iteration": bs.iteration,
		"timestamp": time.Now().Unix(),
		"state":     bs.state.String(),
	}
}

// generateTickData is the internal method that calls the public one
func (bs *BaseSimulation) generateTickData() map[string]interface{} {
	return bs.GenerateTickData()
}

// GetStartTime returns the simulation start time
func (bs *BaseSimulation) GetStartTime() time.Time {
	bs.mu.RLock()
	defer bs.mu.RUnlock()
	return bs.startTime
}

// getMetadata returns metadata about the simulation
func (bs *BaseSimulation) getMetadata() map[string]interface{} {
	return map[string]interface{}{
		"start_time":   bs.startTime,
		"elapsed_time": time.Since(bs.startTime),
		"config":       bs.config,
	}
}

// shouldComplete checks if the simulation should complete
func (bs *BaseSimulation) shouldComplete() bool {
	if bs.config.MaxIterations > 0 && bs.iteration >= bs.config.MaxIterations {
		return true
	}

	if bs.config.Duration > 0 && time.Since(bs.startTime) >= bs.config.Duration {
		return true
	}

	return false
}

// shouldExport checks if data should be exported
func (bs *BaseSimulation) shouldExport() bool {
	return bs.config.ExportInterval > 0 &&
		time.Since(bs.lastExport) >= bs.config.ExportInterval
}

// exportData exports the current simulation data
func (bs *BaseSimulation) exportData() error {
	// Base implementation - subclasses should override
	return nil
}

// complete marks the simulation as completed
func (bs *BaseSimulation) complete() {
	bs.mu.Lock()
	defer bs.mu.Unlock()

	bs.state = StateCompleted

	// Call onComplete callback if set
	if bs.onComplete != nil {
		if err := bs.onComplete(bs.results); err != nil {
			bs.handleError(err)
		}
	}
}

// handleError handles simulation errors
func (bs *BaseSimulation) handleError(err error) {
	bs.mu.Lock()
	defer bs.mu.Unlock()

	bs.state = StateError

	// Call onError callback if set
	if bs.onError != nil {
		if err := bs.onError(err); err != nil {
			// Log the error from the callback
			fmt.Printf("Error in onError callback: %v\n", err)
		}
	}
}

// cleanup performs cleanup when simulation ends
func (bs *BaseSimulation) cleanup() {
	bs.mu.Lock()
	defer bs.mu.Unlock()

	if bs.ticker != nil {
		bs.ticker.Stop()
	}

	if bs.cancel != nil {
		bs.cancel()
	}
}

// Pause pauses the simulation
func (bs *BaseSimulation) Pause() error {
	bs.mu.Lock()
	defer bs.mu.Unlock()

	if bs.state != StateRunning {
		return fmt.Errorf("simulation not running")
	}

	bs.state = StatePaused
	if bs.ticker != nil {
		bs.ticker.Stop()
	}

	return nil
}

// Resume resumes a paused simulation
func (bs *BaseSimulation) Resume() error {
	bs.mu.Lock()
	defer bs.mu.Unlock()

	if bs.state != StatePaused {
		return fmt.Errorf("simulation not paused")
	}

	bs.state = StateRunning
	bs.ticker = time.NewTicker(bs.config.TickRate)

	return nil
}

// Stop stops the simulation
func (bs *BaseSimulation) Stop() error {
	bs.mu.Lock()
	defer bs.mu.Unlock()

	if bs.state == StateCompleted || bs.state == StateError {
		return nil
	}

	if bs.cancel != nil {
		bs.cancel()
	}

	return nil
}

// GetState returns the current simulation state
func (bs *BaseSimulation) GetState() SimulationState {
	bs.mu.RLock()
	defer bs.mu.RUnlock()
	return bs.state
}

// GetResults returns all simulation results
func (bs *BaseSimulation) GetResults() []SimulationResult {
	bs.mu.RLock()
	defer bs.mu.RUnlock()

	// Return a copy to prevent external modification
	results := make([]SimulationResult, len(bs.results))
	copy(results, bs.results)
	return results
}

// GetConfig returns the current configuration
func (bs *BaseSimulation) GetConfig() SimulationConfig {
	bs.mu.RLock()
	defer bs.mu.RUnlock()
	return bs.config
}

// UpdateConfig updates the simulation configuration
func (bs *BaseSimulation) UpdateConfig(config SimulationConfig) error {
	bs.mu.Lock()
	defer bs.mu.Unlock()

	if bs.state == StateRunning {
		return fmt.Errorf("cannot update config while running")
	}

	bs.config = config
	return nil
}

// SetOnTick sets the callback for each tick
func (bs *BaseSimulation) SetOnTick(callback func(iteration int, data map[string]interface{}) error) {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	bs.onTick = callback
}

// SetOnComplete sets the callback for completion
func (bs *BaseSimulation) SetOnComplete(callback func(results []SimulationResult) error) {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	bs.onComplete = callback
}

// SetOnError sets the error callback
func (bs *BaseSimulation) SetOnError(callback func(error) error) {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	bs.onError = callback
}

// SetSimulation sets the simulation reference for method calls
func (bs *BaseSimulation) SetSimulation(sim SimulationEngine) {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	bs.simulation = sim
}

// String returns string representation of SimulationState
func (s SimulationState) String() string {
	switch s {
	case StateInitialized:
		return "Initialized"
	case StateRunning:
		return "Running"
	case StatePaused:
		return "Paused"
	case StateCompleted:
		return "Completed"
	case StateError:
		return "Error"
	default:
		return "Unknown"
	}
}
