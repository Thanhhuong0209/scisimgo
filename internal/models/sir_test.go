package models

import (
	"testing"
	"time"

	"github.com/scisimgo/internal/engine"
)

func TestNewSIRModel(t *testing.T) {
	sir := NewSIRModel()
	
	if sir == nil {
		t.Fatal("NewSIRModel returned nil")
	}
	
	if sir.BaseSimulation == nil {
		t.Fatal("BaseSimulation not initialized")
	}
	
	if sir.rng == nil {
		t.Fatal("Random number generator not initialized")
	}
	
	if sir.PopulationSize != 0 {
		t.Errorf("Expected population size 0, got %d", sir.PopulationSize)
	}
}

func TestSIRModel_Initialize(t *testing.T) {
	sir := NewSIRModel()
	
	config := SIRConfig{
		SimulationConfig: engine.SimulationConfig{
			Duration:      100 * time.Millisecond,
			TickRate:      10 * time.Millisecond,
			MaxIterations: 10,
		},
		PopulationSize:  1000,
		InitialInfected: 100,
		InfectionRate:   0.3,
		RecoveryRate:    0.1,
		EnableStochastic: false,
	}
	
	err := sir.Initialize(config)
	if err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	
	if sir.PopulationSize != 1000 {
		t.Errorf("Expected population size 1000, got %d", sir.PopulationSize)
	}
	
	if sir.InitialInfected != 100 {
		t.Errorf("Expected initial infected 100, got %d", sir.InitialInfected)
	}
	
	if sir.InfectionRate != 0.3 {
		t.Errorf("Expected infection rate 0.3, got %f", sir.InfectionRate)
	}
	
	if sir.RecoveryRate != 0.1 {
		t.Errorf("Expected recovery rate 0.1, got %f", sir.RecoveryRate)
	}
	
	if sir.Susceptible != 900 {
		t.Errorf("Expected susceptible 900, got %d", sir.Susceptible)
	}
	
	if sir.Infected != 100 {
		t.Errorf("Expected infected 100, got %d", sir.Infected)
	}
	
	if sir.Recovered != 0 {
		t.Errorf("Expected recovered 0, got %d", sir.Recovered)
	}
}

func TestSIRModel_Initialize_Validation(t *testing.T) {
	tests := []struct {
		name    string
		config  SIRConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: SIRConfig{
				SimulationConfig: engine.SimulationConfig{
					Duration:      100 * time.Millisecond,
					TickRate:      10 * time.Millisecond,
					MaxIterations: 10,
				},
				PopulationSize:  1000,
				InitialInfected: 100,
				InfectionRate:   0.3,
				RecoveryRate:    0.1,
			},
			wantErr: false,
		},
		{
			name: "negative population size",
			config: SIRConfig{
				SimulationConfig: engine.SimulationConfig{
					Duration:      100 * time.Millisecond,
					TickRate:      10 * time.Millisecond,
					MaxIterations: 10,
				},
				PopulationSize:  -1000,
				InitialInfected: 100,
				InfectionRate:   0.3,
				RecoveryRate:    0.1,
			},
			wantErr: true,
		},
		{
			name: "zero population size",
			config: SIRConfig{
				SimulationConfig: engine.SimulationConfig{
					Duration:      100 * time.Millisecond,
					TickRate:      10 * time.Millisecond,
					MaxIterations: 10,
				},
				PopulationSize:  0,
				InitialInfected: 100,
				InfectionRate:   0.3,
				RecoveryRate:    0.1,
			},
			wantErr: true,
		},
		{
			name: "negative initial infected",
			config: SIRConfig{
				SimulationConfig: engine.SimulationConfig{
					Duration:      100 * time.Millisecond,
					TickRate:      10 * time.Millisecond,
					MaxIterations: 10,
				},
				PopulationSize:  1000,
				InitialInfected: -100,
				InfectionRate:   0.3,
				RecoveryRate:    0.1,
			},
			wantErr: true,
		},
		{
			name: "initial infected exceeds population",
			config: SIRConfig{
				SimulationConfig: engine.SimulationConfig{
					Duration:      100 * time.Millisecond,
					TickRate:      10 * time.Millisecond,
					MaxIterations: 10,
				},
				PopulationSize:  1000,
				InitialInfected: 1500,
				InfectionRate:   0.3,
				RecoveryRate:    0.1,
			},
			wantErr: true,
		},
		{
			name: "negative infection rate",
			config: SIRConfig{
				SimulationConfig: engine.SimulationConfig{
					Duration:      100 * time.Millisecond,
					TickRate:      10 * time.Millisecond,
					MaxIterations: 10,
				},
				PopulationSize:  1000,
				InitialInfected: 100,
				InfectionRate:   -0.3,
				RecoveryRate:    0.1,
			},
			wantErr: true,
		},
		{
			name: "negative recovery rate",
			config: SIRConfig{
				SimulationConfig: engine.SimulationConfig{
					Duration:      100 * time.Millisecond,
					TickRate:      10 * time.Millisecond,
					MaxIterations: 10,
				},
				PopulationSize:  1000,
				InitialInfected: 100,
				InfectionRate:   0.3,
				RecoveryRate:    -0.1,
			},
			wantErr: true,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sir := NewSIRModel()
			err := sir.Initialize(tt.config)
			
			if (err != nil) != tt.wantErr {
				t.Errorf("Initialize() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestSIRModel_CalculateNewInfections(t *testing.T) {
	sir := NewSIRModel()
	
	// Set up model state
	sir.PopulationSize = 1000
	sir.Susceptible = 800
	sir.Infected = 200
	sir.InfectionRate = 0.3
	
	newInfections := sir.calculateNewInfections()
	
	// Expected: 800 * 200 * 0.3 / 1000 = 48
	expected := int(float64(sir.Susceptible) * float64(sir.Infected) * sir.InfectionRate / float64(sir.PopulationSize))
	
	if newInfections != expected {
		t.Errorf("Expected new infections %d, got %d", expected, newInfections)
	}
}

func TestSIRModel_CalculateNewRecoveries(t *testing.T) {
	sir := NewSIRModel()
	
	// Set up model state
	sir.Infected = 200
	sir.RecoveryRate = 0.1
	
	newRecoveries := sir.calculateNewRecoveries()
	
	// Expected: 200 * 0.1 = 20
	expected := int(float64(sir.Infected) * sir.RecoveryRate)
	
	if newRecoveries != expected {
		t.Errorf("Expected new recoveries %d, got %d", expected, newRecoveries)
	}
}

func TestSIRModel_OnTick(t *testing.T) {
	sir := NewSIRModel()
	
	// Initialize with small values for testing
	config := SIRConfig{
		SimulationConfig: engine.SimulationConfig{
			Duration:      100 * time.Millisecond,
			TickRate:      10 * time.Millisecond,
			MaxIterations: 10,
		},
		PopulationSize:  100,
		InitialInfected: 10,
		InfectionRate:   0.3,
		RecoveryRate:    0.1,
	}
	
	err := sir.Initialize(config)
	if err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	
	// Store initial values
	initialSusceptible := sir.Susceptible
	initialInfected := sir.Infected
	initialRecovered := sir.Recovered
	
	data := make(map[string]interface{})
	
	err = sir.onTick(1, data)
	if err != nil {
		t.Fatalf("onTick failed: %v", err)
	}
	
	// Verify state changes
	if sir.Susceptible >= initialSusceptible {
		t.Error("Susceptible should decrease")
	}
	
	if sir.Infected < initialInfected && sir.Infected > initialInfected {
		t.Error("Infected should change (increase or decrease)")
	}
	
	if sir.Recovered < initialRecovered {
		t.Error("Recovered should not decrease")
	}
	
	// Verify data map is populated
	if data["susceptible"] == nil {
		t.Error("Data map should contain susceptible value")
	}
	
	if data["infected"] == nil {
		t.Error("Data map should contain infected value")
	}
	
	if data["recovered"] == nil {
		t.Error("Data map should contain recovered value")
	}
}

func TestSIRModel_GenerateTickData(t *testing.T) {
	sir := NewSIRModel()
	
	// Set up model state
	sir.PopulationSize = 1000
	sir.Susceptible = 800
	sir.Infected = 200
	sir.Recovered = 0
	sir.InfectionRate = 0.3
	sir.RecoveryRate = 0.1
	sir.PeakInfected = 200
	sir.TotalCases = 200
	
	data := sir.generateTickData()
	
	expectedKeys := []string{
		"susceptible",
		"infected", 
		"recovered",
		"total_population",
		"infection_rate",
		"recovery_rate",
		"peak_infected",
		"total_cases",
	}
	
	for _, key := range expectedKeys {
		if data[key] == nil {
			t.Errorf("Data should contain key: %s", key)
		}
	}
	
	if data["susceptible"] != 800 {
		t.Errorf("Expected susceptible 800, got %v", data["susceptible"])
	}
	
	if data["infected"] != 200 {
		t.Errorf("Expected infected 200, got %v", data["infected"])
	}
}

func TestSIRModel_GetCurrentState(t *testing.T) {
	sir := NewSIRModel()
	
	// Set up model state
	sir.PopulationSize = 1000
	sir.Susceptible = 800
	sir.Infected = 200
	sir.Recovered = 0
	sir.InfectionRate = 0.3
	sir.RecoveryRate = 0.1
	sir.PeakInfected = 200
	sir.TotalCases = 200
	
	state := sir.GetCurrentState()
	
	if state.PopulationSize != 1000 {
		t.Errorf("Expected population size 1000, got %d", state.PopulationSize)
	}
	
	if state.Susceptible != 800 {
		t.Errorf("Expected susceptible 800, got %d", state.Susceptible)
	}
	
	if state.Infected != 200 {
		t.Errorf("Expected infected 200, got %d", state.Infected)
	}
}

func TestSIRModel_GetStatistics(t *testing.T) {
	sir := NewSIRModel()
	
	// Set up model state
	sir.PopulationSize = 1000
	sir.Susceptible = 800
	sir.Infected = 200
	sir.Recovered = 0
	sir.PeakInfected = 200
	sir.TotalCases = 200
	
	stats := sir.GetStatistics()
	
	if stats.TotalPopulation != 1000 {
		t.Errorf("Expected total population 1000, got %d", stats.TotalPopulation)
	}
	
	if stats.CurrentSusceptible != 800 {
		t.Errorf("Expected current susceptible 800, got %d", stats.CurrentSusceptible)
	}
	
	if stats.CurrentInfected != 200 {
		t.Errorf("Expected current infected 200, got %d", stats.CurrentInfected)
	}
	
	if stats.PeakInfected != 200 {
		t.Errorf("Expected peak infected 200, got %d", stats.PeakInfected)
	}
	
	if stats.TotalCases != 200 {
		t.Errorf("Expected total cases 200, got %d", stats.TotalCases)
	}
}

func TestValidationError(t *testing.T) {
	message := "test validation error"
	err := NewValidationError(message)
	
	if err.Error() != message {
		t.Errorf("Expected error message '%s', got '%s'", message, err.Error())
	}
	
	if err.Message != message {
		t.Errorf("Expected error message field '%s', got '%s'", message, err.Message)
	}
}
