package engine

import (
	"context"
	"testing"
	"time"
)

func TestNewBaseSimulation(t *testing.T) {
	bs := NewBaseSimulation()
	
	if bs == nil {
		t.Fatal("NewBaseSimulation returned nil")
	}
	
	if bs.state != StateInitialized {
		t.Errorf("Expected state %v, got %v", StateInitialized, bs.state)
	}
	
	if bs.results == nil {
		t.Fatal("Results slice not initialized")
	}
	
	if len(bs.results) != 0 {
		t.Errorf("Expected empty results, got %d", len(bs.results))
	}
}

func TestBaseSimulation_Initialize(t *testing.T) {
	bs := NewBaseSimulation()
	
	config := SimulationConfig{
		Duration:      100 * time.Millisecond,
		TickRate:      10 * time.Millisecond,
		MaxIterations: 10,
		EnableLogging: true,
	}
	
	err := bs.Initialize(config)
	if err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	
	if bs.state != StateInitialized {
		t.Errorf("Expected state %v, got %v", StateInitialized, bs.state)
	}
	
	if bs.config.TickRate != config.TickRate {
		t.Errorf("Expected tick rate %v, got %v", config.TickRate, bs.config.TickRate)
	}
}

func TestBaseSimulation_Initialize_Validation(t *testing.T) {
	bs := NewBaseSimulation()
	
	tests := []struct {
		name    string
		config  SimulationConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: SimulationConfig{
				Duration:      100 * time.Millisecond,
				TickRate:      10 * time.Millisecond,
				MaxIterations: 10,
			},
			wantErr: false,
		},
		{
			name: "invalid tick rate",
			config: SimulationConfig{
				Duration:      100 * time.Millisecond,
				TickRate:      0,
				MaxIterations: 10,
			},
			wantErr: true,
		},
		{
			name: "no duration or max iterations",
			config: SimulationConfig{
				Duration:      0,
				TickRate:      10 * time.Millisecond,
				MaxIterations: 0,
			},
			wantErr: true,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bs := NewBaseSimulation()
			err := bs.Initialize(tt.config)
			
			if (err != nil) != tt.wantErr {
				t.Errorf("Initialize() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestBaseSimulation_Run(t *testing.T) {
	bs := NewBaseSimulation()
	
	config := SimulationConfig{
		Duration:      50 * time.Millisecond,
		TickRate:      10 * time.Millisecond,
		MaxIterations: 5,
		EnableLogging: false,
	}
	
	err := bs.Initialize(config)
	if err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	
	// Set up a simple tick callback
	bs.SetOnTick(func(iteration int, data map[string]interface{}) error {
		data["iteration"] = iteration
		return nil
	})
	
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	
	err = bs.Run(ctx)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	
	// Wait a bit for completion
	time.Sleep(100 * time.Millisecond)
	
	if bs.state != StateCompleted {
		t.Errorf("Expected state %v, got %v", StateCompleted, bs.state)
	}
	
	results := bs.GetResults()
	if len(results) == 0 {
		t.Error("Expected results, got none")
	}
}

func TestBaseSimulation_PauseResume(t *testing.T) {
	bs := NewBaseSimulation()
	
	config := SimulationConfig{
		Duration:      200 * time.Millisecond,
		TickRate:      20 * time.Millisecond,
		MaxIterations: 10,
		EnableLogging: false,
	}
	
	err := bs.Initialize(config)
	if err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	
	bs.SetOnTick(func(iteration int, data map[string]interface{}) error {
		return nil
	})
	
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	
	// Start simulation
	go func() {
		if err := bs.Run(ctx); err != nil {
			t.Errorf("Run failed: %v", err)
		}
	}()
	
	// Wait a bit then pause
	time.Sleep(50 * time.Millisecond)
	
	if err := bs.Pause(); err != nil {
		t.Fatalf("Pause failed: %v", err)
	}
	
	if bs.state != StatePaused {
		t.Errorf("Expected state %v, got %v", StatePaused, bs.state)
	}
	
	// Resume
	if err := bs.Resume(); err != nil {
		t.Fatalf("Resume failed: %v", err)
	}
	
	if bs.state != StateRunning {
		t.Errorf("Expected state %v, got %v", StateRunning, bs.state)
	}
	
	// Wait for completion
	time.Sleep(200 * time.Millisecond)
	
	if bs.state != StateCompleted {
		t.Errorf("Expected state %v, got %v", StateCompleted, bs.state)
	}
}

func TestBaseSimulation_Stop(t *testing.T) {
	bs := NewBaseSimulation()
	
	config := SimulationConfig{
		Duration:      500 * time.Millisecond,
		TickRate:      50 * time.Millisecond,
		MaxIterations: 20,
		EnableLogging: false,
	}
	
	err := bs.Initialize(config)
	if err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	
	bs.SetOnTick(func(iteration int, data map[string]interface{}) error {
		return nil
	})
	
	ctx, cancel := context.WithTimeout(context.Background(), 600*time.Millisecond)
	defer cancel()
	
	// Start simulation
	go func() {
		if err := bs.Run(ctx); err != nil {
			t.Errorf("Run failed: %v", err)
		}
	}()
	
	// Wait a bit then stop
	time.Sleep(100 * time.Millisecond)
	
	if err := bs.Stop(); err != nil {
		t.Fatalf("Stop failed: %v", err)
	}
	
	// Wait a bit for stop to take effect
	time.Sleep(50 * time.Millisecond)
	
	// State should be stopped or completed
	if bs.state == StateRunning {
		t.Error("Simulation should not be running after stop")
	}
}

func TestBaseSimulation_GetResults(t *testing.T) {
	bs := NewBaseSimulation()
	
	// Add some mock results
	bs.results = []SimulationResult{
		{Iteration: 1, Data: map[string]interface{}{"value": 10}},
		{Iteration: 2, Data: map[string]interface{}{"value": 20}},
	}
	
	results := bs.GetResults()
	
	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}
	
	// Verify results are copied (not shared)
	results[0].Data["value"] = 999
	if bs.results[0].Data["value"] == 999 {
		t.Error("Results should be copied, not shared")
	}
}

func TestSimulationState_String(t *testing.T) {
	tests := []struct {
		state SimulationState
		want  string
	}{
		{StateInitialized, "Initialized"},
		{StateRunning, "Running"},
		{StatePaused, "Paused"},
		{StateCompleted, "Completed"},
		{StateError, "Error"},
		{SimulationState(999), "Unknown"},
	}
	
	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := tt.state.String(); got != tt.want {
				t.Errorf("SimulationState.String() = %v, want %v", got, tt.want)
			}
		})
	}
}
