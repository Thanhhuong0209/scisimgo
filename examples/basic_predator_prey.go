package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/scisimgo/internal/models"
)

func main() {
	fmt.Println("🦌 Basic Predator-Prey Simulation Example")
	fmt.Println("==========================================")

	// Create predator-prey model
	pp := models.NewPredatorPreyModel()

	// Configure simulation
	config := models.PredatorPreyConfig{
		InitialPrey:          1000,
		InitialPredator:      100,
		PreyGrowthRate:       0.1,
		PredationRate:        0.01,
		PredatorDeathRate:    0.1,
		ConversionEfficiency: 0.1,
		EnableStochastic:     false,
	}

	// Initialize the model
	if err := pp.Initialize(config); err != nil {
		log.Fatalf("Failed to initialize Predator-Prey model: %v", err)
	}

	// Set up callbacks for real-time monitoring
	pp.SetOnTick(func(result models.SimulationResult) {
		if result.Iteration%10 == 0 { // Print every 10th iteration
			fmt.Printf("Tick %d: Prey=%d, Predator=%d\n",
				result.Iteration,
				int(result.Data["prey"].(float64)),
				int(result.Data["predator"].(float64)))
		}
	})

	pp.SetOnComplete(func() {
		fmt.Println("✅ Simulation completed!")
	})

	// Run simulation
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	fmt.Println("🚀 Starting Predator-Prey simulation...")
	if err := pp.Run(ctx); err != nil {
		log.Fatalf("Simulation failed: %v", err)
	}

	// Get and display statistics
	stats := pp.GetStatistics()
	fmt.Println("\n📊 Simulation Statistics:")
	fmt.Printf("Total Iterations: %v\n", stats["total_iterations"])
	fmt.Printf("Peak Prey: %v\n", stats["peak_prey"])
	fmt.Printf("Peak Predator: %v\n", stats["peak_predator"])
	fmt.Printf("Cycle Count: %v\n", stats["cycle_count"])
	fmt.Printf("Stability Index: %.4f\n", stats["stability_index"])

	// Export results
	if err := pp.ExportData("csv"); err != nil {
		log.Printf("Export failed: %v", err)
	} else {
		fmt.Println("📁 Results exported to CSV")
	}
}
