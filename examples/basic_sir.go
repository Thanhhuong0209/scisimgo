package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/scisimgo/internal/models"
)

func main() {
	fmt.Println("🦠 Basic SIR Simulation Example")
	fmt.Println("================================")

	// Create SIR model
	sir := models.NewSIRModel()

	// Configure simulation
	config := models.SIRConfig{
		PopulationSize:  10000,
		InitialInfected: 100,
		InfectionRate:   0.3,
		RecoveryRate:    0.1,
		EnableStochastic: false,
	}

	// Initialize the model
	if err := sir.Initialize(config); err != nil {
		log.Fatalf("Failed to initialize SIR model: %v", err)
	}

	// Set up callbacks for real-time monitoring
	sir.SetOnTick(func(result models.SimulationResult) {
		if result.Iteration%10 == 0 { // Print every 10th iteration
			fmt.Printf("Tick %d: S=%d, I=%d, R=%d\n",
				result.Iteration,
				int(result.Data["susceptible"].(float64)),
				int(result.Data["infected"].(float64)),
				int(result.Data["recovered"].(float64)))
		}
	})

	sir.SetOnComplete(func() {
		fmt.Println("✅ Simulation completed!")
	})

	// Run simulation
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	fmt.Println("🚀 Starting SIR simulation...")
	if err := sir.Run(ctx); err != nil {
		log.Fatalf("Simulation failed: %v", err)
	}

	// Get and display statistics
	stats := sir.GetStatistics()
	fmt.Println("\n📊 Simulation Statistics:")
	fmt.Printf("Total Iterations: %v\n", stats["total_iterations"])
	fmt.Printf("Peak Infected: %v\n", stats["peak_infected"])
	fmt.Printf("Attack Rate: %.2f%%\n", stats["attack_rate"])
	fmt.Printf("Epidemic Duration: %v\n", stats["epidemic_duration"])

	// Export results
	if err := sir.ExportData("csv"); err != nil {
		log.Printf("Export failed: %v", err)
	} else {
		fmt.Println("📁 Results exported to CSV")
	}
}
