package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/scisimgo/internal/models"
)

func main() {
	fmt.Println("🪐 Basic Orbital Simulation Example")
	fmt.Println("===================================")

	// Create orbital model
	orbital := models.NewOrbitalModel()

	// Configure celestial bodies
	bodies := []models.CelestialBody{
		{
			Name:     "Sun",
			Mass:     1.99e30,
			Position: [3]float64{0, 0, 0},
			Velocity: [3]float64{0, 0, 0},
		},
		{
			Name:     "Earth",
			Mass:     5.97e24,
			Position: [3]float64{1.496e11, 0, 0}, // 1 AU
			Velocity: [3]float64{0, 29780, 0},    // Earth's orbital velocity
		},
		{
			Name:     "Mars",
			Mass:     6.39e23,
			Position: [3]float64{2.28e11, 0, 0}, // 1.52 AU
			Velocity: [3]float64{0, 24070, 0},   // Mars's orbital velocity
		},
	}

	// Configure simulation
	config := models.OrbitalConfig{
		GravitationalConstant: 6.67e-11,
		TimeStep:             1000, // 1000 seconds
		Enable3D:             false,
		CelestialBodies:      bodies,
	}

	// Initialize the model
	if err := orbital.Initialize(config); err != nil {
		log.Fatalf("Failed to initialize Orbital model: %v", err)
	}

	// Set up callbacks for real-time monitoring
	orbital.SetOnTick(func(result models.SimulationResult) {
		if result.Iteration%10 == 0 { // Print every 10th iteration
			fmt.Printf("Tick %d: Earth distance from Sun: %.2e m\n",
				result.Iteration,
				result.Data["earth_distance_from_sun"])
		}
	})

	orbital.SetOnComplete(func() {
		fmt.Println("✅ Simulation completed!")
	})

	// Run simulation
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	fmt.Println("🚀 Starting Orbital simulation...")
	if err := orbital.Run(ctx); err != nil {
		log.Fatalf("Simulation failed: %v", err)
	}

	// Get and display statistics
	stats := orbital.GetStatistics()
	fmt.Println("\n📊 Simulation Statistics:")
	fmt.Printf("Total Iterations: %v\n", stats["total_iterations"])
	fmt.Printf("Time Step: %.1f seconds\n", stats["time_step"])
	fmt.Printf("Number of Bodies: %v\n", stats["number_of_bodies"])

	// Display orbital periods
	if periods, ok := stats["orbital_periods"].(map[string]interface{}); ok {
		fmt.Println("\n🪐 Orbital Periods:")
		for body, period := range periods {
			fmt.Printf("  %s: %.1f days\n", body, period)
		}
	}

	// Display energy conservation
	if energy, ok := stats["energy_conservation"].(map[string]interface{}); ok {
		fmt.Println("\n⚡ Energy Conservation:")
		fmt.Printf("  Initial Energy: %.2e J\n", energy["initial_energy"])
		fmt.Printf("  Final Energy: %.2e J\n", energy["final_energy"])
		fmt.Printf("  Conservation Error: %.2e J\n", energy["conservation_error"])
	}

	// Export results
	if err := orbital.ExportData("csv"); err != nil {
		log.Printf("Export failed: %v", err)
	} else {
		fmt.Println("📁 Results exported to CSV")
	}
}
