package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/scisimgo/internal/models"
)

func main() {
	fmt.Println("🚀 Advanced Multi-Simulation Example")
	fmt.Println("====================================")

	// Run multiple simulations concurrently
	var wg sync.WaitGroup
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// SIR Simulation
	wg.Add(1)
	go func() {
		defer wg.Done()
		runSIRSimulation(ctx)
	}()

	// Predator-Prey Simulation
	wg.Add(1)
	go func() {
		defer wg.Done()
		runPredatorPreySimulation(ctx)
	}()

	// Orbital Simulation
	wg.Add(1)
	go func() {
		defer wg.Done()
		runOrbitalSimulation(ctx)
	}()

	// Wait for all simulations to complete
	wg.Wait()
	fmt.Println("🎉 All simulations completed!")
}

func runSIRSimulation(ctx context.Context) {
	fmt.Println("🦠 Starting SIR simulation...")
	
	sir := models.NewSIRModel()
	config := models.SIRConfig{
		PopulationSize:  50000,
		InitialInfected: 500,
		InfectionRate:   0.25,
		RecoveryRate:    0.08,
		EnableStochastic: true,
	}

	if err := sir.Initialize(config); err != nil {
		log.Printf("SIR initialization failed: %v", err)
		return
	}

	// Set up monitoring
	sir.SetOnTick(func(result models.SimulationResult) {
		if result.Iteration%50 == 0 {
			fmt.Printf("SIR Tick %d: S=%d, I=%d, R=%d\n",
				result.Iteration,
				int(result.Data["susceptible"].(float64)),
				int(result.Data["infected"].(float64)),
				int(result.Data["recovered"].(float64)))
		}
	})

	if err := sir.Run(ctx); err != nil {
		log.Printf("SIR simulation failed: %v", err)
		return
	}

	stats := sir.GetStatistics()
	fmt.Printf("SIR Results: Peak Infected=%v, Attack Rate=%.2f%%\n",
		stats["peak_infected"], stats["attack_rate"])

	sir.ExportData("csv")
}

func runPredatorPreySimulation(ctx context.Context) {
	fmt.Println("🦌 Starting Predator-Prey simulation...")
	
	pp := models.NewPredatorPreyModel()
	config := models.PredatorPreyConfig{
		InitialPrey:          2000,
		InitialPredator:      200,
		PreyGrowthRate:       0.08,
		PredationRate:        0.008,
		PredatorDeathRate:    0.12,
		ConversionEfficiency: 0.08,
		EnableStochastic:     true,
	}

	if err := pp.Initialize(config); err != nil {
		log.Printf("Predator-Prey initialization failed: %v", err)
		return
	}

	// Set up monitoring
	pp.SetOnTick(func(result models.SimulationResult) {
		if result.Iteration%50 == 0 {
			fmt.Printf("PP Tick %d: Prey=%d, Predator=%d\n",
				result.Iteration,
				int(result.Data["prey"].(float64)),
				int(result.Data["predator"].(float64)))
		}
	})

	if err := pp.Run(ctx); err != nil {
		log.Printf("Predator-Prey simulation failed: %v", err)
		return
	}

	stats := pp.GetStatistics()
	fmt.Printf("PP Results: Peak Prey=%v, Peak Predator=%v, Cycles=%v\n",
		stats["peak_prey"], stats["peak_predator"], stats["cycle_count"])

	pp.ExportData("csv")
}

func runOrbitalSimulation(ctx context.Context) {
	fmt.Println("🪐 Starting Orbital simulation...")
	
	orbital := models.NewOrbitalModel()
	
	// Extended solar system
	bodies := []models.CelestialBody{
		{
			Name:     "Sun",
			Mass:     1.99e30,
			Position: [3]float64{0, 0, 0},
			Velocity: [3]float64{0, 0, 0},
		},
		{
			Name:     "Mercury",
			Mass:     3.30e23,
			Position: [3]float64{5.79e10, 0, 0},
			Velocity: [3]float64{0, 47400, 0},
		},
		{
			Name:     "Venus",
			Mass:     4.87e24,
			Position: [3]float64{1.08e11, 0, 0},
			Velocity: [3]float64{0, 35000, 0},
		},
		{
			Name:     "Earth",
			Mass:     5.97e24,
			Position: [3]float64{1.496e11, 0, 0},
			Velocity: [3]float64{0, 29780, 0},
		},
		{
			Name:     "Mars",
			Mass:     6.39e23,
			Position: [3]float64{2.28e11, 0, 0},
			Velocity: [3]float64{0, 24070, 0},
		},
	}

	config := models.OrbitalConfig{
		GravitationalConstant: 6.67e-11,
		TimeStep:             3600, // 1 hour
		Enable3D:             false,
		CelestialBodies:      bodies,
	}

	if err := orbital.Initialize(config); err != nil {
		log.Printf("Orbital initialization failed: %v", err)
		return
	}

	// Set up monitoring
	orbital.SetOnTick(func(result models.SimulationResult) {
		if result.Iteration%50 == 0 {
			fmt.Printf("Orbital Tick %d: Earth distance=%.2e m\n",
				result.Iteration,
				result.Data["earth_distance_from_sun"])
		}
	})

	if err := orbital.Run(ctx); err != nil {
		log.Printf("Orbital simulation failed: %v", err)
		return
	}

	stats := orbital.GetStatistics()
	fmt.Printf("Orbital Results: Bodies=%v, Time Step=%.1f s\n",
		stats["number_of_bodies"], stats["time_step"])

	orbital.ExportData("csv")
}
