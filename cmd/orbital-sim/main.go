package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/scisimgo/internal/engine"
	"github.com/scisimgo/internal/export"
	"github.com/scisimgo/internal/models"
)

func main() {
	// Parse command line flags
	var (
		gravitationalConstant = flag.Float64("gravitational-constant", 6.67430e-11, "Gravitational constant G")
		timeStep = flag.Float64("time-step", 1000.0, "Simulation time step in seconds")
		enable3D = flag.Bool("enable-3d", false, "Enable 3D simulation")
		duration = flag.Duration("duration", 300*time.Second, "Simulation duration")
		tickRate = flag.Duration("tick-rate", 100*time.Millisecond, "Simulation tick rate")
		maxIterations = flag.Int("max-iterations", 0, "Maximum number of iterations (0 = unlimited)")
		outputDir = flag.String("output", "data", "Output directory for CSV files")
		enableLogging = flag.Bool("logging", true, "Enable detailed logging")
		exportInterval = flag.Duration("export-interval", 1*time.Second, "Data export interval")
	)
	flag.Parse()

	// Create output directory
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}

	// Create celestial bodies (Solar System example)
	bodies := createSolarSystemBodies()

	// Create Orbital model
	orbital := models.NewOrbitalModel()

	// Configure simulation
	config := models.OrbitalConfig{
		SimulationConfig: engine.SimulationConfig{
			Duration:       *duration,
			TickRate:       *tickRate,
			MaxIterations:  *maxIterations,
			EnableLogging:  *enableLogging,
			ExportInterval: *exportInterval,
		},
		GravitationalConstant: *gravitationalConstant,
		TimeStep:             *timeStep,
		Enable3D:             *enable3D,
		Bodies:               bodies,
	}

	// Initialize the model
	if err := orbital.Initialize(config); err != nil {
		log.Fatalf("Failed to initialize Orbital model: %v", err)
	}

	// Set up logging callbacks
	if *enableLogging {
		orbital.SetOnTick(func(iteration int, data map[string]interface{}) error {
			log.Printf("Tick %d: Sun distance - Earth: %.2e m, Mars: %.2e m", 
				iteration,
				data["Earth_distance_from_origin"],
				data["Mars_distance_from_origin"])
			return nil
		})
	}

	// Set up completion callback
	orbital.SetOnComplete(func(results []engine.SimulationResult) error {
		log.Printf("Simulation completed! Total iterations: %d", len(results))
		
		// Export final results
		exporter := export.NewCSVExporter(*outputDir)
		
		// Export full results
		if err := exporter.ExportResults(results, "orbital_simulation_results"); err != nil {
			log.Printf("Warning: Failed to export full results: %v", err)
		}
		
		// Export time series data for each body
		for _, body := range bodies {
			columns := []string{
				body.Name + "_position_x",
				body.Name + "_position_y",
				body.Name + "_position_z",
				body.Name + "_velocity_x",
				body.Name + "_velocity_y",
				body.Name + "_velocity_z",
				body.Name + "_distance_from_origin",
			}
			
			filename := fmt.Sprintf("orbital_%s_trajectory", body.Name)
			if err := exporter.ExportTimeSeries(results, filename, "timestamp", columns); err != nil {
				log.Printf("Warning: Failed to export trajectory for %s: %v", body.Name, err)
			}
		}
		
		// Print final statistics
		stats := orbital.GetStatistics()
		printStatistics(stats)
		
		return nil
	})

	// Set up error callback
	orbital.SetOnError(func(err error) error {
		log.Printf("Simulation error: %v", err)
		return err
	})

	// Print simulation configuration
	printConfiguration(config)

	// Run simulation
	log.Printf("Starting Orbital simulation...")
	ctx := context.Background()
	
	startTime := time.Now()
	if err := orbital.Run(ctx); err != nil {
		log.Fatalf("Simulation failed: %v", err)
	}
	
	duration := time.Since(startTime)
	log.Printf("Simulation completed in %v", duration)

	// Export results if not already exported
	if len(orbital.GetResults()) > 0 {
		exporter := export.NewCSVExporter(*outputDir)
		
		// Export with timestamp
		timestamp := time.Now().Format("20060102_150405")
		filename := fmt.Sprintf("orbital_results_%s", timestamp)
		
		if err := exporter.ExportResults(orbital.GetResults(), filename); err != nil {
			log.Printf("Warning: Failed to export results: %v", err)
		} else {
			log.Printf("Results exported to: %s", filepath.Join(*outputDir, filename+".csv"))
		}
	}
}

// createSolarSystemBodies creates a simplified solar system model
func createSolarSystemBodies() []*models.CelestialBody {
	// Constants for realistic scaling (simplified)
	const (
		AU = 1.496e11        // Astronomical Unit in meters
		MSun = 1.989e30      // Solar mass in kg
		MEarth = 5.972e24    // Earth mass in kg
		MMars = 6.39e23      // Mars mass in kg
		RSun = 6.96e8        // Solar radius in meters
		REarth = 6.371e6     // Earth radius in meters
		RMars = 3.39e6       // Mars radius in meters
	)
	
	// Create Sun (at origin)
	sun := &models.CelestialBody{
		Name:     "Sun",
		Mass:     MSun,
		Position: models.Vector3D{X: 0, Y: 0, Z: 0},
		Velocity: models.Vector3D{X: 0, Y: 0, Z: 0},
		Radius:   RSun,
		Color:    "yellow",
	}
	
	// Create Earth (circular orbit)
	earth := &models.CelestialBody{
		Name:     "Earth",
		Mass:     MEarth,
		Position: models.Vector3D{X: AU, Y: 0, Z: 0},
		Velocity: models.Vector3D{X: 0, Y: 29780, Z: 0}, // Orbital velocity ~29.78 km/s
		Radius:   REarth,
		Color:    "blue",
	}
	
	// Create Mars (elliptical orbit)
	mars := &models.CelestialBody{
		Name:     "Mars",
		Mass:     MMars,
		Position: models.Vector3D{X: 1.524 * AU, Y: 0, Z: 0},
		Velocity: models.Vector3D{X: 0, Y: 24100, Z: 0}, // Orbital velocity ~24.1 km/s
		Radius:   RMars,
		Color:    "red",
	}
	
	return []*models.CelestialBody{sun, earth, mars}
}

// printConfiguration prints the simulation configuration
func printConfiguration(config models.OrbitalConfig) {
	fmt.Println("=== Orbital Simulation Configuration ===")
	fmt.Printf("Gravitational Constant (G): %.2e m3/(kg*s2)\n", config.GravitationalConstant)
	fmt.Printf("Time Step: %.1f seconds\n", config.TimeStep)
	fmt.Printf("Enable 3D: %t\n", config.Enable3D)
	fmt.Printf("Number of Bodies: %d\n", len(config.Bodies))
	fmt.Printf("Simulation Duration: %v\n", config.Duration)
	fmt.Printf("Tick Rate: %v\n", config.TickRate)
	fmt.Printf("Max Iterations: %d\n", config.MaxIterations)
	fmt.Printf("Output Directory: %s\n", config.ExportInterval)
	fmt.Printf("Export Interval: %v\n", config.ExportInterval)
	fmt.Printf("Logging Enabled: %t\n", config.EnableLogging)
	
	fmt.Println("\nCelestial Bodies:")
	for _, body := range config.Bodies {
		fmt.Printf("  %s: Mass=%.2e kg, Position=(%.2e, %.2e, %.2e) m\n",
			body.Name, body.Mass, body.Position.X, body.Position.Y, body.Position.Z)
	}
	fmt.Println("=========================================")
}

// printStatistics prints the simulation statistics
func printStatistics(stats map[string]interface{}) {
	fmt.Println("\n=== Simulation Statistics ===")
	
	if totalIterations, ok := stats["total_iterations"]; ok {
		fmt.Printf("Total Iterations: %v\n", totalIterations)
	}
	
	if simulationDuration, ok := stats["simulation_duration"]; ok {
		fmt.Printf("Simulation Duration: %v\n", simulationDuration)
	}
	
	if timeStep, ok := stats["time_step"]; ok {
		fmt.Printf("Time Step: %.1f seconds\n", timeStep)
	}
	
	if bodiesCount, ok := stats["bodies_count"]; ok {
		fmt.Printf("Number of Bodies: %v\n", bodiesCount)
	}
	
	// Print orbital parameters
	if orbitalPeriods, ok := stats["orbital_periods"]; ok {
		if periods, ok := orbitalPeriods.(map[string]float64); ok {
			fmt.Println("\nOrbital Periods:")
			for body, period := range periods {
				fmt.Printf("  %s: %.2e seconds (%.2f days)\n", 
					body, period, period/86400)
			}
		}
	}
	
	if perihelion, ok := stats["perihelion"]; ok {
		if peri, ok := perihelion.(map[string]float64); ok {
			fmt.Println("\nPerihelion (Closest to Sun):")
			for body, distance := range peri {
				fmt.Printf("  %s: %.2e m (%.2f AU)\n", 
					body, distance, distance/1.496e11)
			}
		}
	}
	
	if aphelion, ok := stats["aphelion"]; ok {
		if aph, ok := aphelion.(map[string]float64); ok {
			fmt.Println("\nAphelion (Farthest from Sun):")
			for body, distance := range aph {
				fmt.Printf("  %s: %.2e m (%.2f AU)\n", 
					body, distance, distance/1.496e11)
			}
		}
	}
	
	if eccentricities, ok := stats["eccentricities"]; ok {
		if ecc, ok := eccentricities.(map[string]float64); ok {
			fmt.Println("\nOrbital Eccentricities:")
			for body, eccentricity := range ecc {
				fmt.Printf("  %s: %.4f\n", body, eccentricity)
			}
		}
	}
	
	// Print energy statistics
	if energyStats, ok := stats["energy_stats"]; ok {
		if energy, ok := energyStats.(map[string]interface{}); ok {
			fmt.Println("\nEnergy Conservation:")
			
			if totalEnergy, ok := energy["total_energy"].(map[string]interface{}); ok {
				if initial, ok := totalEnergy["initial"].(float64); ok {
					fmt.Printf("  Initial Total Energy: %.2e J\n", initial)
				}
				if final, ok := totalEnergy["final"].(float64); ok {
					fmt.Printf("  Final Total Energy: %.2e J\n", final)
				}
				if conservationError, ok := totalEnergy["conservation_error"].(float64); ok {
					fmt.Printf("  Energy Conservation Error: %.2e J\n", conservationError)
				}
			}
		}
	}
	
	fmt.Println("================================")
}
