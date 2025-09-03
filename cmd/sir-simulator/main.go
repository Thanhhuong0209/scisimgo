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
		populationSize  = flag.Int("population", 10000, "Total population size")
		initialInfected = flag.Int("initial-infected", 100, "Initial number of infected individuals")
		infectionRate   = flag.Float64("infection-rate", 0.3, "Infection rate (β)")
		recoveryRate    = flag.Float64("recovery-rate", 0.1, "Recovery rate (γ)")
		duration        = flag.Duration("duration", 100*time.Second, "Simulation duration")
		tickRate        = flag.Duration("tick-rate", 100*time.Millisecond, "Simulation tick rate")
		maxIterations   = flag.Int("max-iterations", 0, "Maximum number of iterations (0 = unlimited)")
		outputDir       = flag.String("output", "data", "Output directory for CSV files")
		enableLogging   = flag.Bool("logging", true, "Enable detailed logging")
		exportInterval  = flag.Duration("export-interval", 1*time.Second, "Data export interval")
	)
	flag.Parse()

	// Create output directory
	if err := os.MkdirAll(*outputDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}

	// Create SIR model
	sir := models.NewSIRModel()

	// Configure simulation
	config := models.SIRConfig{
		SimulationConfig: engine.SimulationConfig{
			Duration:       *duration,
			TickRate:       *tickRate,
			MaxIterations:  *maxIterations,
			EnableLogging:  *enableLogging,
			ExportInterval: *exportInterval,
		},
		PopulationSize:   *populationSize,
		InitialInfected:  *initialInfected,
		InfectionRate:    *infectionRate,
		RecoveryRate:     *recoveryRate,
		EnableStochastic: false,
	}

	// Initialize the model
	if err := sir.Initialize(config); err != nil {
		log.Fatalf("Failed to initialize SIR model: %v", err)
	}

	// Set up logging callbacks
	if *enableLogging {
		sir.SetOnTick(func(iteration int, data map[string]interface{}) error {
			log.Printf("Tick %d: S=%d, I=%d, R=%d",
				iteration,
				data["susceptible"],
				data["infected"],
				data["recovered"])
			return nil
		})
	}

	// Set up completion callback
	sir.SetOnComplete(func(results []engine.SimulationResult) error {
		log.Printf("Simulation completed! Total iterations: %d", len(results))

		// Export final results
		exporter := export.NewCSVExporter(*outputDir)

		// Export full results
		if err := exporter.ExportResults(results, "sir_simulation_results"); err != nil {
			log.Printf("Warning: Failed to export full results: %v", err)
		}

		// Export time series data
		if err := exporter.ExportTimeSeries(results, "sir_time_series", "timestamp",
			[]string{"susceptible", "infected", "recovered"}); err != nil {
			log.Printf("Warning: Failed to export time series: %v", err)
		}

		// Print final statistics (moved to after simulation completes)
		log.Printf("Simulation data exported successfully")

		return nil
	})

	// Set up error callback
	sir.SetOnError(func(err error) error {
		log.Printf("Simulation error: %v", err)
		return err
	})

	// Print simulation configuration
	printConfiguration(config)

	// Run simulation
	log.Printf("Starting SIR simulation...")
	ctx := context.Background()

	startTime := time.Now()
	if err := sir.Run(ctx); err != nil {
		log.Fatalf("Simulation failed: %v", err)
	}

	elapsed := time.Since(startTime)
	log.Printf("Simulation completed in %v", elapsed)

	// Print final statistics after simulation completes
	stats := sir.GetStatistics()
	printStatistics(stats)

	// Export results if not already exported
	if len(sir.GetResults()) > 0 {
		exporter := export.NewCSVExporter(*outputDir)

		// Export with timestamp
		timestamp := time.Now().Format("20060102_150405")
		filename := fmt.Sprintf("sir_results_%s", timestamp)

		if err := exporter.ExportResults(sir.GetResults(), filename); err != nil {
			log.Printf("Warning: Failed to export results: %v", err)
		} else {
			log.Printf("Results exported to: %s", filepath.Join(*outputDir, filename+".csv"))
		}
	}
}

// printConfiguration prints the simulation configuration
func printConfiguration(config models.SIRConfig) {
	fmt.Println("=== SIR Simulation Configuration ===")
	fmt.Printf("Population Size: %d\n", config.PopulationSize)
	fmt.Printf("Initial Infected: %d\n", config.InitialInfected)
	fmt.Printf("Infection Rate (β): %.3f\n", config.InfectionRate)
	fmt.Printf("Recovery Rate (γ): %.3f\n", config.RecoveryRate)
	fmt.Printf("Simulation Duration: %v\n", config.Duration)
	fmt.Printf("Tick Rate: %v\n", config.TickRate)
	fmt.Printf("Max Iterations: %d\n", config.MaxIterations)
	fmt.Printf("Output Directory: %s\n", config.ExportInterval)
	fmt.Printf("Export Interval: %v\n", config.ExportInterval)
	fmt.Printf("Logging Enabled: %t\n", config.EnableLogging)
	fmt.Println("=====================================")
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

	if peakInfected, ok := stats["peak_infected"]; ok {
		fmt.Printf("Peak Infected: %v\n", peakInfected)
	}

	if peakTime, ok := stats["peak_time"]; ok {
		fmt.Printf("Peak Time: %v\n", peakTime)
	}

	if totalCases, ok := stats["total_cases"]; ok {
		fmt.Printf("Total Cases: %v\n", totalCases)
	}

	if epidemicDuration, ok := stats["epidemic_duration"]; ok {
		fmt.Printf("Epidemic Duration: %v\n", epidemicDuration)
	}

	if attackRate, ok := stats["attack_rate"]; ok {
		fmt.Printf("Attack Rate: %.2f%%\n", float64(attackRate.(float64))*100)
	}

	if doublingTime, ok := stats["doubling_time"]; ok {
		if doublingTime.(time.Duration) > 0 {
			fmt.Printf("Doubling Time: %v\n", doublingTime)
		}
	}

	if avgGrowthRate, ok := stats["avg_growth_rate"]; ok {
		fmt.Printf("Average Growth Rate: %.3f\n", avgGrowthRate)
	}

	fmt.Println("================================")
}
