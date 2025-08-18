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
		initialPrey = flag.Int("initial-prey", 1000, "Initial prey population")
		initialPredator = flag.Int("initial-predator", 100, "Initial predator population")
		preyGrowthRate = flag.Float64("prey-growth-rate", 0.1, "Prey growth rate (r)")
		predationRate = flag.Float64("predation-rate", 0.01, "Predation rate (a)")
		predatorDeathRate = flag.Float64("predator-death-rate", 0.1, "Predator death rate (m)")
		conversionEfficiency = flag.Float64("conversion-efficiency", 0.1, "Conversion efficiency (b)")
		duration = flag.Duration("duration", 200*time.Second, "Simulation duration")
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

	// Create Predator-Prey model
	pp := models.NewPredatorPreyModel()

	// Configure simulation
	config := models.PredatorPreyConfig{
		SimulationConfig: engine.SimulationConfig{
			Duration:       *duration,
			TickRate:       *tickRate,
			MaxIterations:  *maxIterations,
			EnableLogging:  *enableLogging,
			ExportInterval: *exportInterval,
		},
		InitialPrey:        *initialPrey,
		InitialPredator:    *initialPredator,
		PreyGrowthRate:     *preyGrowthRate,
		PredationRate:      *predationRate,
		PredatorDeathRate:  *predatorDeathRate,
		ConversionEfficiency: *conversionEfficiency,
		EnableStochastic:   false,
	}

	// Initialize the model
	if err := pp.Initialize(config); err != nil {
		log.Fatalf("Failed to initialize Predator-Prey model: %v", err)
	}

	// Set up logging callbacks
	if *enableLogging {
		pp.SetOnTick(func(iteration int, data map[string]interface{}) error {
			log.Printf("Tick %d: Prey=%d, Predator=%d", 
				iteration,
				data["prey"],
				data["predator"])
			return nil
		})
	}

	// Set up completion callback
	pp.SetOnComplete(func(results []engine.SimulationResult) error {
		log.Printf("Simulation completed! Total iterations: %d", len(results))
		
		// Export final results
		exporter := export.NewCSVExporter(*outputDir)
		
		// Export full results
		if err := exporter.ExportResults(results, "predator_prey_simulation_results"); err != nil {
			log.Printf("Warning: Failed to export full results: %v", err)
		}
		
		// Export time series data
		if err := exporter.ExportTimeSeries(results, "predator_prey_time_series", "timestamp", 
			[]string{"prey", "predator"}); err != nil {
			log.Printf("Warning: Failed to export time series: %v", err)
		}
		
		// Print final statistics
		stats := pp.GetStatistics()
		printStatistics(stats)
		
		return nil
	})

	// Set up error callback
	pp.SetOnError(func(err error) error {
		log.Printf("Simulation error: %v", err)
		return err
	})

	// Print simulation configuration
	printConfiguration(config)

	// Run simulation
	log.Printf("Starting Predator-Prey simulation...")
	ctx := context.Background()
	
	startTime := time.Now()
	if err := pp.Run(ctx); err != nil {
		log.Fatalf("Simulation failed: %v", err)
	}
	
	duration := time.Since(startTime)
	log.Printf("Simulation completed in %v", duration)

	// Export results if not already exported
	if len(pp.GetResults()) > 0 {
		exporter := export.NewCSVExporter(*outputDir)
		
		// Export with timestamp
		timestamp := time.Now().Format("20060102_150405")
		filename := fmt.Sprintf("predator_prey_results_%s", timestamp)
		
		if err := exporter.ExportResults(pp.GetResults(), filename); err != nil {
			log.Printf("Warning: Failed to export results: %v", err)
		} else {
			log.Printf("Results exported to: %s", filepath.Join(*outputDir, filename+".csv"))
		}
	}
}

// printConfiguration prints the simulation configuration
func printConfiguration(config models.PredatorPreyConfig) {
	fmt.Println("=== Predator-Prey Simulation Configuration ===")
	fmt.Printf("Initial Prey Population: %d\n", config.InitialPrey)
	fmt.Printf("Initial Predator Population: %d\n", config.InitialPredator)
	fmt.Printf("Prey Growth Rate (r): %.3f\n", config.PreyGrowthRate)
	fmt.Printf("Predation Rate (a): %.3f\n", config.PredationRate)
	fmt.Printf("Predator Death Rate (m): %.3f\n", config.PredatorDeathRate)
	fmt.Printf("Conversion Efficiency (b): %.3f\n", config.ConversionEfficiency)
	fmt.Printf("Simulation Duration: %v\n", config.Duration)
	fmt.Printf("Tick Rate: %v\n", config.TickRate)
	fmt.Printf("Max Iterations: %d\n", config.MaxIterations)
	fmt.Printf("Output Directory: %s\n", config.ExportInterval)
	fmt.Printf("Export Interval: %v\n", config.ExportInterval)
	fmt.Printf("Logging Enabled: %t\n", config.EnableLogging)
	fmt.Println("===============================================")
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
	
	if peakPrey, ok := stats["peak_prey"]; ok {
		fmt.Printf("Peak Prey Population: %v\n", peakPrey)
	}
	
	if peakPredator, ok := stats["peak_predator"]; ok {
		fmt.Printf("Peak Predator Population: %v\n", peakPredator)
	}
	
	if minPrey, ok := stats["min_prey"]; ok {
		fmt.Printf("Minimum Prey Population: %v\n", minPrey)
	}
	
	if minPredator, ok := stats["min_predator"]; ok {
		fmt.Printf("Minimum Predator Population: %v\n", minPredator)
	}
	
	if cycleCount, ok := stats["cycle_count"]; ok {
		fmt.Printf("Detected Cycles: %v\n", cycleCount)
	}
	
	if finalPrey, ok := stats["final_prey"]; ok {
		fmt.Printf("Final Prey Population: %v\n", finalPrey)
	}
	
	if finalPredator, ok := stats["final_predator"]; ok {
		fmt.Printf("Final Predator Population: %v\n", finalPredator)
	}
	
	if avgPrey, ok := stats["avg_prey"]; ok {
		fmt.Printf("Average Prey Population: %.1f\n", avgPrey)
	}
	
	if avgPredator, ok := stats["avg_predator"]; ok {
		fmt.Printf("Average Predator Population: %.1f\n", avgPredator)
	}
	
	if stabilityIndex, ok := stats["stability_index"]; ok {
		fmt.Printf("Stability Index: %.3f (lower = more stable)\n", stabilityIndex)
	}
	
	if avgPreyGrowthRate, ok := stats["avg_prey_growth_rate"]; ok {
		fmt.Printf("Average Prey Growth Rate: %.3f\n", avgPreyGrowthRate)
	}
	
	if avgPredatorGrowthRate, ok := stats["avg_predator_growth_rate"]; ok {
		fmt.Printf("Average Predator Growth Rate: %.3f\n", avgPredatorGrowthRate)
	}
	
	fmt.Println("================================")
}
