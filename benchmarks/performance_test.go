package main

import (
	"context"
	"fmt"
	"log"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/scisimgo/internal/models"
)

// BenchmarkSIRSimulation benchmarks SIR simulation performance
func BenchmarkSIRSimulation(b *testing.B) {
	sir := models.NewSIRModel()
	config := models.SIRConfig{
		PopulationSize:  10000,
		InitialInfected: 100,
		InfectionRate:   0.3,
		RecoveryRate:    0.1,
		EnableStochastic: false,
	}

	if err := sir.Initialize(config); err != nil {
		b.Fatalf("Failed to initialize SIR model: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		sir.Run(ctx)
		cancel()
	}
}

// BenchmarkPredatorPreySimulation benchmarks predator-prey simulation performance
func BenchmarkPredatorPreySimulation(b *testing.B) {
	pp := models.NewPredatorPreyModel()
	config := models.PredatorPreyConfig{
		InitialPrey:          1000,
		InitialPredator:      100,
		PreyGrowthRate:       0.1,
		PredationRate:        0.01,
		PredatorDeathRate:    0.1,
		ConversionEfficiency: 0.1,
		EnableStochastic:     false,
	}

	if err := pp.Initialize(config); err != nil {
		b.Fatalf("Failed to initialize Predator-Prey model: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		pp.Run(ctx)
		cancel()
	}
}

// BenchmarkOrbitalSimulation benchmarks orbital simulation performance
func BenchmarkOrbitalSimulation(b *testing.B) {
	orbital := models.NewOrbitalModel()
	bodies := []models.CelestialBody{
		{Name: "Sun", Mass: 1.99e30, Position: [3]float64{0, 0, 0}, Velocity: [3]float64{0, 0, 0}},
		{Name: "Earth", Mass: 5.97e24, Position: [3]float64{1.496e11, 0, 0}, Velocity: [3]float64{0, 29780, 0}},
		{Name: "Mars", Mass: 6.39e23, Position: [3]float64{2.28e11, 0, 0}, Velocity: [3]float64{0, 24070, 0}},
	}

	config := models.OrbitalConfig{
		GravitationalConstant: 6.67e-11,
		TimeStep:             1000,
		Enable3D:             false,
		CelestialBodies:      bodies,
	}

	if err := orbital.Initialize(config); err != nil {
		b.Fatalf("Failed to initialize Orbital model: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		orbital.Run(ctx)
		cancel()
	}
}

// BenchmarkConcurrentSimulations benchmarks concurrent simulation performance
func BenchmarkConcurrentSimulations(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			var wg sync.WaitGroup
			
			// Run SIR simulation
			wg.Add(1)
			go func() {
				defer wg.Done()
				sir := models.NewSIRModel()
				config := models.SIRConfig{
					PopulationSize:  5000,
					InitialInfected: 50,
					InfectionRate:   0.3,
					RecoveryRate:    0.1,
				}
				if err := sir.Initialize(config); err == nil {
					ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
					sir.Run(ctx)
					cancel()
				}
			}()

			// Run Predator-Prey simulation
			wg.Add(1)
			go func() {
				defer wg.Done()
				pp := models.NewPredatorPreyModel()
				config := models.PredatorPreyConfig{
					InitialPrey:          500,
					InitialPredator:      50,
					PreyGrowthRate:       0.1,
					PredationRate:        0.01,
					PredatorDeathRate:    0.1,
					ConversionEfficiency: 0.1,
				}
				if err := pp.Initialize(config); err == nil {
					ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
					pp.Run(ctx)
					cancel()
				}
			}()

			wg.Wait()
		}
	})
}

// BenchmarkMemoryUsage benchmarks memory usage during simulations
func BenchmarkMemoryUsage(b *testing.B) {
	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)

	sir := models.NewSIRModel()
	config := models.SIRConfig{
		PopulationSize:  100000,
		InitialInfected: 1000,
		InfectionRate:   0.3,
		RecoveryRate:    0.1,
	}

	if err := sir.Initialize(config); err != nil {
		b.Fatalf("Failed to initialize SIR model: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		sir.Run(ctx)
		cancel()
	}

	runtime.GC()
	runtime.ReadMemStats(&m2)
	
	b.ReportMetric(float64(m2.Alloc-m1.Alloc), "bytes/op")
	b.ReportMetric(float64(m2.NumGC-m1.NumGC), "gc/op")
}

// PerformanceTestSuite runs comprehensive performance tests
func PerformanceTestSuite() {
	fmt.Println("🚀 SciSimGo Performance Test Suite")
	fmt.Println("==================================")

	// Test SIR simulation performance
	fmt.Println("\n📊 SIR Simulation Performance:")
	start := time.Now()
	sir := models.NewSIRModel()
	config := models.SIRConfig{
		PopulationSize:  100000,
		InitialInfected: 1000,
		InfectionRate:   0.3,
		RecoveryRate:    0.1,
	}

	if err := sir.Initialize(config); err != nil {
		log.Fatalf("Failed to initialize SIR model: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := sir.Run(ctx); err != nil {
		log.Printf("SIR simulation failed: %v", err)
	}

	elapsed := time.Since(start)
	stats := sir.GetStatistics()
	
	fmt.Printf("  Duration: %v\n", elapsed)
	fmt.Printf("  Iterations: %v\n", stats["total_iterations"])
	fmt.Printf("  Iterations/sec: %.2f\n", float64(stats["total_iterations"].(int))/elapsed.Seconds())
	fmt.Printf("  Peak Infected: %v\n", stats["peak_infected"])

	// Test Predator-Prey simulation performance
	fmt.Println("\n🦌 Predator-Prey Simulation Performance:")
	start = time.Now()
	pp := models.NewPredatorPreyModel()
	ppConfig := models.PredatorPreyConfig{
		InitialPrey:          10000,
		InitialPredator:      1000,
		PreyGrowthRate:       0.1,
		PredationRate:        0.01,
		PredatorDeathRate:    0.1,
		ConversionEfficiency: 0.1,
	}

	if err := pp.Initialize(ppConfig); err != nil {
		log.Fatalf("Failed to initialize Predator-Prey model: %v", err)
	}

	ctx2, cancel2 := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel2()

	if err := pp.Run(ctx2); err != nil {
		log.Printf("Predator-Prey simulation failed: %v", err)
	}

	elapsed = time.Since(start)
	ppStats := pp.GetStatistics()
	
	fmt.Printf("  Duration: %v\n", elapsed)
	fmt.Printf("  Iterations: %v\n", ppStats["total_iterations"])
	fmt.Printf("  Iterations/sec: %.2f\n", float64(ppStats["total_iterations"].(int))/elapsed.Seconds())
	fmt.Printf("  Peak Prey: %v\n", ppStats["peak_prey"])
	fmt.Printf("  Peak Predator: %v\n", ppStats["peak_predator"])

	// Test Orbital simulation performance
	fmt.Println("\n🪐 Orbital Simulation Performance:")
	start = time.Now()
	orbital := models.NewOrbitalModel()
	bodies := []models.CelestialBody{
		{Name: "Sun", Mass: 1.99e30, Position: [3]float64{0, 0, 0}, Velocity: [3]float64{0, 0, 0}},
		{Name: "Mercury", Mass: 3.30e23, Position: [3]float64{5.79e10, 0, 0}, Velocity: [3]float64{0, 47400, 0}},
		{Name: "Venus", Mass: 4.87e24, Position: [3]float64{1.08e11, 0, 0}, Velocity: [3]float64{0, 35000, 0}},
		{Name: "Earth", Mass: 5.97e24, Position: [3]float64{1.496e11, 0, 0}, Velocity: [3]float64{0, 29780, 0}},
		{Name: "Mars", Mass: 6.39e23, Position: [3]float64{2.28e11, 0, 0}, Velocity: [3]float64{0, 24070, 0}},
	}

	orbitalConfig := models.OrbitalConfig{
		GravitationalConstant: 6.67e-11,
		TimeStep:             1000,
		Enable3D:             false,
		CelestialBodies:      bodies,
	}

	if err := orbital.Initialize(orbitalConfig); err != nil {
		log.Fatalf("Failed to initialize Orbital model: %v", err)
	}

	ctx3, cancel3 := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel3()

	if err := orbital.Run(ctx3); err != nil {
		log.Printf("Orbital simulation failed: %v", err)
	}

	elapsed = time.Since(start)
	orbitalStats := orbital.GetStatistics()
	
	fmt.Printf("  Duration: %v\n", elapsed)
	fmt.Printf("  Iterations: %v\n", orbitalStats["total_iterations"])
	fmt.Printf("  Iterations/sec: %.2f\n", float64(orbitalStats["total_iterations"].(int))/elapsed.Seconds())
	fmt.Printf("  Bodies: %v\n", orbitalStats["number_of_bodies"])

	// Test concurrent simulations
	fmt.Println("\n🔄 Concurrent Simulations Performance:")
	start = time.Now()
	
	var wg sync.WaitGroup
	concurrentCtx, concurrentCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer concurrentCancel()

	// Run 3 simulations concurrently
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			sir := models.NewSIRModel()
			config := models.SIRConfig{
				PopulationSize:  50000,
				InitialInfected: 500,
				InfectionRate:   0.3,
				RecoveryRate:    0.1,
			}
			
			if err := sir.Initialize(config); err == nil {
				sir.Run(concurrentCtx)
			}
		}(i)
	}

	wg.Wait()
	elapsed = time.Since(start)
	
	fmt.Printf("  Duration: %v\n", elapsed)
	fmt.Printf("  Concurrent simulations: 3\n")

	// Memory usage analysis
	fmt.Println("\n💾 Memory Usage Analysis:")
	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)

	// Run a large simulation
	largeSir := models.NewSIRModel()
	largeConfig := models.SIRConfig{
		PopulationSize:  1000000,
		InitialInfected: 10000,
		InfectionRate:   0.3,
		RecoveryRate:    0.1,
	}

	if err := largeSir.Initialize(largeConfig); err == nil {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		largeSir.Run(ctx)
		cancel()
	}

	runtime.GC()
	runtime.ReadMemStats(&m2)
	
	fmt.Printf("  Memory allocated: %.2f MB\n", float64(m2.Alloc-m1.Alloc)/1024/1024)
	fmt.Printf("  GC cycles: %d\n", m2.NumGC-m1.NumGC)
	fmt.Printf("  Heap size: %.2f MB\n", float64(m2.HeapAlloc)/1024/1024)

	fmt.Println("\n✅ Performance test suite completed!")
}

func main() {
	PerformanceTestSuite()
}
