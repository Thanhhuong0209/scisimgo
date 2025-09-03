# SciSimGo Performance Benchmarks

This directory contains performance benchmarks and tests for the SciSimGo simulation engine.

## Running Benchmarks

### Go Benchmarks

Run all benchmarks:
```bash
go test -bench=. ./benchmarks/
```

Run specific benchmarks:
```bash
# SIR simulation benchmark
go test -bench=BenchmarkSIRSimulation ./benchmarks/

# Predator-Prey simulation benchmark
go test -bench=BenchmarkPredatorPreySimulation ./benchmarks/

# Orbital simulation benchmark
go test -bench=BenchmarkOrbitalSimulation ./benchmarks/

# Concurrent simulations benchmark
go test -bench=BenchmarkConcurrentSimulations ./benchmarks/

# Memory usage benchmark
go test -bench=BenchmarkMemoryUsage ./benchmarks/
```

### Performance Test Suite

Run the comprehensive performance test suite:
```bash
go run benchmarks/performance_test.go
```

## Benchmark Results

### Expected Performance (on modern hardware)

#### SIR Simulation
- **Small Population (10K)**: ~1000 iterations/second
- **Medium Population (100K)**: ~500 iterations/second
- **Large Population (1M)**: ~100 iterations/second

#### Predator-Prey Simulation
- **Small Population (1K prey, 100 predators)**: ~2000 iterations/second
- **Medium Population (10K prey, 1K predators)**: ~1000 iterations/second
- **Large Population (100K prey, 10K predators)**: ~200 iterations/second

#### Orbital Simulation
- **3 Bodies (Sun, Earth, Mars)**: ~500 iterations/second
- **5 Bodies (Inner planets)**: ~300 iterations/second
- **10 Bodies (Extended solar system)**: ~100 iterations/second

### Memory Usage

- **SIR Model**: ~1MB per 100K population
- **Predator-Prey Model**: ~500KB per 10K prey population
- **Orbital Model**: ~2MB per 10 bodies

## Benchmarking Guidelines

### 1. System Requirements
- Ensure system is not under heavy load
- Close unnecessary applications
- Use consistent hardware for comparisons

### 2. Benchmark Parameters
- Use consistent simulation parameters
- Run multiple iterations for statistical significance
- Consider different population sizes and durations

### 3. Performance Metrics
- **Iterations per second**: Primary performance metric
- **Memory usage**: Memory efficiency
- **CPU usage**: Computational efficiency
- **Concurrent performance**: Scalability

### 4. Optimization Targets
- **SIR**: Optimize for large populations
- **Predator-Prey**: Optimize for long simulations
- **Orbital**: Optimize for many bodies

## Performance Optimization

### Go Optimizations
1. **Compiler optimizations**: Use `-ldflags="-w -s"` for smaller binaries
2. **Goroutine optimization**: Tune goroutine pool sizes
3. **Memory optimization**: Use object pooling for frequent allocations
4. **Algorithm optimization**: Optimize mathematical calculations

### System Optimizations
1. **CPU affinity**: Pin simulations to specific CPU cores
2. **Memory allocation**: Use large pages for better performance
3. **I/O optimization**: Minimize file I/O during simulation
4. **Network optimization**: Optimize data export if using network storage

## Continuous Performance Monitoring

### CI/CD Integration
```yaml
# .github/workflows/performance.yml
name: Performance Tests
on: [push, pull_request]
jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v3
        with:
          go-version: '1.22'
      - run: go test -bench=. ./benchmarks/ -benchmem
```

### Performance Regression Detection
- Set performance thresholds in CI/CD
- Alert on significant performance degradation
- Track performance trends over time

## Profiling

### CPU Profiling
```bash
go test -bench=BenchmarkSIRSimulation -cpuprofile=cpu.prof ./benchmarks/
go tool pprof cpu.prof
```

### Memory Profiling
```bash
go test -bench=BenchmarkMemoryUsage -memprofile=mem.prof ./benchmarks/
go tool pprof mem.prof
```

### Trace Analysis
```bash
go test -bench=BenchmarkSIRSimulation -trace=trace.out ./benchmarks/
go tool trace trace.out
```

## Benchmark Data Analysis

### Statistical Analysis
- Calculate mean, median, and standard deviation
- Identify outliers and performance variations
- Compare performance across different configurations

### Visualization
- Create performance charts and graphs
- Track performance trends over time
- Compare different optimization strategies

## Troubleshooting Performance Issues

### Common Issues
1. **High memory usage**: Check for memory leaks
2. **Slow performance**: Profile CPU usage
3. **Inconsistent results**: Check for race conditions
4. **High GC pressure**: Optimize memory allocation

### Debugging Tools
- `go tool pprof`: CPU and memory profiling
- `go tool trace`: Execution tracing
- `runtime.MemStats`: Memory statistics
- `runtime.GC()`: Force garbage collection

## Performance Best Practices

### Code Optimization
1. **Avoid unnecessary allocations**: Reuse objects when possible
2. **Use appropriate data structures**: Choose efficient data structures
3. **Minimize function calls**: Inline small functions
4. **Optimize hot paths**: Focus on frequently executed code

### Simulation Optimization
1. **Batch operations**: Process multiple items together
2. **Lazy evaluation**: Compute values only when needed
3. **Caching**: Cache expensive calculations
4. **Parallel processing**: Use goroutines for independent operations

### System Optimization
1. **Resource monitoring**: Monitor CPU, memory, and I/O usage
2. **Load balancing**: Distribute work evenly across cores
3. **Cache optimization**: Use CPU caches effectively
4. **I/O optimization**: Minimize disk and network I/O

## Contributing to Benchmarks

### Adding New Benchmarks
1. Create benchmark functions with `Benchmark` prefix
2. Use `b.ResetTimer()` to exclude setup time
3. Report additional metrics with `b.ReportMetric()`
4. Document expected performance ranges

### Benchmark Standards
- Use consistent naming conventions
- Include setup and teardown code
- Handle errors appropriately
- Provide meaningful output

### Performance Targets
- Set realistic performance expectations
- Consider different hardware configurations
- Account for system overhead
- Plan for performance regression detection
