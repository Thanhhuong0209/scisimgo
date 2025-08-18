package export

import (
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/scisimgo/internal/engine"
)

// CSVExporter handles exporting simulation data to CSV format
type CSVExporter struct {
	outputDir string
	headers   []string
}

// NewCSVExporter creates a new CSV exporter
func NewCSVExporter(outputDir string) *CSVExporter {
	return &CSVExporter{
		outputDir: outputDir,
		headers:   make([]string, 0),
	}
}

// ExportResults exports simulation results to CSV
func (ce *CSVExporter) ExportResults(results []engine.SimulationResult, filename string) error {
	if len(results) == 0 {
		return fmt.Errorf("no results to export")
	}

	// Ensure output directory exists
	if err := os.MkdirAll(ce.outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Create full file path
	filepath := filepath.Join(ce.outputDir, filename)
	if filepath.Ext(filepath) != ".csv" {
		filepath += ".csv"
	}

	// Create CSV file
	file, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("failed to create CSV file: %w", err)
	}
	defer file.Close()

	// Create CSV writer
	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Generate headers from first result
	headers := ce.generateHeaders(results[0])
	if err := writer.Write(headers); err != nil {
		return fmt.Errorf("failed to write headers: %w", err)
	}

	// Write data rows
	for _, result := range results {
		row := ce.resultToRow(result, headers)
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("failed to write row: %w", err)
		}
	}

	return nil
}

// generateHeaders generates CSV headers from a simulation result
func (ce *CSVExporter) generateHeaders(result engine.SimulationResult) []string {
	headers := []string{
		"timestamp",
		"iteration",
	}

	// Add data keys
	for key := range result.Data {
		headers = append(headers, key)
	}

	// Add metadata keys
	for key := range result.Metadata {
		headers = append(headers, "meta_"+key)
	}

	return headers
}

// resultToRow converts a simulation result to a CSV row
func (ce *CSVExporter) resultToRow(result engine.SimulationResult, headers []string) []string {
	row := make([]string, len(headers))

	for i, header := range headers {
		switch header {
		case "timestamp":
			row[i] = result.Timestamp.Format(time.RFC3339)
		case "iteration":
			row[i] = strconv.Itoa(result.Iteration)
		default:
			if len(header) > 5 && header[:5] == "meta_" {
				// Metadata field
				metaKey := header[5:]
				if value, exists := result.Metadata[metaKey]; exists {
					row[i] = ce.valueToString(value)
				} else {
					row[i] = ""
				}
			} else {
				// Data field
				if value, exists := result.Data[header]; exists {
					row[i] = ce.valueToString(value)
				} else {
					row[i] = ""
				}
			}
		}
	}

	return row
}

// valueToString converts any value to string for CSV
func (ce *CSVExporter) valueToString(value interface{}) string {
	switch v := value.(type) {
	case string:
		return v
	case int, int8, int16, int32, int64:
		return fmt.Sprintf("%d", v)
	case uint, uint8, uint16, uint32, uint64:
		return fmt.Sprintf("%d", v)
	case float32, float64:
		return fmt.Sprintf("%.6f", v)
	case bool:
		return strconv.FormatBool(v)
	case time.Time:
		return v.Format(time.RFC3339)
	case nil:
		return ""
	default:
		return fmt.Sprintf("%v", v)
	}
}

// ExportWithCustomHeaders exports with custom headers and data mapping
func (ce *CSVExporter) ExportWithCustomHeaders(
	results []engine.SimulationResult,
	filename string,
	headers []string,
	dataMapper func(engine.SimulationResult) []string,
) error {
	if len(results) == 0 {
		return fmt.Errorf("no results to export")
	}

	// Ensure output directory exists
	if err := os.MkdirAll(ce.outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Create full file path
	filepath := filepath.Join(ce.outputDir, filename)
	if filepath.Ext(filepath) != ".csv" {
		filepath += ".csv"
	}

	// Create CSV file
	file, err := os.Create(filepath)
	if err != nil {
		return fmt.Errorf("failed to create CSV file: %w", err)
	}
	defer file.Close()

	// Create CSV writer
	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write headers
	if err := writer.Write(headers); err != nil {
		return fmt.Errorf("failed to write headers: %w", err)
	}

	// Write data rows using custom mapper
	for _, result := range results {
		row := dataMapper(result)
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("failed to write row: %w", err)
		}
	}

	return nil
}

// ExportTimeSeries exports time series data with specific format
func (ce *CSVExporter) ExportTimeSeries(
	results []engine.SimulationResult,
	filename string,
	timeColumn string,
	valueColumns []string,
) error {
	if len(results) == 0 {
		return fmt.Errorf("no results to export")
	}

	// Create headers
	headers := []string{timeColumn}
	headers = append(headers, valueColumns...)

	// Create data mapper
	dataMapper := func(result engine.SimulationResult) []string {
		row := make([]string, len(headers))
		
		// Time column
		row[0] = result.Timestamp.Format(time.RFC3339)
		
		// Value columns
		for i, col := range valueColumns {
			if value, exists := result.Data[col]; exists {
				row[i+1] = ce.valueToString(value)
			} else {
				row[i+1] = ""
			}
		}
		
		return row
	}

	return ce.ExportWithCustomHeaders(results, filename, headers, dataMapper)
}

// GetOutputDir returns the current output directory
func (ce *CSVExporter) GetOutputDir() string {
	return ce.outputDir
}

// SetOutputDir sets a new output directory
func (ce *CSVExporter) SetOutputDir(outputDir string) {
	ce.outputDir = outputDir
}
