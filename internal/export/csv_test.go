package export

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/scisimgo/internal/engine"
)

func TestNewCSVExporter(t *testing.T) {
	outputDir := "test_output"
	exporter := NewCSVExporter(outputDir)
	
	if exporter == nil {
		t.Fatal("NewCSVExporter returned nil")
	}
	
	if exporter.outputDir != outputDir {
		t.Errorf("Expected output directory %s, got %s", outputDir, exporter.outputDir)
	}
}

func TestCSVExporter_ExportResults(t *testing.T) {
	// Create temporary test directory
	testDir := "test_csv_export"
	defer os.RemoveAll(testDir)
	
	exporter := NewCSVExporter(testDir)
	
	// Create test results
	results := []engine.SimulationResult{
		{
			Timestamp:  time.Now(),
			Iteration:  1,
			Data:       map[string]interface{}{"value": 10, "status": "active"},
			Metadata:   map[string]interface{}{"model": "test"},
		},
		{
			Timestamp:  time.Now().Add(time.Second),
			Iteration:  2,
			Data:       map[string]interface{}{"value": 20, "status": "inactive"},
			Metadata:   map[string]interface{}{"model": "test"},
		},
	}
	
	// Test export
	err := exporter.ExportResults(results, "test_results")
	if err != nil {
		t.Fatalf("ExportResults failed: %v", err)
	}
	
	// Verify file was created
	expectedFile := filepath.Join(testDir, "test_results.csv")
	if _, err := os.Stat(expectedFile); os.IsNotExist(err) {
		t.Errorf("Expected file %s to exist", expectedFile)
	}
	
	// Verify file content
	content, err := os.ReadFile(expectedFile)
	if err != nil {
		t.Fatalf("Failed to read exported file: %v", err)
	}
	
	contentStr := string(content)
	if len(contentStr) == 0 {
		t.Error("Exported file is empty")
	}
	
	// Check for expected headers
	expectedHeaders := []string{"timestamp", "iteration", "value", "status", "model"}
	for _, header := range expectedHeaders {
		if !contains(contentStr, header) {
			t.Errorf("Expected header '%s' not found in exported file", header)
		}
	}
}

func TestCSVExporter_ExportResults_EmptyResults(t *testing.T) {
	testDir := "test_empty_export"
	defer os.RemoveAll(testDir)
	
	exporter := NewCSVExporter(testDir)
	
	// Test with empty results
	err := exporter.ExportResults([]engine.SimulationResult{}, "empty_results")
	if err == nil {
		t.Error("Expected error for empty results")
	}
}

func TestCSVExporter_ExportResults_NilResults(t *testing.T) {
	testDir := "test_nil_export"
	defer os.RemoveAll(testDir)
	
	exporter := NewCSVExporter(testDir)
	
	// Test with nil results
	err := exporter.ExportResults(nil, "nil_results")
	if err == nil {
		t.Error("Expected error for nil results")
	}
}

func TestCSVExporter_ExportTimeSeries(t *testing.T) {
	testDir := "test_timeseries_export"
	defer os.RemoveAll(testDir)
	
	exporter := NewCSVExporter(testDir)
	
	// Create test results with time series data
	results := []engine.SimulationResult{
		{
			Timestamp: time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
			Iteration: 1,
			Data: map[string]interface{}{
				"timestamp": time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
				"value1":    10.5,
				"value2":    20.3,
				"status":    "active",
			},
		},
		{
			Timestamp: time.Date(2023, 1, 1, 0, 1, 0, 0, time.UTC),
			Iteration: 2,
			Data: map[string]interface{}{
				"timestamp": time.Date(2023, 1, 1, 0, 1, 0, 0, time.UTC),
				"value1":    15.2,
				"value2":    25.7,
				"status":    "active",
			},
		},
	}
	
	// Test export with custom headers
	headers := []string{"timestamp", "value1", "value2"}
	err := exporter.ExportTimeSeries(results, "test_timeseries", "timestamp", headers)
	if err != nil {
		t.Fatalf("ExportTimeSeries failed: %v", err)
	}
	
	// Verify file was created
	expectedFile := filepath.Join(testDir, "test_timeseries.csv")
	if _, err := os.Stat(expectedFile); os.IsNotExist(err) {
		t.Errorf("Expected file %s to exist", expectedFile)
	}
	
	// Verify file content
	content, err := os.ReadFile(expectedFile)
	if err != nil {
		t.Fatalf("Failed to read exported file: %v", err)
	}
	
	contentStr := string(content)
	if len(contentStr) == 0 {
		t.Error("Exported file is empty")
	}
	
	// Check for expected headers
	for _, header := range headers {
		if !contains(contentStr, header) {
			t.Errorf("Expected header '%s' not found in exported file", header)
		}
	}
}

func TestCSVExporter_ExportTimeSeries_EmptyResults(t *testing.T) {
	testDir := "test_empty_timeseries"
	defer os.RemoveAll(testDir)
	
	exporter := NewCSVExporter(testDir)
	
	// Test with empty results
	err := exporter.ExportTimeSeries([]engine.SimulationResult{}, "empty_timeseries", "timestamp", []string{"timestamp"})
	if err == nil {
		t.Error("Expected error for empty results")
	}
}

func TestCSVExporter_ExportTimeSeries_InvalidTimestampKey(t *testing.T) {
	testDir := "test_invalid_timestamp"
	defer os.RemoveAll(testDir)
	
	exporter := NewCSVExporter(testDir)
	
	results := []engine.SimulationResult{
		{
			Timestamp: time.Now(),
			Iteration: 1,
			Data: map[string]interface{}{
				"value": 10,
			},
		},
	}
	
	// Test with invalid timestamp key
	err := exporter.ExportTimeSeries(results, "invalid_timestamp", "nonexistent_key", []string{"value"})
	if err == nil {
		t.Error("Expected error for invalid timestamp key")
	}
}

func TestCSVExporter_ExportCustomHeaders(t *testing.T) {
	testDir := "test_custom_headers"
	defer os.RemoveAll(testDir)
	
	exporter := NewCSVExporter(testDir)
	
	results := []engine.SimulationResult{
		{
			Timestamp: time.Now(),
			Iteration: 1,
			Data: map[string]interface{}{
				"custom_field1": "value1",
				"custom_field2": 42,
				"custom_field3": 3.14,
			},
		},
	}
	
	// Test export with custom headers
	headers := []string{"custom_field1", "custom_field2", "custom_field3"}
	err := exporter.ExportCustomHeaders(results, "custom_headers", headers)
	if err != nil {
		t.Fatalf("ExportCustomHeaders failed: %v", err)
	}
	
	// Verify file was created
	expectedFile := filepath.Join(testDir, "custom_headers.csv")
	if _, err := os.Stat(expectedFile); os.IsNotExist(err) {
		t.Errorf("Expected file %s to exist", expectedFile)
	}
	
	// Verify file content
	content, err := os.ReadFile(expectedFile)
	if err != nil {
		t.Fatalf("Failed to read exported file: %v", err)
	}
	
	contentStr := string(content)
	if len(contentStr) == 0 {
		t.Error("Exported file is empty")
	}
	
	// Check for expected headers
	for _, header := range headers {
		if !contains(contentStr, header) {
			t.Errorf("Expected header '%s' not found in exported file", header)
		}
	}
}

func TestCSVExporter_ExportCustomHeaders_EmptyResults(t *testing.T) {
	testDir := "test_empty_custom_headers"
	defer os.RemoveAll(testDir)
	
	exporter := NewCSVExporter(testDir)
	
	// Test with empty results
	err := exporter.ExportCustomHeaders([]engine.SimulationResult{}, "empty_custom", []string{"field1"})
	if err == nil {
		t.Error("Expected error for empty results")
	}
}

func TestCSVExporter_ExportCustomHeaders_EmptyHeaders(t *testing.T) {
	testDir := "test_empty_headers"
	defer os.RemoveAll(testDir)
	
	exporter := NewCSVExporter(testDir)
	
	results := []engine.SimulationResult{
		{
			Timestamp: time.Now(),
			Iteration: 1,
			Data: map[string]interface{}{
				"field1": "value1",
			},
		},
	}
	
	// Test with empty headers
	err := exporter.ExportCustomHeaders(results, "empty_headers", []string{})
	if err == nil {
		t.Error("Expected error for empty headers")
	}
}

func TestCSVExporter_ExportCustomHeaders_NilHeaders(t *testing.T) {
	testDir := "test_nil_headers"
	defer os.RemoveAll(testDir)
	
	exporter := NewCSVExporter(testDir)
	
	results := []engine.SimulationResult{
		{
			Timestamp: time.Now(),
			Iteration: 1,
			Data: map[string]interface{}{
				"field1": "value1",
			},
		},
	}
	
	// Test with nil headers
	err := exporter.ExportCustomHeaders(results, "nil_headers", nil)
	if err == nil {
		t.Error("Expected error for nil headers")
	}
}

func TestCSVExporter_ExportCustomHeaders_MissingFields(t *testing.T) {
	testDir := "test_missing_fields"
	defer os.RemoveAll(testDir)
	
	exporter := NewCSVExporter(testDir)
	
	results := []engine.SimulationResult{
		{
			Timestamp: time.Now(),
			Iteration: 1,
			Data: map[string]interface{}{
				"field1": "value1",
				"field2": 42,
			},
		},
	}
	
	// Test with headers that don't exist in data
	headers := []string{"field1", "field2", "nonexistent_field"}
	err := exporter.ExportCustomHeaders(results, "missing_fields", headers)
	if err != nil {
		t.Fatalf("ExportCustomHeaders failed: %v", err)
	}
	
	// Verify file was created
	expectedFile := filepath.Join(testDir, "missing_fields.csv")
	if _, err := os.Stat(expectedFile); os.IsNotExist(err) {
		t.Errorf("Expected file %s to exist", expectedFile)
	}
	
	// Verify file content
	content, err := os.ReadFile(expectedFile)
	if err != nil {
		t.Fatalf("Failed to read exported file: %v", err)
	}
	
	contentStr := string(content)
	if len(contentStr) == 0 {
		t.Error("Exported file is empty")
	}
	
	// Check that missing fields are handled gracefully (empty values)
	if !contains(contentStr, "nonexistent_field") {
		t.Error("Expected header for missing field to be present")
	}
}

func TestCSVExporter_ExportCustomHeaders_ComplexDataTypes(t *testing.T) {
	testDir := "test_complex_data"
	defer os.RemoveAll(testDir)
	
	exporter := NewCSVExporter(testDir)
	
	results := []engine.SimulationResult{
		{
			Timestamp: time.Now(),
			Iteration: 1,
			Data: map[string]interface{}{
				"string_field": "hello world",
				"int_field":    42,
				"float_field":  3.14159,
				"bool_field":   true,
				"nil_field":    nil,
			},
		},
	}
	
	headers := []string{"string_field", "int_field", "float_field", "bool_field", "nil_field"}
	err := exporter.ExportCustomHeaders(results, "complex_data", headers)
	if err != nil {
		t.Fatalf("ExportCustomHeaders failed: %v", err)
	}
	
	// Verify file was created
	expectedFile := filepath.Join(testDir, "complex_data.csv")
	if _, err := os.Stat(expectedFile); os.IsNotExist(err) {
		t.Errorf("Expected file %s to exist", expectedFile)
	}
	
	// Verify file content
	content, err := os.ReadFile(expectedFile)
	if err != nil {
		t.Fatalf("Failed to read exported file: %v", err)
	}
	
	contentStr := string(content)
	if len(contentStr) == 0 {
		t.Error("Exported file is empty")
	}
	
	// Check for expected data
	expectedData := []string{"hello world", "42", "3.14159", "true", ""}
	for _, data := range expectedData {
		if !contains(contentStr, data) {
			t.Errorf("Expected data '%s' not found in exported file", data)
		}
	}
}

// Helper function to check if a string contains a substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 || 
		(len(s) > len(substr) && (s[:len(substr)] == substr || 
		s[len(s)-len(substr):] == substr || 
		func() bool {
			for i := 0; i <= len(s)-len(substr); i++ {
				if s[i:i+len(substr)] == substr {
					return true
				}
			}
			return false
		}())))
}
