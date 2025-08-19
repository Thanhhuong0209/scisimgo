package errors

import (
	"fmt"
	"runtime"
	"strings"
)

// ErrorType represents the type of error
type ErrorType string

const (
	// ErrorTypeValidation represents validation errors
	ErrorTypeValidation ErrorType = "validation"
	
	// ErrorTypeConfiguration represents configuration errors
	ErrorTypeConfiguration ErrorType = "configuration"
	
	// ErrorTypeSimulation represents simulation errors
	ErrorTypeSimulation ErrorType = "simulation"
	
	// ErrorTypeExport represents export errors
	ErrorTypeExport ErrorType = "export"
	
	// ErrorTypeAnalysis represents analysis errors
	ErrorTypeAnalysis ErrorType = "analysis"
	
	// ErrorTypeSystem represents system errors
	ErrorTypeSystem ErrorType = "system"
	
	// ErrorTypeUnknown represents unknown errors
	ErrorTypeUnknown ErrorType = "unknown"
)

// Error represents a structured error with additional context
type Error struct {
	Type        ErrorType
	Message     string
	Cause       error
	Context     map[string]interface{}
	StackTrace  []string
	Component   string
	Operation   string
	Timestamp   string
}

// Error returns the error message
func (e *Error) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%s: %s (caused by: %v)", e.Type, e.Message, e.Cause)
	}
	return fmt.Sprintf("%s: %s", e.Type, e.Message)
}

// Unwrap returns the underlying error
func (e *Error) Unwrap() error {
	return e.Cause
}

// Is checks if the error is of a specific type
func (e *Error) Is(target error) bool {
	if target == nil {
		return e == nil
	}
	
	if t, ok := target.(*Error); ok {
		return e.Type == t.Type
	}
	
	return false
}

// WithContext adds context information to the error
func (e *Error) WithContext(key string, value interface{}) *Error {
	if e.Context == nil {
		e.Context = make(map[string]interface{})
	}
	e.Context[key] = value
	return e
}

// WithContextMap adds multiple context values to the error
func (e *Error) WithContextMap(context map[string]interface{}) *Error {
	if e.Context == nil {
		e.Context = make(map[string]interface{})
	}
	for k, v := range context {
		e.Context[k] = v
	}
	return e
}

// GetContext returns the context value for a key
func (e *Error) GetContext(key string) interface{} {
	if e.Context != nil {
		return e.Context[key]
	}
	return nil
}

// HasContext checks if the error has a specific context key
func (e *Error) HasContext(key string) bool {
	if e.Context != nil {
		_, exists := e.Context[key]
		return exists
	}
	return false
}

// GetStackTrace returns the stack trace as a slice of strings
func (e *Error) GetStackTrace() []string {
	return e.StackTrace
}

// FormatStackTrace formats the stack trace as a single string
func (e *Error) FormatStackTrace() string {
	return strings.Join(e.StackTrace, "\n")
}

// New creates a new error with the specified type and message
func New(errorType ErrorType, message string) *Error {
	return &Error{
		Type:       errorType,
		Message:    message,
		Context:    make(map[string]interface{}),
		StackTrace: captureStackTrace(),
		Timestamp:  getCurrentTimestamp(),
	}
}

// Newf creates a new error with formatted message
func Newf(errorType ErrorType, format string, args ...interface{}) *Error {
	return New(errorType, fmt.Sprintf(format, args...))
}

// Wrap wraps an existing error with additional context
func Wrap(err error, errorType ErrorType, message string) *Error {
	if err == nil {
		return New(errorType, message)
	}
	
	// If it's already our error type, add context
	if e, ok := err.(*Error); ok {
		newErr := *e
		newErr.Message = message
		newErr.Cause = e
		newErr.StackTrace = captureStackTrace()
		newErr.Timestamp = getCurrentTimestamp()
		return &newErr
	}
	
	return &Error{
		Type:       errorType,
		Message:    message,
		Cause:      err,
		Context:    make(map[string]interface{}),
		StackTrace: captureStackTrace(),
		Timestamp:  getCurrentTimestamp(),
	}
}

// Wrapf wraps an existing error with formatted message
func Wrapf(err error, errorType ErrorType, format string, args ...interface{}) *Error {
	return Wrap(err, errorType, fmt.Sprintf(format, args...))
}

// ValidationError creates a validation error
func ValidationError(message string) *Error {
	return New(ErrorTypeValidation, message)
}

// ValidationErrorf creates a validation error with formatted message
func ValidationErrorf(format string, args ...interface{}) *Error {
	return Newf(ErrorTypeValidation, format, args...)
}

// ConfigurationError creates a configuration error
func ConfigurationError(message string) *Error {
	return New(ErrorTypeConfiguration, message)
}

// ConfigurationErrorf creates a configuration error with formatted message
func ConfigurationErrorf(format string, args ...interface{}) *Error {
	return Newf(ErrorTypeConfiguration, format, args...)
}

// SimulationError creates a simulation error
func SimulationError(message string) *Error {
	return New(ErrorTypeSimulation, message)
}

// SimulationErrorf creates a simulation error with formatted message
func SimulationErrorf(format string, args ...interface{}) *Error {
	return Newf(ErrorTypeSimulation, format, args...)
}

// ExportError creates an export error
func ExportError(message string) *Error {
	return New(ErrorTypeExport, message)
}

// ExportErrorf creates an export error with formatted message
func ExportErrorf(format string, args ...interface{}) *Error {
	return Newf(ErrorTypeExport, format, args...)
}

// AnalysisError creates an analysis error
func AnalysisError(message string) *Error {
	return New(ErrorTypeAnalysis, message)
}

// AnalysisErrorf creates an analysis error with formatted message
func AnalysisErrorf(format string, args ...interface{}) *Error {
	return Newf(ErrorTypeAnalysis, format, args...)
}

// SystemError creates a system error
func SystemError(message string) *Error {
	return New(ErrorTypeSystem, message)
}

// SystemErrorf creates a system error with formatted message
func SystemErrorf(format string, args ...interface{}) *Error {
	return Newf(ErrorTypeSystem, format, args...)
}

// IsValidationError checks if an error is a validation error
func IsValidationError(err error) bool {
	return IsErrorType(err, ErrorTypeValidation)
}

// IsConfigurationError checks if an error is a configuration error
func IsConfigurationError(err error) bool {
	return IsErrorType(err, ErrorTypeConfiguration)
}

// IsSimulationError checks if an error is a simulation error
func IsSimulationError(err error) bool {
	return IsErrorType(err, ErrorTypeSimulation)
}

// IsExportError checks if an error is an export error
func IsExportError(err error) bool {
	return IsErrorType(err, ErrorTypeExport)
}

// IsAnalysisError checks if an error is an analysis error
func IsAnalysisError(err error) bool {
	return IsErrorType(err, ErrorTypeAnalysis)
}

// IsSystemError checks if an error is a system error
func IsSystemError(err error) bool {
	return IsErrorType(err, ErrorTypeSystem)
}

// IsErrorType checks if an error is of a specific type
func IsErrorType(err error, errorType ErrorType) bool {
	if err == nil {
		return false
	}
	
	if e, ok := err.(*Error); ok {
		return e.Type == errorType
	}
	
	// Check wrapped errors
	if e, ok := err.(*Error); ok && e.Cause != nil {
		return IsErrorType(e.Cause, errorType)
	}
	
	return false
}

// GetErrorType returns the error type if it's our error type
func GetErrorType(err error) ErrorType {
	if e, ok := err.(*Error); ok {
		return e.Type
	}
	return ErrorTypeUnknown
}

// GetErrorContext returns the error context if it's our error type
func GetErrorContext(err error) map[string]interface{} {
	if e, ok := err.(*Error); ok {
		return e.Context
	}
	return nil
}

// GetErrorStackTrace returns the error stack trace if it's our error type
func GetErrorStackTrace(err error) []string {
	if e, ok := err.(*Error); ok {
		return e.StackTrace
	}
	return nil
}

// AggregateError represents multiple errors
type AggregateError struct {
	Errors []error
	Type   ErrorType
}

// Error returns the error message
func (ae *AggregateError) Error() string {
	if len(ae.Errors) == 0 {
		return "no errors"
	}
	
	if len(ae.Errors) == 1 {
		return ae.Errors[0].Error()
	}
	
	messages := make([]string, len(ae.Errors))
	for i, err := range ae.Errors {
		messages[i] = err.Error()
	}
	
	return fmt.Sprintf("%d errors occurred:\n%s", len(ae.Errors), strings.Join(messages, "\n"))
}

// Add adds an error to the aggregate
func (ae *AggregateError) Add(err error) {
	if err != nil {
		ae.Errors = append(ae.Errors, err)
	}
}

// HasErrors checks if there are any errors
func (ae *AggregateError) HasErrors() bool {
	return len(ae.Errors) > 0
}

// Count returns the number of errors
func (ae *AggregateError) Count() int {
	return len(ae.Errors)
}

// NewAggregateError creates a new aggregate error
func NewAggregateError(errorType ErrorType) *AggregateError {
	return &AggregateError{
		Errors: make([]error, 0),
		Type:   errorType,
	}
}

// Helper functions
func captureStackTrace() []string {
	var stack []string
	
	// Skip the first few frames (this function, New, etc.)
	for i := 3; i < 20; i++ {
		if pc, file, line, ok := runtime.Caller(i); ok {
			fn := runtime.FuncForPC(pc)
			if fn != nil {
				stack = append(stack, fmt.Sprintf("%s:%d %s", file, line, fn.Name()))
			}
		} else {
			break
		}
	}
	
	return stack
}

func getCurrentTimestamp() string {
	// This would typically use a proper timestamp library
	// For now, we'll use a simple format
	return "now"
}

// Error codes for common scenarios
const (
	// Validation error codes
	ErrCodeInvalidParameter    = "INVALID_PARAMETER"
	ErrCodeMissingRequired     = "MISSING_REQUIRED"
	ErrCodeOutOfRange          = "OUT_OF_RANGE"
	ErrCodeInvalidFormat       = "INVALID_FORMAT"
	
	// Configuration error codes
	ErrCodeInvalidConfig       = "INVALID_CONFIG"
	ErrCodeMissingConfig       = "MISSING_CONFIG"
	ErrCodeConfigParseError    = "CONFIG_PARSE_ERROR"
	
	// Simulation error codes
	ErrCodeSimulationFailed    = "SIMULATION_FAILED"
	ErrCodeInvalidState        = "INVALID_STATE"
	ErrCodeTimeout             = "TIMEOUT"
	ErrCodeDivergence          = "DIVERGENCE"
	
	// Export error codes
	ErrCodeExportFailed        = "EXPORT_FAILED"
	ErrCodeInvalidFormat       = "INVALID_FORMAT"
	ErrCodePermissionDenied    = "PERMISSION_DENIED"
	ErrCodeDiskFull            = "DISK_FULL"
	
	// Analysis error codes
	ErrCodeAnalysisFailed      = "ANALYSIS_FAILED"
	ErrCodeInvalidData         = "INVALID_DATA"
	ErrCodeInsufficientData    = "INSUFFICIENT_DATA"
	ErrCodeModelError          = "MODEL_ERROR"
	
	// System error codes
	ErrCodeSystemError         = "SYSTEM_ERROR"
	ErrCodeResourceExhausted   = "RESOURCE_EXHAUSTED"
	ErrCodeNetworkError        = "NETWORK_ERROR"
	ErrCodeDatabaseError       = "DATABASE_ERROR"
)

// ErrorCode represents an error code
type ErrorCode string

// WithCode adds an error code to the error context
func (e *Error) WithCode(code ErrorCode) *Error {
	return e.WithContext("error_code", string(code))
}

// GetCode returns the error code from the context
func (e *Error) GetCode() ErrorCode {
	if code, ok := e.GetContext("error_code").(string); ok {
		return ErrorCode(code)
	}
	return ""
}

// NewWithCode creates a new error with a specific error code
func NewWithCode(errorType ErrorType, message string, code ErrorCode) *Error {
	return New(errorType, message).WithCode(code)
}

// NewfWithCode creates a new formatted error with a specific error code
func NewfWithCode(errorType ErrorType, code ErrorCode, format string, args ...interface{}) *Error {
	return Newf(errorType, format, args...).WithCode(code)
}
