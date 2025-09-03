package logging

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// LogLevel represents the logging level
type LogLevel int

const (
	LevelDebug LogLevel = iota
	LevelInfo
	LevelWarn
	LevelError
	LevelFatal
)

// String returns the string representation of LogLevel
func (l LogLevel) String() string {
	switch l {
	case LevelDebug:
		return "DEBUG"
	case LevelInfo:
		return "INFO"
	case LevelWarn:
		return "WARN"
	case LevelError:
		return "ERROR"
	case LevelFatal:
		return "FATAL"
	default:
		return "UNKNOWN"
	}
}

// ParseLogLevel parses a string to LogLevel
func ParseLogLevel(level string) LogLevel {
	switch strings.ToUpper(level) {
	case "DEBUG":
		return LevelDebug
	case "INFO":
		return LevelInfo
	case "WARN", "WARNING":
		return LevelWarn
	case "ERROR":
		return LevelError
	case "FATAL":
		return LevelFatal
	default:
		return LevelInfo
	}
}

// LogEntry represents a single log entry
type LogEntry struct {
	Timestamp  time.Time
	Level      LogLevel
	Component  string
	Message    string
	Fields     map[string]interface{}
	Caller     string
	CallerLine int
	CallerFunc string
}

// Logger provides structured logging functionality
type Logger struct {
	mu           sync.RWMutex
	level        LogLevel
	component    string
	output       io.Writer
	formatter    LogFormatter
	fields       map[string]interface{}
	enableCaller bool
}

// LogFormatter defines the interface for log formatting
type LogFormatter interface {
	Format(entry LogEntry) string
}

// DefaultFormatter provides basic log formatting
type DefaultFormatter struct{}

// Format formats a log entry as a simple string
func (f *DefaultFormatter) Format(entry LogEntry) string {
	callerInfo := ""
	if entry.Caller != "" {
		callerInfo = fmt.Sprintf(" [%s:%d %s]", entry.Caller, entry.CallerLine, entry.CallerFunc)
	}

	fieldsStr := ""
	if len(entry.Fields) > 0 {
		fieldPairs := make([]string, 0, len(entry.Fields))
		for k, v := range entry.Fields {
			fieldPairs = append(fieldPairs, fmt.Sprintf("%s=%v", k, v))
		}
		fieldsStr = fmt.Sprintf(" {%s}", strings.Join(fieldPairs, ", "))
	}

	return fmt.Sprintf("[%s] %s [%s]%s: %s%s",
		entry.Timestamp.Format("2006-01-02 15:04:05.000"),
		entry.Level.String(),
		entry.Component,
		callerInfo,
		entry.Message,
		fieldsStr)
}

// JSONFormatter provides JSON log formatting
type JSONFormatter struct{}

// Format formats a log entry as JSON
func (f *JSONFormatter) Format(entry LogEntry) string {
	// Simple JSON formatting - in production, use a proper JSON library
	fieldsStr := ""
	if len(entry.Fields) > 0 {
		fieldPairs := make([]string, 0, len(entry.Fields))
		for k, v := range entry.Fields {
			fieldPairs = append(fieldPairs, fmt.Sprintf(`"%s":"%v"`, k, v))
		}
		fieldsStr = fmt.Sprintf(",%s", strings.Join(fieldPairs, ","))
	}

	callerInfo := ""
	if entry.Caller != "" {
		callerInfo = fmt.Sprintf(`,"caller":"%s:%d","function":"%s"`, entry.Caller, entry.CallerLine, entry.CallerFunc)
	}

	return fmt.Sprintf(`{"timestamp":"%s","level":"%s","component":"%s","message":"%s"%s%s}`,
		entry.Timestamp.Format(time.RFC3339Nano),
		entry.Level.String(),
		entry.Component,
		entry.Message,
		fieldsStr,
		callerInfo)
}

// NewLogger creates a new logger instance
func NewLogger(component string, level LogLevel) *Logger {
	return &Logger{
		level:        level,
		component:    component,
		output:       os.Stdout,
		formatter:    &DefaultFormatter{},
		fields:       make(map[string]interface{}),
		enableCaller: true,
	}
}

// SetLevel sets the logging level
func (l *Logger) SetLevel(level LogLevel) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.level = level
}

// SetOutput sets the output writer
func (l *Logger) SetOutput(output io.Writer) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.output = output
}

// SetFormatter sets the log formatter
func (l *Logger) SetFormatter(formatter LogFormatter) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.formatter = formatter
}

// SetFields sets global fields for all log entries
func (l *Logger) SetFields(fields map[string]interface{}) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.fields = fields
}

// AddField adds a single global field
func (l *Logger) AddField(key string, value interface{}) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.fields[key] = value
}

// EnableCaller enables/disables caller information
func (l *Logger) EnableCaller(enable bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.enableCaller = enable
}

// log creates and writes a log entry
func (l *Logger) log(level LogLevel, message string, fields map[string]interface{}) {
	if level < l.level {
		return
	}

	l.mu.RLock()
	defer l.mu.RUnlock()

	// Get caller information
	var caller, callerFunc string
	var callerLine int
	if l.enableCaller {
		if pc, file, line, ok := runtime.Caller(2); ok {
			caller = filepath.Base(file)
			callerLine = line
			if fn := runtime.FuncForPC(pc); fn != nil {
				callerFunc = filepath.Base(fn.Name())
			}
		}
	}

	// Merge global fields with entry fields
	mergedFields := make(map[string]interface{})
	for k, v := range l.fields {
		mergedFields[k] = v
	}
	for k, v := range fields {
		mergedFields[k] = v
	}

	entry := LogEntry{
		Timestamp:  time.Now(),
		Level:      level,
		Component:  l.component,
		Message:    message,
		Fields:     mergedFields,
		Caller:     caller,
		CallerLine: callerLine,
		CallerFunc: callerFunc,
	}

	// Format and write the log entry
	formatted := l.formatter.Format(entry)
	fmt.Fprintln(l.output, formatted)
}

// Debug logs a debug message
func (l *Logger) Debug(message string, fields ...map[string]interface{}) {
	var mergedFields map[string]interface{}
	if len(fields) > 0 {
		mergedFields = fields[0]
	}
	l.log(LevelDebug, message, mergedFields)
}

// Info logs an info message
func (l *Logger) Info(message string, fields ...map[string]interface{}) {
	var mergedFields map[string]interface{}
	if len(fields) > 0 {
		mergedFields = fields[0]
	}
	l.log(LevelInfo, message, mergedFields)
}

// Warn logs a warning message
func (l *Logger) Warn(message string, fields ...map[string]interface{}) {
	var mergedFields map[string]interface{}
	if len(fields) > 0 {
		mergedFields = fields[0]
	}
	l.log(LevelWarn, message, mergedFields)
}

// Error logs an error message
func (l *Logger) Error(message string, fields ...map[string]interface{}) {
	var mergedFields map[string]interface{}
	if len(fields) > 0 {
		mergedFields = fields[0]
	}
	l.log(LevelError, message, mergedFields)
}

// Fatal logs a fatal message and exits
func (l *Logger) Fatal(message string, fields ...map[string]interface{}) {
	var mergedFields map[string]interface{}
	if len(fields) > 0 {
		mergedFields = fields[0]
	}
	l.log(LevelFatal, message, mergedFields)
	os.Exit(1)
}

// WithFields creates a new logger with additional fields
func (l *Logger) WithFields(fields map[string]interface{}) *Logger {
	newLogger := *l
	newLogger.fields = make(map[string]interface{})

	// Copy existing fields
	for k, v := range l.fields {
		newLogger.fields[k] = v
	}

	// Add new fields
	for k, v := range fields {
		newLogger.fields[k] = v
	}

	return &newLogger
}

// WithField creates a new logger with a single additional field
func (l *Logger) WithField(key string, value interface{}) *Logger {
	return l.WithFields(map[string]interface{}{key: value})
}

// FileLogger provides file-based logging with rotation
type FileLogger struct {
	*Logger
	filePath    string
	maxSize     int64
	maxFiles    int
	currentFile *os.File
	mu          sync.Mutex
}

// NewFileLogger creates a new file logger
func NewFileLogger(component, filePath string, level LogLevel, maxSize int64, maxFiles int) (*FileLogger, error) {
	// Ensure directory exists
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create log directory: %w", err)
	}

	// Open log file
	file, err := os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open log file: %w", err)
	}

	logger := NewLogger(component, level)
	logger.SetOutput(file)

	fileLogger := &FileLogger{
		Logger:      logger,
		filePath:    filePath,
		maxSize:     maxSize,
		maxFiles:    maxFiles,
		currentFile: file,
	}

	return fileLogger, nil
}

// Close closes the file logger
func (fl *FileLogger) Close() error {
	fl.mu.Lock()
	defer fl.mu.Unlock()

	if fl.currentFile != nil {
		return fl.currentFile.Close()
	}
	return nil
}

// rotate rotates the log file if it exceeds maxSize
func (fl *FileLogger) rotate() error {
	fl.mu.Lock()
	defer fl.mu.Unlock()

	// Check file size
	info, err := fl.currentFile.Stat()
	if err != nil {
		return err
	}

	if info.Size() < fl.maxSize {
		return nil
	}

	// Close current file
	fl.currentFile.Close()

	// Rotate existing files
	for i := fl.maxFiles - 1; i > 0; i-- {
		oldPath := fmt.Sprintf("%s.%d", fl.filePath, i)
		newPath := fmt.Sprintf("%s.%d", fl.filePath, i+1)

		if _, err := os.Stat(oldPath); err == nil {
			os.Rename(oldPath, newPath)
		}
	}

	// Rename current file
	backupPath := fmt.Sprintf("%s.1", fl.filePath)
	os.Rename(fl.filePath, backupPath)

	// Create new file
	file, err := os.OpenFile(fl.filePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}

	fl.currentFile = file
	fl.Logger.SetOutput(file)

	return nil
}

// Global logger instance
var globalLogger *Logger

// InitGlobalLogger initializes the global logger
func InitGlobalLogger(level LogLevel) {
	globalLogger = NewLogger("global", level)
}

// GetGlobalLogger returns the global logger
func GetGlobalLogger() *Logger {
	if globalLogger == nil {
		InitGlobalLogger(LevelInfo)
	}
	return globalLogger
}

// SetGlobalLogger sets the global logger
func SetGlobalLogger(logger *Logger) {
	globalLogger = logger
}

// Convenience functions for global logging
func Debug(message string, fields ...map[string]interface{}) {
	GetGlobalLogger().Debug(message, fields...)
}

func Info(message string, fields ...map[string]interface{}) {
	GetGlobalLogger().Info(message, fields...)
}

func Warn(message string, fields ...map[string]interface{}) {
	GetGlobalLogger().Warn(message, fields...)
}

func Error(message string, fields ...map[string]interface{}) {
	GetGlobalLogger().Error(message, fields...)
}

func Fatal(message string, fields ...map[string]interface{}) {
	GetGlobalLogger().Fatal(message, fields...)
}
