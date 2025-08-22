# Multi-stage build for SciSimGo
FROM golang:1.25-alpine AS go-builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o sir-simulator ./cmd/sir-simulator
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o predator-prey ./cmd/predator-prey
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o orbital-sim ./cmd/orbital-sim

# Python stage for data analysis
FROM python:3.13-slim AS python-builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM alpine:latest

# Install runtime dependencies
RUN apk --no-cache add ca-certificates tzdata python3 py3-pip

# Create non-root user
RUN addgroup -g 1001 -S scisimgo && \
    adduser -u 1001 -S scisimgo -G scisimgo

# Set working directory
WORKDIR /app

# Copy Go binaries from builder stage
COPY --from=go-builder /app/sir-simulator /app/sir-simulator
COPY --from=go-builder /app/predator-prey /app/predator-prey
COPY --from=go-builder /app/orbital-sim /app/orbital-sim

# Copy Python environment from python-builder stage
COPY --from=python-builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy source code and scripts
COPY --chown=scisimgo:scisimgo . .

# Create necessary directories
RUN mkdir -p data logs profiles && \
    chown -R scisimgo:scisimgo /app

# Switch to non-root user
USER scisimgo

# Set environment variables
ENV PYTHONPATH=/app
ENV GO_ENV=production

# Expose ports (if needed for visualization)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD pgrep -f "sir-simulator|predator-prey|orbital-sim" || exit 1

# Default command
CMD ["/app/sir-simulator", "--help"]
