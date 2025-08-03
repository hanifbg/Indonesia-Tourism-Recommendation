# your-capstone-project/api/Dockerfile
# Multi-stage build for Go API

# Stage 1: Build the Go application
FROM golang:1.23 AS builder

# Set the working directory
WORKDIR /app

# Copy go.mod and go.sum files and download dependencies
COPY go.mod .
COPY go.sum .
RUN go mod download

# Copy the rest of the application source code
COPY . .

# Build the Go application binary
# We use CGO_ENABLED=0 to create a statically linked binary
# -o <output_filename>
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main ./main.go

# Stage 2: Create a minimal, final image
FROM alpine:3.18

# Set the working directory
WORKDIR /root/

# Copy the binary from the builder stage
COPY --from=builder /app/main .

# Expose the port the Go API will run on
EXPOSE 8080

# Set environment variables for database and ML service connections
# These will be set in docker-compose.yml or Kubernetes manifest
# The default values will be for inter-container communication
ENV DB_HOST_GO=postgres
ENV DB_PORT_GO=5432
ENV DB_USER_GO=postgres
ENV DB_PASSWORD_GO=1
ENV DB_NAME_GO=deploycamp
ENV ML_API_HOST=ml-inference-service
ENV ML_API_PORT=8000

# Command to run the application
CMD ["./main"]