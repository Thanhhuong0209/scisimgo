package models

import (
	"math"
	"math/rand"
	"time"

	"github.com/scisimgo/internal/engine"
)

// Vector2D represents a 2D vector
type Vector2D struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// Vector3D represents a 3D vector
type Vector3D struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
	Z float64 `json:"z"`
}

// CelestialBody represents a celestial body in the simulation
type CelestialBody struct {
	Name     string   `json:"name"`
	Mass     float64  `json:"mass"`     // kg
	Position Vector3D `json:"position"` // m
	Velocity Vector3D `json:"velocity"` // m/s
	Radius   float64  `json:"radius"`   // m
	Color    string   `json:"color"`    // For visualization
}

// OrbitalModel represents the orbital mechanics simulation
type OrbitalModel struct {
	*engine.BaseSimulation

	// Model parameters
	GravitationalConstant float64 `json:"gravitational_constant"` // G = 6.67430e-11 m3/(kg*s2)
	TimeStep              float64 `json:"time_step"`              // Simulation time step in seconds
	Enable3D              bool    `json:"enable_3d"`              // Enable 3D simulation

	// Celestial bodies
	Bodies []*CelestialBody `json:"bodies"`

	// Statistical tracking
	OrbitalPeriods map[string]float64 `json:"orbital_periods"` // Period in seconds
	Perihelion     map[string]float64 `json:"perihelion"`      // Closest approach to sun
	Aphelion       map[string]float64 `json:"aphelion"`        // Farthest distance from sun
	Eccentricities map[string]float64 `json:"eccentricities"`  // Orbital eccentricity

	// Random number generator
	rng *rand.Rand
}

// OrbitalConfig extends base simulation config with orbital-specific parameters
type OrbitalConfig struct {
	engine.SimulationConfig
	GravitationalConstant float64          `json:"gravitational_constant"`
	TimeStep              float64          `json:"time_step"`
	Enable3D              bool             `json:"enable_3d"`
	Bodies                []*CelestialBody `json:"bodies"`
}

// NewOrbitalModel creates a new orbital mechanics model
func NewOrbitalModel() *OrbitalModel {
	om := &OrbitalModel{
		BaseSimulation: engine.NewBaseSimulation(),
		rng:            rand.New(rand.NewSource(time.Now().UnixNano())),
		OrbitalPeriods: make(map[string]float64),
		Perihelion:     make(map[string]float64),
		Aphelion:       make(map[string]float64),
		Eccentricities: make(map[string]float64),
	}

	// Override the callbacks
	om.BaseSimulation.SetOnTick(om.onTick)
	om.BaseSimulation.SetOnComplete(om.onComplete)

	return om
}

// Initialize sets up the orbital model with configuration
func (om *OrbitalModel) Initialize(config OrbitalConfig) error {
	// Initialize base simulation
	if err := om.BaseSimulation.Initialize(config.SimulationConfig); err != nil {
		return err
	}

	// Validate model-specific parameters
	if config.GravitationalConstant <= 0 {
		return NewValidationError("gravitational constant must be positive")
	}

	if config.TimeStep <= 0 {
		return NewValidationError("time step must be positive")
	}

	if len(config.Bodies) < 2 {
		return NewValidationError("at least 2 celestial bodies required")
	}

	// Set model parameters
	om.GravitationalConstant = config.GravitationalConstant
	om.TimeStep = config.TimeStep
	om.Enable3D = config.Enable3D

	// Deep copy bodies to avoid external modification
	om.Bodies = make([]*CelestialBody, len(config.Bodies))
	for i, body := range config.Bodies {
		om.Bodies[i] = &CelestialBody{
			Name:     body.Name,
			Mass:     body.Mass,
			Position: body.Position,
			Velocity: body.Velocity,
			Radius:   body.Radius,
			Color:    body.Color,
		}
	}

	// Initialize orbital parameters
	om.initializeOrbitalParameters()

	return nil
}

// initializeOrbitalParameters calculates initial orbital parameters
func (om *OrbitalModel) initializeOrbitalParameters() {
	for _, body := range om.Bodies {
		// Calculate distance from origin (assuming sun at origin)
		distance := om.calculateDistance(body.Position, Vector3D{0, 0, 0})

		// Initialize perihelion and aphelion
		om.Perihelion[body.Name] = distance
		om.Aphelion[body.Name] = distance

		// Calculate eccentricity (simplified)
		om.Eccentricities[body.Name] = 0.0
	}
}

// onTick handles each simulation tick for orbital model
func (om *OrbitalModel) onTick(iteration int, data map[string]interface{}) error {
	// Calculate gravitational forces between all bodies
	forces := om.calculateGravitationalForces()

	// Update velocities and positions
	om.updateBodies(forces)

	// Update orbital parameters
	om.updateOrbitalParameters()

	return nil
}

// calculateGravitationalForces calculates gravitational forces between all bodies
func (om *OrbitalModel) calculateGravitationalForces() map[string]Vector3D {
	forces := make(map[string]Vector3D)

	// Initialize forces to zero
	for _, body := range om.Bodies {
		forces[body.Name] = Vector3D{0, 0, 0}
	}

	// Calculate forces between all pairs of bodies
	for i, body1 := range om.Bodies {
		for j, body2 := range om.Bodies {
			if i != j {
				force := om.calculateGravitationalForce(body1, body2)

				// Add force to body1 (positive)
				currentForce := forces[body1.Name]
				forces[body1.Name] = Vector3D{
					X: currentForce.X + force.X,
					Y: currentForce.Y + force.Y,
					Z: currentForce.Z + force.Z,
				}

				// Subtract force from body2 (negative, Newton's 3rd law)
				currentForce = forces[body2.Name]
				forces[body2.Name] = Vector3D{
					X: currentForce.X - force.X,
					Y: currentForce.Y - force.Y,
					Z: currentForce.Z - force.Z,
				}
			}
		}
	}

	return forces
}

// calculateGravitationalForce calculates gravitational force between two bodies
func (om *OrbitalModel) calculateGravitationalForce(body1, body2 *CelestialBody) Vector3D {
	// Calculate distance vector
	distanceVector := Vector3D{
		X: body2.Position.X - body1.Position.X,
		Y: body2.Position.Y - body1.Position.Y,
		Z: body2.Position.Z - body1.Position.Z,
	}

	// Calculate distance magnitude
	distance := om.calculateDistance(body1.Position, body2.Position)

	// Avoid division by zero
	if distance < 1e-10 {
		return Vector3D{0, 0, 0}
	}

	// Calculate force magnitude: F = G * m1 * m2 / r2
	forceMagnitude := om.GravitationalConstant * body1.Mass * body2.Mass / (distance * distance)

	// Calculate unit vector
	unitVector := Vector3D{
		X: distanceVector.X / distance,
		Y: distanceVector.Y / distance,
		Z: distanceVector.Z / distance,
	}

	// Return force vector
	return Vector3D{
		X: unitVector.X * forceMagnitude,
		Y: unitVector.Y * forceMagnitude,
		Z: unitVector.Z * forceMagnitude,
	}
}

// updateBodies updates the positions and velocities of all bodies
func (om *OrbitalModel) updateBodies(forces map[string]Vector3D) {
	for _, body := range om.Bodies {
		force := forces[body.Name]

		// Calculate acceleration: a = F / m
		acceleration := Vector3D{
			X: force.X / body.Mass,
			Y: force.Y / body.Mass,
			Z: force.Z / body.Mass,
		}

		// Update velocity: v = v0 + a * dt
		body.Velocity.X += acceleration.X * om.TimeStep
		body.Velocity.Y += acceleration.Y * om.TimeStep
		body.Velocity.Z += acceleration.Z * om.TimeStep

		// Update position: r = r0 + v * dt
		body.Position.X += body.Velocity.X * om.TimeStep
		body.Position.Y += body.Velocity.Y * om.TimeStep
		body.Position.Z += body.Velocity.Z * om.TimeStep
	}
}

// updateOrbitalParameters updates orbital parameters for each body
func (om *OrbitalModel) updateOrbitalParameters() {
	for _, body := range om.Bodies {
		// Calculate distance from origin (assuming sun at origin)
		distance := om.calculateDistance(body.Position, Vector3D{0, 0, 0})

		// Update perihelion (closest approach)
		if distance < om.Perihelion[body.Name] {
			om.Perihelion[body.Name] = distance
		}

		// Update aphelion (farthest distance)
		if distance > om.Aphelion[body.Name] {
			om.Aphelion[body.Name] = distance
		}

		// Calculate eccentricity: e = (aphelion - perihelion) / (aphelion + perihelion)
		perihelion := om.Perihelion[body.Name]
		aphelion := om.Aphelion[body.Name]
		if aphelion+perihelion > 0 {
			om.Eccentricities[body.Name] = (aphelion - perihelion) / (aphelion + perihelion)
		}
	}
}

// calculateDistance calculates the distance between two 3D points
func (om *OrbitalModel) calculateDistance(pos1, pos2 Vector3D) float64 {
	dx := pos1.X - pos2.X
	dy := pos1.Y - pos2.Y
	dz := pos1.Z - pos2.Z
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

// onComplete handles simulation completion
func (om *OrbitalModel) onComplete(results []engine.SimulationResult) error {
	// Calculate final orbital periods
	om.calculateOrbitalPeriods(results)
	return nil
}

// calculateOrbitalPeriods calculates orbital periods from simulation data
func (om *OrbitalModel) calculateOrbitalPeriods(results []engine.SimulationResult) {
	// This is a simplified calculation
	// In a real implementation, you'd detect complete orbits
	for _, body := range om.Bodies {
		// Estimate period based on current velocity and distance
		distance := om.calculateDistance(body.Position, Vector3D{0, 0, 0})
		velocity := math.Sqrt(body.Velocity.X*body.Velocity.X +
			body.Velocity.Y*body.Velocity.Y +
			body.Velocity.Z*body.Velocity.Z)

		if velocity > 0 {
			// Approximate period: T = 2πr / v
			period := 2 * math.Pi * distance / velocity
			om.OrbitalPeriods[body.Name] = period
		}
	}
}

// generateTickData overrides the base method to provide orbital-specific data
func (om *OrbitalModel) generateTickData() map[string]interface{} {
	data := map[string]interface{}{
		"iteration": om.BaseSimulation.GetState(),
		"timestamp": time.Now().Unix(),
		"state":     om.BaseSimulation.GetState().String(),
		"time_step": om.TimeStep,
		"enable_3d": om.Enable3D,
	}

	// Add body data
	for _, body := range om.Bodies {
		prefix := body.Name + "_"
		data[prefix+"mass"] = body.Mass
		data[prefix+"position_x"] = body.Position.X
		data[prefix+"position_y"] = body.Position.Y
		data[prefix+"position_z"] = body.Position.Z
		data[prefix+"velocity_x"] = body.Velocity.X
		data[prefix+"velocity_y"] = body.Velocity.Y
		data[prefix+"velocity_z"] = body.Velocity.Z
		data[prefix+"radius"] = body.Radius
		data[prefix+"distance_from_origin"] = om.calculateDistance(body.Position, Vector3D{0, 0, 0})
	}

	// Add orbital parameters
	for name, period := range om.OrbitalPeriods {
		data[name+"_orbital_period"] = period
	}

	for name, perihelion := range om.Perihelion {
		data[name+"_perihelion"] = perihelion
	}

	for name, aphelion := range om.Aphelion {
		data[name+"_aphelion"] = aphelion
	}

	for name, eccentricity := range om.Eccentricities {
		data[name+"_eccentricity"] = eccentricity
	}

	return data
}

// GetCurrentState returns the current orbital state
func (om *OrbitalModel) GetCurrentState() map[string]interface{} {
	state := map[string]interface{}{
		"gravitational_constant": om.GravitationalConstant,
		"time_step":              om.TimeStep,
		"enable_3d":              om.Enable3D,
		"bodies_count":           len(om.Bodies),
		"orbital_periods":        om.OrbitalPeriods,
		"perihelion":             om.Perihelion,
		"aphelion":               om.Aphelion,
		"eccentricities":         om.Eccentricities,
	}

	// Add body states
	for _, body := range om.Bodies {
		state[body.Name] = map[string]interface{}{
			"mass":     body.Mass,
			"position": body.Position,
			"velocity": body.Velocity,
			"radius":   body.Radius,
			"color":    body.Color,
		}
	}

	return state
}

// GetStatistics returns comprehensive orbital statistics
func (om *OrbitalModel) GetStatistics() map[string]interface{} {
	results := om.BaseSimulation.GetResults()

	if len(results) == 0 {
		return map[string]interface{}{}
	}

	// Calculate trajectory statistics
	trajectoryStats := om.calculateTrajectoryStatistics(results)

	// Calculate energy statistics
	energyStats := om.calculateEnergyStatistics(results)

	stats := map[string]interface{}{
		"total_iterations":    len(results),
		"simulation_duration": time.Since(results[0].Timestamp),
		"time_step":           om.TimeStep,
		"bodies_count":        len(om.Bodies),
		"orbital_periods":     om.OrbitalPeriods,
		"perihelion":          om.Perihelion,
		"aphelion":            om.Aphelion,
		"eccentricities":      om.Eccentricities,
		"trajectory_stats":    trajectoryStats,
		"energy_stats":        energyStats,
	}

	return stats
}

// calculateTrajectoryStatistics calculates statistics about body trajectories
func (om *OrbitalModel) calculateTrajectoryStatistics(results []engine.SimulationResult) map[string]interface{} {
	trajectoryStats := make(map[string]interface{})

	for _, body := range om.Bodies {
		var positions []Vector3D
		var velocities []Vector3D
		var distances []float64

		for _, result := range results {
			// Extract position data
			if posX, ok := result.Data[body.Name+"_position_x"]; ok {
				if posY, ok := result.Data[body.Name+"_position_y"]; ok {
					if posZ, ok := result.Data[body.Name+"_position_z"]; ok {
						if x, ok := posX.(float64); ok {
							if y, ok := posY.(float64); ok {
								if z, ok := posZ.(float64); ok {
									positions = append(positions, Vector3D{X: x, Y: y, Z: z})
								}
							}
						}
					}
				}
			}

			// Extract velocity data
			if velX, ok := result.Data[body.Name+"_velocity_x"]; ok {
				if velY, ok := result.Data[body.Name+"_velocity_y"]; ok {
					if velZ, ok := result.Data[body.Name+"_velocity_z"]; ok {
						if x, ok := velX.(float64); ok {
							if y, ok := velY.(float64); ok {
								if z, ok := velZ.(float64); ok {
									velocities = append(velocities, Vector3D{X: x, Y: y, Z: z})
								}
							}
						}
					}
				}
			}

			// Extract distance data
			if dist, ok := result.Data[body.Name+"_distance_from_origin"]; ok {
				if d, ok := dist.(float64); ok {
					distances = append(distances, d)
				}
			}
		}

		// Calculate trajectory statistics for this body
		bodyStats := map[string]interface{}{
			"total_distance_traveled": calculateTotalDistance(positions),
			"max_velocity":            calculateMaxVelocity(velocities),
			"min_velocity":            calculateMinVelocity(velocities),
			"avg_velocity":            calculateAverageVelocity(velocities),
			"max_distance":            calculateMaxFloat(distances),
			"min_distance":            calculateMinFloat(distances),
			"avg_distance":            calculateAverageFloat(distances),
		}

		trajectoryStats[body.Name] = bodyStats
	}

	return trajectoryStats
}

// calculateEnergyStatistics calculates energy conservation statistics
func (om *OrbitalModel) calculateEnergyStatistics(results []engine.SimulationResult) map[string]interface{} {
	var totalKineticEnergy []float64
	var totalPotentialEnergy []float64
	var totalEnergy []float64

	for _, result := range results {
		kinetic := 0.0
		potential := 0.0

		// Calculate kinetic energy for all bodies
		for _, body := range om.Bodies {
			if velX, ok := result.Data[body.Name+"_velocity_x"]; ok {
				if velY, ok := result.Data[body.Name+"_velocity_y"]; ok {
					if velZ, ok := result.Data[body.Name+"_velocity_z"]; ok {
						if x, ok := velX.(float64); ok {
							if y, ok := velY.(float64); ok {
								if z, ok := velZ.(float64); ok {
									velocity := math.Sqrt(x*x + y*y + z*z)
									kinetic += 0.5 * body.Mass * velocity * velocity
								}
							}
						}
					}
				}
			}
		}

		// Calculate potential energy (simplified)
		for i, body1 := range om.Bodies {
			for j, body2 := range om.Bodies {
				if i < j {
					if pos1X, ok := result.Data[body1.Name+"_position_x"]; ok {
						if pos1Y, ok := result.Data[body1.Name+"_position_y"]; ok {
							if pos1Z, ok := result.Data[body1.Name+"_position_z"]; ok {
								if pos2X, ok := result.Data[body2.Name+"_position_x"]; ok {
									if pos2Y, ok := result.Data[body2.Name+"_position_y"]; ok {
										if pos2Z, ok := result.Data[body2.Name+"_position_z"]; ok {
											if x1, ok := pos1X.(float64); ok {
												if y1, ok := pos1Y.(float64); ok {
													if z1, ok := pos1Z.(float64); ok {
														if x2, ok := pos2X.(float64); ok {
															if y2, ok := pos2Y.(float64); ok {
																if z2, ok := pos2Z.(float64); ok {
																	pos1 := Vector3D{X: x1, Y: y1, Z: z1}
																	pos2 := Vector3D{X: x2, Y: y2, Z: z2}
																	distance := om.calculateDistance(pos1, pos2)
																	if distance > 0 {
																		potential -= om.GravitationalConstant * body1.Mass * body2.Mass / distance
																	}
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}

		totalKineticEnergy = append(totalKineticEnergy, kinetic)
		totalPotentialEnergy = append(totalPotentialEnergy, potential)
		totalEnergy = append(totalEnergy, kinetic+potential)
	}

	return map[string]interface{}{
		"kinetic_energy": map[string]interface{}{
			"initial": totalKineticEnergy[0],
			"final":   totalKineticEnergy[len(totalKineticEnergy)-1],
			"max":     calculateMaxFloat(totalKineticEnergy),
			"min":     calculateMinFloat(totalKineticEnergy),
			"avg":     calculateAverageFloat(totalKineticEnergy),
		},
		"potential_energy": map[string]interface{}{
			"initial": totalPotentialEnergy[0],
			"final":   totalPotentialEnergy[len(totalPotentialEnergy)-1],
			"max":     calculateMaxFloat(totalPotentialEnergy),
			"min":     calculateMinFloat(totalPotentialEnergy),
			"avg":     calculateAverageFloat(totalPotentialEnergy),
		},
		"total_energy": map[string]interface{}{
			"initial":            totalEnergy[0],
			"final":              totalEnergy[len(totalEnergy)-1],
			"max":                calculateMaxFloat(totalEnergy),
			"min":                calculateMinFloat(totalEnergy),
			"avg":                calculateAverageFloat(totalEnergy),
			"conservation_error": math.Abs(totalEnergy[0] - totalEnergy[len(totalEnergy)-1]),
		},
	}
}

// Helper functions for calculations
func calculateTotalDistance(positions []Vector3D) float64 {
	if len(positions) < 2 {
		return 0
	}

	total := 0.0
	for i := 1; i < len(positions); i++ {
		dx := positions[i].X - positions[i-1].X
		dy := positions[i].Y - positions[i-1].Y
		dz := positions[i].Z - positions[i-1].Z
		total += math.Sqrt(dx*dx + dy*dy + dz*dz)
	}
	return total
}

func calculateMaxVelocity(velocities []Vector3D) float64 {
	if len(velocities) == 0 {
		return 0
	}

	max := 0.0
	for _, vel := range velocities {
		speed := math.Sqrt(vel.X*vel.X + vel.Y*vel.Y + vel.Z*vel.Z)
		if speed > max {
			max = speed
		}
	}
	return max
}

func calculateMinVelocity(velocities []Vector3D) float64 {
	if len(velocities) == 0 {
		return 0
	}

	min := math.Inf(1)
	for _, vel := range velocities {
		speed := math.Sqrt(vel.X*vel.X + vel.Y*vel.Y + vel.Z*vel.Z)
		if speed < min {
			min = speed
		}
	}
	if math.IsInf(min, 1) {
		return 0
	}
	return min
}

func calculateAverageVelocity(velocities []Vector3D) float64 {
	if len(velocities) == 0 {
		return 0
	}

	sum := 0.0
	for _, vel := range velocities {
		speed := math.Sqrt(vel.X*vel.X + vel.Y*vel.Y + vel.Z*vel.Z)
		sum += speed
	}
	return sum / float64(len(velocities))
}

// calculateMaxFloat finds the maximum value in a slice of floats
func calculateMaxFloat(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	max := values[0]
	for _, val := range values {
		if val > max {
			max = val
		}
	}
	return max
}

// calculateMinFloat finds the minimum value in a slice of floats
func calculateMinFloat(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	min := values[0]
	for _, val := range values {
		if val < min {
			min = val
		}
	}
	return min
}

// calculateAverageFloat calculates the average of a slice of floats
func calculateAverageFloat(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0.0
	for _, val := range values {
		sum += val
	}
	return sum / float64(len(values))
}
