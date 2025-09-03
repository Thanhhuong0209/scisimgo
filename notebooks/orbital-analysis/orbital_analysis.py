#!/usr/bin/env python3
"""
Orbital Mechanics Analysis: Comprehensive Data Science Pipeline
for Celestial Body Dynamics and Gravitational Systems

This script provides comprehensive analysis of orbital simulation data,
including trajectory analysis, energy conservation, and celestial mechanics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class OrbitalAnalyzer:
    """Comprehensive analyzer for orbital mechanics simulation data."""
    
    def __init__(self, data_file=None):
        """Initialize analyzer with data file path."""
        self.data_file = data_file
        self.df = None
        self.stats = {}
        self.models = {}
        self.bodies = {}
        
    def load_data(self, data_file=None):
        """Load orbital simulation data from CSV."""
        if data_file:
            self.data_file = data_file
            
        if not self.data_file:
            raise ValueError("No data file specified")
            
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"✅ Loaded {len(self.df)} records from {self.data_file}")
            print(f"📊 Columns: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_sample_data(self, n_points=1000):
        """Create sample orbital mechanics data for demonstration."""
        print("🔄 Creating sample orbital mechanics data...")
        
        # Physical constants
        G = 6.67e-11  # Gravitational constant
        AU = 1.496e11  # Astronomical unit in meters
        
        # Celestial bodies (Sun, Earth, Mars)
        bodies = {
            'Sun': {'mass': 1.99e30, 'pos': [0, 0, 0], 'vel': [0, 0, 0]},
            'Earth': {'mass': 5.97e24, 'pos': [AU, 0, 0], 'vel': [0, 29780, 0]},
            'Mars': {'mass': 6.39e23, 'pos': [1.52*AU, 0, 0], 'vel': [0, 24070, 0]}
        }
        
        dt = 86400  # 1 day in seconds
        data = []
        
        for i in range(n_points):
            timestamp = i * dt
            
            # Calculate gravitational forces and update positions
            for body_name, body in bodies.items():
                # Calculate total force on this body
                total_force = np.array([0.0, 0.0, 0.0])
                
                for other_name, other_body in bodies.items():
                    if body_name != other_name:
                        # Distance vector
                        r_vec = np.array(other_body['pos']) - np.array(body['pos'])
                        r = np.linalg.norm(r_vec)
                        
                        if r > 0:
                            # Gravitational force
                            force_mag = G * body['mass'] * other_body['mass'] / (r**2)
                            force_vec = force_mag * r_vec / r
                            total_force += force_vec
                
                # Update velocity (F = ma, so a = F/m)
                acceleration = total_force / body['mass']
                body['vel'] = np.array(body['vel']) + acceleration * dt
                
                # Update position
                body['pos'] = np.array(body['pos']) + np.array(body['vel']) * dt
            
            # Store data for this timestep
            for body_name, body in bodies.items():
                data.append({
                    'timestamp': timestamp,
                    'iteration': i,
                    'body': body_name,
                    'mass': body['mass'],
                    'x': body['pos'][0],
                    'y': body['pos'][1],
                    'z': body['pos'][2],
                    'vx': body['vel'][0],
                    'vy': body['vel'][1],
                    'vz': body['vel'][2],
                    'distance_from_sun': np.linalg.norm(body['pos']) if body_name != 'Sun' else 0,
                    'speed': np.linalg.norm(body['vel']),
                    'kinetic_energy': 0.5 * body['mass'] * (np.linalg.norm(body['vel'])**2),
                    'potential_energy': self._calculate_potential_energy(body, bodies, G)
                })
        
        self.df = pd.DataFrame(data)
        print(f"✅ Created sample data with {len(self.df)} records")
        return True
    
    def _calculate_potential_energy(self, body, all_bodies, G):
        """Calculate gravitational potential energy for a body."""
        potential = 0
        for other_name, other_body in all_bodies.items():
            if other_name != body['name']:
                r = np.linalg.norm(np.array(other_body['pos']) - np.array(body['pos']))
                if r > 0:
                    potential -= G * body['mass'] * other_body['mass'] / r
        return potential
    
    def exploratory_data_analysis(self):
        """Perform comprehensive exploratory data analysis."""
        print("\n🔍 EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # Basic statistics
        print("\n📊 Basic Statistics:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numeric_cols].describe())
        
        # Body-specific analysis
        print(f"\n🌍 Celestial Bodies:")
        bodies = self.df['body'].unique()
        for body in bodies:
            body_data = self.df[self.df['body'] == body]
            print(f"{body}: {len(body_data)} records")
            if 'distance_from_sun' in body_data.columns:
                print(f"  Distance from Sun: {body_data['distance_from_sun'].mean():.2e} m")
            if 'speed' in body_data.columns:
                print(f"  Average Speed: {body_data['speed'].mean():.2f} m/s")
        
        # Energy analysis
        self._analyze_energy_conservation()
        
        # Orbital elements analysis
        self._analyze_orbital_elements()
        
        return self.df.describe()
    
    def _analyze_energy_conservation(self):
        """Analyze energy conservation in the system."""
        print(f"\n⚡ Energy Conservation Analysis:")
        
        # Calculate total energy for each timestep
        total_energy = []
        timestamps = []
        
        for timestamp in self.df['timestamp'].unique():
            timestep_data = self.df[self.df['timestamp'] == timestamp]
            total_ke = timestep_data['kinetic_energy'].sum()
            total_pe = timestep_data['potential_energy'].sum()
            total_energy.append(total_ke + total_pe)
            timestamps.append(timestamp)
        
        if len(total_energy) > 1:
            energy_variation = np.std(total_energy) / np.mean(total_energy)
            print(f"Energy Conservation Error: {energy_variation:.2e} (relative)")
            print(f"Total Energy Range: {min(total_energy):.2e} - {max(total_energy):.2e} J")
            
            self.stats['energy_conservation_error'] = energy_variation
            self.stats['total_energy'] = total_energy
            self.stats['energy_timestamps'] = timestamps
    
    def _analyze_orbital_elements(self):
        """Analyze orbital elements for each body."""
        print(f"\n🪐 Orbital Elements Analysis:")
        
        for body in self.df['body'].unique():
            if body == 'Sun':
                continue
                
            body_data = self.df[self.df['body'] == body].copy()
            
            if len(body_data) > 10:
                # Calculate orbital period (simplified)
                distances = body_data['distance_from_sun'].values
                peaks = self._find_peaks(distances)
                
                if len(peaks) > 1:
                    periods = np.diff(peaks) * (body_data['timestamp'].iloc[1] - body_data['timestamp'].iloc[0])
                    avg_period = np.mean(periods)
                    print(f"{body} - Estimated Orbital Period: {avg_period/86400:.1f} days")
                    
                    # Calculate eccentricity (simplified)
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    eccentricity = (max_dist - min_dist) / (max_dist + min_dist)
                    print(f"{body} - Estimated Eccentricity: {eccentricity:.3f}")
                    
                    self.stats[f'{body}_period'] = avg_period
                    self.stats[f'{body}_eccentricity'] = eccentricity
    
    def _find_peaks(self, data, threshold=0.1):
        """Find peaks in time series data."""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(data, height=np.mean(data) * (1 + threshold))
        return peaks
    
    def visualize_orbital_dynamics(self):
        """Create comprehensive visualizations of orbital dynamics."""
        print("\n📊 Creating Orbital Dynamics Visualizations...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Orbital Mechanics Analysis', fontsize=16, fontweight='bold')
        
        # 1. 2D orbital trajectories
        ax1 = axes[0, 0]
        for body in self.df['body'].unique():
            body_data = self.df[self.df['body'] == body]
            ax1.plot(body_data['x'], body_data['y'], label=body, linewidth=2)
            ax1.scatter(body_data['x'].iloc[0], body_data['y'].iloc[0], 
                       s=100, marker='o', zorder=5)
            ax1.scatter(body_data['x'].iloc[-1], body_data['y'].iloc[-1], 
                       s=100, marker='s', zorder=5)
        
        ax1.set_title('2D Orbital Trajectories')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. Distance from Sun over time
        ax2 = axes[0, 1]
        for body in self.df['body'].unique():
            if body != 'Sun':
                body_data = self.df[self.df['body'] == body]
                ax2.plot(body_data['timestamp']/86400, body_data['distance_from_sun']/1.496e11, 
                        label=body, linewidth=2)
        
        ax2.set_title('Distance from Sun Over Time')
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Distance (AU)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Speed over time
        ax3 = axes[0, 2]
        for body in self.df['body'].unique():
            if body != 'Sun':
                body_data = self.df[self.df['body'] == body]
                ax3.plot(body_data['timestamp']/86400, body_data['speed']/1000, 
                        label=body, linewidth=2)
        
        ax3.set_title('Orbital Speed Over Time')
        ax3.set_xlabel('Time (days)')
        ax3.set_ylabel('Speed (km/s)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Energy conservation
        ax4 = axes[1, 0]
        if 'total_energy' in self.stats:
            ax4.plot(np.array(self.stats['energy_timestamps'])/86400, 
                    np.array(self.stats['total_energy'])/1e33, linewidth=2)
            ax4.set_title('Total Energy Conservation')
            ax4.set_xlabel('Time (days)')
            ax4.set_ylabel('Total Energy (×10³³ J)')
            ax4.grid(True, alpha=0.3)
        
        # 5. 3D orbital trajectories
        ax5 = axes[1, 1]
        ax5.remove()
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        
        for body in self.df['body'].unique():
            body_data = self.df[self.df['body'] == body]
            ax5.plot(body_data['x']/1.496e11, body_data['y']/1.496e11, 
                    body_data['z']/1.496e11, label=body, linewidth=2)
        
        ax5.set_title('3D Orbital Trajectories')
        ax5.set_xlabel('X (AU)')
        ax5.set_ylabel('Y (AU)')
        ax5.set_zlabel('Z (AU)')
        ax5.legend()
        
        # 6. Phase space (distance vs speed)
        ax6 = axes[1, 2]
        for body in self.df['body'].unique():
            if body != 'Sun':
                body_data = self.df[self.df['body'] == body]
                ax6.plot(body_data['distance_from_sun']/1.496e11, 
                        body_data['speed']/1000, label=body, linewidth=2)
        
        ax6.set_title('Phase Space (Distance vs Speed)')
        ax6.set_xlabel('Distance from Sun (AU)')
        ax6.set_ylabel('Speed (km/s)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('notebooks/orbital-analysis/orbital_dynamics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Interactive Plotly visualization
        self._create_interactive_plot()
    
    def _create_interactive_plot(self):
        """Create interactive Plotly visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('2D Trajectories', 'Distance from Sun', '3D Trajectories', 'Energy Conservation'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter3d"}, {"type": "scatter"}]]
        )
        
        # 2D trajectories
        for body in self.df['body'].unique():
            body_data = self.df[self.df['body'] == body]
            fig.add_trace(
                go.Scatter(x=body_data['x']/1.496e11, y=body_data['y']/1.496e11, 
                          mode='lines+markers', name=body),
                row=1, col=1
            )
        
        # Distance from Sun
        for body in self.df['body'].unique():
            if body != 'Sun':
                body_data = self.df[self.df['body'] == body]
                fig.add_trace(
                    go.Scatter(x=body_data['timestamp']/86400, 
                              y=body_data['distance_from_sun']/1.496e11,
                              mode='lines', name=f'{body} Distance'),
                    row=1, col=2
                )
        
        # 3D trajectories
        for body in self.df['body'].unique():
            body_data = self.df[self.df['body'] == body]
            fig.add_trace(
                go.Scatter3d(x=body_data['x']/1.496e11, y=body_data['y']/1.496e11, 
                            z=body_data['z']/1.496e11, mode='lines+markers', name=body),
                row=2, col=1
            )
        
        # Energy conservation
        if 'total_energy' in self.stats:
            fig.add_trace(
                go.Scatter(x=np.array(self.stats['energy_timestamps'])/86400,
                          y=np.array(self.stats['total_energy'])/1e33,
                          mode='lines', name='Total Energy'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Interactive Orbital Mechanics Analysis")
        fig.write_html('notebooks/orbital-analysis/interactive_analysis.html')
        print("✅ Interactive plot saved as interactive_analysis.html")
    
    def statistical_analysis(self):
        """Perform comprehensive statistical analysis."""
        print("\n📈 STATISTICAL ANALYSIS")
        print("=" * 50)
        
        # Analyze orbital stability
        self._orbital_stability_analysis()
        
        # Analyze energy conservation
        self._energy_statistics()
        
        # Analyze orbital elements
        self._orbital_elements_statistics()
        
        return self.stats
    
    def _orbital_stability_analysis(self):
        """Analyze orbital stability metrics."""
        print(f"\n🪐 Orbital Stability Analysis:")
        
        for body in self.df['body'].unique():
            if body == 'Sun':
                continue
                
            body_data = self.df[self.df['body'] == body]
            
            # Calculate orbital stability metrics
            distances = body_data['distance_from_sun'].values
            speeds = body_data['speed'].values
            
            # Coefficient of variation for distance
            distance_cv = np.std(distances) / np.mean(distances)
            speed_cv = np.std(speeds) / np.mean(speeds)
            
            print(f"{body}:")
            print(f"  Distance CV: {distance_cv:.4f}")
            print(f"  Speed CV: {speed_cv:.4f}")
            
            self.stats[f'{body}_distance_cv'] = distance_cv
            self.stats[f'{body}_speed_cv'] = speed_cv
    
    def _energy_statistics(self):
        """Analyze energy statistics."""
        print(f"\n⚡ Energy Statistics:")
        
        if 'total_energy' in self.stats:
            total_energy = self.stats['total_energy']
            energy_mean = np.mean(total_energy)
            energy_std = np.std(total_energy)
            energy_cv = energy_std / energy_mean
            
            print(f"Total Energy Mean: {energy_mean:.2e} J")
            print(f"Total Energy Std: {energy_std:.2e} J")
            print(f"Energy Conservation CV: {energy_cv:.2e}")
            
            self.stats['energy_mean'] = energy_mean
            self.stats['energy_std'] = energy_std
            self.stats['energy_cv'] = energy_cv
    
    def _orbital_elements_statistics(self):
        """Analyze orbital elements statistics."""
        print(f"\n📊 Orbital Elements Statistics:")
        
        for body in self.df['body'].unique():
            if body == 'Sun':
                continue
                
            body_data = self.df[self.df['body'] == body]
            
            # Calculate orbital elements
            distances = body_data['distance_from_sun'].values
            speeds = body_data['speed'].values
            
            # Semi-major axis (simplified)
            semi_major_axis = np.mean(distances)
            
            # Eccentricity
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            eccentricity = (max_dist - min_dist) / (max_dist + min_dist)
            
            print(f"{body}:")
            print(f"  Semi-major Axis: {semi_major_axis/1.496e11:.3f} AU")
            print(f"  Eccentricity: {eccentricity:.3f}")
            
            self.stats[f'{body}_semi_major_axis'] = semi_major_axis
            self.stats[f'{body}_eccentricity'] = eccentricity
    
    def machine_learning_analysis(self):
        """Apply machine learning techniques to the data."""
        print("\n🤖 MACHINE LEARNING ANALYSIS")
        print("=" * 50)
        
        # Predict orbital positions
        self._predict_orbital_positions()
        
        # Cluster orbital states
        self._cluster_orbital_states()
        
        return self.models
    
    def _predict_orbital_positions(self):
        """Predict future orbital positions using ML."""
        print(f"\n🔮 Orbital Position Prediction:")
        
        for body in self.df['body'].unique():
            if body == 'Sun':
                continue
                
            body_data = self.df[self.df['body'] == body].copy()
            
            if len(body_data) > 100:
                # Prepare features (current position and velocity)
                X = body_data[['x', 'y', 'z', 'vx', 'vy', 'vz']].values[:-1]
                y = body_data[['x', 'y', 'z']].values[1:]
                
                # Train model
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict
                y_pred = model.predict(X)
                
                # Calculate R² score
                r2 = r2_score(y, y_pred)
                
                print(f"{body} Position Prediction R²: {r2:.4f}")
                
                self.models[f'{body}_position_predictor'] = model
    
    def _cluster_orbital_states(self):
        """Cluster orbital states using K-means."""
        print(f"\n🎯 Orbital State Clustering:")
        
        # Prepare features for clustering
        features = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'distance_from_sun', 'speed']
        available_features = [f for f in features if f in self.df.columns]
        
        X = self.df[available_features].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add clusters to dataframe
        self.df['cluster'] = clusters
        
        print(f"Identified {len(np.unique(clusters))} distinct orbital states")
        
        # Cluster statistics
        for cluster_id in np.unique(clusters):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            print(f"Cluster {cluster_id}: {len(cluster_data)} records")
        
        self.models['orbital_clusterer'] = kmeans
        self.models['orbital_scaler'] = scaler
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("\n📋 GENERATING COMPREHENSIVE REPORT")
        print("=" * 50)
        
        report = f"""
# Orbital Mechanics Analysis Report

## Dataset Overview
- **Total Records**: {len(self.df)}
- **Time Range**: {self.df['timestamp'].min():.2f} - {self.df['timestamp'].max():.2f} seconds
- **Celestial Bodies**: {', '.join(self.df['body'].unique())}

## Energy Conservation Analysis
- **Energy Conservation Error**: {self.stats.get('energy_conservation_error', 0):.2e}
- **Total Energy Mean**: {self.stats.get('energy_mean', 0):.2e} J
- **Energy Coefficient of Variation**: {self.stats.get('energy_cv', 0):.2e}

## Orbital Elements
"""
        
        for body in self.df['body'].unique():
            if body != 'Sun':
                report += f"""
### {body}
- **Semi-major Axis**: {self.stats.get(f'{body}_semi_major_axis', 0)/1.496e11:.3f} AU
- **Eccentricity**: {self.stats.get(f'{body}_eccentricity', 0):.3f}
- **Distance CV**: {self.stats.get(f'{body}_distance_cv', 0):.4f}
- **Speed CV**: {self.stats.get(f'{body}_speed_cv', 0):.4f}
"""
        
        report += f"""
## Machine Learning Results
- **Orbital State Clusters**: {len(np.unique(self.df.get('cluster', [0])))}
- **Position Prediction Models**: {len([k for k in self.models.keys() if 'position_predictor' in k])}

## Conclusions
1. The system exhibits {'excellent' if self.stats.get('energy_conservation_error', 1) < 1e-6 else 'good' if self.stats.get('energy_conservation_error', 1) < 1e-3 else 'poor'} energy conservation
2. Orbital stability varies by body with {'low' if np.mean([self.stats.get(f'{body}_distance_cv', 0) for body in self.df['body'].unique() if body != 'Sun']) < 0.1 else 'moderate' if np.mean([self.stats.get(f'{body}_distance_cv', 0) for body in self.df['body'].unique() if body != 'Sun']) < 0.5 else 'high'} variability
3. The simulation demonstrates realistic celestial mechanics behavior
"""
        
        # Save report
        with open('notebooks/orbital-analysis/analysis_report.md', 'w') as f:
            f.write(report)
        
        print("✅ Report saved as analysis_report.md")
        return report

def main():
    """Main analysis pipeline."""
    print("🪐 ORBITAL MECHANICS ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = OrbitalAnalyzer()
    
    # Load or create data
    if not analyzer.load_data():
        print("📊 Creating sample data for demonstration...")
        analyzer.create_sample_data(1000)
    
    # Run analysis pipeline
    analyzer.exploratory_data_analysis()
    analyzer.visualize_orbital_dynamics()
    analyzer.statistical_analysis()
    analyzer.machine_learning_analysis()
    analyzer.generate_report()
    
    print("\n🎉 Analysis completed successfully!")
    print("📁 Check the following files:")
    print("   - orbital_dynamics.png")
    print("   - interactive_analysis.html")
    print("   - analysis_report.md")

if __name__ == "__main__":
    main()
