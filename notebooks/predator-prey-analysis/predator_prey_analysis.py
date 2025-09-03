#!/usr/bin/env python3
"""
Predator-Prey Analysis: Comprehensive Data Science Pipeline
for Lotka-Volterra Ecological Dynamics

This script provides comprehensive analysis of predator-prey simulation data,
including statistical analysis, visualization, and machine learning applications.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PredatorPreyAnalyzer:
    """Comprehensive analyzer for predator-prey simulation data."""
    
    def __init__(self, data_file=None):
        """Initialize analyzer with data file path."""
        self.data_file = data_file
        self.df = None
        self.stats = {}
        self.models = {}
        
    def load_data(self, data_file=None):
        """Load predator-prey simulation data from CSV."""
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
        """Create sample predator-prey data for demonstration."""
        print("🔄 Creating sample predator-prey data...")
        
        # Lotka-Volterra parameters
        r, a, b, m = 0.1, 0.01, 0.1, 0.1
        
        # Initial conditions
        prey, predator = 1000, 100
        dt = 0.1
        
        data = []
        for i in range(n_points):
            # Lotka-Volterra equations
            dprey_dt = r * prey - a * prey * predator
            dpredator_dt = b * a * prey * predator - m * predator
            
            prey += dprey_dt * dt
            predator += dpredator_dt * dt
            
            # Ensure non-negative populations
            prey = max(0, prey)
            predator = max(0, predator)
            
            data.append({
                'timestamp': i * dt,
                'iteration': i,
                'prey': prey,
                'predator': predator,
                'prey_growth_rate': r,
                'predation_rate': a,
                'predator_death_rate': m,
                'conversion_efficiency': b
            })
        
        self.df = pd.DataFrame(data)
        print(f"✅ Created sample data with {len(self.df)} points")
        return True
    
    def exploratory_data_analysis(self):
        """Perform comprehensive exploratory data analysis."""
        print("\n🔍 EXPLORATORY DATA ANALYSIS")
        print("=" * 50)
        
        # Basic statistics
        print("\n📊 Basic Statistics:")
        print(self.df.describe())
        
        # Population dynamics
        print(f"\n📈 Population Dynamics:")
        print(f"Prey - Min: {self.df['prey'].min():.0f}, Max: {self.df['prey'].max():.0f}, Mean: {self.df['prey'].mean():.1f}")
        print(f"Predator - Min: {self.df['predator'].min():.0f}, Max: {self.df['predator'].max():.0f}, Mean: {self.df['predator'].mean():.1f}")
        
        # Correlation analysis
        correlation = self.df[['prey', 'predator']].corr()
        print(f"\n🔗 Correlation between Prey and Predator: {correlation.loc['prey', 'predator']:.3f}")
        
        # Phase space analysis
        self.df['prey_change'] = self.df['prey'].diff()
        self.df['predator_change'] = self.df['predator'].diff()
        
        # Cycle detection
        self._detect_cycles()
        
        return self.df.describe()
    
    def _detect_cycles(self):
        """Detect population cycles and oscillations."""
        # Find peaks and troughs
        prey_peaks = self._find_peaks(self.df['prey'].values)
        predator_peaks = self._find_peaks(self.df['predator'].values)
        
        print(f"\n🔄 Cycle Analysis:")
        print(f"Prey peaks detected: {len(prey_peaks)}")
        print(f"Predator peaks detected: {len(predator_peaks)}")
        
        if len(prey_peaks) > 1:
            cycle_length = np.mean(np.diff(prey_peaks))
            print(f"Average cycle length: {cycle_length:.1f} time units")
        
        self.stats['prey_peaks'] = prey_peaks
        self.stats['predator_peaks'] = predator_peaks
    
    def _find_peaks(self, data, threshold=0.1):
        """Find peaks in time series data."""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(data, height=np.mean(data) * (1 + threshold))
        return peaks
    
    def visualize_population_dynamics(self):
        """Create comprehensive visualizations of population dynamics."""
        print("\n📊 Creating Population Dynamics Visualizations...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Predator-Prey Population Dynamics Analysis', fontsize=16, fontweight='bold')
        
        # Time series plot
        axes[0, 0].plot(self.df['timestamp'], self.df['prey'], label='Prey', color='green', linewidth=2)
        axes[0, 0].plot(self.df['timestamp'], self.df['predator'], label='Predator', color='red', linewidth=2)
        axes[0, 0].set_title('Population Time Series')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Population')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Phase space plot
        axes[0, 1].plot(self.df['prey'], self.df['predator'], color='purple', linewidth=1.5)
        axes[0, 1].scatter(self.df['prey'].iloc[0], self.df['predator'].iloc[0], 
                          color='green', s=100, label='Start', zorder=5)
        axes[0, 1].scatter(self.df['prey'].iloc[-1], self.df['predator'].iloc[-1], 
                          color='red', s=100, label='End', zorder=5)
        axes[0, 1].set_title('Phase Space (Prey vs Predator)')
        axes[0, 1].set_xlabel('Prey Population')
        axes[0, 1].set_ylabel('Predator Population')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Population ratio
        ratio = self.df['prey'] / (self.df['predator'] + 1)  # Avoid division by zero
        axes[1, 0].plot(self.df['timestamp'], ratio, color='orange', linewidth=2)
        axes[1, 0].set_title('Prey/Predator Ratio Over Time')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Prey/Predator Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distribution analysis
        axes[1, 1].hist(self.df['prey'], bins=30, alpha=0.7, label='Prey', color='green')
        axes[1, 1].hist(self.df['predator'], bins=30, alpha=0.7, label='Predator', color='red')
        axes[1, 1].set_title('Population Distribution')
        axes[1, 1].set_xlabel('Population')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('notebooks/predator-prey-analysis/population_dynamics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Interactive Plotly visualization
        self._create_interactive_plot()
    
    def _create_interactive_plot(self):
        """Create interactive Plotly visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Population Time Series', 'Phase Space', 'Population Ratio', '3D Trajectory'),
            specs=[[{"secondary_y": True}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter3d"}]]
        )
        
        # Time series
        fig.add_trace(
            go.Scatter(x=self.df['timestamp'], y=self.df['prey'], name='Prey', line=dict(color='green')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.df['timestamp'], y=self.df['predator'], name='Predator', line=dict(color='red')),
            row=1, col=1, secondary_y=True
        )
        
        # Phase space
        fig.add_trace(
            go.Scatter(x=self.df['prey'], y=self.df['predator'], mode='lines+markers', name='Trajectory'),
            row=1, col=2
        )
        
        # Population ratio
        ratio = self.df['prey'] / (self.df['predator'] + 1)
        fig.add_trace(
            go.Scatter(x=self.df['timestamp'], y=ratio, name='Prey/Predator Ratio', line=dict(color='orange')),
            row=2, col=1
        )
        
        # 3D trajectory
        fig.add_trace(
            go.Scatter3d(x=self.df['timestamp'], y=self.df['prey'], z=self.df['predator'],
                        mode='lines+markers', name='3D Trajectory', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Interactive Predator-Prey Analysis")
        fig.write_html('notebooks/predator-prey-analysis/interactive_analysis.html')
        print("✅ Interactive plot saved as interactive_analysis.html")
    
    def statistical_analysis(self):
        """Perform comprehensive statistical analysis."""
        print("\n📈 STATISTICAL ANALYSIS")
        print("=" * 50)
        
        # Normality tests
        prey_shapiro = stats.shapiro(self.df['prey'])
        predator_shapiro = stats.shapiro(self.df['predator'])
        
        print(f"\n🔍 Normality Tests (Shapiro-Wilk):")
        print(f"Prey - Statistic: {prey_shapiro.statistic:.4f}, p-value: {prey_shapiro.pvalue:.4f}")
        print(f"Predator - Statistic: {predator_shapiro.statistic:.4f}, p-value: {predator_shapiro.pvalue:.4f}")
        
        # Correlation analysis
        correlation, p_value = stats.pearsonr(self.df['prey'], self.df['predator'])
        print(f"\n🔗 Pearson Correlation: {correlation:.4f} (p-value: {p_value:.4f})")
        
        # Lag analysis
        self._lag_analysis()
        
        # Stability analysis
        self._stability_analysis()
        
        return {
            'prey_normality': prey_shapiro,
            'predator_normality': predator_shapiro,
            'correlation': (correlation, p_value)
        }
    
    def _lag_analysis(self):
        """Analyze lag relationships between prey and predator."""
        from scipy.stats import pearsonr
        
        max_lag = 50
        correlations = []
        lags = range(-max_lag, max_lag + 1)
        
        for lag in lags:
            if lag == 0:
                corr, _ = pearsonr(self.df['prey'], self.df['predator'])
            elif lag > 0:
                corr, _ = pearsonr(self.df['prey'][:-lag], self.df['predator'][lag:])
            else:
                corr, _ = pearsonr(self.df['prey'][-lag:], self.df['predator'][:lag])
            correlations.append(corr)
        
        max_corr_idx = np.argmax(np.abs(correlations))
        optimal_lag = lags[max_corr_idx]
        max_correlation = correlations[max_corr_idx]
        
        print(f"\n⏰ Lag Analysis:")
        print(f"Optimal lag: {optimal_lag} time units")
        print(f"Maximum correlation: {max_correlation:.4f}")
        
        self.stats['optimal_lag'] = optimal_lag
        self.stats['max_correlation'] = max_correlation
    
    def _stability_analysis(self):
        """Analyze system stability and convergence."""
        # Calculate coefficient of variation
        prey_cv = np.std(self.df['prey']) / np.mean(self.df['prey'])
        predator_cv = np.std(self.df['predator']) / np.mean(self.df['predator'])
        
        # Calculate Lyapunov exponent (simplified)
        lyapunov = self._estimate_lyapunov_exponent()
        
        print(f"\n⚖️ Stability Analysis:")
        print(f"Prey Coefficient of Variation: {prey_cv:.4f}")
        print(f"Predator Coefficient of Variation: {predator_cv:.4f}")
        print(f"Estimated Lyapunov Exponent: {lyapunov:.4f}")
        
        self.stats['prey_cv'] = prey_cv
        self.stats['predator_cv'] = predator_cv
        self.stats['lyapunov'] = lyapunov
    
    def _estimate_lyapunov_exponent(self):
        """Estimate Lyapunov exponent for stability analysis."""
        # Simplified estimation using divergence of nearby trajectories
        n = len(self.df)
        if n < 100:
            return 0
        
        # Calculate divergence over time
        divergences = []
        for i in range(10, n-10):
            # Calculate local divergence
            local_div = np.sqrt((self.df['prey'].iloc[i+1] - self.df['prey'].iloc[i])**2 + 
                              (self.df['predator'].iloc[i+1] - self.df['predator'].iloc[i])**2)
            divergences.append(local_div)
        
        # Estimate Lyapunov exponent
        if len(divergences) > 0:
            return np.mean(np.log(np.array(divergences) + 1e-10))
        return 0
    
    def machine_learning_analysis(self):
        """Apply machine learning techniques to the data."""
        print("\n🤖 MACHINE LEARNING ANALYSIS")
        print("=" * 50)
        
        # Prepare features
        X = self.df[['prey', 'predator']].values
        y_prey = self.df['prey'].shift(-1).dropna().values
        y_predator = self.df['predator'].shift(-1).dropna().values
        
        # Remove last row to match lengths
        X = X[:-1]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Predict prey population
        model_prey = LinearRegression()
        model_prey.fit(X_scaled, y_prey)
        y_pred_prey = model_prey.predict(X_scaled)
        
        # Predict predator population
        model_predator = LinearRegression()
        model_predator.fit(X_scaled, y_predator)
        y_pred_predator = model_predator.predict(X_scaled)
        
        # Calculate metrics
        r2_prey = r2_score(y_prey, y_pred_prey)
        r2_predator = r2_score(y_predator, y_pred_predator)
        
        print(f"\n📊 Prediction Performance:")
        print(f"Prey R² Score: {r2_prey:.4f}")
        print(f"Predator R² Score: {r2_predator:.4f}")
        
        # Clustering analysis
        self._clustering_analysis(X_scaled)
        
        # Store models
        self.models['prey_predictor'] = model_prey
        self.models['predator_predictor'] = model_predator
        self.models['scaler'] = scaler
        
        return {
            'prey_r2': r2_prey,
            'predator_r2': r2_predator,
            'models': self.models
        }
    
    def _clustering_analysis(self, X_scaled):
        """Perform clustering analysis on population states."""
        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add clusters to dataframe
        self.df['cluster'] = np.nan
        self.df.loc[:len(clusters)-1, 'cluster'] = clusters
        
        print(f"\n🎯 Clustering Analysis:")
        print(f"Identified {len(np.unique(clusters))} distinct population states")
        
        # Cluster statistics
        for cluster_id in np.unique(clusters):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                print(f"Cluster {cluster_id}: {len(cluster_data)} points, "
                      f"Avg Prey: {cluster_data['prey'].mean():.1f}, "
                      f"Avg Predator: {cluster_data['predator'].mean():.1f}")
        
        self.stats['clusters'] = clusters
        self.stats['kmeans'] = kmeans
    
    def parameter_sensitivity_analysis(self):
        """Analyze sensitivity to model parameters."""
        print("\n🔬 PARAMETER SENSITIVITY ANALYSIS")
        print("=" * 50)
        
        # Get unique parameter values
        params = ['prey_growth_rate', 'predation_rate', 'predator_death_rate', 'conversion_efficiency']
        
        for param in params:
            if param in self.df.columns:
                unique_values = self.df[param].unique()
                print(f"\n📊 {param}: {len(unique_values)} unique values")
                
                # Analyze effect on final populations
                final_prey = self.df.groupby(param)['prey'].last()
                final_predator = self.df.groupby(param)['predator'].last()
                
                print(f"Final Prey Range: {final_prey.min():.1f} - {final_prey.max():.1f}")
                print(f"Final Predator Range: {final_predator.min():.1f} - {final_predator.max():.1f}")
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("\n📋 GENERATING COMPREHENSIVE REPORT")
        print("=" * 50)
        
        report = f"""
# Predator-Prey Analysis Report

## Dataset Overview
- **Total Records**: {len(self.df)}
- **Time Range**: {self.df['timestamp'].min():.2f} - {self.df['timestamp'].max():.2f}
- **Prey Range**: {self.df['prey'].min():.0f} - {self.df['prey'].max():.0f}
- **Predator Range**: {self.df['predator'].min():.0f} - {self.df['predator'].max():.0f}

## Key Statistics
- **Prey Mean**: {self.df['prey'].mean():.1f} ± {self.df['prey'].std():.1f}
- **Predator Mean**: {self.df['predator'].mean():.1f} ± {self.df['predator'].std():.1f}
- **Correlation**: {self.df[['prey', 'predator']].corr().loc['prey', 'predator']:.3f}

## Stability Analysis
- **Prey CV**: {self.stats.get('prey_cv', 0):.4f}
- **Predator CV**: {self.stats.get('predator_cv', 0):.4f}
- **Lyapunov Exponent**: {self.stats.get('lyapunov', 0):.4f}

## Cycle Analysis
- **Prey Peaks**: {len(self.stats.get('prey_peaks', []))}
- **Predator Peaks**: {len(self.stats.get('predator_peaks', []))}
- **Optimal Lag**: {self.stats.get('optimal_lag', 0)}

## Machine Learning Results
- **Prey Prediction R²**: {self.models.get('prey_predictor', {}).score if hasattr(self.models.get('prey_predictor'), 'score') else 'N/A'}
- **Predator Prediction R²**: {self.models.get('predator_predictor', {}).score if hasattr(self.models.get('predator_predictor'), 'score') else 'N/A'}

## Conclusions
1. The system exhibits {'stable' if self.stats.get('lyapunov', 0) < 0 else 'unstable'} dynamics
2. Population oscillations show {'strong' if abs(self.stats.get('max_correlation', 0)) > 0.5 else 'weak'} correlation
3. The model demonstrates {'good' if self.stats.get('prey_cv', 1) < 0.5 else 'high'} variability in prey population
"""
        
        # Save report
        with open('notebooks/predator-prey-analysis/analysis_report.md', 'w') as f:
            f.write(report)
        
        print("✅ Report saved as analysis_report.md")
        return report

def main():
    """Main analysis pipeline."""
    print("🦌 PREDATOR-PREY ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PredatorPreyAnalyzer()
    
    # Load or create data
    if not analyzer.load_data():
        print("📊 Creating sample data for demonstration...")
        analyzer.create_sample_data(1000)
    
    # Run analysis pipeline
    analyzer.exploratory_data_analysis()
    analyzer.visualize_population_dynamics()
    analyzer.statistical_analysis()
    analyzer.machine_learning_analysis()
    analyzer.parameter_sensitivity_analysis()
    analyzer.generate_report()
    
    print("\n🎉 Analysis completed successfully!")
    print("📁 Check the following files:")
    print("   - population_dynamics.png")
    print("   - interactive_analysis.html")
    print("   - analysis_report.md")

if __name__ == "__main__":
    main()
