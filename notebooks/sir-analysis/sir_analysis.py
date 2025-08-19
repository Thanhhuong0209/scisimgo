#!/usr/bin/env python3
"""
SIR Disease Model Analysis Script
SciSimGo - Scientific Simulation & Data Science Playground

This script analyzes SIR simulation data to understand:
- Disease dynamics
- Statistical patterns
- Machine Learning predictions
- Comparison with mathematical models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import glob
import os

def create_sample_sir_data():
    """Create sample SIR data for demonstration"""
    np.random.seed(42)
    
    # Parameters
    population = 10000
    initial_infected = 100
    infection_rate = 0.3
    recovery_rate = 0.1
    
    # Time series
    iterations = 200
    time_steps = np.arange(iterations)
    
    # Initialize arrays
    susceptible = np.zeros(iterations)
    infected = np.zeros(iterations)
    recovered = np.zeros(iterations)
    
    # Set initial conditions
    susceptible[0] = population - initial_infected
    infected[0] = initial_infected
    recovered[0] = 0
    
    # Simulate SIR dynamics
    for i in range(1, iterations):
        # Calculate new infections and recoveries
        new_infections = int(infection_rate * susceptible[i-1] * infected[i-1] / population)
        new_recoveries = int(recovery_rate * infected[i-1])
        
        # Update populations
        susceptible[i] = max(0, susceptible[i-1] - new_infections)
        infected[i] = max(0, infected[i-1] + new_infections - new_recoveries)
        recovered[i] = max(0, recovered[i-1] + new_recoveries)
    
    # Create DataFrame
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=iterations, freq='H'),
        'iteration': time_steps,
        'susceptible': susceptible,
        'infected': infected,
        'recovered': recovered,
        'total_population': [population] * iterations,
        'infection_rate': [infection_rate] * iterations,
        'recovery_rate': [recovery_rate] * iterations
    }
    
    return pd.DataFrame(data)

def load_sir_data():
    """Load SIR simulation data"""
    try:
        # Try to load the most recent SIR results
        sir_files = glob.glob("../../data/sir_results_*.csv")
        if sir_files:
            latest_file = max(sir_files, key=lambda x: x.split('_')[-1].split('.')[0])
            df = pd.read_csv(latest_file)
            print(f"Loaded: {latest_file}")
            return df
        else:
            print("Warning: No SIR CSV files found. Creating sample data for demonstration.")
            return create_sample_sir_data()
            
    except Exception as e:
        print(f"Warning: Error loading data: {e}")
        print("Creating sample data for demonstration.")
        return create_sample_sir_data()

def analyze_sir_data(df):
    """Perform comprehensive SIR data analysis"""
    print("SIR Model Analysis")
    print("=" * 50)
    
    # Basic statistics
    print(f"Dataset Shape: {df.shape}")
    print(f"Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total Iterations: {len(df)}")
    
    # Population dynamics
    print(f"\nPopulation Dynamics:")
    print(f"Initial Susceptible: {df['susceptible'].iloc[0]:,}")
    print(f"Initial Infected: {df['infected'].iloc[0]:,}")
    print(f"Initial Recovered: {df['recovered'].iloc[0]:,}")
    print(f"Total Population: {df['total_population'].iloc[0]:,}")
    
    # Peak analysis
    peak_infected_idx = df['infected'].idxmax()
    peak_infected = df['infected'].max()
    peak_time = df.loc[peak_infected_idx, 'timestamp']
    
    print(f"\nPeak Analysis:")
    print(f"Peak Infected: {peak_infected:,} at iteration {peak_infected_idx}")
    print(f"Peak Time: {peak_time}")
    print(f"Attack Rate: {peak_infected / df['total_population'].iloc[0] * 100:.2f}%")
    
    # Final state
    print(f"\nFinal State:")
    print(f"Final Susceptible: {df['susceptible'].iloc[-1]:,}")
    print(f"Final Infected: {df['infected'].iloc[-1]:,}")
    print(f"Final Recovered: {df['recovered'].iloc[-1]:,}")
    print(f"Total Cases: {df['recovered'].iloc[-1] + df['infected'].iloc[-1]:,}")
    
    return peak_infected_idx, peak_infected

def visualize_sir_dynamics(df, peak_infected_idx):
    """Create comprehensive SIR visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SIR Disease Model Analysis', fontsize=16, fontweight='bold')
    
    # 1. Population over time
    axes[0, 0].plot(df['iteration'], df['susceptible'], label='Susceptible', linewidth=2, color='blue')
    axes[0, 0].plot(df['iteration'], df['infected'], label='Infected', linewidth=2, color='red')
    axes[0, 0].plot(df['iteration'], df['recovered'], label='Recovered', linewidth=2, color='green')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Population')
    axes[0, 0].set_title('SIR Population Dynamics')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Infected population (focus)
    axes[0, 1].plot(df['iteration'], df['infected'], linewidth=3, color='red')
    axes[0, 1].axvline(x=peak_infected_idx, color='black', linestyle='--', alpha=0.7, 
                       label=f'Peak: {df["infected"].max():,}')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Infected Population')
    axes[0, 1].set_title('Infected Population Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Growth rates
    infection_growth = df['infected'].pct_change().dropna()
    axes[1, 0].plot(df['iteration'][1:], infection_growth, linewidth=2, color='orange')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Growth Rate')
    axes[1, 0].set_title('Infection Growth Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Population proportions
    total_pop = df['total_population'].iloc[0]
    axes[1, 1].plot(df['iteration'], df['susceptible']/total_pop*100, label='Susceptible %', linewidth=2)
    axes[1, 1].plot(df['iteration'], df['infected']/total_pop*100, label='Infected %', linewidth=2)
    axes[1, 1].plot(df['iteration'], df['recovered']/total_pop*100, label='Recovered %', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Percentage of Population')
    axes[1, 1].set_title('Population Proportions (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sir_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def machine_learning_analysis(df):
    """Perform machine learning analysis on SIR data"""
    print("\nMachine Learning Analysis")
    print("=" * 50)
    
    # Prepare data for ML
    df_ml = df.copy()
    df_ml['time_squared'] = df_ml['iteration'] ** 2
    df_ml['time_cubed'] = df_ml['iteration'] ** 3
    df_ml['susceptible_infected_ratio'] = df_ml['susceptible'] / (df_ml['infected'] + 1)
    df_ml['total_cases'] = df_ml['infected'] + df_ml['recovered']
    
    # Features for prediction
    feature_columns = ['iteration', 'time_squared', 'time_cubed', 'susceptible_infected_ratio']
    X = df_ml[feature_columns]
    
    # Target variables
    y_infected = df_ml['infected']
    
    # Train multiple models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_infected, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'mse': mse,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': model,
            'scaler': scaler
        }
        
        print(f"  R2 = {r2:.4f}, CV R2 = {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    
    return results

def mathematical_model_comparison(df):
    """Compare simulation with mathematical SIR model"""
    print("\nMathematical Model Comparison")
    print("=" * 50)
    
    def mathematical_sir(t, S0, I0, R0, beta, gamma, N):
        """Mathematical SIR model solution"""
        S = np.zeros_like(t)
        I = np.zeros_like(t)
        R = np.zeros_like(t)
        
        S[0] = S0
        I[0] = I0
        R[0] = R0
        
        dt = t[1] - t[0] if len(t) > 1 else 1
        
        for i in range(1, len(t)):
            # Euler method for SIR equations
            dS = -beta * S[i-1] * I[i-1] / N * dt
            dI = (beta * S[i-1] * I[i-1] / N - gamma * I[i-1]) * dt
            dR = gamma * I[i-1] * dt
            
            S[i] = max(0, S[i-1] + dS)
            I[i] = max(0, I[i-1] + dI)
            R[i] = max(0, R[i-1] + dR)
        
        return S, I, R
    
    # Extract parameters from simulation
    S0 = df['susceptible'].iloc[0]
    I0 = df['infected'].iloc[0]
    R0 = df['recovered'].iloc[0]
    N = df['total_population'].iloc[0]
    beta = df['infection_rate'].iloc[0]
    gamma = df['recovery_rate'].iloc[0]
    
    print(f"Parameters: β={beta:.3f}, γ={gamma:.3f}, N={N:,}")
    
    # Generate mathematical solution
    t_math = np.arange(len(df))
    S_math, I_math, R_math = mathematical_sir(t_math, S0, I0, R0, beta, gamma, N)
    
    # Compare with simulation
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Mathematical vs Simulation SIR Models', fontsize=16, fontweight='bold')
    
    variables = [('Susceptible', 'susceptible', S_math, 'blue'),
                ('Infected', 'infected', I_math, 'red'),
                ('Recovered', 'recovered', R_math, 'green')]
    
    for i, (var_name, col_name, math_vals, color) in enumerate(variables):
        axes[i].plot(df['iteration'], df[col_name], label=f'Simulation ({var_name})', 
                     linewidth=2, color=color, alpha=0.8)
        axes[i].plot(t_math, math_vals, label=f'Mathematical ({var_name})', 
                     linewidth=2, color=color, linestyle='--', alpha=0.8)
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel('Population')
        axes[i].set_title(f'{var_name} Population')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sir_mathematical_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate model fit
    print("\nModel Fit Analysis:")
    for var_name, col_name, math_vals in variables:
        mse = mean_squared_error(df[col_name], math_vals)
        r2 = r2_score(df[col_name], math_vals)
        print(f"{var_name}: MSE = {mse:.2f}, R2 = {r2:.4f}")
    
    return I_math

def main():
    """Main analysis function"""
    print("SciSimGo - SIR Disease Model Analysis")
    print("=" * 50)
    
    # Load data
    df = load_sir_data()
    
    # Analyze data
    peak_infected_idx, peak_infected = analyze_sir_data(df)
    
    # Visualize dynamics
    visualize_sir_dynamics(df, peak_infected_idx)
    
    # Machine learning analysis
    ml_results = machine_learning_analysis(df)
    
    # Mathematical model comparison
    I_math = mathematical_model_comparison(df)
    
    # Summary
    print("\nAnalysis Summary")
    print("=" * 50)
    print(f" SIR simulation analyzed successfully")
    print(f" Machine learning models trained")
    print(f" Mathematical model comparison completed")
    print(f" Visualizations saved as PNG files")
    
    # Key insights
    R0 = df['infection_rate'].iloc[0] / df['recovery_rate'].iloc[0]
    print(f"\nKey Insights:")
    print(f"- Reproductive Number (R0): {R0:.2f}")
    if R0 > 1:
        print("- R0 > 1: Epidemic will spread (confirmed by simulation)")
    else:
        print("- R0 <= 1: Disease will not spread significantly")
    
    best_ml_score = max([ml_results[model]['cv_mean'] for model in ml_results.keys()])
    print(f"- Best ML Model CV R2: {best_ml_score:.4f}")
    
    math_r2 = r2_score(df['infected'], I_math)
    print(f"- Mathematical Model R2: {math_r2:.4f}")

if __name__ == "__main__":
    main()
