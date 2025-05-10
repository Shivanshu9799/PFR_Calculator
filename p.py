import streamlit as st
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calculate_pressure_profile(P0, beta0, L):
    z_points = np.linspace(0, L, 100)
    P_points = []
    delta_P_points = []
    
    for z in z_points:
        P_ratio = (1 - (2 * beta0 * z) / P0)**0.5
        P = P0 * P_ratio
        delta_P = P0 - P
        
        P_points.append(P)
        delta_P_points.append(delta_P)
    
    return z_points, P_points, delta_P_points

st.title("Packed Bed Reactor Pressure Drop Calculator")

# Basic parameters
P0 = st.number_input("Inlet pressure (atm):", min_value=0.1, value=1.0, step=0.1)
L = st.number_input("Length of packed bed (ft):", min_value=0.1, value=5.0, step=0.1)

# Choose calculation method
beta0_choice = st.radio("β₀ Input Method:", ["Calculate β₀", "Input β₀ directly"])

if beta0_choice == "Input β₀ directly":
    beta0_atm_per_ft = st.number_input("Enter β₀ value (atm/ft):", min_value=0.0001, value=0.01, format="%f")
else:
    # Schedule 40 pipe sizes and areas (ft²)
    pipe_options = {
        "1/8 inch (0.269\" ID)": 0.000395,
        "1/4 inch (0.364\" ID)": 0.000723,
        "3/8 inch (0.493\" ID)": 0.001327,
        "1/2 inch (0.622\" ID)": 0.002110,
        "3/4 inch (0.824\" ID)": 0.003707,
        "1 inch (1.049\" ID)": 0.006013,
        "1-1/4 inch (1.380\" ID)": 0.010406,
        "1-1/2 inch (1.610\" ID)": 0.014150,
        "2 inch (2.067\" ID)": 0.023350,
        "2-1/2 inch (2.469\" ID)": 0.033260,
        "3 inch (3.068\" ID)": 0.051500,
        "4 inch (4.026\" ID)": 0.088600
    }
    
    # Pipe selection
    pipe_selection = st.selectbox("Select Schedule 40 Pipe Size:", list(pipe_options.keys()))
    Ac = pipe_options[pipe_selection]
    st.write(f"Pipe cross-sectional area: {Ac:.6f} ft²")
    
    # Bed parameters
    Dp = st.number_input("Particle diameter (inches):", min_value=0.01, value=0.25, step=0.01) / 12
    porosity = st.number_input("Void fraction (0-1):", min_value=0.01, max_value=0.99, value=0.4, step=0.01)
    
    # Flow parameters
    mass_flow = st.number_input("Mass flow rate (lb/h):", min_value=0.1, value=100.0, step=0.1)
    G = mass_flow / Ac
    st.write(f"Superficial mass velocity (G): {G:.2f} lb_m/(h·ft²)")
    
    # Gas properties
    use_default = st.checkbox("Use default gas properties for air at 260°C", value=True)
    if use_default:
        mu = 0.0673
        rho0 = 0.413
        st.write(f"Using: μ = {mu} lb_m/(ft·h), ρ₀ = {rho0} lb_m/ft³")
    else:
        mu = st.number_input("Gas viscosity (lb_m/(ft·h)):", min_value=0.0001, value=0.0673, format="%f")
        rho0 = st.number_input("Gas density (lb_m/ft³):", min_value=0.0001, value=0.413, format="%f")
    
    # Calculate β₀
    gc = 4.17 * 10**8
    term1 = G * (1 - porosity) / (gc * rho0 * Dp * porosity**3)
    term2 = (150 * (1 - porosity) * mu / Dp) + (1.75 * G)
    beta0 = term1 * term2
    beta0_atm_per_ft = beta0 / 2116.2
    st.write(f"Calculated β₀: {beta0_atm_per_ft:.6f} atm/ft")

# Calculate results
if st.button("Calculate Pressure Drop"):
    try:
        P_ratio = (1 - (2 * beta0_atm_per_ft * L) / P0)**0.5
        
        if P_ratio <= 0:
            st.error("Error: Invalid pressure ratio. The pressure drop is too large.")
        else:
            P_final = P0 * P_ratio
            delta_P = P0 - P_final
            
            # Display results
            st.success("Calculation Complete!")
            st.metric("Final pressure", f"{P_final:.2f} atm")
            st.metric("Pressure drop", f"{delta_P:.2f} atm")
            
            # Generate pressure profile
            z_points, P_points, delta_P_points = calculate_pressure_profile(P0, beta0_atm_per_ft, L)
            
            # Create plots
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(z_points, P_points, 'b-', linewidth=2)
            ax.set_title('Pressure Along Packed Bed')
            ax.set_xlabel('Bed Length (ft)')
            ax.set_ylabel('Pressure (atm)')
            ax.grid(True)
            st.pyplot(fig)
            
            # Data table
            df = pd.DataFrame({
                'Bed Length (ft)': z_points,
                'Pressure (atm)': P_points,
                'Pressure Drop (atm)': delta_P_points
            })
            st.dataframe(df.iloc[::10])
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
