import streamlit as st
import numpy as np
from pfr_solver import solve_pfr_system # assuming you save your main code as pfr_solver.py

st.set_page_config(page_title="PFR Simulator", layout="wide")
st.title("Plug Flow Reactor (PFR) Simulator")


# Sidebar Inputs
st.sidebar.header("Reaction Parameters")
XA_target = st.sidebar.slider("Target Conversion (X_A)", 0.01, 1.0, 0.45, step=0.01)
use_direct_A = st.sidebar.checkbox("Provide Pre-exponential Factor (A)?", value=False)

if use_direct_A:
    A_arrhenius = st.sidebar.number_input("A (1/s)", min_value=1e0, value=1e5)
    k1_ref_temp = None
else:
    A_arrhenius = None
    k_val = st.sidebar.number_input("k_ref (1/s)", min_value=1e-6, value=0.05)
    T_val = st.sidebar.number_input("T_ref (K)", min_value=1.0, value=320.0)
    k1_ref_temp = (k_val, T_val)

Ea = st.sidebar.number_input("Activation Energy Ea (J/mol)", value=75000)

st.sidebar.header("Heat Transfer & Reactor Settings")
U = st.sidebar.number_input("Overall Heat Transfer Coefficient U (W/m².K)", value=300.0)
a_v = st.sidebar.number_input("Area per Unit Volume a_v (1/m)", value=15.0)
T_a = st.sidebar.number_input("Ambient Temperature T_a (K)", value=293.15)

delta_H_rxn = st.sidebar.number_input("ΔH_rxn (J/mol)", value=-55000.0)

st.sidebar.header("Feed Properties")
rho = st.sidebar.number_input("Density ρ (kg/m³)", value=980.0)
F_A0 = st.sidebar.number_input("Inlet Molar Flow Rate F_A0 (mol/s)", value=0.1)
C_A0 = st.sidebar.number_input("Inlet Concentration C_A0 (mol/m³)", value=1.0)
T0 = st.sidebar.number_input("Inlet Temperature T₀ (K)", value=303.15)

st.sidebar.header("Cp(T) Parameters")
alpha = st.sidebar.number_input("Cp α", value=2000.0)
beta = st.sidebar.number_input("Cp β", value=0.5)
gamma = st.sidebar.number_input("Cp γ", value=0.0001)

V_max_integration = st.sidebar.slider("Max Reactor Volume to Simulate (m³)", 0.1, 20.0, 10.0, step=0.1)

if st.button("Simulate PFR"):
    with st.spinner("Solving..."):
        try:
            V_reactor, T_outlet, V_profile, X_profile, T_profile, X_achieved = solve_pfr_system(
                XA_target=XA_target,
                A_arrhenius=A_arrhenius,
                Ea=Ea,
                k1_ref_temp=k1_ref_temp,
                U=U,
                a_v=a_v,
                T_a=T_a,
                delta_H_rxn=delta_H_rxn,
                rho=rho,
                Cp_params={'alpha': alpha, 'beta': beta, 'gamma': gamma},
                F_A0=F_A0,
                C_A0=C_A0,
                T0=T0,
                V_max_integration=V_max_integration
            )

            st.success(f"✅ Simulation complete!")
            st.metric("Required Reactor Volume", f"{V_reactor:.3f} m³")
            st.metric("Outlet Temperature", f"{T_outlet:.2f} K")
            st.metric("Achieved Conversion", f"{X_achieved:.3f}")

            # st.subheader("Conversion Profile")
            # st.line_chart(data={'Conversion X': X_profile}, x=V_profile)

            # st.subheader("Temperature Profile")
            # st.line_chart(data={'Temperature (K)': T_profile}, x=V_profile)
              # Assumptions Table
            st.subheader("Assumptions Used in Simulation")
            st.markdown(
            """
| Category      | Assumption                                               |
|---------------|----------------------------------------------------------|
| Plug Flow     | No axial dispersion or back-mixing                       |
| Stoichiometry | Single A → B reaction, 1:1 molar ratio                   |
| Kinetics      | Rate = k(T)·C_A·C_B; k(T)=k0·exp((–Ea/R)/(1/T - 1/T)))   |
| Density       | Constant ρ; volumetric flow = F_A0/C_A0                  |
| Heat Capacity | Cₚ,i(T)=αᵢ+βᵢT+γᵢT² ; Cp_A = Cp_B ; (del Cp=0)            |
| Heat Transfer | Lumped U·a_v·(T_amb–T) to infinite‐sink ambient          |
| Numerics      | Stop when X hits target (after tiny V>1e–6 to avoid V=0) |
            """
        ) 

        except Exception as e:
            st.error("Error: The required reactor volume exceeds the maximum capacity we can provide. "  
                     "The maximum allowable volume is 20 m³, but the requested volume is greater than this limit.")
