import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp

# Constants
R = 8.314  # J/(mol*K)

st.set_page_config(page_title="Reactor Comparison Tool", layout="wide")
st.title("PFR vs CSTR Reactor Comparison")
st.markdown(
    "<h4 style='text-align:center'>Reversible Reaction A ↔ B</h4>",
    unsafe_allow_html=True
)

# Sidebar Inputs
st.sidebar.header("Reaction Parameters")

# Main reaction parameters
delHnotrx = st.sidebar.number_input(
    "ΔH°rxn (J/mol)", value=-80000.0, format="%.1f"
)
To = st.sidebar.number_input(
    "Inlet Temperature T₀ (K)", min_value=100.0, value=300.0
)
Ea = st.sidebar.number_input(
    "Activation Energy Ea (J/mol)", min_value=0.0, value=60000.0
)

# Kinetic parameters
col1, col2 = st.sidebar.columns(2)
with col1:
    Kc2 = st.number_input(
        "Kc₂ at T₂", min_value=0.001, value=0.5, step=0.1
    )
with col2:
    T2 = st.number_input(
        "T₂ (K)", min_value=100.0, value=400.0
    )

col3, col4 = st.sidebar.columns(2)
with col3:
    k1 = st.number_input(
        "k₁ at T₁", min_value=0.001, value=0.1, step=0.01
    )
with col4:
    T1 = st.number_input(
        "T₁ (K)", min_value=100.0, value=300.0
    )

# Concentration and flow parameters
Cao = st.sidebar.number_input(
    "Initial Concentration Cₐ₀ (mol/m³)", min_value=0.001, value=1.0
)
Fao = st.sidebar.number_input(
    "Molar Flow Rate Fₐ₀ (mol/s)", min_value=0.001, value=10.0
)

# Heat capacity parameters
st.sidebar.subheader("Heat Capacities (J/mol·K)")
Cpa = st.sidebar.number_input(
    "Cp(A)", min_value=0.0, value=100.0
)
Cpb = st.sidebar.number_input(
    "Cp(B)", min_value=0.0, value=100.0
)
Cp_inert = st.sidebar.number_input(
    "Cp(Inert)", min_value=0.0, value=50.0
)
theta_inert = st.sidebar.number_input(
    "θ Inert", min_value=0.0, value=0.5, step=0.1
)

# Conversion target
X = st.sidebar.slider(
    "Target Conversion X", min_value=0.01, max_value=0.99, value=0.8, step=0.01
)

# Reactor cost rates
st.sidebar.header("Reactor Cost Parameters")
pfr_rate = st.sidebar.number_input(
    "PFR Cost Rate ($/m³)", min_value=0.01, value=1000.0, step=10.0
)
cstr_rate = st.sidebar.number_input(
    "CSTR Cost Rate ($/m³)", min_value=0.01, value=800.0, step=10.0
)

def calculate_reactors(delHnotrx, To, Ea, Kc2, T2, k1, T1, Cao, Cpa, Cpb, Cp_inert, theta_inert, Fao, X):
    # Calculate outlet temperature (same for both reactors)
    sum_Cp = Cpa + theta_inert * Cp_inert
    T_out = To + X * (-delHnotrx) / sum_Cp

    # CSTR calculations
    try:
        k_cstr = k1 * np.exp((Ea / R) * (1/T1 - 1/T_out))
        Kc_cstr = Kc2 * np.exp((delHnotrx / R) * (1/T2 - 1/T_out))
        denominator = 1 - (1 + 1/Kc_cstr) * X
        
        if denominator <= 0:
            V_cstr = np.inf
        else:
            V_cstr = (Fao * X) / (k_cstr * Cao * denominator)
    except:
        V_cstr = np.inf

    # PFR calculations
    def pfr_ode(V, X_current):
        T_current = To + X_current[0] * (-delHnotrx) / sum_Cp
        k = k1 * np.exp((Ea / R) * (1/T1 - 1/T_current))
        Kc = Kc2 * np.exp((delHnotrx / R) * (1/T2 - 1/T_current))
        dXdV_val = (k * Cao * (1 - (1 + 1/Kc) * X_current[0])) / Fao
        return [dXdV_val]

    def event(V, X_current):
        return X_current[0] - X
    event.terminal = True
    event.direction = 1

    try:
        sol = solve_ivp(pfr_ode, [0, 1e6], [0.0], events=event, max_step=0.1)
        if sol.status == 1 and len(sol.t_events[0]) > 0:
            V_pfr = sol.t_events[0][0]
        else:
            V_pfr = np.inf
    except:
        V_pfr = np.inf

    return V_pfr, T_out, V_cstr, T_out

# Compute button
if st.sidebar.button("Calculate Reactor Volumes and Costs"):
    V_pfr, T_out_pfr, V_cstr, T_out_cstr = calculate_reactors(
        delHnotrx, To, Ea, Kc2, T2, k1, T1, Cao, Cpa, Cpb, 
        Cp_inert, theta_inert, Fao, X
    )
    
    # Calculate costs
    pfr_cost = V_pfr * pfr_rate if np.isfinite(V_pfr) else np.inf
    cstr_cost = V_cstr * cstr_rate if np.isfinite(V_cstr) else np.inf
    
    # Display results
    st.subheader("Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### PFR Results")
        if np.isfinite(V_pfr):
            st.metric("Required Volume", f"{V_pfr:.2f} m³")
            st.metric("Total Cost", f"${pfr_cost:,.2f}")
        else:
            st.error("Target conversion not achievable in PFR")
        st.metric("Outlet Temperature", f"{T_out_pfr:.2f} K")
    
    with col2:
        st.markdown("### CSTR Results")
        if np.isfinite(V_cstr):
            st.metric("Required Volume", f"{V_cstr:.2f} m³")
            st.metric("Total Cost", f"${cstr_cost:,.2f}")
        else:
            st.error("Target conversion not achievable in CSTR")
        st.metric("Outlet Temperature", f"{T_out_cstr:.2f} K")
    
    # Additional information
    st.markdown("---")
    st.markdown("### Design Considerations")
    if np.isfinite(V_pfr) and np.isfinite(V_cstr):
        if V_pfr < V_cstr:
            vol_diff = V_cstr - V_pfr
            cost_diff = cstr_cost - pfr_cost
            st.success(f"PFR requires {vol_diff:.2f} m³ less volume (${cost_diff:,.2f} cheaper)")
        else:
            vol_diff = V_pfr - V_cstr
            cost_diff = pfr_cost - cstr_cost
            st.success(f"CSTR requires {vol_diff:.2f} m³ less volume (${cost_diff:,.2f} cheaper)")
    elif np.isfinite(V_pfr):
        st.success("Only PFR can achieve this conversion")
    elif np.isfinite(V_cstr):
        st.success("Only CSTR can achieve this conversion")
    else:
        st.error("Target conversion not achievable with either reactor type")
    
    st.markdown("_*Calculations based on reversible reaction kinetics*_")