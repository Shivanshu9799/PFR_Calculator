# main_app.py
import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from pfr_solver import solve_pfr_system

# starting with page setup 
st.set_page_config(
    page_title="Reactor Engineering Toolkit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title dalte
st.title("Reactor Engineering Toolkit")

# calc choice add karne ke liye 
app_choice = st.sidebar.radio(
    "Select Calculator:",
    ("PFR vs CSTR Comparison", "Packed Bed Pressure Drop", "PFR Simulator"),
    index=0
)
# CALC 1 
# ==============================================
# PFR vs CSTR Comparison
# ==============================================
def run_pfr_cstr_comparison():
    st.header("PFR vs CSTR Reactor Comparison")
    st.markdown("<h4 style='text-align:center'>Reversible Reaction A ↔ B</h4>", unsafe_allow_html=True)

    # Sidebar Inputs
    with st.sidebar:
        st.header("Reaction Parameters")
        delHnotrx = st.number_input("ΔH°rxn (J/mol)", value=-6900.0, format="%.1f")
        To = st.number_input("Inlet Temperature T₀ (K)", min_value=100.0, value=330.0)
        Ea = st.number_input("Activation Energy Ea (J/mol)", min_value=0.0, value=65700.0)
    # basically kuch nahi kar raha bas input le rahe user se with the limit on input edge cases ke liye
        st.header("Kinetic Parameters")
        col1, col2 = st.columns(2)
        with col1:
            Kc2 = st.number_input("Kc₂ at T₂", min_value=0.0001, value=3.03, step=0.1)
        with col2:
            T2 = st.number_input("T₂ (K)", min_value=1.0, value=333.0)

        col3, col4 = st.columns(2)
        with col3:
            k1 = st.number_input("k₁ at T₁", min_value=0.0001, value=0.1, step=0.001)
        with col4:
            T1 = st.number_input("T₁ (K)", min_value=1.0, value=360.00)
   # yaha thoda coloumn format mai  input liya hai 
        st.header("Concentration Parameters")
        Cao = st.number_input("Initial Concentration Cₐ₀ (mol/m³)", min_value=0.001, value=9300.0)
        Fao = st.number_input("Molar Flow Rate Fₐ₀ (mol/s)", min_value=0.001, value=40.75)

        st.header("Heat Capacities (J/mol·K)")
        Cpa = st.number_input("Cp(A)", min_value=0.0, value=141.0)
        Cpb = st.number_input("Cp(B)", min_value=0.0, value=141.0)
        Cp_inert = st.number_input("Cp(Inert)", min_value=0.0, value=161.0)
        theta_inert = st.number_input("θ Inert", min_value=0.0, value=0.5, step=0.1)

        st.header("Conversion Target")
        X = st.slider("Target Conversion X", min_value=0.01, max_value=0.99, value=0.5, step=0.01)

        st.header("Reactor Cost Parameters")
        pfr_rate = st.number_input("PFR Cost Rate (Rs./m³)", min_value=0.01, value=1000.0, step=10.0)
        cstr_rate = st.number_input("CSTR Cost Rate (Rs./m³)", min_value=0.01, value=1000.0, step=10.0)

    # Calculation function
    def calculate_reactors():
        R = 8.314  # J/(mol*K)
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

        return V_pfr, T_out, V_cstr

    # Calculate and display results
    if st.sidebar.button("Calculate Reactor Volumes and Costs"):
        V_pfr, T_out, V_cstr = calculate_reactors()
        
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
                st.metric("Total Cost", f"Rs.{pfr_cost:,.2f}")
            else:
                st.error("Target conversion not achievable in PFR")
            st.metric("Outlet Temperature", f"{T_out:.2f} K")
        
        with col2:
            st.markdown("### CSTR Results")
            if np.isfinite(V_cstr):
                st.metric("Required Volume", f"{V_cstr:.2f} m³")
                st.metric("Total Cost", f"Rs.{cstr_cost:,.2f}")
            else:
                st.error("Target conversion not achievable in CSTR")
            st.metric("Outlet Temperature", f"{T_out:.2f} K")


#starting the second calc.
# ==============================================
# Pressure Drop Calculator
# ==============================================
def run_pressure_drop_calculator():
    st.header("Packed Bed Reactor Pressure Drop Calculator")

    def calculate_pressure_profile(P0, beta0, L):
        z_points = np.linspace(0, L, 100)
        P_points = P0 * (1 - (2 * beta0 * z_points) / P0)**0.5
        delta_P_points = P0 - P_points
        return z_points, P_points, delta_P_points

    # Basic parameters
    P0 = st.number_input("Inlet pressure (atm):", min_value=0.1, value=10.0, step=0.1)
    L = st.number_input("Length of packed bed (ft):", min_value=0.1, value=60.0, step=0.1)
#beta0 par choices de rahe ki dirct input or calaculate 
    # Beta0 input method
    beta0_choice = st.radio("β₀ Input Method:", ["Calculate β₀", "Input β₀ directly"])

    if beta0_choice == "Input β₀ directly":
        beta0_atm_per_ft = st.number_input("Enter β₀ value (atm/ft):", min_value=0.0001, value=0.01, format="%f")

    else:
        pipe_options = {
            "1/8 inch (0.269\" ID)": 0.000395,   #b aa d mai pata chala ki area input circle nahi le sakte .
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

        #list banadi valid area input ki 
        
        pipe_selection = st.selectbox("Select Schedule 40 Pipe Size:", list(pipe_options.keys()))
        Ac = pipe_options[pipe_selection]
        
        Dp = st.number_input("Particle diameter (inches):", min_value=0.01, value=0.25, step=0.01) / 12 #diameter hai catalyst  particles ka
        porosity = st.number_input("Void fraction (0-1):", min_value=0.01, max_value=0.99, value=0.45, step=0.01)
        mass_flow = st.number_input("Mass flow rate (lb/h):", min_value=0.1, value=100.0, step=0.1)
        G = mass_flow / Ac #g is superficial mass velocity jo ki m./ac hojaati hai
        
        use_default = st.checkbox("Use default gas properties for air at 260°C", value=True)
        if use_default:
            mu, rho0 = 0.0673, 0.413
        else:
            mu = st.number_input("Gas viscosity (lb_m/(ft·h)):", value=0.0673)
            rho0 = st.number_input("Gas density (lb_m/ft³):", value=0.413)

        gc = 4.17e8 #unit conversion factor hai for keeping units consistent  lmb * ft/h2 . lbf
        term1 = G * (1 - porosity) / (gc * rho0 * Dp * porosity**3)
        term2 = (150 * (1 - porosity) * mu / Dp) + (1.75 * G)
        beta0 = term1 * term2
        beta0_atm_per_ft = beta0 / 2116.2 # division to convert in these units

    if st.button("Calculate Pressure Drop"):
        try:
            z_points, P_points, delta_P_points = calculate_pressure_profile(P0, beta0_atm_per_ft, L) # aage graph mein use hoga #
            
            st.success("Calculation Complete!")
            if np.isnan(P_points[-1]) or np.isnan(delta_P_points[-1]):
             st.error("Pressure calculation resulted in NaN. Please check your input values.")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final pressure", f"{P_points[-1]:.2f} atm")
            with col2:
                st.metric("Pressure drop", f"{delta_P_points[-1]:.2f} atm")
 #figure and axis create kara hu 
            fig, ax = plt.subplots()
            ax.plot(z_points, P_points, 'b-', linewidth=2) # b-  blue line ke liye hai #
            ax.set_title('Pressure Along Packed Bed') # title for plot
            ax.set_xlabel('Bed Length (ft)') # x & y axix  label #
            ax.set_ylabel('Pressure (atm)')
            ax.grid(True) # turning  on grid for better readibilty #
            st.pyplot(fig)
#Create a pandas DataFrame from the calculated data
            df = pd.DataFrame({
                'Bed Length (ft)': z_points,
                'Pressure (atm)': P_points,
                'Pressure Drop (atm)': delta_P_points
            })
            st.dataframe(df.iloc[::10]) # har 10 step ke baad ki values kar rahe till 10 points to plot the graph#

        except Exception as e:
            st.error(f"Error: {str(e)}")
#calc 3 
# ==============================================
# PFR Simulator
# ==============================================
def run_pfr_simulator():
    st.header("NonIsothermal  Plug Flow Reactor (PFR) Simulator with Cp Varying with T")

    with st.sidebar:
        st.header("Reaction Parameters")
        XA_target = st.slider("Target Conversion (X_A)", 0.01, 1.0, 0.45, step=0.01)
        use_direct_A = st.checkbox("Provide Pre-exponential Factor (A)?", value=False)

        if use_direct_A:
            A_arrhenius = st.number_input("A (1/s)", min_value=1e0, value=1e3)
            k1_ref_temp = None
        else:
            A_arrhenius = None
            k_val = st.number_input("k_ref (1/s)", min_value=1e-6, value=0.05)
            T_val = st.number_input("T_ref (K)", min_value=1.0, value=320.0)
            k1_ref_temp = (k_val, T_val)

        Ea = st.number_input("Activation Energy Ea (J/mol)", value=750)
        
        st.header("Heat Transfer & Reactor Settings")
        U = st.number_input("Overall Heat Transfer Coefficient U (W/m².K)", value=300.0)
        a_v = st.number_input("Area per Unit Volume a_v (1/m)",min_value=0.1, value=15.0)
        T_a = st.number_input("Ambient Temperature T_a (K)",min_value=0.1, value=293.15)
        delta_H_rxn = st.number_input("ΔH_rxn (J/mol)", value=-55000.0)
        
        st.header("Feed Properties")
        rho = st.number_input("Density ρ (kg/m³)",min_value=0.1, value=980.0)
        F_A0 = st.number_input("Inlet Molar Flow Rate F_A0 (mol/s)",min_value=0.0, value=0.1)
        C_A0 = st.number_input("Inlet Concentration C_A0 (mol/m³)",min_value=0.0, value=1.0)
        T0 = st.number_input("Inlet Temperature T₀ (K)", value=303.15)
        
        st.header("Cp(T) Parameters (Cₚ,i(T)=αᵢ+βᵢT+γᵢT²)")
        alpha = st.number_input("Cp α", value=2000.0)
        beta = st.number_input("Cp β", value=0.5)
        gamma = st.number_input("Cp γ", value=0.0001)
        
        V_max_integration = st.slider("Max Reactor Volume to Simulate (m³)", 0.1, 20.0, 10.0, step=0.1)

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

            # --- Custom error checks ---
            # 1. Check if output arrays are empty or have non-realistic values
            if (len(V_profile) == 0 or len(X_profile) == 0 or len(T_profile) == 0 or
                np.any(np.isnan(V_profile)) or np.any(np.isnan(X_profile)) or np.any(np.isnan(T_profile))):
                st.error("Non-realistic values, please check input again.")
            # 2. Check if integration stopped at max volume (slider limit)
            elif np.isclose(V_profile[-1], V_max_integration, atol=1e-6):
                st.error("Maximum reactor volume capacity reached. Try increasing the V_max_integration slider.")
            else:
                st.success("✅ Simulation complete!")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Reactor Volume", f"{V_reactor:.3f} m³")
                with col2:
                    st.metric("Outlet Temperature", f"{T_outlet:.2f} K")
                with col3:
                    st.metric("Achieved Conversion", f"{X_achieved:.3f}")

                # ---- Assumptions Table (Markdown) ----
                st.markdown("#### Model Assumptions")
                st.markdown(
                    """
| Category      | Assumption                                               |
|---------------|----------------------------------------------------------|
| Plug Flow     | No axial dispersion or back-mixing                       |
| Stoichiometry | Single A → B reaction, 1:1 molar ratio                   |
| Kinetics      | Rate = k(T)·C_A·C_B; k(T)=A·exp(–Ea/(R·T))              |
| Density       | Constant ρ; volumetric flow = F_A0/C_A0                  |
| Heat Capacity | Cₚ,i(T)=αᵢ+βᵢT+γᵢT²; mixture Cₚ = (C_A·Cₚ,A + C_B·Cₚ,B)/(C_A+C_B) |
| Enthalpy      | H_A, H_B constant ⇒ ΔH_rxn = H_B – H_A                    |
| Heat Transfer | Lumped U·a_v·(T_amb–T) to infinite‐sink ambient          |
| Numerics      | Stop when X hits target (after tiny V>1e–6 to avoid V=0) |
                    """
                )

        except Exception as e:
            st.error("Non-realistic values, please check input again.\n\nDetails: " + str(e))


# ==============================================
# Main App Router
# ==============================================
if app_choice == "PFR vs CSTR Comparison":
    run_pfr_cstr_comparison()
elif app_choice == "Packed Bed Pressure Drop":
    run_pressure_drop_calculator()
elif app_choice == "PFR Simulator":
    run_pfr_simulator()
