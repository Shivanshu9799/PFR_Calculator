from scipy.integrate import solve_ivp
import numpy as np

# Global Constants
R_GAS_CONST = 8.314  # J/mol.K

# --- Helper Functions from previous CSTR code (reused) ---
def calculate_A_arrhenius_factor(A_direct, k_ref, T_ref, Ea_val):
    if A_direct is not None and A_direct > 0:
        return A_direct
    elif k_ref is not None and T_ref is not None and Ea_val is not None:
        if k_ref <= 0: raise ValueError("k_ref must be > 0")
        if T_ref <= 0: raise ValueError("T_ref must be > 0 K")
        return k_ref / np.exp(-Ea_val / (R_GAS_CONST * T_ref))
    else:
        raise ValueError("Provide A_arrhenius or (k_ref, T_ref, Ea_val).")

def rate_constant_calc(T_kelvin, A_factor, Ea_val):
    if T_kelvin <= 0: return 0.0
    return A_factor * np.exp(-Ea_val / (R_GAS_CONST * T_kelvin))

def cp_mass_calc(T_kelvin, alpha, beta, gamma):
    """ Calculates mass-based heat capacity Cp(T) = alpha + beta*T + gamma*T^2 """
    return alpha + beta * T_kelvin + gamma * T_kelvin**2

# --- PFR ODE System Definition ---
def pfr_odes(V_reactor, y_vars, # y_vars = [X, T]
             # Parameters
             A_factor, Ea_val, C_A0_val, F_A0_val,
             delta_H_rxn_val, U_val, a_v_val, T_a_kelvin_val,
             rho_val, alpha_cp, beta_cp, gamma_cp):
    """
    Defines the system of ODEs for the PFR:
    dX/dV = ...
    dT/dV = ...

    y_vars[0] = X (conversion of A)
    y_vars[1] = T (temperature in Kelvin)

    a_v_val: Heat transfer area per unit reactor volume (m^2/m^3 = 1/m)
             This was A_ht in your input, interpreted as a_v for PFR.
    """
    X = y_vars[0]
    T_kelvin = y_vars[1]

    # Prevent X from going beyond 1 due to numerical issues
    X = min(max(X, 0.0), 1.0)
    if T_kelvin <=0: # Non-physical temperature, stop reaction
        k_val = 0.0
    else:
        k_val = rate_constant_calc(T_kelvin, A_factor, Ea_val)

    # Rate of reaction for 1st order: -rA = k * CA
    # CA = C_A0 * (1 - X) for liquid phase, constant density
    neg_rA = k_val * C_A0_val * (1.0 - X)

    # --- Mole Balance: dX/dV ---
    if F_A0_val == 0: # Avoid division by zero if no flow
        dX_dV = 0.0
    else:
        dX_dV = neg_rA / F_A0_val

    # --- Energy Balance: dT/dV ---
    # Heat capacity of the mixture (mass-based)
    Cp_mixture_mass = cp_mass_calc(T_kelvin, alpha_cp, beta_cp, gamma_cp)

    # Inlet volumetric flow rate v0 = F_A0 / C_A0
    v0_val = F_A0_val / C_A0_val # m^3/s

    # Total mass flow rate = v0 * rho (kg/s)
    # (assuming rho is constant throughout the reactor for liquid phase)
    mass_flow_rate = v0_val * rho_val

    # Numerator for dT/dV: Heat_Reaction_per_Volume + Heat_Exchange_per_Volume
    # Heat_Reaction_per_Volume = (-delta_H_rxn) * neg_rA  (J/m^3.s)
    # Heat_Exchange_per_Volume = U * a_v * (T_a - T) (J/m^3.s)
    numerator_dT_dV = ((-delta_H_rxn_val) * neg_rA) + (U_val * a_v_val * (T_a_kelvin_val - T_kelvin))

    # Denominator for dT/dV: mass_flow_rate * Cp_mixture_mass (J/s.K)
    denominator_dT_dV = mass_flow_rate * Cp_mixture_mass

    if denominator_dT_dV == 0: # Avoid division by zero
        dT_dV = 0.0
    else:
        dT_dV = numerator_dT_dV / denominator_dT_dV
        
    # Safety: if X is already 1, no more reaction, dX_dV should be 0
    if X >= 1.0 - 1e-9: # Using a small tolerance
        dX_dV = 0.0
        # If no reaction, heat generation term is zero
        numerator_dT_dV_no_rxn = (U_val * a_v_val * (T_a_kelvin_val - T_kelvin))
        if denominator_dT_dV == 0:
            dT_dV = 0.0
        else:
            dT_dV = numerator_dT_dV_no_rxn / denominator_dT_dV


    return [dX_dV, dT_dV]

# --- Main PFR Solver ---
def solve_pfr_system(
        XA_target,
        A_arrhenius,     # Pre-exponential factor (can be None if k1_ref_temp used)
        Ea,              # Activation energy (J/mol)
        k1_ref_temp,     # Tuple: (k_ref_value, T_ref_kelvin) e.g., (0.04, 300.0)
        U,               # Overall heat transfer coefficient (J/s.m^2.K or W/m^2.K)
        a_v,             # Heat transfer area PER UNIT VOLUME (m^2/m^3 = 1/m). This was A_ht.
        T_a,             # Coolant temperature (K)
        delta_H_rxn,     # Heat of reaction (J/mol of A reacted, negative for exothermic)
        rho,             # Density of fluid (kg/m^3)
        Cp_params,       # Dict: {'alpha': <val>, 'beta': <val>, 'gamma': <val>} for Cp in J/kg.K
        F_A0,            # Inlet molar flow rate of A (mol/s)
        C_A0,            # Inlet concentration of A (mol/m^3 or mol/L - ensure consistency)
        T0,              # Inlet temperature (K)
        V_max_integration = 10.0 # Max reactor volume to integrate up to (m^3)
    ):
    """
    Solves for the PFR volume (V_reactor) required to achieve XA_target,
    and the outlet temperature (T_outlet).
    Integrates ODEs: dX/dV and dT/dV.
    """
    if not (0 < XA_target <= 1.0): # XA_target > 0 for PFR
        raise ValueError("XA_target must be between > 0 and 1 (inclusive). If XA_target=0, V=0.")
    if XA_target == 0:
        return 0.0, T0, [], [] # No volume needed for zero conversion

    # Determine Arrhenius factor
    k_ref_val, T_ref_kelvin_val = (None, None)
    if k1_ref_temp:
        k_ref_val, T_ref_kelvin_val = k1_ref_temp
    A_factor_actual = calculate_A_arrhenius_factor(A_arrhenius, k_ref_val, T_ref_kelvin_val, Ea)
    if A_factor_actual <= 0:
        raise ValueError("Arrhenius pre-exponential factor 'A' must be positive.")

    # Initial conditions for integration [X, T] at V=0
    y0 = [0.0, T0]

    # Parameters tuple for the ODE function
    ode_args = (A_factor_actual, Ea, C_A0, F_A0,
                delta_H_rxn, U, a_v, T_a,
                rho, Cp_params['alpha'], Cp_params['beta'], Cp_params['gamma'])

    # --- Event function to stop integration when XA_target is reached ---
    def stop_at_XA_target(V_reactor, y_vars, *args):
        return y_vars[0] - XA_target
    stop_at_XA_target.terminal = True  # Stop integration when event occurs
    stop_at_XA_target.direction = 1    # Event when value goes from negative to positive

    # Integrate the ODEs
    # V_span defines the range of independent variable (Volume) for integration
    # We integrate up to V_max_integration or until XA_target is met
    sol = solve_ivp(
        pfr_odes,
        t_span=[0, V_max_integration], # t is Volume here
        y0=y0,
        method='RK45', # Or 'LSODA' for stiff problems, 'BDF'
        args=ode_args,
        events=stop_at_XA_target,
        dense_output=True # Allows interpolation
    )

    if not sol.success:
        raise RuntimeError(f"PFR ODE integration failed: {sol.message}")

    if not sol.t_events or not sol.t_events[0]:
        # Target conversion not reached within V_max_integration
        print(f"Warning: Target conversion XA_target={XA_target} was not reached "
              f"within V_max_integration={V_max_integration} m^3.")
        print(f"  Max conversion achieved: {sol.y[0, -1]:.4f} at V = {sol.t[-1]:.4f} m^3, T_out = {sol.y[1, -1]:.2f} K")
        V_final = sol.t[-1]
        T_final = sol.y[1, -1]
        X_achieved = sol.y[0,-1]
        # You might want to return all profiles in this case
        return V_final, T_final, sol.t, sol.y[0], sol.y[1], X_achieved # Volume, Temp, V_profile, X_profile, T_profile

    # If event occurred, sol.t_events[0][0] is the volume where XA_target was met
    V_at_target = sol.t_events[0][0]
    # Get X and T at this specific volume using dense output
    state_at_target = sol.sol(V_at_target)
    X_final = state_at_target[0]
    T_final = state_at_target[1]

    # For plotting, you can get profiles up to V_at_target
    V_profile = np.linspace(0, V_at_target, 100)
    profiles = sol.sol(V_profile)
    X_profile = profiles[0]
    T_profile = profiles[1]

    return V_at_target, T_final, V_profile, X_profile, T_profile, X_final

# --- Example Usage ---
if __name__ == "__main__":
    # Define your parameters here
    # Ensure C_A0 units match F_A0 for v0 (e.g., F_A0 mol/s, C_A0 mol/m^3 => v0 m^3/s)
    # Cp_params for J/kg.K if rho is kg/m^3
    example_pfr_inputs = {
        'XA_target': 0.85,
        'A_arrhenius': None,       # Use k1_ref_temp instead
        'Ea': 75000,               # J/mol
        'k1_ref_temp': (0.05, 320.0), # k=0.05 1/s at T=320 K
        'U': 300,                  # J/s.m^2.K (W/m^2.K)
        'a_v': 15.0,                # Heat transfer area PER UNIT VOLUME (m^2/m^3 = 1/m)
                                   # Example: For a tube of D=0.267m, a_v = 4/D = 15 1/m
        'T_a': 293.15,             # K (20 C) - Coolant temperature
        'delta_H_rxn': -55000,     # J/mol (exothermic reaction)
        'rho': 980,                # kg/m^3 (e.g., organic liquid)
        'Cp_params': {'alpha': 2000, 'beta': 0.5, 'gamma': 0.0001}, # J/kg.K based
        'F_A0': 0.1,               # mol/s
        'C_A0': 1000,              # mol/m^3 (1.0 mol/L)
        'T0': 303.15,              # K (30 C) - Inlet temperature
        'V_max_integration': 5.0   # Max PFR volume to try integrating up to (m^3)
    }

    print("Solving PFR System with the following example inputs:")
    for key, value in example_pfr_inputs.items():
        print(f"  {key}: {value}")
    print("-" * 30)

    try:
        V_reactor, T_outlet, V_prof, X_prof, T_prof, X_final_check = solve_pfr_system(**example_pfr_inputs)

        print(f"\n--- PFR Results ---")
        print(f"Target Conversion (XA_target): {example_pfr_inputs['XA_target']:.4f}")
        print(f"Achieved Conversion at Outlet: {X_final_check:.4f}")
        print(f"Required PFR Volume (V): {V_reactor:.4f} m^3 ({V_reactor * 1000:.2f} L)")
        print(f"Outlet Temperature (T_outlet): {T_outlet:.2f} K ({T_outlet - 273.15:.2f} °C)")

        # Optional: Plotting the profiles
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Reactor Volume (m^3)')
        ax1.set_ylabel('Conversion (X)', color=color)
        ax1.plot(V_prof, X_prof, color=color, linestyle='-')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Temperature (K)', color=color)
        ax2.plot(V_prof, T_prof, color=color, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('PFR Conversion and Temperature Profiles')
        plt.show()

    except (ValueError, RuntimeError) as e:
        print(f"\nError: {e}")
    except Exception as e_gen:
        print(f"\nAn unexpected general error occurred: {e_gen}")

