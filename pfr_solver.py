from scipy.integrate import solve_ivp
import numpy as np

R_GAS_CONST = 8.314  # J/mol.K

def calculate_A_arrhenius_factor(A_direct, k_ref, T_ref, Ea_val):
    # Agar direct A diya hai aur sahi hai toh wahi le lo
    if A_direct is not None and A_direct > 0:
        return A_direct
    # Nahi toh reference se nikal lo
    elif k_ref is not None and T_ref is not None and Ea_val is not None:
        if k_ref <= 0: raise ValueError("k_ref must be > 0")
        if T_ref <= 0: raise ValueError("T_ref must be > 0 K")
        return k_ref / np.exp(-Ea_val / (R_GAS_CONST * T_ref))
    else:
        raise ValueError("A_arrhenius ya (k_ref, T_ref, Ea_val) mein se kuch toh do bhai.")

def rate_constant_calc(T_kelvin, A_factor, Ea_val):
    # Temperature zero ya negative hua toh reaction band
    if T_kelvin <= 0: return 0.0
    return A_factor * np.exp(-Ea_val / (R_GAS_CONST * T_kelvin))

def cp_mass_calc(T_kelvin, alpha, beta, gamma):
    # Cp ka formula, simple polynomial hai
    return alpha + beta * T_kelvin + gamma * T_kelvin**2

# --- PFR ODE System Definition ---
def pfr_odes(V_reactor, y_vars, # y_vars = [X, T]
             A_factor, Ea_val, C_A0_val, F_A0_val,
             delta_H_rxn_val, U_val, a_v_val, T_a_kelvin_val,
             rho_val, alpha_cp, beta_cp, gamma_cp):
    """
    Yahan PFR ke ODE define ho rahe hain:
    dX/dV aur dT/dV ka scene hai

    y_vars[0] = X (conversion of A)
    y_vars[1] = T (temperature in Kelvin)

    a_v_val: Heat transfer area per unit volume (m^2/m^3 = 1/m)
    """
    X = y_vars[0]
    T_kelvin = y_vars[1]

    # X ko 0-1 ke beech hi rakhna hai, warna gadbad ho jayegi
    X = min(max(X, 0.0), 1.0)
    if T_kelvin <=0: # Temperature zero ya negative hua toh reaction ruk gayi
        k_val = 0.0
    else:
        k_val = rate_constant_calc(T_kelvin, A_factor, Ea_val)

    # 1st order reaction ka rate: -rA = k * CA
    # CA = C_A0 * (1 - X), liquid phase, density constant maan lo
    neg_rA = k_val * C_A0_val * (1.0 - X)

    # --- Mole Balance: dX/dV ---
    if F_A0_val == 0: # Flow zero hai toh division by zero se bachna hai
        dX_dV = 0.0
    else:
        dX_dV = neg_rA / F_A0_val

    # --- Energy Balance: dT/dV ---
    # Mixture ka Cp nikal lo
    Cp_mixture_mass = cp_mass_calc(T_kelvin, alpha_cp, beta_cp, gamma_cp)

    # Inlet volumetric flow rate v0 = F_A0 / C_A0
    v0_val = F_A0_val / C_A0_val # m^3/s

    # Total mass flow rate = v0 * rho (kg/s), density constant maan rahe hain
    mass_flow_rate = v0_val * rho_val

    # Numerator: Reaction ki heat + heat exchange
    numerator_dT_dV = ((-delta_H_rxn_val) * neg_rA) + (U_val * a_v_val * (T_a_kelvin_val - T_kelvin))

    # Denominator: mass_flow_rate * Cp_mixture_mass
    denominator_dT_dV = mass_flow_rate * Cp_mixture_mass
    if denominator_dT_dV == 0: # Division by zero se bachna hai
        dT_dV = 0.0
    else:
        dT_dV = numerator_dT_dV / denominator_dT_dV

    # Agar X already 1 ke paas hai toh reaction khatam, dX_dV zero kar do
    if X >= 1.0 - 1e-9:
        dX_dV = 0.0
        # Reaction band, sirf heat exchange bacha
        numerator_dT_dV_no_rxn = (U_val * a_v_val * (T_a_kelvin_val - T_kelvin))
        if denominator_dT_dV == 0:
            dT_dV = 0.0
        else:
            dT_dV = numerator_dT_dV_no_rxn / denominator_dT_dV

    return [dX_dV, dT_dV]

# --- Main PFR Solver ---
def solve_pfr_system(
        XA_target,
        A_arrhenius,
        Ea,
        k1_ref_temp,
        U,
        a_v,
        T_a,
        delta_H_rxn,
        rho,
        Cp_params,
        F_A0,
        C_A0,
        T0,
        V_max_integration = 10.0
    ):
    """
    Yahan PFR ka volume nikal rahe hain jo chahiye target conversion ke liye,
    aur outlet temperature bhi mil jayega.
    ODEs: dX/dV aur dT/dV solve ho rahe hain.
    """
    if not (0 < XA_target <= 1.0): # XA_target 0 se 1 ke beech hi hona chahiye
        raise ValueError("XA_target 0 se bada aur 1 se chhota ya barabar hona chahiye. Agar XA_target=0 hai toh V=0.")
    if XA_target == 0:
        return 0.0, T0, [], []

    # Arrhenius factor decide karo
    k_ref_val, T_ref_kelvin_val = (None, None)
    if k1_ref_temp:
        k_ref_val, T_ref_kelvin_val = k1_ref_temp
    A_factor_actual = calculate_A_arrhenius_factor(A_arrhenius, k_ref_val, T_ref_kelvin_val, Ea)
    if A_factor_actual <= 0:
        raise ValueError("Arrhenius pre-exponential factor 'A' positive hona chahiye.")

    # Initial conditions [X, T] at V=0
    y0 = [0.0, T0]

    # ODE function ke liye parameters
    ode_args = (A_factor_actual, Ea, C_A0, F_A0,
                delta_H_rxn, U, a_v, T_a,
                rho, Cp_params['alpha'], Cp_params['beta'], Cp_params['gamma'])

    # --- Event function: XA_target milte hi integration rok do ---
    def stop_at_XA_target(V_reactor, y_vars, *args):
        return y_vars[0] - XA_target
    stop_at_XA_target.terminal = True
    stop_at_XA_target.direction = 1

    # ODEs integrate karo
    sol = solve_ivp(
        pfr_odes,
        t_span=[0, V_max_integration], # Volume ka range
        y0=y0,
        method='RK45',
        args=ode_args,
        events=stop_at_XA_target,
        dense_output=True
    )

    if not sol.success:
        raise RuntimeError(f"PFR ODE integration fail ho gaya: {sol.message}")

    if not sol.t_events or not sol.t_events[0]:
        # Target conversion nahi mila max volume tak
        print(f"Warning: Target conversion XA_target={XA_target} nahi mila V_max_integration={V_max_integration} m^3 tak.")
        print(f"  Max conversion mila: {sol.y[0, -1]:.4f} at V = {sol.t[-1]:.4f} m^3, T_out = {sol.y[1, -1]:.2f} K")
        V_final = sol.t[-1]
        T_final = sol.y[1, -1]
        X_achieved = sol.y[0,-1]
        return V_final, T_final, sol.t, sol.y[0], sol.y[1], X_achieved

    # Event hua toh yahan XA_target mil gaya
    V_at_target = sol.t_events[0][0]
    state_at_target = sol.sol(V_at_target)
    X_final = state_at_target[0]
    T_final = state_at_target[1]

    # Plotting ke liye profiles nikal lo
    V_profile = np.linspace(0, V_at_target, 100)
    profiles = sol.sol(V_profile)
    X_profile = profiles[0]
    T_profile = profiles[1]
    return V_at_target, T_final, V_profile, X_profile, T_profile, X_final

# --- Example Usage ---
if __name__ == "__main__":
    # Yahan apne parameters daal do
    example_pfr_inputs = {
        'XA_target': 0.85,
        'A_arrhenius': None,
        'Ea': 75000,
        'k1_ref_temp': (0.05, 320.0),
        'U': 300,
        'a_v': 15.0,
        'T_a': 293.15,
        'delta_H_rxn': -55000,
        'rho': 980,
        'Cp_params': {'alpha': 2000, 'beta': 0.5, 'gamma': 0.0001},
        'F_A0': 0.1,
        'C_A0': 1,
        'T0': 303.15,
        'V_max_integration': 5.0
    }

    print("PFR System solve kar rahe hain, yeh inputs diye hain:")
    for key, value in example_pfr_inputs.items():
        print(f"  {key}: {value}")
    print("-" * 30)

    try:
        V_reactor, T_outlet, V_prof, X_prof, T_prof, X_final_check = solve_pfr_system(**example_pfr_inputs)

        print(f"\n--- PFR Results ---")
        print(f"Target Conversion (XA_target): {example_pfr_inputs['XA_target']:.4f}")
        print(f"Outlet pe Conversion mila: {X_final_check:.4f}")
        print(f"Required PFR Volume (V): {V_reactor:.4f} m^3 ({V_reactor * 1000:.2f} L)")
        print(f"Outlet Temperature (T_outlet): {T_outlet:.2f} K ({T_outlet - 273.15:.2f} °C)")

    except (ValueError, RuntimeError) as e:
        print(f"\nError: {e}")
    except Exception as e_gen:
        print(f"\n Unexpected Error  {e_gen}")
