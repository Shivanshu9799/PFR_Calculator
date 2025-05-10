import numpy as np
from scipy.integrate import solve_ivp

# Constants
R = 8.314  # J/(mol*K)

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

def get_user_input():
    """Get user input for all parameters including reactor rates"""
    print("Enter reaction parameters:")
    params = {
        'delHnotrx': float(input("ΔH°rxn (J/mol) [default: -80000.0]: ") or -80000.0),
        'To': float(input("Inlet Temperature T₀ (K) [default: 300.0]: ") or 300.0),
        'Ea': float(input("Activation Energy Ea (J/mol) [default: 60000.0]: ") or 60000.0),
        'Kc2': float(input("Kc₂ at T₂ [default: 0.5]: ") or 0.5),
        'T2': float(input("T₂ (K) [default: 400.0]: ") or 400.0),
        'k1': float(input("k₁ at T₁ [default: 0.1]: ") or 0.1),
        'T1': float(input("T₁ (K) [default: 300.0]: ") or 300.0),
        'Cao': float(input("Initial Concentration Cₐ₀ (mol/m³) [default: 1.0]: ") or 1.0),
        'Cpa': float(input("Cp(A) (J/mol·K) [default: 100.0]: ") or 100.0),
        'Cpb': float(input("Cp(B) (J/mol·K) [default: 100.0]: ") or 100.0),
        'Cp_inert': float(input("Cp(Inert) (J/mol·K) [default: 50.0]: ") or 50.0),
        'theta_inert': float(input("θ Inert [default: 0.5]: ") or 0.5),
        'Fao': float(input("Molar Flow Rate Fₐ₀ (mol/s) [default: 10.0]: ") or 10.0),
        'X': float(input("Target Conversion X [default: 0.8]: ") or 0.8)
    }
    
    # Get reactor rates
    print("\nEnter reactor cost rates:")
    pfr_rate = float(input("PFR cost rate ($/m³) [default: 1000.0]: ") or 1000.0)
    cstr_rate = float(input("CSTR cost rate ($/m³) [default: 800.0]: ") or 800.0)
    
    return params, pfr_rate, cstr_rate

# Main execution
if __name__ == "__main__":
    # Get user input
    params, pfr_rate, cstr_rate = get_user_input()
    
    # Calculate reactor volumes
    V_pfr, T_out, V_cstr, _ = calculate_reactors(**params)
    
    # Calculate costs
    pfr_cost = V_pfr * pfr_rate if np.isfinite(V_pfr) else np.inf
    cstr_cost = V_cstr * cstr_rate if np.isfinite(V_cstr) else np.inf
    
    # Print results
    print("\nResults:")
    print(f"PFR Volume: {V_pfr:.2f} m³")
    print(f"PFR Cost: ${pfr_cost:.2f}" if np.isfinite(pfr_cost) else "PFR Cost: Infeasible")
    print(f"CSTR Volume: {V_cstr:.2f} m³")
    print(f"CSTR Cost: ${cstr_cost:.2f}" if np.isfinite(cstr_cost) else "CSTR Cost: Infeasible")
    print(f"Outlet Temperature: {T_out:.2f} K")
    
    # Comparison
    if np.isfinite(pfr_cost) and np.isfinite(cstr_cost):
        if pfr_cost < cstr_cost:
            print("\nPFR is more economical for this conversion")
        else:
            print("\nCSTR is more economical for this conversion")
    elif np.isfinite(pfr_cost):
        print("\nOnly PFR can achieve this conversion")
    elif np.isfinite(cstr_cost):
        print("\nOnly CSTR can achieve this conversion")
    else:
        print("\nTarget conversion not achievable with either reactor type")