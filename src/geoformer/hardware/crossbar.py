"""
Physical Memristor Crossbar Hardware Simulator.
Models 1T1R non-volatile memristors with discrete 4-bit states, cycle-to-cycle variability,
parasitic interconnect line resistance, and Johnson-Nyquist thermal read noise.
"""

import numpy as np

class PhysicalMemristorCrossbar:
    """
    Simulates a 2D passive memristor crossbar array with 1T1R cells
    implementing the plastic conductance lattice W in physical analog domain.
    """
    def __init__(
        self,
        n_inputs=64,
        n_outputs=64,
        g_min=10e-6,      # 10 uS (High-Resistance State, HRS)
        g_max=150e-6,     # 150 uS (Low-Resistance State, LRS)
        b_levels=16,      # 4-bit quantization (16 discrete states)
        v_read=0.2,       # 200 mV read pulse voltage
        t_pulse=10e-9,    # 10 ns pulse width
        r_wire=3.2,       # 3.2 Ohms metal wire resistance per cell
        drift_exp=0.045,  # Conductance drift exponent
    ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.g_min = g_min
        self.g_max = g_max
        self.b_levels = b_levels
        self.v_read = v_read
        self.t_pulse = t_pulse
        self.r_wire = r_wire
        self.drift_exp = drift_exp
        
        self.delta_g = (g_max - g_min) / (b_levels - 1)
        self.G_ideal = np.full((n_outputs, n_inputs), g_min)
        self.G_actual = np.copy(self.G_ideal)

    def program_conductance_matrix(self, W_norm, noise_std=0.05):
        """
        Programs normalized conductance W in [0, 1] into physical memristors.
        Applies 4-bit discretization and cycle-to-cycle programming variation.
        """
        w_levels = np.clip(np.round(W_norm * (self.b_levels - 1)), 0, self.b_levels - 1)
        self.G_ideal = self.g_min + w_levels * self.delta_g
        
        variation = np.random.normal(loc=0.0, scale=noise_std, size=self.G_ideal.shape)
        self.G_actual = np.clip(self.G_ideal * (1.0 + variation), self.g_min * 0.5, self.g_max * 1.5)

    def apply_drift(self, t_seconds=1000.0, t_0=1.0):
        """Simulates conductance drift: G(t) = G_0 * (t / t_0)^(-nu)."""
        drift_factor = (max(1.0, t_seconds) / t_0) ** (-self.drift_exp)
        self.G_actual = self.G_actual * drift_factor

    def compute_analog_vmm(self, v_in, apply_parasitics=True):
        """
        Computes analog Vector-Matrix Multiplication I_out = G_actual @ v_in
        via Ohm's Law and Kirchhoff's Current Law.
        """
        i_ideal = np.dot(self.G_actual, v_in)
        
        if apply_parasitics:
            ir_attenuation = 1.0 / (1.0 + 0.5 * self.r_wire * self.n_inputs * np.mean(self.G_actual))
            i_actual = i_ideal * ir_attenuation
            read_noise = np.random.normal(loc=0.0, scale=0.005 * np.max(i_actual + 1e-12), size=i_actual.shape)
            i_actual = np.maximum(0.0, i_actual + read_noise)
            return i_actual
        return i_ideal

    def compute_dynamic_energy_per_inference(self):
        """Calculates dynamic energy for a 2-stage MVM relaxation in nanojoules (nJ)."""
        total_conductance = np.sum(self.G_actual)
        e_stage1 = (self.v_read ** 2) * total_conductance * self.t_pulse
        e_stage2 = (self.v_read ** 2) * total_conductance * self.t_pulse
        total_joules = (e_stage1 + e_stage2) * 1.6 # 60% peripheral overhead
        return total_joules * 1e9 # nJ
