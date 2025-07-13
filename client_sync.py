from client import Client

class SyncClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_gradient_norm = 1.0  # For EST-P estimation
        self.energy_consumed = 0.0
        
    def compute_gradient(self):
        # Existing gradient computation
        super().compute_gradient()
        # Update EST-P estimator
        self.last_gradient_norm = self.gradient_norm
        
    def report_energy(self, sigma_t):
        """Calculate actual energy consumption with proper scaling"""
        if abs(self.h_t_k) < 1e-8:
            return 0.0
            
        # Computation energy (fixed)
        E_comp = self.mu_k * (self.fk ** 2) * self.C * self.Ak
        
        # Communication energy - SCALE BY GRADIENT DIMENSION
        gradient_dim = self.last_gradient.numel()
        E_comm = (sigma_t ** 2) * (self.gradient_norm ** 2) / (abs(self.h_t_k) ** 2) / gradient_dim
        
        total_energy = E_comp + E_comm
        self.energy_consumed += total_energy
        return total_energy