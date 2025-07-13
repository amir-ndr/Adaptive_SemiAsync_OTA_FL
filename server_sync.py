import torch
from server import Server
import numpy as np
import logging

class SyncServer(Server):
    def __init__(self, global_model, clients, gamma0=10, G=1.0, l=0.1, 
                 V=1.0, T_total_rounds=50, **kwargs):
        super().__init__(global_model, clients, **kwargs)
        # Lyapunov parameters
        self.gamma0 = gamma0  # Target SNR
        self.G = G            # Gradient bound
        self.l = l            # Smoothness constant
        self.V = V            # Lyapunov trade-off
        self.T_total_rounds = T_total_rounds
        
        # Virtual energy queues
        self.Q_e = {client.client_id: 0.0 for client in clients}
        
    def schedule_devices(self, round_idx):
        """Fixed device scheduling with power constraints"""
        channel_gains = {}
        for client in self.clients:
            gain = client.set_channel_gain()
            channel_gains[client.client_id] = gain
        
        # Calculate sigma_t with clamping
        grad_norms = [c.last_gradient_norm for c in self.clients if c.last_gradient_norm > 0]
        min_grad_norm = min(grad_norms) if grad_norms else 1.0
        min_grad_norm = max(min_grad_norm, 1e-3)  # Prevent division by zero
        
        sigma_t = np.sqrt(
            (self.gamma0 * self.sigma_n**2 * np.sqrt(self.d)) / 
            min_grad_norm
        )
        
        # Estimate energy with power constraints
        E_est = {}
        for client in self.clients:
            grad_norm_est = client.last_gradient_norm
            E_comp = client.mu_k * (client.fk ** 2) * client.C * client.Ak
            
            h_abs = abs(channel_gains[client.client_id])
            if h_abs < 1e-8:
                E_comm = float('inf')
            else:
                required_power = sigma_t / h_abs
                if required_power > client.P_max:
                    E_comm = float('inf')
                else:
                    E_comm = (sigma_t ** 2) * (grad_norm_est ** 2) / (h_abs ** 2)
            
            E_est[client.client_id] = E_comp + E_comm
        
        # Calculate C_n = Q_e[n] * E_est[n]
        C = {cid: self.Q_e[cid] * E_est[cid] for cid in self.Q_e}
        sorted_clients = sorted(self.clients, key=lambda c: C[c.client_id])
        
        # Find optimal k with valid clients only
        best_cost = float('inf')
        best_k = 0
        best_set = []
        
        eta_t = 0.1
        valid_clients = [c for c in sorted_clients if not np.isinf(C[c.client_id])]
        
        for k in range(1, len(valid_clients) + 1):
            candidate_set = valid_clients[:k]
            
            term1 = self.G**2 / (candidate_set[0].Ak * k)
            term2 = (self.sigma_n**2 * self.d) / (sigma_t**2 * k**2)
            Ut = (self.l * eta_t**2 / 2) * (term1 + term2)
            
            sum_C = sum(C[c.client_id] for c in candidate_set)
            cost = self.V * Ut + sum_C
            
            if cost < best_cost:
                best_cost = cost
                best_k = k
                best_set = candidate_set
        
        logging.info(f"Selected {best_k} devices | σ_t: {sigma_t:.4f}")
        return best_set, sigma_t
    
    # In server.py
    def aggregate(self, selected, sigma_t):  # Add sigma_t parameter
        """OTA aggregation with noise injection"""
        if not selected:
            return torch.zeros(self.d, device=self.device)
            
        # Sum of gradients with power scaling
        aggregated = torch.zeros(self.d, device=self.device)
        for client in selected:
            # Apply channel inversion scaling
            scaled_grad = client.last_gradient * sigma_t / abs(client.h_t_k)
            aggregated += scaled_grad
        
        # Add Gaussian noise
        noise = torch.randn(self.d, device=self.device) * self.sigma_n
        aggregated += noise
        
        # Normalize by number of devices
        return aggregated / (sigma_t * len(selected))
    
    def update_queues(self, selected, actual_energy):
        """Update virtual queues based on actual energy consumption"""
        for client in selected:
            cid = client.client_id
            # Per-round energy budget
            per_round_budget = self.E_max[cid] / self.T_total_rounds
            # Update virtual queue
            self.Q_e[cid] = max(0, self.Q_e[cid] + actual_energy[cid] - per_round_budget)