# server_sync.py
import numpy as np
import logging
import torch
import time
from collections import defaultdict
import math

class Server:
    def __init__(self, global_model, clients, V=5000.0, sigma_n=0.01, 
                 tau_cm=0.1, T_total_rounds=300, E_max=1.0, 
                 gamma0=0.01, G=0.1, l=0.1, device='cpu'):
        self.global_model = global_model
        self.clients = clients
        self.device = device
        self.V = V  # Increased Lyapunov parameter
        self.sigma_n = sigma_n  # Reduced noise
        self.tau_cm = tau_cm
        self.T_total_rounds = T_total_rounds
        self.gamma0 = gamma0  # Reduced target SNR
        self.G = G  # Reduced gradient variance
        self.l = l  # Reduced smoothness
        self.d = self._get_model_dimension()
        self.s = self.d  # Gradient dimension = model dimension
        self.logger = logging.getLogger("Server")
        
        # Virtual queues for energy
        self.Q_e = {client.client_id: 0.0 for client in clients}
        self.E_max = E_max if isinstance(E_max, dict) else \
                    {c.client_id: E_max for c in clients}
        
        # Gradient norm history
        self.grad_norm_history = defaultdict(lambda: 0.1)  # Start with small value

        # Critical parameters
        self.min_grad_norm = 1e-8
        self.max_sigma_t = 100.0  # Reduced cap
        
        # Tracking
        self.total_energy_per_round = []
        self.cumulative_energy = {c.client_id: 0.0 for c in clients}
        self.selected_history = []
        self.queue_history = []
        self.round_idx = 0

    def _get_model_dimension(self):
        return sum(p.numel() for p in self.global_model.parameters())
    
    def broadcast_model(self, selected_clients, sigma_t):
        global_state = self.global_model.state_dict()
        for client in selected_clients:
            client.load_model(global_state)
            client.sigma_t = sigma_t

    def select_devices(self):
        """Improved device selection with balanced Lyapunov terms"""
        self.round_idx += 1
        
        # Set channels for all clients
        for client in self.clients:
            channel_gain = client.set_channel_gain(self.round_idx)
            self.logger.debug(f"Client {client.client_id} | h = {channel_gain:.4f}")
            
            # Update gradient norm estimate
            self.grad_norm_history[client.client_id] = max(
                client.estimate_gradient_norm(), 
                self.min_grad_norm
            )
        
        # Compute σₜ according to Eq. 29: σₜ = √(γ₀σ₀²s) / minₙ‖g̃ₙ‖
        min_grad_norm = min(self.grad_norm_history.values())
        raw_sigma_t = np.sqrt(self.gamma0 * self.sigma_n**2 * self.s) / min_grad_norm
        self.sigma_t = min(max(raw_sigma_t, 1e-8), self.max_sigma_t)
        
        self.logger.info(f"σₜ = {self.sigma_t:.4e} | min ‖g̃‖ = {min_grad_norm:.4e} | s = {self.s}")
        
        # Compute costs with normalized energy estimates
        client_costs = []
        for client in self.clients:
            E_est = client.compute_energy_estimate(self.sigma_t)
            
            # Normalize energy cost by budget
            norm_factor = self.E_max[client.client_id] / self.T_total_rounds
            norm_energy = E_est / max(norm_factor, 1e-8)
            
            cost = self.Q_e[client.client_id] * norm_energy
            client_costs.append((client, E_est, cost))
            
            self.logger.debug(f"Client {client.client_id} | "
                             f"Q_e = {self.Q_e[client.client_id]:.2f} | "
                             f"E_est = {E_est:.4e} | cost = {cost:.4e}")

        # Sort clients by cost (ascending)
        sorted_client_costs = sorted(client_costs, key=lambda x: x[2])
        sorted_clients = [item[0] for item in sorted_client_costs]
        sorted_costs = [item[2] for item in sorted_client_costs]
        
        # Dynamic learning rate with minimum bound
        base_lr = 0.1
        eta_t = base_lr * (1 + math.cos(math.pi * self.round_idx / self.T_total_rounds)) / 2
        eta_t = max(eta_t, 0.01)  # Prevent too small LR
        
        # Precompute vₜ(k) for all k with balanced terms
        min_v = float('inf')
        best_k = 0
        best_Ut = 0
        
        self.logger.debug("k analysis:")
        for k in range(1, len(sorted_clients)+1):
            # Compute Ut according to Eq. 19 with reduced impact
            G_term = self.G**2 / (self.clients[0].Ak * k)
            noise_term = (self.sigma_n**2 * self.s) / (self.sigma_t**2 * k**2)
            Ut = (self.l * eta_t**2 / 2) * (G_term + noise_term)
            
            sum_cost = sum(sorted_costs[:k])
            
            # Balance the terms using adaptive weighting
            weight = 1.0 + self.round_idx / self.T_total_rounds  # Favor convergence later
            v_tk = self.V * Ut * weight + sum_cost / math.sqrt(k)
            
            self.logger.debug(
                f"k={k}: Ut={Ut:.4e} | Σcost={sum_cost:.4e} | v_tk={v_tk:.4e}"
            )
            
            if v_tk < min_v:
                min_v = v_tk
                best_k = k
                best_Ut = Ut
        
        # Adaptive k-min constraint based on training progress
        min_k = 1 if self.round_idx < 0.2 * self.T_total_rounds else max(2, int(0.1 * len(self.clients)))
        best_k = max(best_k, min_k)
        
        selected = sorted_clients[:best_k]
        self.selected_history.append([c.client_id for c in selected])
        
        self.logger.info(f"Selected {best_k} clients | min_v={min_v:.4e} | Ut={best_Ut:.4e}")
        return selected

    def aggregate_gradients(self, selected_clients):
        """OTA aggregation with noise scaling"""
        if not selected_clients:
            return torch.zeros(self.d, device=self.device)
        
        aggregated = torch.zeros(self.d, device=self.device)
        
        for client in selected_clients:
            gradient = client.compute_gradient(self.round_idx)
            aggregated += gradient
        
        # Apply scaled noise
        k = len(selected_clients)
        noise_scale = min(1.0, 10.0 / math.sqrt(k))  # Reduce noise for larger k
        noise = torch.randn(self.d, device=self.device) * self.sigma_n * noise_scale
        received_signal = self.sigma_t * aggregated + noise
        
        # Normalize with stability
        if k > 0 and self.sigma_t > 1e-8:
            normalized_update = received_signal / (self.sigma_t * k)
        else:
            normalized_update = torch.zeros_like(aggregated)
        
        update_norm = torch.norm(normalized_update).item()
        self.logger.info(f"Aggregated {k} clients | Update norm: {update_norm:.4f}")
        return normalized_update

    def update_model(self, aggregated_update):
        """Update with stabilized learning rate"""
        k = len(self.selected_history[-1])
        base_lr = 0.1
        lr = base_lr
        
        with torch.no_grad():
            params = torch.nn.utils.parameters_to_vector(self.global_model.parameters())
            params -= lr * aggregated_update
            torch.nn.utils.vector_to_parameters(params, self.global_model.parameters())
        
        self.logger.info(f"Model updated | LR: {lr:.4f}")

    def update_queues(self, selected_clients):
        """Update queues with normalized energy"""
        round_energy = 0
        
        for client in selected_clients:
            actual_energy = client.compute_actual_energy(client.sigma_t)
            self.cumulative_energy[client.client_id] += actual_energy
            round_energy += actual_energy
            
            per_round_budget = self.E_max[client.client_id] / self.T_total_rounds
            energy_increment = actual_energy - per_round_budget
            prev_q = self.Q_e[client.client_id]
            
            # Soft queue update to prevent explosion
            self.Q_e[client.client_id] = max(0, prev_q + 0.1 * energy_increment)
            
            self.logger.info(f"Client {client.client_id} | "
                             f"Energy: {actual_energy:.4e} J | "
                             f"Budget: {per_round_budget:.4e} J | "
                             f"Q_e: {prev_q:.2f} → {self.Q_e[client.client_id]:.2f}")
        
        self.total_energy_per_round.append(round_energy)
        self.queue_history.append(dict(self.Q_e))
        
        return round_energy