# server_sync.py
import numpy as np
import logging
import torch
import time
from collections import defaultdict
import math

class Server:
    def __init__(self, global_model, clients, V=50.0, sigma_n=0.5, 
                 tau_cm=0.1, T_total_rounds=300, E_max=1.0, 
                 gamma0=10.0, G=1.0, l=1.0, device='cpu'):
        self.global_model = global_model
        self.clients = clients
        self.device = device
        self.V = V
        self.sigma_n = sigma_n
        self.tau_cm = tau_cm
        self.T_total_rounds = T_total_rounds
        self.gamma0 = gamma0
        self.G = G
        self.l = l
        self.d = self._get_model_dimension()
        self.s = self._get_gradient_dimension()  # Gradient dimension
        self.logger = logging.getLogger("Server")
        
        # Virtual queues for energy
        self.Q_e = {client.client_id: 0.0 for client in clients}
        self.E_max = E_max if isinstance(E_max, dict) else \
                    {c.client_id: E_max for c in clients}
        
        # Gradient norm history
        self.grad_norm_history = defaultdict(lambda: 0.1)  # Start with small value

        # Critical parameters (no scaling)
        self.min_grad_norm = 1e-8
        self.max_sigma_t = 10.0  # Reduced from 1000.0
        
        # Tracking
        self.total_energy_per_round = []
        self.cumulative_energy = {c.client_id: 0.0 for c in clients}
        self.selected_history = []
        self.queue_history = []
        self.round_idx = 0

    def _get_model_dimension(self):
        return sum(p.numel() for p in self.global_model.parameters())
    
    def _get_gradient_dimension(self):
        """Get the dimension of the gradient vector (s)"""
        dummy_input = torch.randn(1, 1, 28, 28, device=self.device)
        output = self.global_model(dummy_input)
        return output.numel()
    
    def broadcast_model(self, selected_clients, sigma_t):
        global_state = self.global_model.state_dict()
        for client in selected_clients:
            client.load_model(global_state)
            client.sigma_t = sigma_t

    def select_devices(self):
        """Device selection with Lyapunov optimization"""
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
        
        # Compute costs with true energy estimates
        client_costs = []
        for client in self.clients:
            E_est = client.compute_energy_estimate(self.sigma_t)
            cost = self.Q_e[client.client_id] * E_est
            client_costs.append((client, E_est, cost))
            
            self.logger.debug(f"Client {client.client_id} | "
                             f"Q_e = {self.Q_e[client.client_id]:.2f} | "
                             f"E_est = {E_est:.4e} | cost = {cost:.4e}")

        # Sort clients by cost (ascending)
        sorted_client_costs = sorted(client_costs, key=lambda x: x[2])
        sorted_clients = [item[0] for item in sorted_client_costs]
        sorted_costs = [item[2] for item in sorted_client_costs]
        
        # Cosine annealing learning rate
        eta_t = 0.1 * (1 + math.cos(math.pi * self.round_idx / self.T_total_rounds)) / 2
        eta_t = max(eta_t, 0.01)  # Ensure minimum learning rate
        
        # Precompute vₜ(k) for all k
        min_v = float('inf')
        best_k = 0
        
        self.logger.debug("k analysis:")
        for k in range(1, len(sorted_clients)+1):
            # Compute Ut according to Eq. 19
            G_term = self.G**2 / (self.clients[0].Ak * k)
            noise_term = (self.sigma_n**2 * self.s) / (self.sigma_t**2 * k**2)
            Ut = (self.l * eta_t**2 / 2) * (G_term + noise_term)
            
            # Sum costs for the first k clients
            sum_cost = sum(cost for _, _, cost in sorted_client_costs[:k])
            v_tk = self.V * Ut + sum_cost
            
            self.logger.debug(
                f"k={k}: Ut={Ut:.4e} | Σcost={sum_cost:.4e} | v_tk={v_tk:.4e}"
            )
            
            if v_tk < min_v:
                min_v = v_tk
                best_k = k
        
        # Ensure at least 1 client is selected
        best_k = max(best_k, 1)
        selected = sorted_clients[:best_k]
        self.selected_history.append([c.client_id for c in selected])
        
        self.logger.info(f"Selected {best_k} clients | min_v={min_v:.4e}")
        return selected

    def aggregate_gradients(self, selected_clients):
        """OTA aggregation with true gradient values"""
        if not selected_clients:
            self.logger.warning("No clients selected - returning zero update")
            return torch.zeros(self.d, device=self.device)
        
        aggregated = torch.zeros(self.d, device=self.device)
        comp_times = []
        
        for client in selected_clients:
            start_time = time.time()
            gradient = client.compute_gradient(self.round_idx)
            comp_times.append(time.time() - start_time)
            
            # Use true gradient without scaling
            aggregated += gradient
        
        # Apply OTA effects: yₜ = σₜΣg̃ₙ + zₜ
        noise = torch.randn(self.d, device=self.device) * self.sigma_n
        received_signal = self.sigma_t * aggregated + noise
        
        # Normalize: ĝ = yₜ/(σₜ|Bₜ|)
        k = len(selected_clients)
        if k == 0 or self.sigma_t < 1e-8:
            normalized_update = torch.zeros_like(aggregated)
        else:
            normalized_update = received_signal / (self.sigma_t * k)
        
        # Check for invalid updates
        if not torch.isfinite(normalized_update).all():
            self.logger.error("Non-finite update detected! Using zero update")
            normalized_update = torch.zeros_like(normalized_update)
        
        update_norm = torch.norm(normalized_update).item()
        self.logger.info(f"Aggregated {k} clients | Update norm: {update_norm:.4f}")
        return normalized_update

    def update_model(self, aggregated_update):
        """Update global model with adaptive learning rate"""
        # Effective SNR-based learning rate
        k = len(self.selected_history[-1])
        if k == 0:
            effective_snr = 0
        else:
            # SNR = (σₜ²k) / (σ₀²s)
            effective_snr = (self.sigma_t**2 * k) / (self.sigma_n**2 * self.s)
        
        base_lr = 0.1 * (1 + math.cos(math.pi * self.round_idx / self.T_total_rounds)) / 2
        base_lr = max(base_lr, 0.01)  # Minimum learning rate
        
        # SNR-based adjustment with clipping
        snr_factor = min(1.0, effective_snr / self.gamma0)
        snr_factor = max(snr_factor, 0.1)  # Minimum factor
        lr = base_lr * snr_factor
        
        with torch.no_grad():
            params = torch.nn.utils.parameters_to_vector(self.global_model.parameters())
            params -= lr * aggregated_update
            torch.nn.utils.vector_to_parameters(params, self.global_model.parameters())
        
        self.logger.info(f"Model updated | LR: {lr:.4f} (base: {base_lr:.4f}, SNR: {effective_snr:.2f}/{self.gamma0})")

    def update_queues(self, selected_clients):
        """Update virtual queues with true energy values"""
        round_energy = 0
        
        for client in selected_clients:
            # Get actual energy consumption
            actual_energy = client.compute_actual_energy(client.sigma_t)
            
            # Update cumulative energy
            self.cumulative_energy[client.client_id] += actual_energy
            round_energy += actual_energy
            
            # Update queue according to Eq. 25
            per_round_budget = self.E_max[client.client_id] / self.T_total_rounds
            energy_increment = actual_energy - per_round_budget
            prev_q = self.Q_e[client.client_id]
            self.Q_e[client.client_id] = max(0, prev_q + energy_increment)
            
            # Log update
            self.logger.info(f"Client {client.client_id} | "
                             f"Energy: {actual_energy:.4e} J | "
                             f"Budget: {per_round_budget:.4e} J | "
                             f"Q_e: {prev_q:.2f} → {self.Q_e[client.client_id]:.2f}")
        
        self.total_energy_per_round.append(round_energy)
        self.queue_history.append(dict(self.Q_e))
        
        return round_energy