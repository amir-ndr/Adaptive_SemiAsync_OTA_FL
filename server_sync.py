import numpy as np
import torch
import logging
from collections import defaultdict

class SyncServer:
    def __init__(self, global_model, clients, total_rounds,
                 batch_size, gamma0, sigma_n, G2, l_smooth,
                 energy_budgets, device='cpu'):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.total_rounds = total_rounds
        self.batch_size = batch_size
        self.gamma0 = gamma0  # Target SNR
        self.sigma_n = sigma_n  # Noise std dev
        self.G2 = G2  # Gradient variance bound
        self.l_smooth = l_smooth  # Smoothness constant
        self.device = device
        self.d = self._model_dimension()
        
        # Energy management
        self.energy_budgets = energy_budgets
        self.virtual_queues = {c.client_id: 0.0 for c in clients}
        self.energy_consumed = {c.client_id: 0.0 for c in clients}
        self.actual_snr_history = []
        
        # History tracking
        self.selection_history = []
        self.queue_history = []
        
        logging.info(f"SyncServer initialized | "
                     f"Model dim: {self.d} | Target SNR: {gamma0} | "
                     f"Noise std: {sigma_n} | Total rounds: {total_rounds}")
    
    def _model_dimension(self):
        return sum(p.numel() for p in self.global_model.parameters())
    
    def select_clients(self, round_idx, V):
        # Update gradient estimates
        for client in self.clients:
            client.update_gradient_estimate(round_idx)
        
        # Get channel gains with clamping
        channel_gains = [max(1e-3, abs(c.set_channel_gain())) for c in self.clients]
        min_channel_gain = min(channel_gains)
        
        # Compute sigma_t with physical constraints
        grad_norms = [max(1.0, c.last_gradient_norm_est) for c in self.clients]
        min_grad_norm = min(grad_norms)
        desired_sigma = (self.gamma0 * (self.sigma_n ** 2) * np.sqrt(self.d)) / min_grad_norm
        
        # Get minimum P_max * |h| among clients - FIXED
        min_power_limit = min(c.P_max * abs(c.channel_gain) for c in self.clients)
        max_sigma = np.sqrt(min_power_limit)
        
        sigma_t = min(desired_sigma, max_sigma)
        
        # Compute and normalize Ct
        Ct = {}
        for client in self.clients:
            E_est = client.estimate_energy(sigma_t, self.batch_size)
            Ct[client.client_id] = self.virtual_queues[client.client_id] * E_est
        
        # Normalize Ct values
        median_ct = np.median(list(Ct.values())) + 1e-8
        normalized_ct = {cid: q/median_ct for cid, q in Ct.items()}
        
        # Sort clients by normalized Ct
        sorted_clients = sorted(self.clients, key=lambda c: normalized_ct[c.client_id])
        sorted_ct = [normalized_ct[c.client_id] for c in sorted_clients]
        cumsum_ct = np.cumsum(sorted_ct)
        
        # Calculate vt(k) with balanced terms
        vt = []
        eta_t = self._learning_rate(round_idx)
        
        for k in range(1, len(self.clients) + 1):
            term1 = self.G2 / (self.batch_size * k)
            term2 = (self.sigma_n ** 2 * self.d) / (sigma_t ** 2 * k ** 2)
            Ut = (self.l_smooth * eta_t ** 2 / 2) * (term1 + term2)
            cost = V * Ut + cumsum_ct[k-1]  # Use normalized Ct
            vt.append(cost)
        
        # Find optimal k with minimum cost
        k_star = np.argmin(vt) + 1
        selected = sorted_clients[:k_star]
        
        # Debug logging
        logging.info(f"Round {round_idx}: Selected {k_star} clients")
        logging.info(f"σ_t: {sigma_t:.2f} (desired: {desired_sigma:.2f}, max: {max_sigma:.2f})")
        logging.info(f"Min grad norm: {min_grad_norm:.4f}, Min channel: {min_channel_gain:.4f}")
        logging.info(f"Virtual queues: {max(self.virtual_queues.values()):.1f} (max)")
        
        return selected, sigma_t
    
    def aggregate_gradients(self, selected, sigma_t):
        # Sum of gradients
        sum_grad = None
        actual_norms = {}
        signal_power = 0.0
        
        for client in selected:
            grad_norm = client.compute_gradient(self.batch_size)
            actual_norms[client.client_id] = grad_norm
            
            if sum_grad is None:
                sum_grad = client.last_gradient.clone()
            else:
                sum_grad += client.last_gradient
            
            # For SNR calculation
            signal_power += (sigma_t * grad_norm) ** 2
        
        # Simulate OTA transmission
        noise = torch.randn_like(sum_grad) * self.sigma_n
        received = sigma_t * sum_grad + noise
        
        # Calculate actual SNR
        noise_power = torch.sum(noise ** 2).item()
        actual_snr = signal_power / noise_power if noise_power > 0 else float('inf')
        self.actual_snr_history.append(actual_snr)
        
        # Normalize (eq 10)
        aggregated = received / (sigma_t * len(selected))
        return aggregated, actual_norms
    
    def update_model(self, update, round_idx):
        lr = self._learning_rate(round_idx)
        with torch.no_grad():
            params = list(self.global_model.parameters())
            for i, param in enumerate(params):
                update_slice = update[:param.numel()].view_as(param)
                param -= lr * update_slice
                update = update[param.numel():]
    
    def update_queues(self, selected, actual_norms, sigma_t, round_idx):
        per_round_budget = {cid: budget / self.total_rounds 
                           for cid, budget in self.energy_budgets.items()}
        
        for client in self.clients:
            cid = client.client_id
            prev_q = self.virtual_queues[cid]
            
            if client in selected:
                # Calculate actual energy used
                E_actual = client.actual_energy(
                    sigma_t, self.batch_size, actual_norms[cid]
                )
                self.energy_consumed[cid] += E_actual
                energy_diff = E_actual - per_round_budget[cid]
            else:
                # Unscheduled clients only subtract budget
                energy_diff = -per_round_budget[cid]
            
            # Update virtual queue (eq 25)
            self.virtual_queues[cid] = max(0, prev_q + energy_diff)
        
        # Record history
        self.selection_history.append([c.client_id for c in selected])
        self.queue_history.append(self.virtual_queues.copy())
    
    def _learning_rate(self, round_idx):
        # Decaying learning rate
        base_lr = 0.1
        return base_lr * (0.95 ** (round_idx // 30))