import torch
import numpy as np
import copy
import logging
import math

class COTAFServer:
    def __init__(self, global_model, clients, P_max, noise_var, H_local=1, device='cpu'):
        self.global_model = global_model
        self.clients = clients
        self.P_max = P_max
        self.noise_var = noise_var
        self.H_local = H_local
        self.device = device
        self.global_model_prev = None
        self.energy_tracker = {
            'per_round_total': [],
            'per_client_per_round': [],
            'cumulative_per_client': {c.client_id: 0.0 for c in clients}
        }
        
    def broadcast_model(self):
        state_dict = copy.deepcopy(self.global_model.state_dict())
        for client in self.clients:
            client.set_model(state_dict)
        self.global_model_prev = state_dict
    
    def compute_alpha_t(self, updates):
        max_norm_sq = 1e-8  # Start with epsilon
        for update in updates:
            if update is None:  # Skip None updates
                continue
            total_norm = 0
            for param in update.values():
                total_norm += torch.norm(param).item() ** 2
            if total_norm > max_norm_sq:
                max_norm_sq = total_norm
        return self.P_max / (max_norm_sq + 1e-8)
    
    def aggregate(self):
        updates = [client.get_update() for client in self.clients]
        alpha_t = self.compute_alpha_t(updates)
        aggregated = None
        
        for update in updates:
            if update is None:  # Skip None updates
                continue
            precoded_update = {}
            for key, param in update.items():
                # Ensure proper type and device
                param = param.to(self.device).float()
                precoded_update[key] = math.sqrt(alpha_t) * param
            
            if aggregated is None:
                aggregated = precoded_update
            else:
                for key in aggregated:
                    aggregated[key] += precoded_update[key]
        
        # Handle case where all updates are None
        if aggregated is None:
            return copy.deepcopy(self.global_model_prev)
        
        # Add noise
        noise = {}
        for key, param in aggregated.items():
            noise[key] = torch.randn_like(param) * math.sqrt(self.noise_var)
            aggregated[key] += noise[key]
        
        # Server-side scaling
        N = len([u for u in updates if u is not None]) or 1  # Handle zero case
        scaled_update = {}
        for key in aggregated:
            scaled_update[key] = aggregated[key] / (N * math.sqrt(alpha_t)) + self.global_model_prev[key]
        
        # Energy tracking
        round_energy = 0.0
        per_client_energy = {}
        
        for client in self.clients:
            total_energy = client.get_energy_consumption(alpha_t)
            self.energy_tracker['cumulative_per_client'][client.client_id] += total_energy
            round_energy += total_energy
            per_client_energy[client.client_id] = total_energy
            
        self.energy_tracker['per_round_total'].append(round_energy)
        self.energy_tracker['per_client_per_round'].append(per_client_energy)
        
        return scaled_update
    
    def update_model(self, state_dict):
        current_keys = set(self.global_model.state_dict().keys())
        new_keys = set(state_dict.keys())
        if current_keys != new_keys:
            logging.warning(f"Model key mismatch: {current_keys - new_keys}")
        self.global_model.load_state_dict(state_dict)