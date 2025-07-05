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
        self.max_update_norm = 1.0
        self.device = device
        self.global_model_prev = None
        self.energy_tracker = {
            'per_round_total': [],
            'per_client_per_round': [],
            'cumulative_per_client': {c.client_id: 0.0 for c in clients}
        }
        self.model_dim = sum(p.numel() for p in global_model.parameters())  # Precompute
        
    def broadcast_model(self):
        state_dict = copy.deepcopy(self.global_model.state_dict())
        for client in self.clients:
            client.set_model(state_dict)
        self.global_model_prev = state_dict
    
    def compute_alpha_t(self, updates):
        current_max = 1e-8
        for update in updates:
            if update is None: 
                continue
            total_norm = sum(torch.norm(p).item()**2 for p in update.values())
            if total_norm > current_max:
                current_max = total_norm
                
        # Update running maximum (paper uses historical expectation)
        if current_max > self.max_update_norm:
            self.max_update_norm = current_max * 1.1  # Smooth update
        else:
            self.max_update_norm = self.max_update_norm * 0.9 + current_max * 0.1
            
        return self.P_max / self.max_update_norm
    
    def aggregate(self):
        updates = [client.get_update() for client in self.clients]
        alpha_t = self.compute_alpha_t(updates)
        aggregated = None
        
        # 1. Precoding (Eq 9)
        for update in updates:
            if update is None: 
                continue
            precoded_update = {}
            for key, param in update.items():
                precoded_update[key] = math.sqrt(alpha_t) * param.to(self.device).float()
            
            if aggregated is None:
                aggregated = precoded_update
            else:
                for key in aggregated:
                    aggregated[key] += precoded_update[key]
        
        if aggregated is None:
            return copy.deepcopy(self.global_model_prev)
        
        # 2. Add channel noise (Eq 11)
        for key in aggregated:
            # Paper uses w̃t ~ N(0, σ_w² I)
            noise = torch.randn_like(aggregated[key]) * math.sqrt(self.noise_var)
            aggregated[key] += noise
        
        # 3. Server-side scaling (Eq 12)
        N = len([u for u in updates if u is not None]) or 1
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
        self.global_model.load_state_dict(state_dict)