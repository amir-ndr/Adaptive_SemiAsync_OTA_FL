import torch
import numpy as np
import copy
import logging
import math
import time
from collections import defaultdict

class COTAFServer:
    def __init__(self, global_model, clients, P_max, noise_var, H_local=1, device='cpu'):
        self.global_model = global_model
        self.clients = clients
        self.P_max = P_max
        self.noise_var = noise_var
        self.H_local = H_local
        self.device = device
        
        # Norm tracking
        self.norm_history = []
        self.ema_max_norm = None  # Initialize EMA tracker
        
        # Energy tracking
        self.energy_tracker = {
            'per_round_total': [],
            'cumulative_per_client': defaultdict(float),
            'divergence_history': [],
            'round_times': []
        }
        
        # Fading channel support
        self.fading_coefficients = None
        
    def broadcast_model(self):
        """Broadcast current global model to all clients"""
        state_dict = copy.deepcopy(self.global_model.state_dict())
        for client in self.clients:
            client.set_model(state_dict)
    
    def compute_alpha_t(self, updates):
        """
        Compute precoding factor α_t per paper specification:
        α_t = P / max_n E[||θ_t^n - θ_{t-H}^n||^2]
        """
        # Calculate squared norms for all valid updates
        squared_norms = []
        for update in updates:
            if update is None: 
                continue
            total_norm = 0.0
            for param in update.values():
                total_norm += torch.norm(param).item() ** 2
            squared_norms.append(total_norm)
        
        if not squared_norms:
            return self.P_max / 1e-8  # Avoid division by zero
        
        # Track max norm for EMA calculation
        current_max = max(squared_norms)
        self.norm_history.append(current_max)
        
        # Paper uses expectation - approximate with EMA
        decay = 0.9
        if self.ema_max_norm is None:
            self.ema_max_norm = current_max
        else:
            self.ema_max_norm = decay * self.ema_max_norm + (1 - decay) * current_max
            
        return self.P_max / (self.ema_max_norm + 1e-12)
    
    def aggregate(self):
        """Perform COTAF aggregation with noise mitigation"""
        aggregation_start = time.time()
        
        # 1. Collect updates from clients
        updates = []
        for client in self.clients:
            # Use actual data size for duration estimation
            tx_duration = 0.1 * len(client.data_indices) / 1000
            update = client.get_update(tx_duration)
            updates.append(update)
        
        # 2. Compute precoding factor α_t
        alpha_t = self.compute_alpha_t(updates)
        
        # 3. Apply precoding (Eq 9) and aggregate
        aggregated = None
        active_clients = 0
        
        for i, update in enumerate(updates):
            if update is None: 
                continue
                
            precoded_update = {}
            for key, param in update.items():
                # Apply fading if available
                if self.fading_coefficients:
                    h = self.fading_coefficients[i]
                    precoding = h.conjugate() / abs(h) if abs(h) > 0 else 1
                else:
                    precoding = 1
                    
                precoded_update[key] = math.sqrt(alpha_t) * precoding * param.to(self.device).float()
            
            if aggregated is None:
                aggregated = precoded_update
            else:
                for key in aggregated:
                    aggregated[key] += precoded_update[key]
            active_clients += 1
        
        # Handle case where no updates are available
        if aggregated is None:
            return copy.deepcopy(self.global_model.state_dict())
        
        # 4. Add channel noise with proper scaling (Eq 12)
        for key in aggregated:
            # Calculate noise scaling factor: w_t = w̃_t/(N√α_t)
            noise_scale = math.sqrt(self.noise_var / (active_clients**2 * alpha_t))
            noise = torch.randn_like(aggregated[key]) * noise_scale
            aggregated[key] += noise
        
        # 5. Server-side scaling (Eq 12)
        scaled_update = {}
        global_model_prev = self.global_model.state_dict()
        for key in aggregated:
            scaled_update[key] = aggregated[key] / (active_clients * math.sqrt(alpha_t)) + global_model_prev[key]
        
        # 6. Track client divergence
        self._track_divergence(scaled_update)
        
        # 7. Track energy consumption
        self._track_energy(alpha_t)
        
        # Record aggregation time
        self.energy_tracker['round_times'].append(time.time() - aggregation_start)
        
        return scaled_update
    
    def _track_divergence(self, global_update):
        """Measure client divergence for heterogeneity analysis"""
        divergence = 0.0
        global_state = global_update
        
        for client in self.clients:
            # Only check clients that completed training
            if not hasattr(client, 'local_model_trained') or client.local_model_trained is None:
                continue
                
            client_state = client.local_model_trained
            for key in global_state:
                if key in client_state:
                    # Calculate difference between global and client model
                    diff = global_state[key] - client_state[key].to(self.device)
                    divergence += torch.norm(diff).item() ** 2
                    
        self.energy_tracker['divergence_history'].append(divergence)
    
    def _track_energy(self, alpha_t):
        """Track energy consumption across all clients"""
        round_energy = 0.0
        per_client_energy = {}
        
        for client in self.clients:
            total_energy = client.get_energy_consumption(alpha_t)
            self.energy_tracker['cumulative_per_client'][client.client_id] += total_energy
            round_energy += total_energy
            per_client_energy[client.client_id] = total_energy
            
        self.energy_tracker['per_round_total'].append(round_energy)
    
    def update_model(self, state_dict):
        """Update global model and reset client states"""
        self.global_model.load_state_dict(state_dict)
        
        # Reset client states for next round
        for client in self.clients:
            client.reset_round()
    
    def apply_fading(self, coefficients):
        """Apply fading coefficients for current round (Eq 7 extension)"""
        if len(coefficients) != len(self.clients):
            logging.warning(f"Fading coefficients count mismatch. Expected {len(self.clients)}, got {len(coefficients)}")
        self.fading_coefficients = coefficients