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
        logging.info(f"Server initialized with {len(clients)} clients on {device}")
        
        # Norm tracking
        self.norm_history = []
        self.ema_max_norm = None
        
        # Energy tracking
        self.energy_tracker = {
            'per_round_total': [],
            'cumulative_per_client': defaultdict(float),
            'divergence_history': [],
            'round_times': []
        }
        
        # Fading channel support
        self.fading_coefficients = None

    def _get_client_channel(self, client_idx):
        """Get channel coefficient for client (1.0 if no fading)"""
        if self.fading_coefficients and client_idx < len(self.fading_coefficients):
            return self.fading_coefficients[client_idx]
        return 1.2 + 0j  # Default channel gain = 1
        # magnitude = np.random.rayleigh(scale=1/np.sqrt(2))
        # phase = np.random.uniform(0, 2*np.pi)
        # return magnitude * np.exp(1j * phase)

        
    def broadcast_model(self):
        logging.info("Broadcasting global model to clients")
        try:
            state_dict = copy.deepcopy(self.global_model.state_dict())
            for client in self.clients:
                client.set_model(state_dict)
            logging.info("Broadcast completed")
        except Exception as e:
            logging.error(f"Broadcast failed: {str(e)}")
            raise
    
    def compute_alpha_t(self, updates):
        logging.info("Computing alpha_t")
        # Calculate squared norms for all valid updates
        squared_norms = []
        for i, update in enumerate(updates):
            if update is None: 
                logging.warning(f"Update {i} is None")
                continue
            total_norm = 0.0
            for param in update.values():
                total_norm += torch.norm(param).item() ** 2
            squared_norms.append(total_norm)
            logging.info(f"Update {i} norm: {np.sqrt(total_norm):.4f}")
        
        if not squared_norms:
            logging.warning("No valid updates for alpha_t calculation")
            return self.P_max / 1e-8
        
        # Track max norm for EMA calculation
        current_max = max(squared_norms)
        self.norm_history.append(current_max)
        logging.info(f"Current max norm: {np.sqrt(current_max):.4f}")
        
        # EMA calculation
        decay = 0.9
        if self.ema_max_norm is None:
            self.ema_max_norm = current_max
        else:
            self.ema_max_norm = decay * self.ema_max_norm + (1 - decay) * current_max
            
        alpha = self.P_max / (self.ema_max_norm + 1e-12)
        logging.info(f"Computed alpha_t: {alpha:.6f} (EMA norm: {np.sqrt(self.ema_max_norm):.4f})")
        return alpha
    
    def aggregate(self):
        logging.info("Starting aggregation")
        aggregation_start = time.time()
        
        # 1. Collect updates from clients
        updates = []
        for i, client in enumerate(self.clients):
            tx_duration = 0.1 * len(client.data_indices) / 1000
            update = client.get_update(tx_duration)
            updates.append(update)
            logging.info(f"Client {i} update: {'Received' if update is not None else 'None'}")
        
        # 2. Compute precoding factor α_t
        alpha_t = self.compute_alpha_t(updates)
        
        # 3. Apply precoding and aggregate
        aggregated = None
        active_clients = 0
        logging.info("Applying precoding and aggregation")
        
        for i, update in enumerate(updates):
            if update is None: 
                continue
                
            precoded_update = {}
            for key, param in update.items():
                # Apply fading if available
                if self.fading_coefficients:
                    h = self.fading_coefficients[i]
                    # precoding = h.conjugate() / abs(h) if abs(h) > 0 else 1
                    precoding = 1.0 / h if abs(h) > 1e-3 else 0
                else:
                    precoding = 1
                    
                precoded_update[key] = math.sqrt(alpha_t) * precoding * param.to(self.device).float()
            
            if aggregated is None:
                aggregated = precoded_update
            else:
                for key in aggregated:
                    aggregated[key] += precoded_update[key]
            active_clients += 1
        
        if aggregated is None:
            logging.warning("No updates available, returning previous model")
            return copy.deepcopy(self.global_model.state_dict())
        
        logging.info(f"Aggregated {active_clients} client updates")
        
        # 4. Add channel noise
        logging.info("Adding channel noise")
        for key in aggregated:
            noise_scale = math.sqrt(self.noise_var / (active_clients**2 * alpha_t))
            noise = torch.randn_like(aggregated[key]) * noise_scale
            aggregated[key] += noise
            noise_level = torch.norm(noise).item()
            if noise_level > 0.1:
                logging.warning(f"Large noise for {key}: {noise_level:.4f}")
        
        # 5. Server-side scaling
        logging.info("Applying server-side scaling")
        scaled_update = {}
        global_model_prev = self.global_model.state_dict()
        for key in aggregated:
            update_val = aggregated[key] / (active_clients * math.sqrt(alpha_t))
            
            # Clip large updates
            update_norm = torch.norm(update_val).item()
            if update_norm > 1.0:
                logging.warning(f"Large update for {key}: {update_norm:.4f}, clipping")
                update_val = update_val / update_norm
                
            scaled_update[key] = update_val + global_model_prev[key]
        
        # 6. Track client divergence
        self._track_divergence(scaled_update)
        
        # 7. Track energy consumption
        self._track_energy(alpha_t)
        
        # Record aggregation time
        agg_time = time.time() - aggregation_start
        self.energy_tracker['round_times'].append(agg_time)
        logging.info(f"Aggregation completed in {agg_time:.4f}s")
        
        return scaled_update
    
    def _track_divergence(self, global_update):
        """Measure client divergence for heterogeneity analysis"""
        try:
            divergence = 0.0
            global_state = global_update
            
            for client in self.clients:
                if not hasattr(client, 'local_model_trained') or client.local_model_trained is None:
                    continue
                    
                client_state = client.local_model_trained
                for key in global_state:
                    if key in client_state:
                        diff = global_state[key] - client_state[key].to(self.device)
                        divergence += torch.norm(diff).item() ** 2
            
            self.energy_tracker['divergence_history'].append(divergence)
            logging.info(f"Divergence tracked: {divergence:.4f}")
        except Exception as e:
            logging.error(f"Error tracking divergence: {str(e)}")
    
    def _track_energy(self, alpha_t):
        """Track energy consumption across all clients (PAPER-COMPATIBLE)"""
        try:
            round_energy = 0.0
            per_client_energy = {}
            
            for i, client in enumerate(self.clients):
                # 1. Get channel coefficient
                h_k = self._get_client_channel(i)
                
                # 2. Compute transmit power (p_k = sqrt(alpha_t * norm_squared))
                p_k = math.sqrt(alpha_t * client.norm_squared)
                
                # 3. Get energy using paper's formula (NO gradient norm needed)
                total_energy = client.get_energy_consumption(p_k, h_k)
                
                # 4. Update tracking
                self.energy_tracker['cumulative_per_client'][client.client_id] += total_energy
                round_energy += total_energy
                per_client_energy[client.client_id] = total_energy
            
            self.energy_tracker['per_round_total'].append(round_energy)
            logging.info(f"Round energy: {round_energy:.2f}J")
        except Exception as e:
            logging.error(f"Error tracking energy: {str(e)}")

    
    def update_model(self, state_dict):
        """Update global model and reset client states"""
        logging.info("Updating global model")
        try:
            self.global_model.load_state_dict(state_dict)
            
            # Reset client states for next round
            for client in self.clients:
                client.reset_round()
        except Exception as e:
            logging.error(f"Error updating global model: {str(e)}")
            raise
    
    def apply_fading(self, coefficients):
        """Apply fading coefficients for current round"""
        logging.info("Applying fading coefficients")
        if len(coefficients) != len(self.clients):
            logging.warning(f"Fading coefficients count mismatch. Expected {len(self.clients)}, got {len(coefficients)}")
        self.fading_coefficients = coefficients