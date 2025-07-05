import torch
import numpy as np
import copy
import time
import logging
import random
from torch.utils.data import Subset, DataLoader

class COTAFClient:
    def __init__(self, client_id, data_indices, model, fk, mu_k, P_max, C, Ak, 
                 train_dataset, device='cpu', local_epochs=1):
        self.client_id = client_id
        self.data_indices = data_indices
        self.local_model = copy.deepcopy(model)
        self.stale_model = copy.deepcopy(model)
        self.fk = fk
        self.mu_k = mu_k
        self.P_max = P_max
        self.C = C  # FLOPs per sample (precomputed)
        self.Ak = Ak
        self.train_dataset = train_dataset
        self.device = device
        self.local_epochs = local_epochs
        
        # State management
        self.global_model_start = None
        self.local_model_trained = None
        self.train_loader = self._create_data_loader()
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.1)
        
        # Energy tracking
        self.comp_energy = 0.0
        self.comm_energy = 0.0
        self.norm_squared = 0.0
        self.flops_completed = 0
        self.last_tx_duration = 0.0
        
    def _create_data_loader(self):
        """Create DataLoader for client's local data"""
        subset = Subset(self.train_dataset, self.data_indices)
        return DataLoader(subset, batch_size=self.Ak, shuffle=True)
    
    def set_model(self, state_dict):
        """Set starting model for current round"""
        self.global_model_start = copy.deepcopy(state_dict)
        self.local_model.load_state_dict(state_dict)
        self.stale_model.load_state_dict(state_dict)
        
    def local_train(self):
        """Perform local SGD training for H steps"""
        self.local_model.train()
        total_loss = 0.0
        num_samples = 0
        
        # Training loop
        start_time = time.time()
        for epoch in range(self.local_epochs):
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                batch_size = data.size(0)
                
                self.optimizer.zero_grad()
                outputs = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 5.0)
                self.optimizer.step()
                
                total_loss += loss.item() * batch_size
                num_samples += batch_size
        
        # Compute FLOPs: C (FLOPs/sample) * samples processed
        self.flops_completed = self.C * num_samples * self.local_epochs
        
        # Compute computation energy
        duration = time.time() - start_time
        self.comp_energy = self.mu_k * (self.fk ** 2) * self.flops_completed
        
        # Store trained model
        self.local_model_trained = copy.deepcopy(self.local_model.state_dict())
        return total_loss / num_samples if num_samples > 0 else 0.0
    
    def get_update(self, tx_duration=0.1):
        """Compute model update relative to start of round"""
        self.last_tx_duration = tx_duration
        
        if self.global_model_start is None or self.local_model_trained is None:
            logging.warning(f"Client {self.client_id}: Missing model states for update")
            return None
            
        update = {}
        self.norm_squared = 0.0
        for key in self.local_model_trained:
            # Compute delta: trained model - start model
            delta = self.local_model_trained[key] - self.global_model_start[key]
            update[key] = delta
            self.norm_squared += torch.norm(delta).item() ** 2
            
        return update
    
    def get_energy_consumption(self, alpha_t):
        """Compute total energy consumption with circuit energy"""
        # Communication energy: ||x_n||^2 = α_t * ||Δθ||^2
        tx_power = alpha_t * self.norm_squared / self.last_tx_duration
        comm_energy = tx_power * self.last_tx_duration
        
        # Circuit energy (base cost for RF components)
        circuit_energy = 0.1 * self.last_tx_duration  # Fixed cost in Joules
        
        # Total energy = computation + transmission + circuit
        total_energy = self.comp_energy + comm_energy + circuit_energy
        return total_energy
    
    def reset_round(self):
        """Reset round-specific state"""
        self.comp_energy = 0.0
        self.comm_energy = 0.0
        self.norm_squared = 0.0
        self.flops_completed = 0
        self.global_model_start = None
        self.local_model_trained = None