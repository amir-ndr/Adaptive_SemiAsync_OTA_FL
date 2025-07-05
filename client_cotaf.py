import torch
import numpy as np
import copy
import time
import logging
import random
from client import Client

class COTAFClient(Client):
    def __init__(self, client_id, data_indices, model, fk, mu_k, P_max, C, Ak, 
                 train_dataset, device='cpu', local_epochs=1):
        super().__init__(client_id, data_indices, model, fk, mu_k, P_max, C, Ak,
                         train_dataset, device, local_epochs)
        
        # Additional state for COTAF
        self.theta_prev = None
        self.theta_current = None
        self.train_loader = self._create_data_loader()
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.1)

        
        # Energy tracking
        self.comp_energy = 0.0
        self.comm_energy = 0.0
        self.norm_squared = 0.0  # For communication energy calculation
        
    def _create_data_loader(self):
        """Create DataLoader for client's local data"""
        from torch.utils.data import Subset, DataLoader
        subset = Subset(self.train_dataset, self.data_indices)
        return DataLoader(subset, batch_size=self.Ak, shuffle=True)
    
    def set_model(self, state_dict):
        """Set current model and store previous state"""
        self.theta_prev = copy.deepcopy(self.theta_current) if self.theta_current else None
        self.theta_current = copy.deepcopy(state_dict)
        self.local_model.load_state_dict(state_dict)
        self.stale_model.load_state_dict(state_dict)
        
    def local_train(self):
        """Perform local SGD training for H steps and track computation energy"""
        self.local_model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Compute FLOPs and energy
        total_samples = len(self.train_loader.dataset) * self.local_epochs
        total_flops = self.C * total_samples
        self.comp_energy = self.mu_k * (self.fk ** 2) * total_flops
        
        # Training loop
        for epoch in range(self.local_epochs):
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 5.0)
                self.optimizer.step()
                
                total_loss += loss.item() * data.size(0)  # Weight by batch size
                num_batches += data.size(0)

        # Update model state
        self.theta_current = copy.deepcopy(self.local_model.state_dict())
        return total_loss / num_batches  # Return average loss
    
    def get_update(self):
        # Remove zero-update special case
        if self.theta_prev is None or self.theta_current is None:
            return None  # Shouldn't happen after initialization
            
        update = {}
        self.norm_squared = 0.0
        for key in self.theta_current:
            delta = self.theta_current[key] - self.theta_prev[key]
            update[key] = delta
            self.norm_squared += torch.norm(delta).item() ** 2
        return update
    
    def get_energy_consumption(self, alpha_t):
        """Compute communication energy for given precoding factor"""
        # Communication energy: ||x_n||^2 = α_t * ||Δθ||^2
        self.comm_energy = alpha_t * self.norm_squared
        return self.comp_energy + self.comm_energy