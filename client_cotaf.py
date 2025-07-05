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
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.01)

        
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
        
        # Reset energy counters
        self.comp_energy = 0.0
        
        # Compute number of FLOPs
        num_batches = len(self.train_loader)
        total_samples = num_batches * self.Ak
        total_flops = self.C * total_samples * self.local_epochs
        
        # Calculate computation energy
        self.comp_energy = self.mu_k * (self.fk ** 2) * total_flops
        
        # Actual training
        for epoch in range(self.local_epochs):
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                self.optimizer.step()
        
        # Update current state after training
        self.theta_current = copy.deepcopy(self.local_model.state_dict())
    
    def get_update(self):
        """Compute model update and its norm for communication energy"""

        if self.theta_prev is None:  # First round case
            # Return zero update with correct structure
            update = {}
            for key in self.theta_current:
                update[key] = torch.zeros_like(self.theta_current[key])
            return update

        if self.theta_prev is None or self.theta_current is None:
            return None
            
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