import torch
import numpy as np
from torch.utils.data import DataLoader
from model import CNNMnist
from dataloader import load_mnist, partition_mnist_noniid
import matplotlib.pyplot as plt
import time
import copy

# Client class without threads
class Client:
    def __init__(self, client_id, data_indices, model, fk, mu_k, P_max, C, Ak, 
                 train_dataset, device='cpu'):
        self.client_id = client_id
        self.data_indices = data_indices
        self.local_model = copy.deepcopy(model)
        self.fk = fk
        self.mu_k = mu_k
        self.P_max = P_max
        self.C = C
        self.Ak = Ak
        self.train_dataset = train_dataset
        self.device = device
        
        # State variables
        self.dt_k = 0  # Remaining computation time
        self.tau_k = 0  # Staleness counter
        self.last_gradient = None
        self.gradient_norm = 0
        self.h_t_k = None  # Current channel gain
        self.next_available = 0  # Time when client will be available

    def compute_gradient(self, current_time, global_model):
        """Compute gradient if client is ready"""
        if current_time < self.next_available:
            return False
            
        # Update model and compute gradient
        self.local_model.load_state_dict(global_model.state_dict())
        
        # Sample mini-batch
        indices = np.random.choice(self.data_indices, self.Ak, replace=False)
        batch = [self.train_dataset[i] for i in indices]
        images = torch.stack([item[0] for item in batch]).to(self.device)
        labels = torch.tensor([item[1] for item in batch]).to(self.device)
        
        # Forward pass
        self.local_model.zero_grad()
        outputs = self.local_model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        
        # Extract gradients
        gradient = []
        for param in self.local_model.parameters():
            gradient.append(param.grad.clone())
        
        # Flatten and store
        flat_gradient = torch.cat([g.view(-1) for g in gradient])
        self.last_gradient = flat_gradient
        self.gradient_norm = torch.norm(flat_gradient).item()
        
        # Schedule next availability
        comp_time = self.C * self.Ak / self.fk
        self.next_available = current_time + comp_time
        self.dt_k = comp_time
        return True

    def set_channel_gain(self):
        """Simulate channel gain"""
        real = np.random.normal(0, 1)
        imag = np.random.normal(0, 1)
        self.h_t_k = complex(real, imag)
        return self.h_t_k
