import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy
import time

class Client:
    def __init__(self, client_id, data_indices, model, fk, mu_k, P_max, C, Ak, 
                 train_dataset, device='cpu', local_epochs=1):
        self.client_id = client_id
        self.data_indices = data_indices
        self.stale_model = copy.deepcopy(model).to(device)  # Stale global model
        self.local_model = copy.deepcopy(model).to(device)  # For local training
        self.fk = fk
        self.mu_k = mu_k
        self.P_max = P_max
        self.C = C
        self.Ak = Ak
        self.train_dataset = train_dataset
        self.device = device
        self.local_epochs = local_epochs
        
        # State variables
        self.dt_k = self._full_computation_time()  # Initial computation time
        self.tau_k = 0  # Staleness counter
        self.last_gradient = None
        self.gradient_norm = 0
        self.h_t_k = None  # Current channel gain
        self.ready = (self.dt_k <= 0)  # Ready if computation complete

    def _full_computation_time(self):
        """Calculate full computation time based on data size and epochs"""
        num_samples = len(self.data_indices)
        total_ops = self.C * num_samples * self.local_epochs
        return total_ops / self.fk

    def update_stale_model(self, model_state_dict):
        """Update stale model from server (Sec III.B)"""
        self.stale_model.load_state_dict(model_state_dict)
        self.local_model.load_state_dict(model_state_dict)  # Sync local model
        self.tau_k = 0  # Reset staleness counter
        self.reset_computation()  # Reset computation time

    def compute_gradient_multiple(self):
        """Compute gradient using stale model w^{t-τ_t^k}"""
        # Start from stale model version
        self.local_model.load_state_dict(self.stale_model.state_dict())
        initial_weights = [param.clone().detach() for param in self.local_model.parameters()]
        
        # Create DataLoader for local data
        dataset = [self.train_dataset[i] for i in self.data_indices]
        images = torch.stack([item[0] for item in dataset])
        labels = torch.tensor([item[1] for item in dataset])
        data_loader = DataLoader(TensorDataset(images, labels), 
                                batch_size=self.Ak, shuffle=True)
        
        # Create optimizer
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.01)
        
        # Run local training
        for epoch in range(self.local_epochs):
            for batch in data_loader:
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad()
                outputs = self.local_model(images)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Compute delta (w^{t-τ_t^k} - w_local)
        with torch.no_grad():
            gradient = []
            for init, current in zip(initial_weights, self.local_model.parameters()):
                gradient.append((init - current).view(-1))
            
            flat_gradient = torch.cat(gradient)
            self.last_gradient = flat_gradient
            self.gradient_norm = torch.norm(flat_gradient).item()
        
        return True

    def compute_gradient(self):
        """Compute gradient using stale model (paper: Sec III.B)"""
        self.local_model.load_state_dict(self.stale_model.state_dict())
        
        # Sample a single mini-batch (not full dataset)
        indices = np.random.choice(self.data_indices, size=self.Ak, replace=False)
        batch = [self.train_dataset[i] for i in indices]
        images = torch.stack([item[0] for item in batch]).to(self.device)
        labels = torch.tensor([item[1] for item in batch]).to(self.device)
        
        # Zero gradients
        self.local_model.zero_grad()
        
        # Forward and backward pass
        outputs = self.local_model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        
        # Extract gradients directly
        gradient = []
        for param in self.local_model.parameters():
            gradient.append(param.grad.clone().view(-1))
        
        flat_gradient = torch.cat(gradient)
        self.last_gradient = flat_gradient
        self.gradient_norm = torch.norm(flat_gradient).item()
        
        return True

    def update_computation_time(self, D_t):
        """Update remaining computation time (Eq. 21c)"""
        self.dt_k = max(0, self.dt_k - D_t)
        self.ready = (self.dt_k <= 0)
        return self.ready

    def reset_computation(self):
        """Reset computation time when selected"""
        self.dt_k = self._full_computation_time()
        self.ready = (self.dt_k <= 0)

    def increment_staleness(self):
        """Increment staleness counter when not selected"""
        self.tau_k += 1

    def set_channel_gain(self):
        """Simulate Rayleigh fading channel"""
        gain = complex(np.random.normal(0, 1), np.random.normal(0, 1))
        self.h_t_k = gain / np.sqrt(2)  # Normalized complex gain
        return abs(self.h_t_k)
    
    def reset_staleness(self):
        """Reset staleness counter after selection (paper: Sec III.B)"""
        self.tau_k = 0  # Reset staleness