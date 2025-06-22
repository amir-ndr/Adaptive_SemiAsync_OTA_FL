import torch
import numpy as np

class Client:
    def __init__(self, client_id, data_indices, model, fk, mu_k, P_max, C, Ak, 
                 train_dataset, device='cpu'):
        self.client_id = client_id
        self.data_indices = data_indices
        self.local_model = model
        self.fk = fk
        self.mu_k = mu_k
        self.P_max = P_max
        self.C = C
        self.Ak = Ak
        self.train_dataset = train_dataset
        self.device = device
        
        # State variables (paper: Sec III.B)
        # Initialize with full computation time
        self.dt_k = C * Ak / fk  # Initial computation time
        self.tau_k = 0  # Staleness counter
        self.last_gradient = None
        self.gradient_norm = 0
        self.h_t_k = None  # Current channel gain
        self.ready = False  # Not ready initially

    def compute_gradient(self):
        """Compute gradient using client's local model"""
        # Sample mini-batch
        indices = np.random.choice(self.data_indices, self.Ak, replace=False)
        batch = [self.train_dataset[i] for i in indices]
        images = torch.stack([item[0] for item in batch]).to(self.device)
        labels = torch.tensor([item[1] for item in batch]).to(self.device)
        
        # Compute gradient
        self.local_model.zero_grad()
        outputs = self.local_model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        
        # Extract and store gradient
        gradient = []
        for param in self.local_model.parameters():
            if param.grad is not None:
                gradient.append(param.grad.clone())
        
        flat_gradient = torch.cat([g.view(-1) for g in gradient])
        self.last_gradient = flat_gradient
        self.gradient_norm = torch.norm(flat_gradient).item()
        return True

    def update_computation_time(self, D_t):
        """Update remaining computation time according to paper (Eq. 21c)"""
        # If not selected, reduce remaining time
        if not self.ready:
            self.dt_k = max(0, self.dt_k - D_t)
            self.ready = (self.dt_k <= 0)
        return self.ready

    def reset_computation(self):
        """Reset computation time when selected (paper: Eq. 21c)"""
        self.dt_k = self.C * self.Ak / self.fk
        self.ready = False

    def reset_staleness(self):
        """Reset after being selected (paper: Sec III.B)"""
        self.tau_k = 0

    def increment_staleness(self):
        """Increment when not selected (paper: Sec III.B)"""
        self.tau_k += 1

    def set_channel_gain(self):
        """Simulate channel gain (Rayleigh fading)"""
        real = np.random.normal(0, 1)
        imag = np.random.normal(0, 1)
        self.h_t_k = complex(real, imag)
        return abs(self.h_t_k)