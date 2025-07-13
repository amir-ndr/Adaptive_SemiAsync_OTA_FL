import torch
import numpy as np
import logging

class SyncClient:
    def __init__(self, client_id, data_indices, model, fk, en, P_max, 
                 train_dataset, device='cpu'):
        self.client_id = client_id
        self.data_indices = data_indices
        self.model = model.to(device)
        self.device = device
        self.fk = fk  # CPU frequency (Hz)
        self.en = en  # Energy per sample (J/sample)
        self.P_max = P_max  # Max transmit power
        self.train_dataset = train_dataset
        
        # State variables
        self.last_gradient = None
        self.last_gradient_norm = 1.0  # For EST-P estimation
        self.channel_gain = None
        self.energy_consumed = 0.0
        self.last_gradient_norm_est = 1.0  # Estimated gradient norm
        self.gradient_history = []
        
        logging.info(f"Client {client_id} initialized | "
                     f"CPU: {fk/1e9:.2f} GHz | "
                     f"Energy/sample: {en:.2e} J | "
                     f"Data samples: {len(data_indices)}")
    
    def update_model(self, model_state_dict):
        self.model.load_state_dict(model_state_dict)
        
    def set_channel_gain(self):
        # Rayleigh fading with random phase
        magnitude = np.random.rayleigh(scale=1/np.sqrt(2))
        phase = np.random.uniform(0, 2*np.pi)
        self.channel_gain = magnitude * np.exp(1j * phase)
        return abs(self.channel_gain)
    
    def compute_gradient(self, batch_size):
        if len(self.data_indices) == 0:
            logging.warning(f"Client {self.client_id} has no data!")
            self.last_gradient = torch.zeros(self._model_dimension())
            self.last_gradient_norm = 0.0
            return 0.0
        
        # Random mini-batch
        indices = np.random.choice(self.data_indices, 
                                  size=min(batch_size, len(self.data_indices)),
                                  replace=False)
        batch = [self.train_dataset[i] for i in indices]
        
        # Prepare data
        images = torch.stack([x[0] for x in batch]).to(self.device)
        labels = torch.tensor([x[1] for x in batch]).to(self.device)
        
        # Compute gradient
        self.model.zero_grad()
        outputs = self.model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        
        # Flatten gradients
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.detach().view(-1))
        flat_grad = torch.cat(gradients)
        
        # Store results
        self.last_gradient = flat_grad
        self.last_gradient_norm = torch.norm(flat_grad).item()
        self.gradient_history.append(self.last_gradient_norm)
        return self.last_gradient_norm
    
    def estimate_energy(self, sigma_t, batch_size):
        if self.channel_gain is None:
            return float('inf')
        
        h_sq = abs(self.channel_gain) ** 2
        comp_energy = self.en * batch_size
        comm_energy = (sigma_t ** 2) / h_sq * (self.last_gradient_norm ** 2)
        return comp_energy + comm_energy
    
    def actual_energy(self, sigma_t, batch_size, actual_grad_norm):
        if self.channel_gain is None:
            return 0.0
        
        h_sq = abs(self.channel_gain) ** 2
        comp_energy = self.en * batch_size
        comm_energy = (sigma_t ** 2) / h_sq * (actual_grad_norm ** 2)
        return comp_energy + comm_energy
    
    def _model_dimension(self):
        return sum(p.numel() for p in self.model.parameters())
    
    def update_gradient_estimate(self, round_idx, decay=0.8):
        """Update gradient norm estimate with exponential smoothing"""
        if not self.gradient_history:
            self.last_gradient_norm_est = self.last_gradient_norm
        else:
            # Use weighted average of historical norms
            weights = np.array([decay ** i for i in range(len(self.gradient_history))])
            weights /= weights.sum()
            self.last_gradient_norm_est = np.dot(weights, self.gradient_history[-len(weights):])
        
        # Record current for future reference
        self.gradient_history.append(self.last_gradient_norm)