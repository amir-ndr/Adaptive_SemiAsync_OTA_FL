# client_sync.py
import torch
import math
import numpy as np
import random
import logging
from collections import deque
from torch.utils.data import Subset, DataLoader

class Client:
    def __init__(self, client_id, data_indices, model, fk, mu_k, P_max, C, Ak, 
                 train_dataset, device='cpu', local_epochs=1):
        self.client_id = client_id
        self.data_indices = list(data_indices)
        self.model = model.to(device)
        self.fk = fk  # in Hz (1e9 = 1GHz)
        self.mu_k = mu_k  # Joules/FLOP
        self.P_max = P_max  # Watts (transmit power limit)
        self.C = C  # FLOPs per sample
        self.Ak = Ak  # Batch size
        self.train_dataset = train_dataset
        self.device = device
        self.local_epochs = local_epochs
        
        # Energy constants
        self.en = mu_k * fk**2 * C  # Per-sample computation energy (Joules)
        
        # State tracking
        self.grad_history = deque(maxlen=5)  # Track last 5 gradients for EST-P
        self.last_gradient = None
        self.last_gradient_norm = 0.1  # Initialize with small value
        self.h_t_k = None
        self.sigma_t = None
        self.logger = logging.getLogger(f"Client-{client_id}")
        self.last_update_round = -10  # Track when last updated
        
        # Initialize with dummy gradient
        self._initialize_gradient_history()
        
    def _initialize_gradient_history(self):
        """Initialize gradient history with small random values"""
        dim = self.model_dimension()
        for _ in range(3):
            dummy_grad = torch.randn(dim, device=self.device) * 0.01
            self.grad_history.append(dummy_grad)
            self.last_gradient = dummy_grad
            self.last_gradient_norm = max(torch.norm(dummy_grad).item(), 1e-8)
    
    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)
        
    def estimate_gradient_norm(self):
        """Improved EST-P with gradient history"""
        if not self.grad_history:
            return self.last_gradient_norm
            
        try:
            # Use exponentially weighted average
            weights = [0.5**i for i in range(len(self.grad_history))]
            total_weight = sum(weights)
            weighted_sum = 0.0
            
            for i, grad in enumerate(reversed(self.grad_history)):
                grad_norm = torch.norm(grad).item()
                if not np.isfinite(grad_norm):
                    grad_norm = 1e-8
                weighted_sum += weights[i] * grad_norm
                
            return max(weighted_sum / total_weight, 1e-8)
        except:
            return self.last_gradient_norm

    def compute_gradient(self, current_round):
        if not self.data_indices:
            self.logger.warning("No data available, returning zero gradient")
            zero_grad = torch.zeros(self.model_dimension(), device=self.device)
            self._update_gradient(zero_grad, current_round)
            return zero_grad
        
        try:
            # Dynamically adjust batch size if needed
            actual_batch_size = min(self.Ak, len(self.data_indices))
            if actual_batch_size < 5:  # Minimum viable batch size
                actual_batch_size = min(5, len(self.data_indices))
                self.logger.warning(f"Client {self.client_id} using reduced batch size {actual_batch_size}")
            
            # Select random mini-batch
            indices = random.sample(self.data_indices, actual_batch_size)
            subset = Subset(self.train_dataset, indices)
            loader = DataLoader(subset, batch_size=actual_batch_size)
            images, labels = next(iter(loader))
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.model.zero_grad()
            outputs = self.model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Flatten gradients
            gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.detach().view(-1))
            
            if gradients:
                flat_gradient = torch.cat(gradients)
            else:
                flat_gradient = torch.zeros(self.model_dimension(), device=self.device)
            
            # Gradient clipping to prevent explosions (max norm = 1.0)
            grad_norm = torch.norm(flat_gradient)
            max_norm = 1.0
            if grad_norm > max_norm:
                flat_gradient = flat_gradient * (max_norm / grad_norm)
                self.logger.info(f"Gradient clipped: {grad_norm.item():.2f} > {max_norm}")
            
            self._update_gradient(flat_gradient, current_round)
            return flat_gradient
            
        except Exception as e:
            self.logger.error(f"Gradient computation failed: {str(e)}")
            zero_grad = torch.zeros(self.model_dimension(), device=self.device)
            self._update_gradient(zero_grad, current_round)
            return zero_grad
    
    def _update_gradient(self, gradient, current_round):
        """Update gradient state and history"""
        self.last_gradient = gradient
        norm = torch.norm(gradient).item()
        
        if not np.isfinite(norm) or norm < 1e-8:
            self.logger.warning(f"Invalid gradient norm: {norm}, using 0.1")
            norm = 0.1
            
        self.last_gradient_norm = norm
        self.last_update_round = current_round
        
        # Only keep meaningful updates in history
        if norm > 1e-4:  
            self.grad_history.append(gradient.clone())

    def model_dimension(self):
        return sum(p.numel() for p in self.model.parameters())

    def compute_energy_estimate(self, sigma_t):
        """Estimated energy before computation (Joules)"""
        if self.h_t_k is None:
            return 0.0
            
        try:
            h_mag = max(abs(self.h_t_k), 1e-8)
            grad_norm_est = self.estimate_gradient_norm()
            
            # Communication energy (Eq.6 in paper)
            E_comm = (sigma_t**2 / h_mag**2) * (grad_norm_est**2)
            
            # Computation energy (Eq.4 in paper)
            E_comp = self.en * self.Ak
            
            # Total energy (Eq.7 in paper)
            return E_comm + E_comp
        except Exception as e:
            self.logger.error(f"Energy estimate failed: {str(e)}")
            return 0.0

    # client_sync.py
    def compute_actual_energy(self, sigma_t):
        if self.h_t_k is None or self.last_gradient is None:
            return 0.0
        
        # PHYSICS-CORRECTED FORMULA (Eq.6 with dimension scaling)
        actual_norm = torch.norm(self.last_gradient).item()
        h_mag = max(abs(self.h_t_k), 1e-8)
        
        # Critical scaling factor for large models
        scaling_factor = 1 / math.sqrt(self.model_dimension())
        
        E_comm = (sigma_t**2 / h_mag**2) * (actual_norm**2) * scaling_factor
        E_comp = self.en * self.Ak
        return E_comm + E_comp

    def set_channel_gain(self, current_round):
        """Improved channel model with temporal correlation"""
        try:
            # Add temporal correlation (0.7 correlation factor)
            if self.h_t_k is None or current_round - self.last_update_round > 3:
                # Full randomization if no recent updates
                magnitude = np.random.rayleigh(scale=0.5)  # Reduced scale
            else:
                # Correlated with previous state
                prev_mag = abs(self.h_t_k)
                magnitude = 0.7 * prev_mag + 0.3 * np.random.rayleigh(scale=0.5)
            
            phase = np.random.uniform(0, 2*np.pi)
            self.h_t_k = magnitude * np.exp(1j * phase)
            return abs(self.h_t_k)
        except:
            self.h_t_k = 1.0 + 0.1j  # Default channel
            return 1.0