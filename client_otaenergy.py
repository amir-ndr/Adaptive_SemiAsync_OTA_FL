import torch
import numpy as np
import logging
import random
from torch.utils.data import Dataset
import copy

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SyncOTAClient:
    def __init__(self, client_id: int, data_indices: list, 
                 model: torch.nn.Module, en: float, Lb: int,
                 train_dataset: Dataset, device: str = 'cpu'):
        """
        Initialize a synchronous OTA client.
        
        Args:
            client_id: Unique client identifier
            data_indices: Indices of client's data in the global dataset
            model: Initial global model
            en: Energy per data sample (Joules)
            Lb: Batch size for local gradient computation
            train_dataset: Reference to training dataset
            device: Computation device ('cpu' or 'cuda')
        """
        self.client_id = client_id
        self.data_indices = list(data_indices)  # Ensure list type
        self.model = copy.deepcopy(model).to(device)
        self.en = en
        self.Lb = Lb
        self.train_dataset = train_dataset
        self.device = device
        
        # State variables
        self.last_grad_norm = 1.0  # For EST-P estimation (initial safe value)
        self.last_gradient = None   # Initialize gradient storage
        self.h_t_k = None  # Current channel gain
        
        logger.info(f"Client {client_id} initialized | "
                    f"Data samples: {len(data_indices)} | "
                    f"Energy/sample: {en:.2e} J | "
                    f"Batch size: {Lb}")

    def update_model(self, model_state_dict: dict):
        """Update local model with global parameters"""
        self.model.load_state_dict(model_state_dict)
        logger.debug(f"Client {self.client_id}: Model updated")

    def set_channel_gain(self):
        """Set current channel gain using Rayleigh fading (real-valued)"""
        # Real-valued channel (magnitude only)
        self.h_t_k = np.random.rayleigh(scale=1.0)
        logger.debug(f"Client {self.client_id}: Channel set | "
                     f"|h|: {self.h_t_k:.4f}")
        return self.h_t_k

    def compute_gradient(self) -> torch.Tensor:
        """
        Compute local gradient using current model and local data.
        Stores gradient norm for future EST-P estimation.
        
        Returns:
            Flat gradient tensor
        """
        # Handle insufficient data
        n_available = len(self.data_indices)
        if n_available == 0:
            logger.warning(f"Client {self.client_id} has no data! Returning zero gradient")
            self.last_gradient = torch.zeros(self._model_dimension(), device=self.device)
            self.last_grad_norm = 0.0
            return self.last_gradient
        
        # Select random mini-batch
        batch_size = min(self.Lb, n_available)
        indices = random.sample(self.data_indices, batch_size)
        batch = [self.train_dataset[i] for i in indices]
        
        # Prepare data
        images = torch.stack([x[0] for x in batch]).to(self.device)
        if images.dim() == 3:
            images = images.unsqueeze(1)  # Add channel dimension for MNIST
            
        labels = torch.tensor([x[1] for x in batch], dtype=torch.long).to(self.device)
        
        # Compute gradient
        self.model.zero_grad()
        outputs = self.model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        
        # Extract and flatten gradients
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.detach().clone().view(-1))
        flat_gradient = torch.cat(gradients)
        
        # Store results
        self.last_gradient = flat_gradient
        self.last_grad_norm = torch.norm(flat_gradient).item()
        
        # Gradient sanity checks
        if torch.isnan(flat_gradient).any():
            logger.error(f"Client {self.client_id}: NaN gradient detected!")
            self.last_gradient = torch.zeros_like(flat_gradient)
            self.last_grad_norm = 0.0
        elif self.last_grad_norm < 1e-8:
            logger.warning(f"Client {self.client_id}: Near-zero gradient!")
        
        logger.info(f"Client {self.client_id}: Gradient computed | "
                    f"Samples: {batch_size}/{n_available} | "
                    f"Norm: {self.last_grad_norm:.4f}")
        
        return flat_gradient

    def compute_and_transmit(self, sigma_t: float) -> tuple:
        """
        Compute gradient and prepare transmission signal with energy calculation.
        
        Args:
            sigma_t: Power scalar from server
            
        Returns:
            (transmit_signal, actual_energy, gradient_norm)
        """
        # Compute gradient if not available
        if self.last_gradient is None:
            self.compute_gradient()
        
        # Get current channel magnitude
        if self.h_t_k is None:
            self.set_channel_gain()
        
        # Prepare transmission signal (Eq 5) - RAW GRADIENT
        transmit_signal = (sigma_t / self.h_t_k) * self.last_gradient
        
        # Calculate actual energy consumption (Eq 7)
        E_comp = self.en * self.Lb
        E_comm = torch.norm(transmit_signal).item() ** 2
        actual_energy = E_comp + E_comm
        
        logger.info(f"Client {self.client_id}: Transmission prepared | "
                    f"Comp energy: {E_comp:.4e} J | "
                    f"Comm energy: {E_comm:.4e} J | "
                    f"Total energy: {actual_energy:.4e} J")
        
        return transmit_signal, actual_energy, self.last_grad_norm

    def estimate_energy(self, sigma_t: float) -> float:
        """
        Estimate energy consumption using EST-P method (Eq 26)
        
        Args:
            sigma_t: Power scalar from server
            
        Returns:
            Estimated energy consumption
        """
        if self.h_t_k is None:
            self.set_channel_gain()
        
        E_comp = self.en * self.Lb
        
        # Protect against extreme gradient values
        grad_norm = min(self.last_grad_norm, 10.0)  # Clip large gradients
        grad_norm = max(grad_norm, 0.1)             # Prevent underflow
        
        # EST-P: Use last gradient norm for estimation
        E_comm = (sigma_t**2 / (self.h_t_k**2)) * (grad_norm**2)
        
        return E_comp + E_comm

    def _model_dimension(self) -> int:
        """Get model parameter dimension"""
        return sum(p.numel() for p in self.model.parameters())