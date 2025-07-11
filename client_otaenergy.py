import torch
import numpy as np
import copy
import logging
import random
from torch.utils.data import Dataset

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
        self.data_indices = list(data_indices)  # Ensure list conversion
        self.model = copy.deepcopy(model).to(device)
        self.en = en
        self.Lb = Lb
        self.train_dataset = train_dataset
        self.device = device
        
        # State variables
        self.last_grad_norm = 1.0
        self.true_h = None      # True channel gain (for transmission)
        self.observed_h = None  # Observed channel gain (for scheduling)
        
        logger.info(f"Client {client_id} initialized | "
                    f"Data samples: {len(data_indices)} | "
                    f"Energy/sample: {en:.2e} J | "
                    f"Batch size: {Lb}")

    def update_model(self, model_state_dict: dict):
        """Update local model with global parameters"""
        self.model.load_state_dict(model_state_dict)
        logger.debug(f"Client {self.client_id}: Model updated")

    def set_channel_gain(self):
        """Set current channel gains with observation error"""
        # True channel (Rayleigh fading)
        true_magnitude = np.random.rayleigh(scale=1/np.sqrt(2))
        true_phase = np.random.uniform(0, 2*np.pi)
        self.true_h = true_magnitude * np.exp(1j * true_phase)
        
        # Observed channel (with error)
        error_factor = 1 + np.random.normal(0, 0.1)  # 10% observation error
        observed_magnitude = true_magnitude * error_factor
        self.observed_h = observed_magnitude * np.exp(1j * true_phase)
        
        logger.debug(f"Client {self.client_id}: Channel set | "
                     f"True |h|: {true_magnitude:.4f} | "
                     f"Observed |h|: {observed_magnitude:.4f}")
        return abs(self.observed_h)  # Return observed for scheduling

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
            self.last_grad_norm = 1e-8  # Safe minimum instead of 0
            return self.last_gradient
        
        # Select random mini-batch
        batch_size = min(self.Lb, n_available)
        indices = random.sample(self.data_indices, batch_size)
        batch = [self.train_dataset[i] for i in indices]
        
        # Prepare data
        images = torch.stack([x[0] for x in batch]).to(self.device)
        labels = torch.tensor([x[1] for x in batch]).to(self.device)
        
        # Compute gradient
        self.model.zero_grad()
        outputs = self.model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        
        # Extract and flatten gradients
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.detach().view(-1))
        flat_gradient = torch.cat(gradients)
        
        # Store results
        self.last_gradient = flat_gradient
        self.last_grad_norm = torch.norm(flat_gradient).item()
        
        logger.info(f"Client {self.client_id}: Gradient computed | "
                    f"Norm: {self.last_grad_norm:.4f}")
        
        return flat_gradient

    def compute_and_transmit(self, sigma_t: float) -> tuple:
        """
        Compute gradient and prepare transmission signal with energy calculation
        
        Args:
            sigma_t: Power scalar from server
            
        Returns:
            (transmit_signal, actual_energy, gradient_norm)
        """
        # Compute gradient (sets self.last_gradient and self.last_grad_norm)
        gradient = self.compute_gradient()
        
        # Use TRUE channel for transmission
        if self.true_h is None:
            self.set_channel_gain()
        h_mag = max(abs(self.true_h), 1e-8)  # Use true channel, prevent div0
        
        # Prepare transmission signal (Eq 5)
        transmit_signal = (sigma_t / h_mag) * gradient
        
        # Calculate actual energy consumption (Eq 7)
        E_comp = self.en * self.Lb
        E_comm = (sigma_t**2 / (h_mag**2)) * (self.last_grad_norm**2)
        actual_energy = E_comp + E_comm
        
        logger.info(f"Client {self.client_id}: Transmission prepared | "
                    f"Comp energy: {E_comp:.4e} J | "
                    f"Comm energy: {E_comm:.4e} J | "
                    f"Total energy: {actual_energy:.4e} J")
        
        return transmit_signal, actual_energy, self.last_grad_norm

    def estimate_energy(self, sigma_t: float) -> float:
        """
        Estimate energy consumption using EST-P method (Eq 26)
        Uses last known gradient norm and OBSERVED channel gain
        
        Args:
            sigma_t: Power scalar from server
            
        Returns:
            Estimated energy consumption
        """
        # Use OBSERVED channel for estimation
        if self.observed_h is None:
            self.set_channel_gain()
        h_mag = max(abs(self.observed_h), 1e-8)  # Prevent div0
        
        E_comp = self.en * self.Lb
        # EST-P: Use last gradient norm for estimation
        E_comm = (sigma_t**2 / (h_mag**2)) * (self.last_grad_norm**2)
        
        return E_comp + E_comm

    def _model_dimension(self) -> int:
        """Get model parameter dimension"""
        return sum(p.numel() for p in self.model.parameters())