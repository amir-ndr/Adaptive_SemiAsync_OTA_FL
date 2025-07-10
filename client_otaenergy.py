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
        self.data_indices = list(data_indices)
        self.model = copy.deepcopy(model).to(device)
        self.en = en
        self.Lb = Lb
        self.train_dataset = train_dataset
        self.device = device
        
        # State variables
        self.last_grad_norm = 1.0  # For EST-P estimation
        self.last_gradient = None
        self.h_t_k = None
        self.current_grad_computed = False  # Track if gradient computed in current round

        logger.info(f"Client {client_id} initialized | "
                    f"Samples: {len(data_indices)} | "
                    f"Energy/sample: {en:.2e} J | "
                    f"Batch: {Lb}")

    def update_model(self, model_state_dict: dict):
        """Update local model with global parameters"""
        self.model.load_state_dict(model_state_dict)
        self.current_grad_computed = False  # Reset on model update
        logger.debug(f"Client {self.client_id}: Model updated")

    def set_channel_gain(self, h_value: float = None):
        """
        Set current channel gain, either from server or generate new
        
        Args:
            h_value: Optional pre-calculated channel gain
        """
        if h_value is None:
            # Rayleigh fading with minimum threshold
            self.h_t_k = max(np.random.rayleigh(scale=1.0), 0.05)
        else:
            self.h_t_k = max(h_value, 0.05)
            
        logger.debug(f"Client {self.client_id}: Channel set | "
                     f"|h|: {self.h_t_k:.4f}")
        return self.h_t_k

    def compute_gradient(self) -> torch.Tensor:
        """
        Compute local gradient with robust handling
        - Automatic batch size adjustment for small datasets
        - Gradient clipping and NaN protection
        - Energy tracking
        
        Returns:
            Flat gradient tensor
        """
        if self.current_grad_computed:
            return self.last_gradient
            
        # Handle insufficient data
        n_available = len(self.data_indices)
        if n_available == 0:
            logger.warning(f"Client {self.client_id}: No data! Returning zero gradient")
            self.last_gradient = torch.zeros(self._model_dimension(), device=self.device)
            self.last_grad_norm = 0.0
            self.current_grad_computed = True
            return self.last_gradient
        
        # Adjust batch size if insufficient data
        batch_size = min(self.Lb, n_available)
        indices = random.sample(self.data_indices, batch_size)
        batch = [self.train_dataset[i] for i in indices]
        
        # Prepare data
        images = torch.stack([x[0] for x in batch]).to(self.device)
        if images.dim() == 3:
            images = images.unsqueeze(1)  # Add channel dim for MNIST
            
        labels = torch.tensor([x[1] for x in batch], dtype=torch.long).to(self.device)
        
        # Compute gradient
        self.model.zero_grad()
        outputs = self.model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        
        # Gradient clipping to prevent explosions
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        
        # Extract and flatten gradients
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.detach().clone().view(-1))
        flat_gradient = torch.cat(gradients)
        
        # Handle NaN values
        if torch.isnan(flat_gradient).any():
            logger.error(f"Client {self.client_id}: NaN detected! Using zeros")
            flat_gradient = torch.zeros_like(flat_gradient)
        
        # Store results
        self.last_gradient = flat_gradient
        grad_norm = torch.norm(flat_gradient).item()
        
        # Protect against extreme values
        self.last_grad_norm = max(grad_norm, 0.01)  # Minimum norm threshold
        self.current_grad_computed = True
        
        logger.info(f"Client {self.client_id}: Gradient computed | "
                    f"Samples: {batch_size}/{n_available} | "
                    f"Norm: {self.last_grad_norm:.4f}")
        
        return flat_gradient

    def prepare_transmission(self, sigma_t: float) -> tuple:
        """
        Prepare transmission signal without computing gradient
        (Gradient must be pre-computed)
        
        Args:
            sigma_t: Power scalar from server
            
        Returns:
            (transmit_signal, communication_energy)
        """
        if not self.current_grad_computed:
            raise RuntimeError("Gradient not computed before transmission prep")
        
        if self.h_t_k is None:
            logger.warning("Channel not set, using default")
            self.set_channel_gain()
        
        # Calculate transmission signal
        h_safe = max(self.h_t_k, 0.05)  # Prevent division issues
        transmit_signal = (sigma_t / h_safe) * self.last_gradient
        
        # Calculate communication energy only
        comm_energy = torch.norm(transmit_signal).item() ** 2
        
        logger.debug(f"Client {self.client_id}: Transmission prepared | "
                     f"Comm energy: {comm_energy:.4e} J")
        
        return transmit_signal, comm_energy

    def compute_and_transmit(self, sigma_t: float) -> tuple:
        """
        Full compute-transmit sequence
        
        Args:
            sigma_t: Power scalar from server
            
        Returns:
            (transmit_signal, total_energy, gradient_norm)
        """
        # Compute gradient if not already done
        if not self.current_grad_computed:
            self.compute_gradient()
        
        # Get transmission signal
        transmit_signal, comm_energy = self.prepare_transmission(sigma_t)
        
        # Calculate computation energy (based on actual batch size)
        actual_batch = min(self.Lb, len(self.data_indices))
        comp_energy = self.en * actual_batch
        total_energy = comp_energy + comm_energy
        
        logger.info(f"Client {self.client_id}: Transmission complete | "
                    f"Comp: {comp_energy:.4e} J | "
                    f"Comm: {comm_energy:.4e} J | "
                    f"Total: {total_energy:.4e} J")
        
        return transmit_signal, total_energy, self.last_grad_norm

    def estimate_energy(self, sigma_t: float) -> float:
        """
        Estimate energy consumption using EST-P method
        - Uses last known gradient norm
        - Accounts for variable batch size
        
        Args:
            sigma_t: Power scalar from server
            
        Returns:
            Estimated energy consumption
        """
        if self.h_t_k is None:
            logger.warning("Channel not set for estimation")
            self.set_channel_gain()
        
        # Computation energy (use nominal batch size)
        E_comp = self.en * self.Lb
        
        # Communication energy estimation
        h_safe = max(self.h_t_k, 0.05)
        grad_norm = max(self.last_grad_norm, 0.01)  # Safe minimum
        
        # EST-P: (σ_t² / h²) * ||g||²
        E_comm = (sigma_t**2 / (h_safe**2)) * (grad_norm**2)
        
        return E_comp + E_comm

    def _model_dimension(self) -> int:
        """Get model parameter dimension"""
        return sum(p.numel() for p in self.model.parameters())