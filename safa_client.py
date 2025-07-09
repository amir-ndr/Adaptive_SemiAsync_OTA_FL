import torch
import numpy as np
import copy
import time
import logging
import random
from typing import Tuple, Optional, Dict, List
from torch.utils.data import Subset, DataLoader

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SAFAClient:
    def __init__(self, client_id: int, data_indices: list, model: torch.nn.Module,
                 fk: float, mu_k: float, P_max: float, C: float, Ak: int,
                 train_dataset: torch.utils.data.Dataset, device: str = 'cpu',
                 local_epochs: int = 1, crash_prob: float = 0.1):
        """
        Fixed SAFA Client implementation with proper training and delta calculation
        
        Args:
            client_id: Unique identifier for the client
            data_indices: Indices of local training data
            model: Initial model architecture
            fk: CPU frequency in Hz
            mu_k: Hardware efficiency coefficient (J/FLOP)
            P_max: Maximum transmission power in Watts
            C: FLOPs per data sample
            Ak: Local batch size
            train_dataset: Reference to training dataset
            device: Computation device ('cpu' or 'cuda')
            local_epochs: Number of local epochs
            crash_prob: Probability of client failure during round
        """
        self.client_id = client_id
        self.data_indices = list(data_indices)
        self.original_model = copy.deepcopy(model).to(device)
        self.local_model = copy.deepcopy(model).to(device)
        self.device = device
        self.local_epochs = local_epochs
        
        # Hardware parameters
        self.fk = float(fk)
        self.mu_k = float(mu_k)
        self.C = float(C)  # FLOPs per sample
        self.Ak = int(Ak)
        
        # Communication parameters
        self.P_max = float(P_max)
        self.bandwidth = 20e6  # 20 Mbps
        self.tx_power = min(0.2, self.P_max)  # Realistic mobile power
        
        # Training data
        self.train_dataset = train_dataset
        
        # SAFA state tracking
        self.version = 0
        self.crash_prob = min(max(float(crash_prob), 0), 1)
        self.last_selected_round = -1
        self.rounds_missed = 0
        self.staleness = 0
        self.state = "active"
        self.ready = True
        self.cached_update = None
        
        # Training artifacts
        self.last_update = None
        self.update_norm = 0.0
        self.last_comp_time = 0.0
        
        # Energy tracking
        self.total_energy = 0.0
        self.computation_energy = 0.0
        self.communication_energy = 0.0
        
        logger.info(f"SAFA Client {client_id} initialized | "
                   f"CPU: {fk/1e9:.2f}GHz | "
                   f"Crash: {crash_prob:.1%} | "
                   f"Data: {len(data_indices)} samples")

    def update_model(self, model_state_dict: dict, new_version: int) -> None:
        """Synchronize client model with server version"""
        self.original_model.load_state_dict(model_state_dict)
        self.local_model.load_state_dict(model_state_dict)
        old_version = self.version
        self.version = new_version
        self.staleness = 0
        logger.debug(f"Client {self.client_id} updated v{old_version}→v{new_version}")

    def get_state(self, global_version: int, tau: int) -> str:
        """Determine client state based on SAFA definitions"""
        lag = global_version - self.version
        if lag == 0:
            return "up-to-date"
        elif lag > tau:
            return "deprecated"
        else:
            return "tolerable"

    def compute_update(self, current_global_version: int) -> Tuple[Optional[torch.Tensor], float, bool]:
        """Perform full local training and compute model delta"""
        self.staleness = current_global_version - self.version
        self.last_update = None
        self.update_norm = 0.0
        start_time = time.time()
        
        try:
            # Pre-computation crash check
            if random.random() < self.crash_prob:
                raise RuntimeError("Pre-computation crash")
            
            # Handle empty local dataset
            if not self.data_indices:
                self.last_update = torch.zeros(self._model_dimension(), device=self.device)
                return self.last_update, 0.0, True
            
            # Store initial state for delta calculation
            initial_state = copy.deepcopy(self.local_model.state_dict())
            
            # Create DataLoader for local data
            subset = Subset(self.train_dataset, self.data_indices)
            loader = DataLoader(subset, batch_size=self.Ak, shuffle=True)
            
            # Training setup
            optimizer = torch.optim.Adam(self.local_model.parameters(), lr=0.001)
            
            # Full local training
            for epoch in range(self.local_epochs):
                for images, labels in loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # Mid-batch crash check
                    if random.random() < self.crash_prob/10:
                        raise RuntimeError(f"Mid-batch crash @ epoch {epoch+1}")
                    
                    optimizer.zero_grad()
                    outputs = self.local_model(images)
                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                    loss.backward()
                    
                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 1.0)
                    
                    optimizer.step()
            
            # Calculate model delta (current weights - initial weights)
            current_state = self.local_model.state_dict()
            model_delta = []
            for key in initial_state:
                model_delta.append((current_state[key] - initial_state[key]).view(-1))
            model_delta = torch.cat(model_delta)
            
            # Track artifacts
            self.last_update = model_delta
            self.update_norm = torch.norm(model_delta).item()
            self.last_comp_time = time.time() - start_time
            
            # Calculate computation energy (FLOP-based)
            samples_processed = len(self.data_indices) * self.local_epochs
            total_flops = samples_processed * self.C
            self.computation_energy = self.mu_k * total_flops * (self.fk**2)
            
            # Increment local version after successful training
            self.version += 1
            
            logger.info(f"Client {self.client_id} update ready | "
                       f"Norm: {self.update_norm:.4f} | "
                       f"Time: {self.last_comp_time:.4f}s | "
                       f"Energy: {self.computation_energy:.4f}J")
            
            return self.last_update, self.last_comp_time, True
            
        except RuntimeError as e:
            logger.error(f"Client {self.client_id} failed: {str(e)}")
            self.state = "crashed"
            return None, 0.0, False

    def transmit_update(self) -> Tuple[float, bool]:
        """Transmit model update to server"""
        if self.state == "crashed" or self.last_update is None:
            return 0.0, False
            
        try:
            # Transmission failure simulation
            if random.random() < self.crash_prob/2:
                raise RuntimeError("Transmission failure")
            
            # Calculate transmission metrics
            model_size_bits = self.last_update.numel() * 32  # 32 bits per float
            tx_time = model_size_bits / self.bandwidth
            self.communication_energy = self.tx_power * tx_time
            self.total_energy = self.computation_energy + self.communication_energy
            
            logger.debug(f"Client {self.client_id} transmitted | "
                         f"Size: {model_size_bits/8e3:.2f}KB | "
                         f"Time: {tx_time:.4f}s | "
                         f"Power: {self.tx_power:.4f}W")
            
            return tx_time, True
            
        except RuntimeError as e:
            logger.error(f"Client {self.client_id} transmission failed: {str(e)}")
            self.state = "crashed"
            return 0.0, False

    def cache_update(self, update: torch.Tensor) -> None:
        """Store update in bypass cache"""
        self.cached_update = update.clone()
        logger.debug(f"Client {self.client_id} cached update")

    def get_cached_update(self) -> Optional[torch.Tensor]:
        """Retrieve cached update"""
        return self.cached_update.clone() if self.cached_update is not None else None

    def _model_dimension(self) -> int:
        """Calculate total model parameter count"""
        return sum(p.numel() for p in self.local_model.parameters())

    def reset_for_round(self) -> None:
        """Prepare client for new training round"""
        self.state = "active"
        self.computation_energy = 0.0
        self.communication_energy = 0.0
        self.total_energy = 0.0
        logger.debug(f"Client {self.client_id} reset for new round")

    def mark_round_missed(self) -> None:
        """Increment missed round counter"""
        self.rounds_missed += 1
        logger.debug(f"Client {self.client_id} missed round (total {self.rounds_missed})")

    def get_priority_score(self) -> float:
        """Priority for compensatory selection"""
        return 1.0 + self.rounds_missed * 0.5

    def get_energy_report(self) -> Dict[str, float]:
        """Get energy consumption report"""
        return {
            "total": float(self.total_energy),
            "computation": float(self.computation_energy),
            "communication": float(self.communication_energy),
            "efficiency": float(self.total_energy / (self._model_dimension() + 1e-8))
        }

    def get_client_status(self) -> Dict[str, float]:
        """Get client state metrics"""
        return {
            "version": int(self.version),
            "staleness": int(self.staleness),
            "update_norm": float(self.update_norm),
            "last_comp_time": float(self.last_comp_time),
            "state": str(self.state),
            "rounds_missed": int(self.rounds_missed)
        }