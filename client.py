import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy
import time
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Client:
    def __init__(self, client_id, data_indices, model, fk, mu_k, P_max, C, Ak, 
                 train_dataset, device='cpu', local_epochs=1):
        self.client_id = client_id
        self.data_indices = data_indices
        self.stale_model = copy.deepcopy(model).to(device)
        self.local_model = copy.deepcopy(model).to(device)
        self.fk = fk
        self.mu_k = mu_k
        self.P_max = P_max
        self.C = C
        self.Ak = Ak
        self.train_dataset = train_dataset
        self.device = device
        self.local_epochs = local_epochs
        
        # State variables
        self.dt_k = self._full_computation_time()
        self.tau_k = 0
        self.last_gradient = None
        self.gradient_norm = 1.0
        self.h_t_k = None
        self.ready = (self.dt_k <= 0)

        logger.info(f"Client {client_id} initialized | "
                    f"CPU: {fk/1e9:.2f} GHz | "
                    f"Batch: {Ak} samples | "
                    f"Comp Time: {self.dt_k:.4f}s")

    def _full_computation_time(self):
        return (self.C * self.Ak * self.local_epochs) / self.fk

    def update_stale_model(self, model_state_dict):
        self.stale_model.load_state_dict(model_state_dict)
        self.local_model.load_state_dict(model_state_dict)
        self.tau_k = 0
        logger.debug(f"Client {self.client_id}: Model updated | "
                     f"Staleness reset to 0")

    def compute_gradient(self):
        start_time = time.time()
        self.local_model.load_state_dict(self.stale_model.state_dict())
        
        # Create DataLoader for entire local dataset
        dataset = [self.train_dataset[i] for i in self.data_indices]
        images = torch.stack([item[0] for item in dataset])
        labels = torch.tensor([item[1] for item in dataset])
        data_loader = DataLoader(TensorDataset(images, labels), 
                                batch_size=self.Ak, shuffle=True)
        
        total_gradient = None
        total_samples = 0
        
        for batch in data_loader:
            batch_images, batch_labels = batch[0].to(self.device), batch[1].to(self.device)
            
            # Forward pass
            self.local_model.zero_grad()
            outputs = self.local_model(batch_images)
            loss = torch.nn.functional.cross_entropy(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradients
            with torch.no_grad():
                batch_gradient = []
                for param in self.local_model.parameters():
                    batch_gradient.append(param.grad.clone().view(-1))
                flat_batch_grad = torch.cat(batch_gradient)
                
                if total_gradient is None:
                    total_gradient = flat_batch_grad
                else:
                    total_gradient += flat_batch_grad
                    
                total_samples += batch_images.size(0)
        
        # Compute average gradient
        avg_gradient = total_gradient / total_samples
        self.last_gradient = avg_gradient
        self.gradient_norm = torch.norm(avg_gradient).item()
        
        logger.info(f"Client {self.client_id}: Gradient computed | "
                    f"Norm: {self.gradient_norm:.4f} | "
                    f"Batches: {len(data_loader)}")
        return True

    def compute_gradient_(self):
        """Compute average gradient over multiple local iterations"""
        start_time = time.time()
        self.local_model.load_state_dict(self.stale_model.state_dict())
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.01)
        
        # Store gradients from all batches
        all_gradients = []
        
        for epoch in range(self.local_epochs):
            indices = np.random.choice(self.data_indices, size=self.Ak, replace=False)
            batch = [self.train_dataset[i] for i in indices]
            images = torch.stack([x[0] for x in batch]).to(self.device)
            labels = torch.tensor([x[1] for x in batch]).to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.local_model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            
            # Backward pass - compute gradients WITHOUT updating model
            loss.backward()
            
            # Extract and store current gradients
            epoch_grad = []
            for param in self.local_model.parameters():
                epoch_grad.append(param.grad.clone().view(-1))
            all_gradients.append(torch.cat(epoch_grad))
        
        # Compute average gradient across all iterations
        avg_gradient = torch.stack(all_gradients).mean(dim=0)
        
        self.last_gradient = avg_gradient
        self.gradient_norm = torch.norm(avg_gradient).item()
        self.actual_comp_time = time.time() - start_time

        return True

    def update_computation_time(self, D_t):
        self.dt_k = max(0, self.dt_k - D_t)
        self.ready = (self.dt_k <= 0)
        logger.debug(f"Client {self.client_id}: Comp time updated | "
                     f"Remaining: {self.dt_k:.4f}s | "
                     f"Ready: {self.ready}")
        return self.ready

    def reset_computation(self):
        self.dt_k = self._full_computation_time()
        self.ready = (self.dt_k <= 0)
        logger.debug(f"Client {self.client_id}: Comp time reset | "
                     f"New: {self.dt_k:.4f}s")

    def increment_staleness(self):
        self.tau_k += 1
        logger.debug(f"Client {self.client_id}: Staleness incremented | "
                     f"New: {self.tau_k}")

    def set_channel_gain(self):
        gain = complex(np.random.normal(0, 1), np.random.normal(0, 1))
        self.h_t_k = gain / np.sqrt(2)
        channel_mag = abs(self.h_t_k)
        logger.debug(f"Client {self.client_id}: Channel set | "
                     f"|h|: {channel_mag:.4f}")
        return channel_mag
    
    def reset_staleness(self):
        self.tau_k = 0
        logger.debug(f"Client {self.client_id}: Staleness reset to 0")