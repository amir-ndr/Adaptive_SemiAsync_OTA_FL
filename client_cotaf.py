import torch
import numpy as np
import copy
import time
import math
import logging
from torch.utils.data import Subset, DataLoader

class COTAFClient:
    def __init__(self, client_id, data_indices, model, fk, mu_k, P_max, C, Ak, 
                 train_dataset, device='cpu', local_epochs=1):
        self.client_id = client_id
        self.data_indices = data_indices
        self.device = device
        self.last_grad_norm = 0.0  # Track gradient norm
        self.local_model = copy.deepcopy(model).to(self.device)
        logging.info(f"Client {client_id}: Model initialized on {device}")
        
        self.fk = fk
        self.mu_k = mu_k
        self.P_max = P_max
        self.C = C
        self.Ak = Ak
        self.train_dataset = train_dataset
        self.local_epochs = local_epochs
        
        # State management
        self.global_model_start = None
        self.local_model_trained = None
        self.train_loader = self._create_data_loader()
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.1)
        logging.info(f"Client {client_id}: Optimizer initialized")
        
        # Energy tracking
        self.comp_energy = 0.0
        self.comm_energy = 0.0
        self.norm_squared = 0.0
        self.flops_completed = 0
        self.last_tx_duration = 0.0
        
    def _create_data_loader(self):
        subset = Subset(self.train_dataset, self.data_indices)
        loader = DataLoader(subset, batch_size=self.Ak, shuffle=True)
        logging.info(f"Client {self.client_id}: Created loader with {len(subset)} samples")
        return loader
    
    def set_model(self, state_dict):
        logging.info(f"Client {self.client_id}: Setting new model")
        try:
            # Log model parameter stats before update
            pre_update = next(iter(self.local_model.parameters())).data.mean().item()
            
            # Move state_dict to the same device as model
            device_state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
            self.local_model.load_state_dict(device_state_dict)
            
            # Log model parameter stats after update
            post_update = next(iter(self.local_model.parameters())).data.mean().item()
            logging.info(f"Client {self.client_id}: Model updated. Param change: {pre_update:.4f} → {post_update:.4f}")
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error setting model: {str(e)}")
            raise

    # def compute_gradient_norm(self):
    #     """Compute L2 norm of model gradients"""
    #     total_norm = 0.0
    #     for param in self.local_model.parameters():
    #         if param.grad is not None:
    #             param_norm = param.grad.data.norm(2)
    #             total_norm += param_norm.item() ** 2
    #     return math.sqrt(total_norm)
    def compute_gradient_norm(self):
        """Compute true L2 norm of full gradient vector"""
        gradients = []
        for param in self.local_model.parameters():
            if param.grad is not None:
                # Flatten and concatenate all gradients
                gradients.append(param.grad.detach().view(-1))
        if not gradients:
            return 0.0
        full_gradient = torch.cat(gradients)
        return torch.norm(full_gradient).item()
        
    def local_train(self):
        logging.info(f"Client {self.client_id}: Starting local training")
        self.local_model.train()
        total_loss = 0.0
        num_samples = 0
        
        # Save initial model BEFORE training
        initial_state = copy.deepcopy(self.local_model.state_dict())
        self.global_model_start = initial_state
        logging.info(f"Client {self.client_id}: Saved initial model state")
        
        # Training loop - PAPER-COMPATIBLE: ONE BATCH PER ROUND
        start_time = time.time()
        max_grad = 0.0
        try:
            # Get single batch (matches paper's one SGD step)
            data, labels = next(iter(self.train_loader))
            data, labels = data.to(self.device), labels.to(self.device)
            batch_size = data.size(0)
            
            self.optimizer.zero_grad()
            outputs = self.local_model(data)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            
            # Store gradient norm BEFORE stepping
            self.last_grad_norm = self.compute_gradient_norm()
            
            # Gradient monitoring
            for param in self.local_model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    if grad_norm > max_grad:
                        max_grad = grad_norm
            
            self.optimizer.step()
            total_loss = loss.item() * batch_size
            num_samples = batch_size
                
        except Exception as e:
            logging.error(f"Client {self.client_id}: Training error: {str(e)}")
            return float('inf')
        
        # Compute FLOPs and energy - PAPER-COMPATIBLE: ONE BATCH
        self.flops_completed = self.C * num_samples  # No local_epochs multiplier
        duration = time.time() - start_time
        self.comp_energy = self.mu_k * (self.fk ** 2) * self.flops_completed
        
        # Update model states
        self.local_model_trained = copy.deepcopy(self.local_model.state_dict())
        logging.info(f"Client {self.client_id}: Training complete. Loss: {loss.item():.4f}, "
                     f"Grad norm: {self.last_grad_norm:.4f}, Samples: {num_samples}, FLOPs: {self.flops_completed/1e6:.2f}M")
        
        return loss.item()
    
    def get_update(self, tx_duration=0.1):
        logging.info(f"Client {self.client_id}: Getting update")
        self.last_tx_duration = tx_duration
        
        if self.global_model_start is None or self.local_model_trained is None:
            logging.warning(f"Client {self.client_id}: Missing model states for update")
            return None
            
        update = {}
        self.norm_squared = 0.0
        try:
            # Create gradient-mimic update: θ_new - θ_old
            delta_dict = {}
            for key in self.local_model_trained:
                start_cpu = self.global_model_start[key].cpu()
                trained_cpu = self.local_model_trained[key].cpu()
                delta_dict[key] = trained_cpu - start_cpu
            
            # Compute full-model norm (for power constraint)
            full_norm = torch.norm(torch.cat([t.flatten() for t in delta_dict.values()])).item()
            scaling = min(1.0, math.sqrt(self.P_max) / (full_norm + 1e-8))
            
            # Apply uniform scaling
            for key in delta_dict:
                update[key] = scaling * delta_dict[key]
                param_norm = torch.norm(update[key]).item()
                self.norm_squared += param_norm ** 2
                
                # Log large updates
                if param_norm > 1.0:
                    logging.warning(f"Client {self.client_id}: Large update for {key}: {param_norm:.4f}")
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error creating update: {str(e)}")
            return None
            
        logging.info(f"Client {self.client_id}: Update norm: {np.sqrt(self.norm_squared):.4f}")
        return update
    
    def get_energy_consumption(self, p_k, h_k):
        h_mag = max(abs(h_k), 1e-8)
        true_grad_norm = math.sqrt(self.last_grad_norm)  # Already true norm
        
        # PAPER FORMULA: E_comm = (p_k^2 * ||g_k||^2) / |h_k|^2
        comm_energy = (p_k ** 2) * (true_grad_norm ** 2) / (h_mag ** 2)
        
        total_energy = self.comp_energy + comm_energy
        
        logging.info(f"Client {self.client_id}: Energy | "
                    f"Comp: {self.comp_energy:.2f}J | "
                    f"Comm: {comm_energy:.2f}J | "
                    f"Total: {total_energy:.2f}J | "
                    f"P_k: {p_k:.4f} |H|: {h_mag:.4f} ||g||: {true_grad_norm:.4f}")
        return total_energy

    def reset_round(self):
        logging.info(f"Client {self.client_id}: Resetting round")
        self.comp_energy = 0.0
        self.comm_energy = 0.0
        self.norm_squared = 0.0
        self.flops_completed = 0
        self.last_tx_duration = 0.0
        self.last_grad_norm = 0.0