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

    def compute_gradient_norm(self):
        """Compute L2 norm of model gradients"""
        total_norm = 0.0
        for param in self.local_model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return math.sqrt(total_norm)
        
    def local_train(self):
        logging.info(f"Client {self.client_id}: Starting local training")
        self.local_model.train()
        total_loss = 0.0
        num_samples = 0
        
        # Save initial model BEFORE training
        initial_state = copy.deepcopy(self.local_model.state_dict())
        self.global_model_start = initial_state
        logging.info(f"Client {self.client_id}: Saved initial model state")
        
        # Training loop
        start_time = time.time()
        max_grad = 0.0
        try:
            for epoch in range(self.local_epochs):
                logging.info(f"Client {self.client_id}: Epoch {epoch+1}/{self.local_epochs}")
                for batch_idx, (data, labels) in enumerate(self.train_loader):
                    # Move data to same device as model
                    data, labels = data.to(self.device), labels.to(self.device)
                    batch_size = data.size(0)
                    
                    self.optimizer.zero_grad()
                    outputs = self.local_model(data)
                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                    loss.backward()
                    
                    # Gradient monitoring
                    for param in self.local_model.parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2).item()
                            if grad_norm > max_grad:
                                max_grad = grad_norm
                    
                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 5.0)
                    self.optimizer.step()
                    
                    total_loss += loss.item() * batch_size
                    num_samples += batch_size
                    
                    # if batch_idx % 10 == 0:
                    #     logging.info(f"Client {self.client_id}: Batch {batch_idx} loss: {loss.item():.4f}")
        except Exception as e:
            logging.error(f"Client {self.client_id}: Training error: {str(e)}")
            return float('inf')
        
        # Compute FLOPs and energy
        self.flops_completed = self.C * num_samples * self.local_epochs
        duration = time.time() - start_time
        self.comp_energy = self.mu_k * (self.fk ** 2) * self.flops_completed
        
        # Update model states
        self.local_model_trained = copy.deepcopy(self.local_model.state_dict())
        logging.info(f"Client {self.client_id}: Training complete. Avg loss: {total_loss/num_samples:.4f}, "
                     f"Max grad: {max_grad:.4f}, Samples: {num_samples}, FLOPs: {self.flops_completed/1e6:.2f}M")
        
        return total_loss / num_samples if num_samples > 0 else float('inf')
    
    def get_update(self, tx_duration=0.1):
        logging.info(f"Client {self.client_id}: Getting update")
        self.last_tx_duration = tx_duration
        
        # Debug state before getting update
        state_debug = f"global_start: {self.global_model_start is not None}, "
        state_debug += f"local_trained: {self.local_model_trained is not None}"
        logging.info(f"Client {self.client_id}: State - {state_debug}")
        
        if self.global_model_start is None or self.local_model_trained is None:
            logging.warning(f"Client {self.client_id}: Missing model states for update")
            return None
            
        update = {}
        self.norm_squared = 0.0
        try:
            for key in self.local_model_trained:
                # Move tensors to CPU for norm calculation
                start_cpu = self.global_model_start[key].cpu()
                trained_cpu = self.local_model_trained[key].cpu()
                
                # scaling = min(1.0, math.sqrt(self.P_max / (self.norm_squared + 1e-8)))
                # delta = scaling * (trained_cpu - start_cpu)  # NEW: Enforce power constraint
                # update[key] = delta
                delta = trained_cpu - start_cpu
                full_norm = torch.norm(torch.cat([t.flatten() for t in delta.values()])).item()
                scaling = min(1.0, math.sqrt(self.P_max) / (full_norm + 1e-8))
                for key in delta:
                    update[key] = scaling * delta[key]


                param_norm = torch.norm(delta).item()
                self.norm_squared += param_norm ** 2
                
                # Log large parameter updates
                if param_norm > 1.0:
                    logging.warning(f"Client {self.client_id}: Large update for {key}: {param_norm:.4f}")
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error creating update: {str(e)}")
            return None
            
        logging.info(f"Client {self.client_id}: Update norm: {np.sqrt(self.norm_squared):.4f}")
        return update
    
    def get_energy_consumption(self, p_k, h_k, grad_norm):
        """
        PAPER-COMPATIBLE ENERGY MODEL
        p_k: Transmit power (scalar)
        h_k: Complex channel coefficient
        grad_norm: ||g_k|| (scalar)
        """
        # 1. Communication energy (Eq. 5 in paper)
        h_mag = max(abs(h_k), 1e-8)  # Avoid division by zero
        comm_energy = (p_k ** 2) * (grad_norm ** 2) / (h_mag ** 2)
        
        # 2. Computation energy (unchanged)
        total_energy = self.comp_energy + comm_energy
        
        logging.info(f"Client {self.client_id}: Energy | "
                    f"Comp: {self.comp_energy:.2f}J | "
                    f"Comm: {comm_energy:.2f}J | "
                    f"Total: {total_energy:.2f}J | "
                    f"P_k: {p_k:.4f} |H|: {h_mag:.4f} ||g||: {grad_norm:.4f}")
        return total_energy

    def reset_round(self):
        logging.info(f"Client {self.client_id}: Resetting round")
        self.comp_energy = 0.0
        self.comm_energy = 0.0
        self.norm_squared = 0.0
        self.flops_completed = 0
        self.last_tx_duration = 0.0
        # Note: Not resetting model states intentionally