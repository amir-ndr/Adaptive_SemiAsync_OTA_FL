import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import threading
import time
import logging

class Client(threading.Thread):
    def __init__(self, client_id, model_fn, dataset, dataloader, device, local_lr=0.01):
        super(Client, self).__init__()
        self.client_id = client_id
        self.device = device
        self.model_fn = model_fn
        self.dataloader = dataloader
        self.lock = threading.Lock()
        
        # Model management
        self.current_model = model_fn().to(device)
        self.reference_model = None  # Model version for gradient calculation
        
        # Hardware characteristics
        self.f_k = np.random.uniform(1e9, 3e9)  # Compute capacity (FLOPS)
        self.p_k = np.random.uniform(0.1, 5.0)  # Transmit power (W)
        self.mu_k = 1e-28  # Energy efficiency coefficient
        self.C = np.random.uniform(1e4, 5e4)  # FLOPs per sample

        # Wireless channel
        model_size = sum(p.numel() for p in self.current_model.parameters())
        self.h_k = torch.randn(model_size, dtype=torch.cfloat, device=device)

        # Training state
        self.staleness = 0
        self.selected = False
        self.ready = False
        self.training_progress = 0.0
        self.total_rounds = len(dataloader)
        self.current_round = 0
        
        # Communication
        self.grad_ready_event = threading.Event()
        self.alive = True
        self.notify_done_callback = None
        self.send_gradient_callback = None

    def receive_global_model(self, global_model_state, staleness):
        """Update model reference for gradient calculation"""
        with self.lock:
            self.staleness = staleness
            self.reference_model = self.model_fn().to(self.device)
            self.reference_model.load_state_dict(global_model_state)
            self.selected = True
            self.current_round = 0  # Reset training progress

    def estimated_training_time(self):
        """Remaining time for current training task"""
        remaining_batches = len(self.dataloader) - self.current_round
        batch_flops = self.C * self.dataloader.batch_size
        return (remaining_batches * batch_flops) / self.f_k

    def estimated_energy(self):
        """Energy estimate for current training + transmission"""
        comp_energy = self.mu_k * (self.f_k**2) * self.C * len(self.dataloader.dataset)
        gamma = 1 / (1 + self.staleness)
        comm_energy = (self.p_k**2) * (gamma**2) * torch.norm(self.h_k).item()
        return comp_energy + comm_energy

    def run(self):
        """Continuous training process"""
        dataloader_iter = iter(self.dataloader)
        
        while self.alive:
            if self.selected and self.reference_model is not None:
                # Start new training task with fresh reference
                self._perform_scheduled_training()
            else:
                # Continue current training trajectory
                self._perform_background_training(dataloader_iter)

    def _perform_scheduled_training(self):
        """Complete training task with reference model"""
        try:
            self.current_round = 0
            self.training_progress = 0.0
            
            # Train for full local epochs
            for batch_idx, (x, y) in enumerate(self.dataloader):
                if not self.selected: break  # Allow interruption
                
                x, y = x.to(self.device), y.to(self.device)
                self._train_batch(x, y)
                self.current_round += 1
                self.training_progress = batch_idx / len(self.dataloader)

            if self.selected:  # Only notify if completed fully
                self._handle_training_completion()
                
        except Exception as e:
            logging.error(f"Client {self.client_id} training failed: {str(e)}")
        finally:
            self.selected = False

    def _perform_background_training(self, dataloader_iter):
        """Continuous background training"""
        try:
            x, y = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(self.dataloader)
            x, y = next(dataloader_iter)
            
        x, y = x.to(self.device), y.to(self.device)
        self._train_batch(x, y)
        time.sleep(0.01)  # Prevent CPU overuse

    def _train_batch(self, x, y):
        """Core training logic for one batch"""
        with self.lock:
            model = self.model_fn().to(self.device)
            model.load_state_dict(self.current_model.state_dict())
            optimizer = optim.SGD(model.parameters(), lr=self.local_lr)
            
            optimizer.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optimizer.step()
            
            # Update current model
            self.current_model.load_state_dict(model.state_dict())

    def _handle_training_completion(self):
        """Finalize training task and prepare transmission"""
        with self.lock:
            # Calculate gradient relative to reference model
            current_params = torch.nn.utils.parameters_to_vector(
                self.current_model.parameters()
            )
            reference_params = torch.nn.utils.parameters_to_vector(
                self.reference_model.parameters()
            )
            self.local_gradient = (current_params - reference_params) / self.local_lr

        # Notify server and wait for transmission
        if self.notify_done_callback:
            self.notify_done_callback(self.client_id)
        self.grad_ready_event.wait()
        self.grad_ready_event.clear()
        self._transmit_gradient()

    def _transmit_gradient(self):
        """OTA transmission implementation"""
        with self.lock:
            if self.local_gradient is None:
                return

            # Staleness-aware signal construction
            gamma = 1 / (1 + self.staleness)
            h_inv = torch.conj(self.h_k) / (self.h_k.abs().square() + 1e-6)
            tx_signal = (self.p_k * gamma) * h_inv * self.local_gradient.to(torch.cfloat)

            # Energy calculation
            comp_energy = self.mu_k * (self.f_k**2) * self.C * len(self.dataloader.dataset)
            comm_energy = torch.norm(tx_signal).item() ** 2
            total_energy = comp_energy + comm_energy

            logging.info(
                f"Client {self.client_id} TX | "
                f"Staleness: {self.staleness} | "
                f"Energy: {total_energy:.2f}J | "
                f"Grad Norm: {torch.norm(self.local_gradient).item():.2f}"
            )

            if self.send_gradient_callback:
                self.send_gradient_callback(
                    client_id=self.client_id,
                    signal=tx_signal.detach().cpu(),
                    energy=total_energy
                )

    def set_callbacks(self, done_cb, send_cb):
        self.notify_done_callback = done_cb
        self.send_gradient_callback = send_cb

    def stop(self):
        self.alive = False
        self.grad_ready_event.set()