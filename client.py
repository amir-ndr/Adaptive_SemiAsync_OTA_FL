import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import threading
import time

class Client(threading.Thread):
    def __init__(self, client_id, model_fn, dataset, dataloader, device, local_lr=0.01, channel_dim=10):
        super(Client, self).__init__()
        self.client_id = client_id
        self.device = device
        self.model_fn = model_fn
        self.model = model_fn().to(device)
        self.dataset = dataset
        self.dataloader = dataloader
        self.lock = threading.Lock()

        self.local_lr = local_lr
        self.f_k = np.random.uniform(1e8, 3e8)  # FLOPS
        self.p_k = np.random.uniform(0.01, 1.0)  # transmit power
        self.h_k = torch.randn(channel_dim, dtype=torch.cfloat)  # channel vector
        self.mu_k = 1e-28  # CPU efficiency constant
        self.C = np.random.uniform(5e5, 2e6)  # FLOPs per sample

        self.staleness = 0
        self.received_model = None
        self.selected = False
        self.grad_ready_event = threading.Event()

        self.local_gradient = None
        self.notify_done_callback = None  # callback to notify server of training completion
        self.send_gradient_callback = None  # callback to send OTA signal
        self.alive = True  # control client lifecycle

    def receive_global_model(self, global_model, staleness):
        self.staleness = staleness
        self.received_model = global_model.to(self.device)
        self.model.load_state_dict(global_model.state_dict())
        self.selected = True

    def estimated_training_time(self):
        return (self.C * len(self.dataloader.dataset)) / self.f_k

    def estimated_energy(self):
        comp_energy = self.mu_k * (self.f_k ** 2) * self.C * len(self.dataloader.dataset)
        comm_energy = self.p_k  # Simplified, actual comm energy depends on ||s_k||²
        return comp_energy + comm_energy

    def run(self):
        while self.alive:
            if self.selected and self.received_model is not None:
                self.local_training()
                if self.notify_done_callback:
                    self.notify_done_callback(self.client_id)
                self.grad_ready_event.wait()  # wait until server signals OTA
                self.grad_ready_event.clear()
                self.transmit_gradient()
                self.selected = False
                self.staleness = 0
            else:
                self.local_training(step_only=True)
                self.staleness += 1
                time.sleep(0.1)

    def local_training(self, E=5, step_only=False):
        optimizer = optim.SGD(self.model.parameters(), lr=self.local_lr)
        self.model.train()

        data_iter = iter(self.dataloader)
        for _ in range(E):
            try:
                x, y = next(data_iter)
            except StopIteration:
                break
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            output = self.model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optimizer.step()

        if not step_only and self.received_model:
            new_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
            old_params = torch.nn.utils.parameters_to_vector(self.received_model.parameters())
            self.local_gradient = (new_params - old_params) / self.local_lr

            print(f"Client {self.client_id} | Staleness: {self.staleness}")

    def transmit_gradient(self):
        if self.local_gradient is None:
            return

        h = self.h_k
        p = self.p_k
        tau = self.staleness if self.staleness > 0 else 1  # avoid div by 0
        gamma = 1.0 / tau

        b_k = torch.conj(h) / (h.abs() ** 2 + 1e-6)
        if b_k.shape != self.local_gradient.shape:
            b_k = b_k.expand_as(self.local_gradient)

        s_k = (p * gamma) * b_k * self.local_gradient.to(torch.cfloat)

        # Communication energy (more realistic):
        E_comm = torch.norm(s_k).item() ** 2
        E_comp = self.mu_k * (self.f_k ** 2) * self.C * len(self.dataloader.dataset)
        total_energy = E_comp + E_comm

        if self.send_gradient_callback:
            self.send_gradient_callback(self.client_id, s_k, total_energy)

    def set_callbacks(self, done_cb, send_cb):
        self.notify_done_callback = done_cb
        self.send_gradient_callback = send_cb

    def trigger_transmission(self):
        self.grad_ready_event.set()

    def stop(self):
        self.alive = False