import torch
from queue import Queue

class Server:
    
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.diffusion_model = None
        self.diffusion_seed = None
        self.device = None
        self.queue = Queue() # logits stored on cpu
        self.busy = False

    def generate_diffusion_seed(self):
        pass
        
    def get_diffusion(self):
        pass

    def generate_diffusion(self):
        pass

    def knowledge_distillation(self):
        pass

    def send_logits(self, device):
        pass

    def get_logits(self, logits, device):
        if not self.busy:
            if self.queue.empty():
                self.busy = True
                self.queue.put((logits, device))

                return None
        if self.busy:
            self.queue.put((logits, device))
         