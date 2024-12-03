from abc import ABC, abstractmethod
import torch

# set cpu or cuda for default option
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device.type)


class BaseEmbedding(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self, **kwargs):
        raise NotImplementedError

    def train(self, model, optimizer, **kwargs):
        model.train()
        loader = model.loader(batch_size=kwargs["batch_size"], shuffle=True)
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(loader)
        return model, total_loss
