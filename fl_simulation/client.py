
import flwr as fl
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import SimpleCNN
from src.train import train
from src.evaluate import evaluate
from src.metrics import mae
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CTClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data):
        self.model = model.to(DEVICE)
        self.train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=8)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss = train(self.model, self.train_loader, self.optimizer, self.criterion, DEVICE)
        return self.get_parameters(config), len(self.train_loader.dataset), {"loss": loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = evaluate(self.model, self.test_loader, self.criterion, DEVICE)
        return float(loss), len(self.test_loader.dataset), {"mae": loss}

def load_dummy_data():
    x_train = torch.randn(100, 1, 128, 128)
    y_train = torch.rand(100)
    x_test = torch.randn(20, 1, 128, 128)
    y_test = torch.rand(20)
    return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)

if __name__ == "__main__":
    model = SimpleCNN()
    train_data, test_data = load_dummy_data()
    client = CTClient(model, train_data, test_data)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
