
import flwr as fl
from fl_simulation.strategy import CustomFedAvg

if __name__ == "__main__":
    strategy = CustomFedAvg()
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)
