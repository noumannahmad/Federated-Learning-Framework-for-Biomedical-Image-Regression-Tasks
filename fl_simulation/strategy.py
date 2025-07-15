
from flwr.server.strategy import FedAvg

class CustomFedAvg(FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        print(f"Aggregating round {rnd} with {len(results)} results")
        return super().aggregate_fit(rnd, results, failures)
