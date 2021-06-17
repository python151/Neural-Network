from main import run_network, input_nodes, worker_nodes, output_nodes, get_mock_training_data
import pickle
import copy
import matplotlib.pyplot as plt

with open("top_network.obj", "rb") as f:
  network = pickle.load(f)

inputs = []
print( run_network(network, inputs, input_nodes, worker_nodes, output_nodes) )