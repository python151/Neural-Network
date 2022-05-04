import numpy as np
import networkx as nx
from copy_network import copy_network

class Network:
    input_layer, hidden_layers, output_layer = None, None, None
    network = None
    layer_size, max_node = None, None

    def __init__(self, num_hidden_layers=3, layer_size=5, input_shape=1, output_shape=1, duplicates=None, network=None):
        self.network = network if network is not None else None
        if duplicates is not None:
            self.network = duplicates.network if self.network is None else network
            self.input_layer, self.hidden_layers, self.output_layer = duplicates.input_layer, duplicates.hidden_layers, duplicates.output_layer
            self.layer_size, self.max_node = duplicates.layer_size, duplicates.max_node
        else:
            self.network = nx.DiGraph()
            self.hidden_layers = [ [input_shape + (k*layer_size) + j for j in range(layer_size)] for k in range(num_hidden_layers) ]
            self.input_layer, self.output_layer = [i for i in range(input_shape)], [input_shape+(num_hidden_layers * layer_size)+i for i in range(output_shape)]
            self.layer_size = layer_size
            self.max_node = input_shape + output_shape + num_hidden_layers*layer_size

            for i in range(self.max_node):
                self.network.add_node(i)
                self.network.nodes[i].update(value=.5)
    
    @copy_network
    def _run_network(self, inputs, network=None):
        for i, node in enumerate(self.input_layer):
            network.nodes[node].update(value=inputs[i])
        
        for node in self.input_layer + [i for i in range(self.hidden_layers[0][0], self.max_node-len(self.output_layer)+i)]:
            for edge in network.out_edges(node):
                network.nodes[edge[1]].update(value=(edge[1] * network.nodes[node]["value"] * network.edges[edge]["weight"]))

        return [network.nodes[node]["value"] for node in self.output_layer]
    
    def compare_to_dataset(self, data):
        return np.sum([np.sum( [np.abs(n - d[1][i]) for i, n in enumerate(self._run_network(d[0]))] ) for d in data])
    
    @copy_network
    def generate_permuation(self, network=None):
        for node in np.random.random_integers(0, high=self.max_node, size=5):
            next_row = np.floor(node / self.layer_size)+1
            network.add_edge(node, next_row*self.layer_size+np.random.randint(0, high=self.layer_size), weight=np.random.uniform(0, 1))
        return Network(duplicates=self, network=network)
    
    def recursive_permutation(self, num):
        if num == 0:
            return self.generate_permuation()
        return self.generate_permuation().recursive_permutation(num-1)
