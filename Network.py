import numpy as np
import random
import networkx as nx
from copy_network import copy_network

class Network:
    input_layer, hidden_layers, output_layer = None, None, None
    network = None
    layer_size, max_node = None, None

    def __init__(self, num_hidden_layers=2, layer_size=4, input_shape=2, output_shape=2, duplicates=None, network=None):
        self.network = network if network is not None else None
        if duplicates is not None:
            # Copying data from duplicate argument
            self.network = duplicates.network if self.network is None else network
            self.input_layer, self.hidden_layers, self.output_layer = duplicates.input_layer, duplicates.hidden_layers, duplicates.output_layer
            self.layer_size, self.max_node = duplicates.layer_size, duplicates.max_node
        else:
            # Defining some defaults
            self.network = nx.DiGraph()
            self.hidden_layers = [ [input_shape + (k*layer_size) + j for j in range(layer_size)] for k in range(num_hidden_layers) ]
            self.input_layer, self.output_layer = [i for i in range(input_shape)], [input_shape+(num_hidden_layers * layer_size)+i-1 for i in range(output_shape)]
            self.layer_size = layer_size
            self.max_node = input_shape + output_shape + num_hidden_layers*layer_size - 1

            for i in range(self.max_node+1):
                self.network.add_node(i)
                self.network.nodes[i].update(value=0, bias=0)

            for node in self.input_layer:
                next_row = np.floor(node / self.layer_size)+1
                [self.network.add_edge(node, (next_row * self.layer_size) + np.random.randint(0, high=self.layer_size), weight=np.random.uniform(-10, 10)) for x in range(layer_size*2)]
    
    @copy_network
    def _run_network(self, inputs: list, network=None) -> list:
        # Sets up input nodes
        for i, node in enumerate(self.input_layer):
            network.nodes[node].update(value=inputs[i])
        
        non_linearity = lambda z : 1/(1 + np.exp(-z))#(-1 if x < 0 else 1) * np.log(np.abs(x)) / 10
        # Evaluates rest of network
        for node in range(self.max_node):
            for edge in network.in_edges(node):
                network.nodes[node].update(value=(network.nodes[node]["value"] + (network.nodes[edge[0]]["value"] * network.edges[edge]["weight"])))
            network.nodes[node].update(value=non_linearity(network.nodes[node]["value"]+network.nodes[node]["bias"]))
        return [network.nodes[node]["value"] for node in self.output_layer]
    
    def compare_to_dataset(self, data: list) -> float:
        return np.sum([np.sum( [np.abs(n - d[1][i]) for i, n in enumerate(self._run_network(d[0]))] ) for d in data])
    
    @copy_network
    def generate_permutation(self, network=None) -> object:
        # Randomly picks 1 node and creates/overrides randomly selected edges
        for node in np.random.random_integers(0, high=self.max_node, size=1):
            next_row = np.floor(node / self.layer_size)+1
            next_node = (next_row * self.layer_size) + np.random.randint(0, high=self.layer_size)
            network.add_edge(node, next_node if next_node < self.max_node else self.max_node, weight=np.random.uniform(-1, 1))
            network.nodes[node].update(bias=np.random.uniform(-1, 1))
        return Network(duplicates=self, network=network)
    
    @copy_network
    def micro_permutation(self, macro_chance=.1, network=None) -> object:
        if np.random.uniform(0, 1) < macro_chance:
            return self.generate_permutation()
        # Adjusting random edge
        edge = random.choice(list(network.edges))
        network.edges[edge].update(weight=network.edges[edge]['weight']+(np.random.uniform(-1, 1)/10))
        return Network(duplicates=self, network=network)
    
    def recursive_permutation(self, num: int, macro=True) -> object:
        if num == 0 and macro:
            return self.generate_permutation()
        elif num == 0 and not macro:
            return self.micro_permutation()
        return self.generate_permutation().recursive_permutation(num-1, macro=macro) if macro else self.micro_permutation().recursive_permutation(num-1, macro=macro)
