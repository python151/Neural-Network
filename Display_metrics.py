import timeit
import matplotlib.pyplot as plt
import pickle
import numpy as np
from alive_progress import alive_it

# FAKE DATA HERE #2
data =  [([i, i*2], [i*2, i*4]) for i in range(100)]/np.linalg.norm([([i, i*2], [i*2, i*4]) for i in range(100)])
training_data = data 

with open("top_network.obj", "rb") as f:
  network = pickle.load(f)

def run_timing():
    num_iterations = 250

    start = timeit.default_timer()
    results = [network.compare_to_dataset(data) for i in alive_it(range(num_iterations))]
    end = timeit.default_timer()

    print(f"network took {end-start}s to complete {num_iterations} iterations on a dataset of {len(data)} or an average of {(end-start)/num_iterations}s per iteration or {(end-start)*1000/(num_iterations*len(data))}ms per data point")

print(network.input_layer)
print(network.hidden_layers)
print(network.output_layer)
print(network.max_node)
print(data[0][0], network._run_network(data[0][0]))
print(data[23][0], network._run_network(data[23][0]))
print(data[65][0], network._run_network(data[65][0]))

def plot_data():
    x = [d[0] for d in data]
    y = [d[1] for d in data]

    grab_node = lambda nodes, n : [point[n] for point in nodes]
    figure, axis = plt.subplots(len(x[0]), len(y[0]))
    for i in range(len(x[0])):
        for j in range(len(y[0])):
            axis[i, j].plot(grab_node(x, i), grab_node(y, j), 'g', label='expected')
            axis[i, j].plot(grab_node(x, i), grab_node([network._run_network(point) for point in x], j), 'b', label='network')
            axis[i, j].set_title(f'Input Node {i} - Output Node {j}')
            axis[i, j].legend()


    plt.tight_layout()
    plt.savefig(f'in-out.png')

if __name__ == '__main__':
    plot_data()