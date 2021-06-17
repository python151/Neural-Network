from main import run_network, input_nodes, worker_nodes, output_nodes, get_mock_training_data
import pickle
import copy
import matplotlib.pyplot as plt

with open("top_network.obj", "rb") as f:
  network = pickle.load(f)

#print(list(network.edges(data=True)))

def run_network_modified(graph, inputs, input_nodes, worker_nodes, output_nodes):
  graph = graph.copy()
  input_nodes = input_nodes.copy()
  worker_nodes = copy.deepcopy(worker_nodes)
  output_nodes = output_nodes.copy()
  
  for i, node in enumerate(input_nodes):
    graph.nodes[node].update(value=inputs[i])
  worker_nodes.append(output_nodes)

  last_layer = input_nodes
  for current_layer in worker_nodes:
    for node in last_layer:
      for edge in list(graph.out_edges(node, data=True)):
        weight = edge[2]["weight"]
        graph.nodes[edge[1]].update(value= (list(graph.nodes(data="value", default=.5))[edge[0]][1] * weight) + list(graph.nodes(data="value", default=.5))[edge[1]][1])
    
    for n in list(graph.nodes(data="value")):
      print(n)
    
    print(" ----- ")
    last_layer = current_layer
  
  ret = []
  for n in output_nodes:
    ret.append(graph.nodes(data=True, default=.5)[n]["value"])
  
  return ret

print(list(network.edges(data=True)))


n1_tests = []
n2_tests = []

node1_inputs = []
node2_inputs = []


mock_data = get_mock_training_data()
for i in range(10000):
  if i % 100 == 0:
    print("=", end="")
  output = run_network(network, mock_data[i][0], input_nodes, worker_nodes, output_nodes)
  n1_tests.append([abs(output[0] - mock_data[i][1][0]), mock_data[i][0][0]])
  n2_tests.append([abs(output[1] - mock_data[i][1][1]), mock_data[i][0][1]])

  node1_inputs.append(mock_data[i][0][0])
  node2_inputs.append(mock_data[i][0][1])
  #print(f"1: {abs(output[0] - mock_data[i][1][0])} 2: {abs(output[1] - mock_data[i][1][1])} 3: {output[0]}, {output[1]} 4: {mock_data[i][0]}")

n1_tests = sorted(n1_tests, key=lambda x : x[1])
t = list(map(list, zip(*n1_tests)))

plt.plot(t[1], t[0])
#plt.plot(node2_inputs, n2_tests, 'node 2')

plt.savefig("Graph1.png", format="PNG")
plt.show()

#print(f"\n\n\n\n")
#run_network_modified(network, [.3, .8], input_nodes, worker_nodes, output_nodes)