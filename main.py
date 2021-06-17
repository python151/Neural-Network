# networkx
import networkx as nx

# standard library packages
import random
import math
import copy
import statistics
from fractions import Fraction
import pickle

input_nodes = [0, 1]

worker_nodes = []

current = 2
for layer in range(2):
  layer = []
  for node in range(10):
    layer.append(current)
    current += 1
  worker_nodes.append(layer)

output_nodes = [current+1, current+2]

print(output_nodes)

def create_random_network(shape):
  input_nodes, worker_nodes, output_nodes = shape

  graph = nx.DiGraph()

  worker_nodes = copy.deepcopy(worker_nodes)
  worker_nodes.append(output_nodes)

  for n in input_nodes:
    graph.add_node(n, value=.5)
  for i in worker_nodes:
    for j in i: 
      graph.add_node(j, value=.5)
  for n in output_nodes:
    graph.add_node(n, value=.5)
  graph.add_node(output_nodes[::-1][0]+1)

  last_layer = input_nodes.copy()
  for current_layer in worker_nodes:
    for edge in range(random.randint(220, 300)):
      weight = random.random()-.5
      graph.add_edge(random.choice(last_layer), random.choice(current_layer), weight=weight)
    last_layer = current_layer.copy()
  
  return graph

def get_layers(shape, edge_layer):
  shape = copy.deepcopy(shape)

  new_shape = [shape[0]]
  for worker_layer in shape[1]:
    new_shape.append(worker_layer)
  new_shape.append(shape[2])

  return new_shape[edge_layer], new_shape[edge_layer+1]

def modify_network_at_random(graph, shape, modifications=2):
  graph = graph.copy()
  number_of_edge_layers = len(shape[1]) + 1
  modifications_per_edge_layer = math.ceil(modifications/number_of_edge_layers)
  for edge_layer in range(number_of_edge_layers):
    for modification in range(modifications_per_edge_layer):
      layer1, layer2 = get_layers(shape, edge_layer)
      
      modified = False
      while not modified:
        try:
          graph.remove_edge(random.choice(layer1), random.choice(layer2))
          # 20% chance (1/5th) of new edge NOT BEING added
          if random.choice([0, 0, 0, 0, 1]) == 0:
            graph.add_edge(random.choice(layer1), random.choice(layer2), weight=random.random()-.5)
          
          # 22% chance (2/9ths) of new edge BEING added
          if random.choice([0, 0, 0, 0, 0, 0, 0, 1, 1]) == 1:
            graph.add_edge(random.choice(layer1), random.choice(layer2), weight=random.random()-.5)

          modified = True
        except:
          pass
  return graph

def run_network(graph, inputs, input_nodes, worker_nodes, output_nodes):
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
    last_layer = current_layer
  
  ret = []
  for n in output_nodes:
    ret.append(graph.nodes(data=True, default=.5)[n]["value"])
  
  return ret

def generate_generation_from_survivors(generation_to_replicate, generation_size, shape_of_network):
  return_generation = []

  replication_ratio = math.ceil(generation_size / len(generation_to_replicate))
  print(f'{len(generation_to_replicate)} {replication_ratio}')
  for n, network in enumerate(generation_to_replicate):
    print(f'{(n/len(generation_to_replicate))*100}% COMPLETE')
    for i in range(replication_ratio):
      new_network = modify_network_at_random(network, shape_of_network)
      return_generation.append(new_network)

  return return_generation

def get_accuracy_score(correct, output):
  if len(correct) != len(output):
    raise ValueError("Correct answer and output lists must be the same length.")

  listOfDifferences = []
  for i, o in enumerate(output):
    c = correct[i]
    listOfDifferences.append(abs(o - c))

  fractionForm = listOfDifferences[0].as_integer_ratio()
  currentSum = Fraction(numerator=fractionForm[1], denominator=fractionForm[0])
  listOfDifferences.pop(0)
  for i in listOfDifferences:
    fractionForm = i.as_integer_ratio()
    newFraction = Fraction(numerator=fractionForm[1], denominator=fractionForm[0])
    currentSum = currentSum + newFraction
  
  return currentSum.denominator / currentSum.numerator

def checkOutputs(outputs):
  flag = False
  for o in outputs:
    for o1 in outputs:
      if o != o1:
        flag = True
  if flag != True:
    raise ValueError("it's the same again...")

def similarity_score(outputs):
  # swap rows and columns
  transposed = list(map(list, zip(*outputs)))
  # finding which node has the highest similarity in it's answers
  max_similarity_score = 0
  for node_answers in transposed:
    node_answers = [n * 1000 for n in node_answers]
    similarity = .5 - statistics.stdev(node_answers) / 80
    if similarity < 0: similarity = 0
    if similarity > max_similarity_score:
      max_similarity_score = similarity
  return max_similarity_score

def get_net_accuracy(scores, outputs, spiratic_multiplyer=1.9, similarity_score_multiplyer=.1, accuracy_multiplyer = 2):
  # punishes a difference between average and median which would indicate more randomized accuracy
  avg = statistics.mean(scores)
  median = statistics.median(scores)
  variance = statistics.variance(scores, avg)
  return (spiratic_multiplyer * abs(avg - median) * (1 + variance)) + (avg * accuracy_multiplyer) + (similarity_score_multiplyer * similarity_score(outputs))

def train_with_genetic_algorithm(training_data, generation_size, number_of_generations, kill_percentage=99, shape_of_network=[input_nodes, worker_nodes, output_nodes]):
    generations = []

    p_generation = []
    for i in range(math.ceil(generation_size/10)):
      p_generation.append(create_random_network(shape_of_network))
    generations.append(p_generation)

    last_generation_survivors = p_generation
    for g in range(number_of_generations):
      print(f'GENERATION {g} STARTING...')
      generation = generate_generation_from_survivors(last_generation_survivors, generation_size, shape_of_network)

      network_scores = []
      for i, network in enumerate(generation):
        print("=", end="")
        accuracy_scores = []
        outputs = []
        for i in range(40):
          training_data_chosen = random.choice(training_data)

          output = run_network(network, training_data_chosen[0], 
          shape_of_network[0], shape_of_network[1], shape_of_network[2])
          
          outputs.append(output)

          accuracy_scores.append( get_accuracy_score(training_data_chosen[1], output) )
        checkOutputs(outputs)
        net_accuracy_score = get_net_accuracy(accuracy_scores, outputs)
        network_scores.append([net_accuracy_score, network.copy()])

      number_to_survive = math.ceil( (generation_size * (100-kill_percentage))/100 )
      survivors = sorted(network_scores, key=lambda x : x[0])[:number_to_survive]
      print(len(survivors))
      print(survivors)

      new_survivors = []
      for s in survivors:
        new_survivors.append(s[1])
      
      filehandler = open("top_network.obj", 'wb') 
      pickle.dump(new_survivors[0], filehandler)

      last_generation_survivors = new_survivors
      print(f'GENERATION {g} COMPLETE.')

    return last_generation_survivors

def continue_training_with_genetic_algorithm(training_data, network, generation_size, number_of_generations, kill_percentage=99, shape_of_network=[input_nodes, worker_nodes, output_nodes]):
    p_generation = generate_generation_from_survivors([network], generation_size, shape_of_network)

    last_generation_survivors = p_generation
    for g in range(number_of_generations):
      print(f'GENERATION {g} STARTING...')
      generation = generate_generation_from_survivors(last_generation_survivors, generation_size, shape_of_network)

      network_scores = []
      # swap this for an asynchronous for loop
      for i, network in enumerate(generation):
        print("=", end="")
        accuracy_scores = []
        outputs = []
        # swap this for an asynchronous for loop
        for i in range(40):
          training_data_chosen = random.choice(training_data)

          output = run_network(network, training_data_chosen[0], 
          shape_of_network[0], shape_of_network[1], shape_of_network[2])
          
          outputs.append(output)

          accuracy_scores.append( get_accuracy_score(training_data_chosen[1], output) )
        checkOutputs(outputs)
        net_accuracy_score = get_net_accuracy(accuracy_scores, outputs)
        network_scores.append([net_accuracy_score, network.copy()])

      number_to_survive = math.ceil( (generation_size * (100-kill_percentage))/100 )
      survivors = sorted(network_scores, key=lambda x : x[0])[:number_to_survive]
      print(survivors)

      new_survivors = []
      for s in survivors:
        new_survivors.append(s[1])
      
      filehandler = open("top_network.obj", 'wb') 
      pickle.dump(new_survivors[0], filehandler)

      last_generation_survivors = new_survivors
      print(f'GENERATION {g} COMPLETE.')

    return last_generation_survivors

def get_mock_training_data():
  data = []
  for i in range(10000):
    sample = []
    x, y = random.random(), random.random()
    sample.append([x, y])
    sample.append([math.sqrt((x**2) + (y**2)),
                   math.atan(math.radians(y/x))
                   ])
    data.append(sample)
  return data
    
def main(new=True):
  testing_data = get_mock_training_data()
  if new:
      networks = train_with_genetic_algorithm(get_mock_training_data(), 1000, 100)
  elif not new:
    with open("top_network.obj", "rb") as f:
      top_network = pickle.load(f)
    networks = continue_training_with_genetic_algorithm(get_mock_training_data(), top_network, 1250, 50)
  else:
    raise ValueError("'new' variable should be a boolean value but other found.")

  print("testing top network: ------------")
  for i in range(10):
    print(testing_data[i])
    print(run_network(networks[0], testing_data[i][0], input_nodes, worker_nodes, output_nodes))

if __name__ == "__main__":
  main(new=False)