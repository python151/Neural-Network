from Network import Network

import numpy as np
import pickle

# Loading in Dataset
'''
# FAKE DATA HERE
data =  [([i], [i+1]) for i in range(100)]/np.linalg.norm([([i], [i+1]) for i in range(100)])
training_data = data 
'''

# FAKE DATA HERE #2
data =  [([i], [i**2]) for i in range(100)]/np.linalg.norm([([i], [i**2]) for i in range(100)])
training_data = data 


# Configuring training
generation_size, num_generations, filter_size = 50, 10, 10

# Actually training
generation = [Network().recursive_permutation(50) for i in range(generation_size)]
for gen in range(num_generations):
    # Evaluation and filter of current network
    generation = sorted(generation, key=lambda member: member.compare_to_dataset(training_data))[0:filter_size:]

    # Permuation of top 'filter_size' to be of size 'generation_size'
    new_generation = generation
    for member in generation:
        for i in range( int(generation_size / len(generation)) - 1):
            new_generation.append( member.generate_permuation() )
    generation = new_generation

    print(f"the best of gen {gen} is {generation[0].compare_to_dataset(data)}")

# Saving network
filehandler = open("top_network.obj", 'wb') 
pickle.dump(generation[0].network, filehandler)