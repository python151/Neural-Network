from Network import Network

import numpy as np
import pickle
from alive_progress import alive_it

# Loading in Dataset

# FAKE DATA HERE
data =  [([i, i*2], [i*2, i*4]) for i in range(100)]/np.linalg.norm([([i, i*2], [i*2, i*4]) for i in range(100)])
training_data = data 

'''
# FAKE DATA HERE #2
data =  [([i, i+1], [i**2, i**3]) for i in range(100)]/np.linalg.norm([([i, i+1], [i**2, i**3]) for i in range(100)])
training_data = data 
'''

def genetic_algo(training_data, first_generation_size=1000, generation_size=250, num_generations=750, filter_size=10) -> object:
    '''
    This is a Genetic Algorithm model training algorithm that uses the idea of natural selection to it's advantage.
    '''
    # Actually training
    generation = [Network().recursive_permutation(50) for i in alive_it(range(first_generation_size))]
    for gen in alive_it(range(num_generations)):
        # Evaluation and filter of current network
        generation = sorted(generation, key=lambda member: member.compare_to_dataset(training_data))[0:filter_size:]

        # Permuation of top 'filter_size' to be of size 'generation_size'
        new_generation = generation
        for member in generation:
            for i in range( int(generation_size / len(generation)) - 1):
                new_generation.append( member.generate_permuation() )
        generation = new_generation

        if gen % 25 == 0:
            # Saving network
            filehandler = open("top_network.obj", 'wb') 
            pickle.dump(generation[0], filehandler)
            
            print(f"Accuracy: {generation[0].compare_to_dataset(data)}")
    return generation[0]

def MCMC(training_data, model=None, r=.025, num_permutations_per_step=2, iterations=100) -> object:
    '''
    This is a Monte-Carlo Markov Chain model training algorithm that uses statistical tendencies to it's advantage.
    '''
    if model is None:
        model = Network()
    if iterations == 0:
        return model
    possible_permute = model.recursive_permutation(num_permutations_per_step)
    print(f"Accuracy #1: {model.compare_to_dataset(training_data)} \nAccuracy #2: {possible_permute.compare_to_dataset(training_data)}")
    return MCMC(training_data, model=possible_permute if model.compare_to_dataset(training_data) > possible_permute.compare_to_dataset(training_data) or np.random.uniform(0, 1) < r else model, r=r, iterations=iterations-1, num_permutations_per_step=2)
    

if __name__ == '__main__':
    best = MCMC(training_data)
    # Saving network
    filehandler = open("top_network.obj", 'wb') 
    pickle.dump(best, filehandler)