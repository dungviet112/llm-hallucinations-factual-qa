import pickle
import numpy as np

def load_hidden_states(results_file):
    with open(results_file, "rb") as infile:
        results = pickle.loads(infile.read())
    correct = np.array(results['correct'])
    infile.close()
    return results, correct