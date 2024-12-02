#%% chdir
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
print(os.getcwd())

#%%

from utility.response_selection import keyword_based
from utility.response_selection import method
from utils_JL import to_method_object

from tqdm import tqdm
import pandas as pd
import numpy as np

#%%
import torch
cuda_torch_available = torch.cuda.is_available()
print('Cuda Torch available:', cuda_torch_available)

import tensorflow as tf
cuda_available = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
print('Cuda TF available:', cuda_available)

#%%
args_dataset = 'conansp' # default='sample', choices=['reddit', 'gab', 'conan']
args_kpq = 300
#%%
import os
import pickle

# Function to save data with a flag to prevent overwriting
def save_data(file_path, data):
    if os.path.exists(file_path):
        print(f"The file {file_path} exist and will returned")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    else: 
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data has been saved to {file_path}.")
        return data
#%%

contexts_train = save_data('backup/contexts_train_SP.pkl', None)
responses_train = save_data('backup/responses_train_SP.pkl', None)

train_x_text = save_data('backup/train_x_text_SP.pkl', None)
train_y_text = save_data('backup/train_y_text_SP.pkl', None)
test_x_text = save_data('backup/test_x_text_SP.pkl', None)
test_y_text = save_data('backup/test_y_text_SP.pkl', None)

candidates = save_data('backup/extracted_good_candidates_by_LQ_SP.pkl', None)
good_candidates = save_data('backup/good_candidates.pkl', None)

#%%
from utility.response_selection import vector_based_JL as vector_based

# method = vector_based.VectorMappingMethod(encoder=vector_based.TfHubEncoder("https://tfhub.dev/google/universal-sentence-encoder/2"))
# method = vector_based.VectorSimilarityMethod(encoder=vector_based.TfHubEncoder("https://tfhub.dev/google/universal-sentence-encoder/2"))
method = vector_based.VectorMappingMethod(encoder=vector_based.SentenceTransformerEncoder('sentence-transformers/use-cmlm-multilingual'))
# method = vector_based.VectorSimilarityMethod(encoder=vector_based.SentenceTransformerEncoder('sentence-transformers/use-cmlm-multilingual'))

method.train(contexts_train, responses_train)

output = []
for i, test_i in enumerate(tqdm(test_x_text)):
    # print(i)
    predictions = method.rank_responses([test_i], good_candidates[i])
    output.append(good_candidates[i][predictions.item()])
print(output)

df_output = pd.DataFrame({'x_text': test_x_text, 'output': output})
df_output = save_data('backup/df_output_MAP_SP.pkl', df_output)

#%%

contexts = [test_i] 
responses = good_candidates[i]

def rank_responses(self, contexts, responses):
    """Rank the responses for each context, using cosine similarity."""
    contexts_matrix = method._encoder.encode_context(contexts)
    responses_matrix = method._encoder.encode_response(responses)
    responses_matrix /= np.linalg.norm(responses_matrix, axis=1, keepdims=True)
    similarities = np.matmul(contexts_matrix, responses_matrix.T)
    return np.argmax(similarities, axis=1)


#%% Dummy SentenceTransformer
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/use-cmlm-multilingual')
                  
sentences = ["This is an example sentence", "Each sentence is converted"]
embeddings = model.encode(sentences, show_progress_bar=False)
print(embeddings)

type(embeddings)
print(embeddings.shape)

#%% class Encoder(object)
import abc

class Encoder(object):
    """A model that maps from text to dense vectors."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode_context(self, contexts):
        """Encode the given texts as vectors.
        Args:
            contexts: a list of N strings, to be encoded.
        Returns:
            an (N, d) numpy matrix of encodings.
        """
        pass

    def encode_response(self, responses):
        """Encode the given response texts as vectors.
        Args:
            responses: a list of N strings, to be encoded.
        Returns:
            an (N, d) numpy matrix of encodings.
        """
        # Default to using the context encoding.
        return self.encode_context(responses)


#%% SentenceTransformerEncoder
import abc
from sentence_transformers import SentenceTransformer

class SentenceTransformerEncoder(Encoder):
    """An encoder that uses SentenceTransformer to encode texts into vectors.

    Args:
        model_name: (string) the name of the pre-trained SentenceTransformer model.
    """
    def __init__(self, model_name):
        """Create a new `SentenceTransformerEncoder` object."""
        self.model = SentenceTransformer(model_name)

    def encode_context(self, contexts):
        """Encode the given texts."""
        return self.model.encode(contexts)

# Example usage
model_name = 'sentence-transformers/use-cmlm-multilingual'
encoder = SentenceTransformerEncoder(model_name)

sentences = ["This is an example sentence", "Each sentence is converted"]
embeddings = encoder.encode_context(sentences)

print(embeddings.shape)


#%% METHODS = ['USE_SIM_M']
METHODS = ['USE_SIM_M']
for method_name in METHODS:
    print(method_name)
    method = to_method_object(method_name)
    method.train(contexts_train, responses_train)
    output = []
    for i, test_i in enumerate(tqdm(test_x_text)):
        predictions = method.rank_responses([test_i], good_candidates[i])
        output.append(good_candidates[i][predictions.item()])
    print(output)
print('*' * 80)
print(f'After filtering by LQ, there are {len(candidates)} candidates.\n')