print("Loading word embeddings...")

import glove_embeddings

print("Loading other libraries...")

import mygrad as mg
import numpy as np

from mynn.layers.dense import dense
from mynn.optimizers.adam import Adam

from mygrad.nnet.activations import relu
from mygrad.nnet.initializers import glorot_normal
from mygrad.nnet.losses import softmax_crossentropy

print("Loading model...")

import pickle

class Model:
  def __init__(self, dim_in, num_hidden, dim_out):
    self.d1 = dense(dim_in, num_hidden, weight_initializer=glorot_normal)
    self.d2 = dense(num_hidden, dim_out, weight_initializer=glorot_normal)

  def __call__(self, x):
    """ The model's forward pass functionality.
    
    Parameters
    ----------
    x : Union[numpy.ndarray, mygrad.Tensor], shape=(N, T)
        The batch of size-N.
        
    Returns
    -------
    mygrad.Tensor, shape=(N, 2)
        The model's predictions for each of the N pieces of data in the batch.
    """
    return self.d2(relu(self.d1(x)))

  @property
  def parameters(self):
    """ A convenience function for getting all the parameters of our model. """
    return self.d1.parameters + self.d2.parameters

  def load_model(self, path):
    with open(path, "rb") as f:
      for param, (name, array) in zip(self.parameters, np.load(f).items()):
        param.data[:] = array

model = Model(50, 100, 2)
model.load_model("./model_data.npz")

with open("model_idf.pkl", "rb") as f:
  idf = pickle.load(f)

print("Sentiment Analysis is ready. Type phrases to analyze or /exit to exit.")

while True:
  phrase = input("Enter phrase to classify: ")
  
  if phrase == "/exit":
    print("Goodbye!")
    break

  result = model(glove_embeddings.get_phrase_embedding(phrase, idf)).data.reshape(2)
  classification = result.argmax()

  if np.abs(result[0] - result[1]) < 0.1:
    print("🤏 It's rather close, but it is more " + 
    ("positive" if classification == 1 else "negative") + 
    ".")
  else:
    print("👍 Positive" if classification == 1 else "👎 Negative")