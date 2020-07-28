import numpy as np
import sentiment140
import glove_embeddings

def generate_training_data():
  """Generates training data from the Sentiment140 dataset

  Returns
  -------
  data : Tuple[np.ndarray - shape(N, 50), shape(N, 3)]
    Training data tuple with the first element being the embeddings of every
    tweet in the dataset and the second being a one-hot encoding of [negative,
    neutral, positive]
  """
  print("Generating IDF...")
  idf = glove_embeddings.generate_idf(sentiment140.tweets)
  print("Done")
  embeddings = []
  
  for tweet in sentiment140.tweets:
    embeddings.append(glove_embeddings.get_phrase_embedding(tweet, idf))

  print("Embeddings done")

  one_hots = []

  for polarity in sentiment140.polarities:
    one_hot = np.zeros((3,))
    one_hot[int(int(polarity) / 2)] = 1
    one_hots.append(one_hot)

  return (np.vstack(embeddings), np.vstack(one_hots))