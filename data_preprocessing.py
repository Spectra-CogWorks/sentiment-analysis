import numpy as np
import sentiment140
import glove_embeddings

def generate_training_data():
  """Generates training data from the Sentiment140 dataset

  Returns
  -------
  data : Tuple[np.ndarray - shape(N, 50), shape(N,)]
    Training data tuple with the first element being the embeddings of every
    tweet in the dataset and the second being the category of the data
  """
  print("Generating IDF...")
  idf = glove_embeddings.generate_idf(sentiment140.tweets)
  print("Done")
  embeddings = []
  
  for tweet in sentiment140.tweets:
    embeddings.append(glove_embeddings.get_phrase_embedding(tweet, idf))

  print("Embeddings done")

  categories = []

  for polarity in sentiment140.polarities:
    categories.append(1 if polarity == "4" else 0)

  return (np.vstack(embeddings), np.array(categories))