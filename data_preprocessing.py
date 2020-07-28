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
  categories = []
  
  for index, tweet in enumerate(sentiment140.tweets):
    embedding = glove_embeddings.get_phrase_embedding(tweet, idf)
    if embedding is not None:
      embeddings.append(embedding)
      categories.append(1 if sentiment140.polarities[index] == "4" else 0)

  return (np.vstack(embeddings), np.array(categories))