from gensim.models.keyedvectors import KeyedVectors
from collections import Counter
import numpy as np

_PUNC_REGEX = re.compile('[{}]'.format(re.escape(string.punctuation)))

_glove = KeyedVectors.load_word2vec_format("./glove.6B.50d.txt.w2v", binary=False)

def get_words(text):
  """Returns all the words in a string, removing punctuation and capitalization. 
  (copied from jekthewarrior/Findr)

  Parameters
  ----------
  text : str
    The text whose words to return.

  Returns
  -------
  words : List[str]
    The words in `text`, with no punctuation.
  """

  return _PUNC_REGEX.sub(" ", text.lower()).split()

def get_word_embedding(word):
  """Returns the GloVe embedding for the given `word`

  Parameters
  ----------
  word : str
    The word for which to find a GloVe embedding

  Returns
  -------
  embedding : Union[NoneType, np.ndarray] - shape(50,)
    The GloVe embedding if found or `None` otherwise.
  """

  if word in _glove:
    return _glove[word]
  else:
    return None

def get_phrase_embedding(phrase, idf):
  """Returns a summed phrase embedding for `phrase` given `idf`

  Parameters
  ----------
  phrase : str
    The phrase for which to calculate an embedding for

  idf : Dict[str, float]
    A dictionary mapping words to their inverse document frequency

  Returns
  -------
  embedding : np.ndarray - shape(50,)
    The phrase embedding for `phrase`
  """
  phrase_embedding = np.zeros((50,))

  for word in get_words(phrase):
    word_embedding = get_word_embedding(word)

    if word_embedding is None:
      continue
    
    phrase_embedding += word_embedding * idf[word]

  phrase_embedding /= np.linalg.norm(embedding)

  return phrase_embedding
