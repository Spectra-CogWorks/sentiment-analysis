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

def get_phrase_embedding(phrase):
  """Returns 

  """

  word_embeddings = []
