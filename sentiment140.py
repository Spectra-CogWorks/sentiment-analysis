from gensim.models.keyedvectors import KeyedVectors
from collections import Counter
import csv
from pathlib import Path

"""
This file loads in the traing data from the sentiment 140 dataset, it is stored to `train_data`
as a list of lists, the fields are as follows (as copied from http://help.sentiment140.com/for-students/):
  0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
  1 - the id of the tweet (2087)
  2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
  3 - the query (lyx). If there is no query, then this value is NO_QUERY.
  4 - the user that tweeted (robotickilldozr)
  5 - the text of the tweet (Lyx is cool)
"""

with open(Path("./trainingandtestdata/training.1600000.processed.noemoticon.csv")) as train_data_file:
  train_data = list(csv.reader(train_data_file))