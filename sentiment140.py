from gensim.models.keyedvectors import KeyedVectors
from collections import Counter
import csv
from pathlib import Path

with open(Path("./trainingandtestdata/training.1600000.processed.noemoticon.csv")) as train_data_file:
  train_data = csv.DictReader(train_data_file)
  for row in train_data:
    pass