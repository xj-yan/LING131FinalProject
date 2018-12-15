import re
import pandas as pd
from nltk.corpus import stopwords


# read the data
data = pd.read_csv('spam.csv',encoding = "ISO-8859-1")

# store the raw_text and label in seperate lists, and then zip them together
raw_text = list(data['v2'])
label = list(data['v1'])
labeled_data = list(zip(raw, label))
