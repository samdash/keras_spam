# coding:utf-8
import argparse
#from spam_model import load_model
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
from utils import sentence_to_index
import json

with open ('index.json', 'rb') as fp:
    data = json.load(fp)

from keras.models import load_model
model = load_model('spam_cnn.h5')

# predict
text = "how should i restart my apple watch?"
text = "if you are having any issues please contact jack from accounts at 8 6 6 9 8 9 8 9 0 0"
raw_x = sentence_to_index(text,data['word_to_id'],256)

y_pred = model.predict(raw_x)

print(text)

#y_classes = y_pred.argmax(axis=-1)

output_json = {}

output_json['spam_or_ham'] = {"ham": y_pred[0][0],
                                      "spam": y_pred[0][1]}

print(output_json)