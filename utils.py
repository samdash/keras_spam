import numpy as np
import pandas as pd
import nltk
import re
from sklearn.utils import shuffle
from collections import Counter
import itertools

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_spam_data_from_disk():
    # Load dataset from file
    df_ham = pd.read_table("./data/ham.csv", sep=",",encoding='latin-1', low_memory=False)
    df_spam = pd.read_table("./data/spam.csv", sep=",",encoding='latin-1',low_memory=False)

    # remove all Unnamed Columns form the CSV File
    df_ham.drop(list(df_ham.filter(regex='Unnamed')), axis=1, inplace=True)
    df_spam.drop(list(df_spam.filter(regex='Unnamed')), axis=1, inplace=True)

    # concatenate both SUBJECT and BODY
    df_ham['message'] = df_ham.SUBJECT.str.cat(df_ham.BODY)
    df_spam['message'] = df_spam.SUBJECT.str.cat(df_spam.BODY)

    # drop the columns SUBJECT from both ham and spam files
    df_ham.drop(['SUBJECT', 'BODY'], axis=1, inplace=True)
    df_spam.drop(['SUBJECT', 'BODY'], axis=1, inplace=True)

    # adding labels
    df_ham['label'] = 'ham'
    df_spam['label'] = 'spam'

    # merge both data frames HAM and SPAM into One.
    df = df_ham.append(df_spam, ignore_index=True)
    df = shuffle(df)

    # very important otherwise df[0]->(message) length and df[1]->(label) length are mismatched
    df = df[pd.notnull(df['message'])]
    # drop all NaN rows from the data frame
    df.dropna()

    labels = sorted(list(set(df['label'].tolist())))
    print(labels)
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    # Split by words
    X = [clean_str(sentence) for sentence in df['message']]
    X = [list(sentence) for sentence in X]
    Y = [[0, 1] if (label == 'spam') else [1, 0] for label in df['label']]

    return [X, Y,labels]

def pad_sentences(sentences, padding_word="<PAD/>", maxlen=0): #maxlen=256
    """
    Pads all the sentences to the same length. The length is defined by the longest sentence.
     Returns padded sentences.
    """
    print('padding sentences ...')
    if maxlen > 0:
        sequence_length = maxlen
    else:
        sequence_length = max(len(s) for s in sentences)

    print('max sentence length is ', sequence_length)
    print('number of sentences ',len(sentences))
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        #sentence = (sentence[:sequence_length]) if len(sentence) > sequence_length else sentence

        num_padding = sequence_length - len(sentence)

        replaced_newline_sentence = []
        for char in list(sentence):
            if char == "\n":
                replaced_newline_sentence.append("<NEWLINE/>")
            elif char == " ":
                replaced_newline_sentence.append("<SPACE/>")
            else:
                replaced_newline_sentence.append(char)

        new_sentence = replaced_newline_sentence + [padding_word] * num_padding

        # new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """

    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))

    # Map from index to word
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))

    # Map from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary
    """
    x = np.array([[vocabulary[word] if word in vocabulary else 0 for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def sentence_to_index(sentence, vocabulary, maxlen):
    sentence = clean_str(sentence)
    raw_input = [list(sentence)]
    sentences_padded = pad_sentences(raw_input, maxlen=maxlen)
    raw_x, dummy_y = build_input_data(sentences_padded, [0], vocabulary)
    return raw_x


def load_spam_data():
    x_raw, y_raw,labels = load_spam_data_from_disk()#load_auto_tag_data('pt') #load_data_from_disk()

    sentences_padded = pad_sentences(x_raw)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, y_raw, vocabulary)
    print('data loaded ......')
    return [x, y, vocabulary, vocabulary_inv,labels]

def load_auto_tag_data():
    x_raw, y_raw,labels = load_auto_tag_data_from_disk()

    sentences_padded = pad_sentences(x_raw)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, y_raw, vocabulary)
    print('data loaded ......')
    return [x, y, vocabulary, vocabulary_inv,labels]


if __name__ == "__main__":
    load_spam_data_from_disk()
