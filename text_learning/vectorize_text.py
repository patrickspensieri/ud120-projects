#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText
from sklearn.feature_extraction.text import TfidfVectorizer

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []
# temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        # only look at first 200 emails when developing
        # once everything is working, remove this line to run over full dataset
        # temp_counter += 1
        # if temp_counter < 200:
            path = os.path.join('../enron', path[:-1])
            print path
            email = open(path, "r")

            # use parseOutText to extract the text from the opened email
            # remove all instances of given words
            # note: sshacklensf, cgermannsf added to list since causes overfitting in future quiz
            text = parseOutText(email)
            replaceDic = ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"]
            for word in replaceDic:
                text = text.replace(word, "")

            # append to word_data
            word_data.append(text)
            if name == "sara":
                from_data.append(0)
            else:
                from_data.append(1)
            email.close()

print word_data[152]
print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )


# Tfidf vectorization of word_data
# like a bag of words, except weights are attributed to words depending on frequency
# rare words get higher weight, while common words get a lower weight
# frequency determined relative to the corpus fed into the transformer
vectorizer = TfidfVectorizer(stop_words="english")
vectorizer.fit_transform(word_data)
features = vectorizer.get_feature_names()
print "word num 34597 : ", features[34597]

# get number of words
print "number of words : ", len(vectorizer.get_feature_names())

