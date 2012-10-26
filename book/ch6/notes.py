# pg. 221 

# Supervised Classification 
#  Classification is the task of choosing the correct class label for a given input

# pg. 222

# Examples of Classification 
#  1. Deciding whether an email is spam.
#  2. Deciding what topic a news article is, "sports", "technology", etc.

# The basic classification task has a number of variants.
#  1. Multi-class classification - each instance may be assigned multiple labels
#  2. Open-class classification - set of labels in not defined in advance
#  3. Sequence classification - a list of inputs are jointly classified

# pg. 223

def gender_features(word):
    return {'last_letter': word[-1]}

def gender_features_better(word):
    return {
            'first_letter': word[0],
            'length': len(word),
            'last_letter': word[-1]
           }

gender_features("Shrek")

import nltk
from nltk.corpus import names
import random

names = ([(name, 'male') for name in names.words('male.txt')] +
          [(name, 'female') for name in names.words('female.txt')])
import random
random.shuffle(names)

featuresets = [(gender_features_better(n), g) for (n,g) in names]
train_set, test_set = featuresets[2500:], featuresets[:2500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

classifier.classify(gender_features_better('Neo'))

classifier.classify(gender_features_better('Trinity'))

# pg. 224

print nltk.classify.accuracy(classifier, test_set)

classifier.show_most_informative_features(5)

from nltk.classify import apply_features
train_set = apply_features(gender_features, names[2500:])
test_set = apply_features(gender_features, names[:2500])

# pg. 225

def gender_features2(name):
    features = {}
    features["firstletter"] = name[0].lower()
    features["lastletter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count(%s)" % letter] = name.lower().count(letter)
        features["has(%s)" % letter] = (letter in name.lower())
    return features

gender_features2('John')

featuresets = [(gender_features2(n), g) for (n,g) in names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print nltk.classify.accuracy(classifier, test_set)

train_names = names[1500:]
devtest_names = names[500:1500]
test_names = names[:500]

train_set = [(gender_features(n), g) for (n,g) in train_names]
devtest_set = [(gender_features(n), g) for (n,g) in devtest_names]
test_set = [(gender_features(n), g) for (n,g) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set) 

# pg. 226

print nltk.classify.accuracy(classifier, devtest_set)

errors = []
for (name, tag) in devtest_names:
    guess = classifier.classify(gender_features(name))
    if guess != tag:
        errors.append( (tag, guess, name))

for (tag, guess, name) in sorted(errors): # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    print 'correct=%-8s guess=%-8s name=%-30s' % (tag, guess, name)

# pg. 227

def gender_features(word):
    return {'suffix1': word[-1],
            'suffix2': word[-2],
            "count(m)": name.lower().count('m'),
            "count(o)": name.lower().count('o'),
            "count(n)": name.lower().count('n')}

train_set = [(gender_features(n), g) for (n, g) in train_names]
devtest_set = [(gender_features(n), g) for (n, g) in devtest_names]
classifier = nltk.NaiveBayesClassifier.train(train_set) 
print nltk.classify.accuracy(classifier, devtest_set)

# Document Classification

from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# pg. 228


