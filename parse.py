import csv
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
import nltk
import numpy as np
from nltk import NaiveBayesClassifier
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import pickle
from nltk.tokenize import word_tokenize

def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict

neg_rev = []
pos_rev = []
bad_rev = []


bad = open('bad-words.csv','r').read().strip().split('\n')

neg = open('negative-words.csv','r').read().strip().split('\n')

pos = open('positive-words.csv','r').read().strip().split('\n')

bad_set = set(bad)

neg_set = set(neg)

pos_set = set(pos)

for words in bad_set:
    bad_rev.append((create_word_features(bad_set), "bad"))

for words in pos_set:
    pos_rev.append((create_word_features(pos_set), "pos"))

for words in neg_set:
    neg_rev.append((create_word_features(neg_set), "neg"))

poslen = int(len(pos_rev)*0.75)
neglen = int(len(neg_rev)*0.75)
badlen  = int(len(bad_rev)*0.75)

train_set = neg_rev[:neglen] + pos_rev[:poslen] + bad_rev[:badlen]
test_set = neg_rev[neglen:] + pos_rev[poslen:] + bad_rev[badlen:]
print(len(train_set),  len(test_set))

classifier = NaiveBayesClassifier.train(train_set)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
classifier1 = LogisticRegression_classifier.train(test_set)
accuracy = nltk.classify.util.accuracy(classifier, test_set)
print(accuracy * 100)

save_classifier = open ("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

save_classifier = open ("Logistic.pickle","wb")
pickle.dump(classifier1, save_classifier)
save_classifier.close()

result_list = []
result_list1 = []
result_list2 = []

with open('testset.csv') as csvfile2:
    readCSV1 = csv.reader(csvfile2, delimiter=',')
    for row in readCSV1:
        templist = row[1].split()
        str1 = ' '.join(templist)
        s_analysis = TextBlob(str1, analyzer=NaiveBayesAnalyzer())
        sentiment_score = (s_analysis.sentiment.p_pos)
        print sentiment_score
        result_list2.append(sentiment_score)
        words = word_tokenize(str1)
        words = create_word_features(words)
        result = (classifier.classify(words))
        result_list1.append(result)
        if sentiment_score > 0.65:
            if result == "pos":
                result_list.append('Negative')
            elif result == "neg":
                result_list.append('Negative')
            else:
                result_list.append('Positive')
        if sentiment_score < 0.65:
            if result == "pos":
                result_list.append('Negative')
            elif result == "neg":
                result_list.append('Positive')
            else:
                result_list.append('Positive')

rows = zip(result_list,result_list1,result_list2)

with open("output.csv", "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

result_list = []
result_list1 = []
result_list2 = []

with open('testset.csv') as csvfile2:
    readCSV1 = csv.reader(csvfile2, delimiter=',')
    for row in readCSV1:
        templist = row[1].split()
        str1 = ' '.join(templist)
        s_analysis = TextBlob(str1, analyzer=NaiveBayesAnalyzer())
        sentiment_score_1 = (s_analysis.sentiment.p_pos)
        result_list2.append(sentiment_score_1)
        words = word_tokenize(str1)
        words = create_word_features(words)
        result1 = (classifier1.classify(words))
        result_list1.append(result1)
        print (nltk.classify.util.accuracy(classifier1, test_set))
        if sentiment_score > 0.65:
            if result1 == "pos":
                result_list.append('Negative')
            elif result1 == "neg":
                result_list.append('Negative')
            else:
                result_list.append('Positive')
        if sentiment_score < 0.65:
            if result1 == "pos":
                result_list.append('Negative')
            elif result1 == "neg":
                result_list.append('Positive')
            else:
                result_list.append('Positive')

rows = zip(result_list,result_list1,result_list2)

with open("output.csv", "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
