# Name: Ashwini Kadam
# Student ID: 800967986
# email: akadam3@uncc.edu
# Group No. : 09
# Project title: Email Classification using Naive Bayes Approach


from __future__ import division
from collections import defaultdict
from pyspark import SparkContext,SparkConf
import math
import operator
import os
import pickle
import itertools
from operator import add

import sys
import nltk
from nltk.corpus import stopwords


conf = SparkConf().setAppName("NaiveBayes")
sc = SparkContext(conf=conf)

# Class to evaluate performance of algorithm
class PerformanceParams:
    def __init__(self):
        self.t_p = 0.
        self.t_n = 0.
        self.f_p = 0.
        self.f_n = 0.
        self.accuracy = 0.
        self.precision = 0.
        self.recall = 0.
        self.f_measure = 0.

    def print_performance(self):
        print 'Overall Performance ...'
        print 'True Positive: ', self.t_p
        print 'True Negative: ', self.t_n
        print 'False Positive: ', self.f_p
        print 'False Negative: ', self.f_n
        print 'Accuracy: ', self.accuracy
        print 'Precision: ', self.precision
        print 'Recall: ', self.recall
        print 'F-measure: ', self.f_measure

    def calculate_all(self):
        self.get_accuracy()
        self.get_precision()
        self.get_recall()
        self.get_f_measure()

    def get_accuracy(self):
        self.accuracy = (self.t_p + self.t_n) / (self.t_p + self.f_p +
                                                 self.t_n + self.f_n)

    def get_precision(self):
        if self.t_p and self.f_p is not 0:
            self.precision = self.t_p / (self.t_p + self.f_p)

    def get_recall(self):
        if self.t_p and self.f_n is not 0:
            self.recall = self.t_p / (self.t_p + self.f_n)

    def get_f_measure(self):
        if self.precision and self.recall is not 0.:
            self.f_measure = 2 * self.precision * self.recall / \
                             (self.precision + self.recall)

# Class to train and test email classfication algorithm using Naive Bayes Approach
class NaiveBayes:
    def __init__(self):
        reload(sys)  
        sys.setdefaultencoding('ISO-8859-1') # To cope with Unicode standard issues
        self.dictionary = set()
        self.priors = {}  # count of 'categories given classes'
        self.word_counts = defaultdict(dict)  # use to compute likelihood of a 'word' given a class ;
                                              # {cateogry: (word, count) }
        self.perf_params = PerformanceParams()
        self.d_path = sys.argv[1] # Data path for train and test data set of emails
        self.m_path = sys.argv[2]
        self.cachedStopWords = stopwords.words('english') # Removing unneccessary stopwords from dictonary

    # Training data set from both 'Ham' and 'Spam' labels
    def train(self, text, category):
	if category not in self.priors.keys():	# Keeping track of categories
            self.priors[category] = 1
        else:
            self.priors[category] += 1

        terms = text.flatMap(lambda x: x.split()).map(lambda x: (x,1)).reduceByKey(lambda a,b: a+b).collect() # Converting text into terms and count
	unique_terms = [word for word in terms if word[0] not in self.cachedStopWords]	# Collecting only unique terms
	u_terms = [term[0] for term in unique_terms]	# Adding only terms in dictionary, not count
	self.dictionary = self.dictionary.union(u_terms)

	# Formulating matrix of words in each category with thier count
	for term in unique_terms:
            if term[0] in self.word_counts[category]:
		self.word_counts[category][term[0]] += term[1] 
            else:
                self.word_counts[category][term[0]] = term[1]
        return	

    # Claassify given test data into categories
    def classify_text(self, path, label, orig_class):
	text = {}
        # Read the test data
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.endswith('.txt'):
                        text[f] = sc.textFile(os.path.abspath(os.path.join(root, f)))
        else:
            print path, ' is invalid !'
	    return 

	print '# of ', label, ' files: ', len(text)

        if label == orig_class:
	    for t in text:
            	predict = self.predict(text[t])
            	prediction = max(predict.iteritems(), key=operator.itemgetter(1))[0]
	    	if prediction == label:
		    self.perf_params.t_p += 1	# If email is classfied as it is in real, then count it as true positive
            	else:
                    self.perf_params.f_p += 1	# else count it as false positive
        else:
            for t in text:
            	predict = self.predict(text[t])
            	prediction = max(predict.iteritems(), key=operator.itemgetter(1))[0]
	    	if prediction == label:
		    self.perf_params.t_n += 1	# If email is predicted as given 'fake' lable, then consider as true negetive
            	else:
                    self.perf_params.f_n += 1	# Else it is false negetive

    # Function to predict the category of input email
    def predict(self, text):
        terms = text.flatMap(lambda x: x.split()).collect()
	u_terms = [word for word in terms if word not in self.cachedStopWords]

        posterior = {}
        total_cats = 0.

        for cat in self.priors.keys():
            total_cats += self.priors[cat]

        for cat in self.priors.keys():
            p_w_given_cat = 0.  # probability of a word given category, i.e likelihood
            for term in u_terms:
                if term in self.word_counts[cat]:
                    p_w_given_cat += math.log((self.word_counts[cat][term] + 1.) /
                                              (len(self.word_counts[cat]) + len(self.dictionary) + 1), 2)
                # Assign a very low probability to the term that don't exist in word_counts[cat]
                else:
                    p_w_given_cat += math.log(1. / (len(self.word_counts[cat]) +
                                                    len(self.dictionary) + 1), 2)

            posterior[cat] = math.log(self.priors[cat] / total_cats, 2) + p_w_given_cat	# calculating posterior probability of email
        return posterior

    # Initialize the data of a specific category
    def initialize_data(self, path, label):
        # Read all files in a directory
        text = {}
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    if f.endswith('.txt'):
                        text[f] = sc.textFile(os.path.abspath(os.path.join(root, f)))
        else:
            print path, ' is invalid !'
	    return
 
	for t in text:
	    datamap = self.train(text[t], label)

    # Initilizing data set
    def initialize_input_data(self):
	ham_train_path = self.d_path + 'train/ham'
        spam_train_path = self.d_path + 'train/spam'
	print 'Initializing Training Data'
	self.initialize_data(ham_train_path, 'ham')
	self.initialize_data(spam_train_path, 'spam')

    # Classifying data set
    def classify_input_text(self):
        ham_test_path = self.d_path + 'test/ham'
        spam_test_path = self.d_path + 'test/spam'
        print 'Classifying Test Data'
        self.classify_text(ham_test_path, 'ham', 'ham')
        self.classify_text(spam_test_path, 'spam', 'spam')
    
    # Evaluating perfromance of algorithm
    def calculate_performance(self):
	self.perf_params.calculate_all()
        self.perf_params.print_performance()

    def init_naive_bayes(self):
        self.initialize_input_data()
        self.classify_input_text()
	self.calculate_performance()

if __name__ == '__main__':
    NaiveBayes().init_naive_bayes()
    
sc.stop()
