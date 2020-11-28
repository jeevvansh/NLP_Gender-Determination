#!/usr/bin/env python
# coding: utf-8

#Identifying the gender of a name... We will use the heuristic that the last few characters in a name is its defining characteristic. For example, if the name ends with "la", it's most likely a female name, such as "Angela" or "Layla". On the other hand, if the name ends with "im", it's most likely a male name, such as "Tim" or "Jim". Using this program we will be able to compare the accuracy of classifiers that use varying numbers of ending letters to determine the gender of a name.

# In[1]:


#imports
import nltk
nltk.download('names')
import random
from nltk.corpus import names
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy


# In[2]:



#Defining a function to extract the features from the input name
# For this method, considering the last 2 letters will provide the most accuracy as we will see in the end
def gender_features(word, num_letters = 2):
    return {'feature': word[-num_letters:].lower()}


# In[3]:


#Define main function using labeled training data
if __name__ == '__main__':
    #Extract labeled names
    labeled_names = ([(name, 'male') for name in names.words('male.txt')]) + [(name, 'female') for name in names.words('female.txt')]
    #seed random number generator and shuffle the training data
    random.seed(9)
    random.shuffle(labeled_names)
    #Lets provide a list of input names to test our classifiers
    input_names = ['Dwight', 'Andrew', 'Jim', 'Kelly', 'Angela']
    #We don't know how many ending characters to consider, for this reason we will sweep the parameter space from 1 to 5 and each
    #time we will extract the features
    for i in range(1, 5):
        print('\nNumber of ending letters used:', i)
        featuresets = [(gender_features(name, i), gender) for (name, gender) in labeled_names]
        #splitting into train and test datasets
        train_set, test_set = featuresets[500:], featuresets[:500]
        #Using Naive Bayes Classifier
        classifier = NaiveBayesClassifier.train(train_set)
        #Evaluate the classifier for each value in the parameter space
        #Print classifier accuracy
        print('Classifier Accuracy (against training set) ==>', str(100 * nltk_accuracy(classifier, test_set)) + str('%'))
    
        #Predict outputs for new inputs
        for name in input_names:
            print(name, '==>', classifier.classify(gender_features(name, i)))


print ("\nAs we can see, the most accurate classifier is the second one. This classifier uses the two ending characters to make its predictions. The accuracy score is a measure of each classifier's accuracy against the test_set created from the second half of the shuffled featuresets(500 feature sets).") 
