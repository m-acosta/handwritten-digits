#-------------------------------------------------------------------------
# AUTHOR: Michael Acosta
# FILENAME: perceptron.py
# SPECIFICATION: In order to classify handwritten digits, a perceptron is
#                   compared to the ground truth for an accuracy. The 
#                   model and hyperparameters with the highest accuracy 
#                   values are shown for a single layer and multi-layer 
#                   configuration.
# FOR: CS 4210- Assignment #4
# TIME SPENT: 45 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

single_accuracy = 0
multi_accuracy = 0

for i in n: #iterates over n

    for j in r: #iterates over r

        #iterates over both algorithms
        single_layer = [True, False]

        for k in single_layer: #iterates over the algorithms

            #Create a Neural Network classifier
            if k == True:
                #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
                clf = Perceptron(eta0=i, shuffle=j, max_iter=1000)
            else:
                #use those hyperparameters: activation='logistic', learning_rate_init = learning rate, hidden_layer_sizes = number of neurons in the ith hidden layer,
                #                          shuffle = shuffle the training data, max_iter=1000
                clf = MLPClassifier(activation='logistic', learning_rate_init=i, hidden_layer_sizes=100, shuffle=j, max_iter=1000)

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            positives = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                if clf.predict([x_testSample])[0] == y_testSample:
                    positives += 1

            current_accuracy = positives / len(y_test)

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            if k == True:
                if current_accuracy > single_accuracy:
                    print('\nHighest Perceptron accuracy so far: {:.3f}, Parameters: learning rate={}, shuffle={}'
                            .format(current_accuracy, i, j))
                    single_accuracy = current_accuracy
            else:
                if current_accuracy > multi_accuracy:
                    print('\nHighest MLP accuracy so far: {:.3f}, Parameters: learning rate={}, shuffle={}'
                            .format(current_accuracy, i, j))
                    multi_accuracy = current_accuracy










