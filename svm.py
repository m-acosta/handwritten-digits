#-------------------------------------------------------------------------
# AUTHOR: Michael Acosta
# FILENAME: svm.py
# SPECIFICATION: In order to classify handwritten digits, an SVM classifier 
#                   built by sklearn is compared to the ground truth for an 
#                   accuracy. The model and hyperparameters with the highest 
#                   accuracy values are shown.
# FOR: CS 4210- Assignment #3
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df_test = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the testing data by using Pandas library

X_test = np.array(df_test.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df_test.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
high_accuracy = 0

for iterable_c in c:
    for iterable_deg in degree:
        for iterable_ker in kernel:
           for iterable_shape in decision_function_shape:

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                clf = svm.SVC(C=iterable_c,degree=iterable_deg, kernel=iterable_ker, decision_function_shape=iterable_shape)

                #Fit SVM to the training data
                clf.fit(X_training, y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                prediction = []
                for (x_testSample, y_testSample) in zip(X_test, y_test):
                    prediction.append(clf.predict([x_testSample])[0])

                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                current_accuracy = accuracy_score(y_test, prediction)
                if current_accuracy > high_accuracy:
                    print('\nHighest SVM accuracy so far: {:.3f}, Parameters: c={}, degree={}, kernel={}, decision_function_shape={}'
                          .format(current_accuracy, iterable_c, iterable_deg, iterable_ker, iterable_shape))
                    high_accuracy = current_accuracy
