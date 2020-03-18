import csv                  #Import csv library for reading data form provided csv file
import numpy as np          #Import numpy for data processing
import random               #Import random for randomly taking samples from data file and for other random functions
from cvxopt import solvers  #Import cvxopt to be used for finding optimum solution for Lagrange Multipliers
from cvxopt import matrix


with open('./creditcard.csv', 'r') as csvFile:             #Open the CSV file containg credit card data
 data = list(csv.reader(csvFile,delimiter=','))

#Below for loop removes the Time parameter from list as it was observed that dropping this variable is increasing accuracy
for row in data:
    del row[0]

#==================Following block of code declares various variables to be used in the training and testing of SVM==================

p_samples = []   #Empty list for storing positive samples from credit card data file
n_samples = []   #Empty list for storing negarive samples form credit card data file
no_of_test_samples = 200  #Total number of samples of data to be taken for the data file which will be further divided into training and testing data
no_of_runs = 5            #Number of times to run the training and testing of SVM by picking data randomly from the data file
train_length = int(no_of_test_samples*0.8) #No of samples of data to be used for training which is 80% of the all sampled data from file.
test_length = int(no_of_test_samples*0.2)  #No of samples of data to be used for testing of the trained model of SVM
samp_length = int(no_of_test_samples/2)    #No of samples to be taken for each positive or negative class of data which is equal so half of all data.
train_samp_length = int(samp_length*0.8)   #No of samples of each positive or negative class used for training
test_samp_length = int(samp_length*0.2)    #No of samples of each positive or negative class used for testing of model.
header = data.pop(0)                       # Header variable stored the labels of parameters used for training and testing
n_columns = len(header)                    # Stores the number of parameters used for training and testing (Time is removed from data so 29 + 1(class))
accuracies = np.zeros(no_of_runs)          #Array for storing the accuracies obtained in each run of SVM


#=====Following bloack of code replaces 1's and 0's in class parameter of data which denotes fraud with -1's and 1's which is suitable for SVM======

for i in range(len(data)):
    if data[i][(len(data[0])-1)] == '1':  #Replace 1's of class with -1's
        data[i][(len(data[0])-1)] = '-1'
        p_samples.append(data[i])
    elif data[i][(len(data[0])-1)] == '0':  #Replace -1's of class with 1's
        data[i][(len(data[0])-1)] = '1'
        n_samples.append(data[i])

#==========================Following block of code randomly take samples of data and trains and tests the SVM multiple times==========================

for k in range(no_of_runs):                        #Train and test by picking up data randomly from file as many times as the value fo variable no_of_runs
    samp_p = random.sample(p_samples,samp_length)  #Randomly take samples of data from list corresponding to positive class
    samp_n = random.sample(n_samples,samp_length)  #Randomly take samples of data from list corresponding to negative class


    train_samp_p = np.asarray((samp_p[0:train_samp_length]),dtype='float64') #Take 80% of positive class samples for training
    train_samp_n = np.asarray((samp_n[0:train_samp_length]),dtype='float64') #Take 80% of negative class samples for training
    test_samp_p = np.asarray((samp_p[train_samp_length:samp_length]),dtype='float64') #Take 20% of positive class samples for testing
    test_samp_n = np.asarray((samp_n[train_samp_length:samp_length]),dtype='float64') #Take 20% of negative class samples for testing
    
    train_mat = np.concatenate((train_samp_n,train_samp_p),axis=0) #Concatenate positive and negative samples for training to obtain a mix
    test_mat =  np.concatenate((test_samp_n,test_samp_p),axis=0)   #Concatenate positive and negative samples for testing to obtain a mix

    x_train_p = train_samp_p[:,0:(n_columns-1)] #Extract training parameters from positive sample for obtaining value of hypothesis parameter b later
    x_train_n = train_samp_n[:,0:(n_columns-1)] #Extract training parameters from negative sample for obtaining value of hypothesis parameter b later

    np.random.shuffle(train_mat)   #Randomly shuffle the training samples matrix to mix positive and negative samples
    np.random.shuffle(test_mat)    #Randomly shuffle the test samples matrix to mix positive and negative samples
    
    x_train = train_mat[:,0:(n_columns-1)]#Get training parameters from randomly shuffled mix of p and n for training
    x_test = test_mat[:,0:(n_columns-1)]  #Get testing parameters from randomly shuffled mix of p and n for testing
    y_train = train_mat[:,(n_columns-1)]  #Get class variables(yi's) from randomly shuffled mix of p and n samples for training
    y_test = test_mat[:,(n_columns-1)]    #Get class variable(yi's ) from randomly shuffled mix for testing

    G = -1*(np.identity(train_length))  #G matrix to be input to cvxopt function for finding optimum value of alphas
    q = (-1)*(np.ones(train_length))    #q matrix to be input to cvxopt function for finding optimum value of alphas
    h = np.zeros(train_length)          #h matrix representing contraint alpha greater than zero    
    b = np.zeros(1)                     #b to be input to cvxopt representing 2nd constraint
    A = y_train                         #A matrix to be input to cvxopt representing yi's
  
    
    P = np.zeros((train_length,train_length))  #Create empty P matrix for storing values of ymynxmxn

    for i in range(train_length):           #Calculate values of P matrix to be input to cvxopt function
        for j in range(train_length):
            P[i][j] = y_train[i]*y_train[j]*(np.matmul((x_train[i]).T,x_train[j]))

    #Following line of code obtains optimum values of alpha coefficients corresponding to lagrange multipliers
    sol = solvers.qp(matrix(P),matrix(q),matrix(G),matrix(h),matrix(A.reshape(1,train_length)),matrix(b)) 
    alpha = np.asarray(sol['x'],dtype='float64')#Get values of alpha from optimum solution
    w = np.zeros(n_columns-1)  #Create empty matrix for storing values of hypothesis vector w
    
    for i in range(train_length): #Calculate values of corresponding elemets of hypothesis vector w
        w = w + (alpha[i])*(y_train[i])*(x_train[i])

    
    b = ((-1/2))*(np.max(np.matmul(x_train_p,w))+np.min(np.matmul(x_train_n,w))) #obtain value of hypothesis parameter b

    y_predict = [x + b for x in np.matmul(x_test,w)]  #Get output class label for test data


    for i in range(len(y_predict)):  #Threshold the values of y to convert the vector to only 1's and -1's
       if y_predict[i] < 0:
           y_predict[i] = -1
       elif y_predict[i]>0:
           y_predict[i]= 1
     
    no_of_mismatch = 0          #Declare variable for storing no. of incorrect output for obtaining accuracy

    for i in range(len(y_predict)): #Calculate no. of incorrect ouputs of trained model for test data
        if y_predict[i] != y_test[i]:
            no_of_mismatch +=1

    
    accuracies[k] = (1-(no_of_mismatch/(test_length)))*100 #Calculate accuracy for each run of main loop
    print(alpha)       #Prints the values of obtained Lagrange multipliers
 

print("Obtained accuracies are as below") #Print the obtained accuracies for various runs of training and testing
print(accuracies)
print("Average accuracy:") #Prints obtained average accuracy
print(np.sum(accuracies)/no_of_runs) 
