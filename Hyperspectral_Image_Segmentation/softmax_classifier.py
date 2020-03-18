#Amit Lohan(183079033)

import numpy as np 
import pandas as pd
from scipy.io import loadmat
import sys


###############---------Import data from mat files containing hyperspectral image data and groundtruth data-----------------------
im_data = loadmat('ipc.mat')
gt_data = loadmat('gt.mat')


#################--------------Convert imported data from dictionary into a numpy array for further processing--------------------
im_data2 = np.array(im_data["indian_pines_corrected"])
gt_data2 = np.array(gt_data["indian_pines_gt"])

s_data = np.dstack((im_data2,gt_data2)) #Concatenate groundtruth file lables to corresponding hyperspectral image data bands.
super_df = pd.DataFrame(s_data[:,0,:]) #Convert the first plane of column 0 into rows of a dataframe out of 145 columns to
# -transform it into a dataframe containing all the bands and respective groundtruth label in rows.  


#------------------------------Convert all the hyperspectral pixel bands and ground truth labels to rows of a pandas dataframe.--
for i in range(len(s_data[0,:,0])-1):
    super_df = super_df.append(pd.DataFrame(s_data[:,i+1,:]),ignore_index=True)


#--------------------------Declare empty dataframes for storing training and test data-------------------------------------------
train_frame = pd.DataFrame(columns=super_df.columns)
test_frame = pd.DataFrame(columns=super_df.columns)


#-------------------------Below block of code Normalize the features by first separating features and labels, then normalizing---
#--------------------------------------features and then reassembling features data to corresponding labels----------------------
dfm = (super_df.sort_values(200,ascending=True)) #Sort the rows by classes (for the purpose of observing the data classes)-------
dfy = dfm[200] #Separate labels to a separate Dataframe
dfynp = np.array(dfy) #Convert to numpy array for reassembly with normalized array of features in following lines
df3 = dfm.drop(200,axis = 1) #Separate features from groundtruth labels
df3np = np.array(df3) #Convert to numpy array for normalizing
df2np = (df3np - df3np.mean(axis = 0)) / (df3np.std(axis = 0)) #Normalize the matrix features stored in numpy array
df2 = pd.concat((pd.DataFrame(df2np),pd.DataFrame(dfynp,columns=[200])),axis=1) #Reassemble groundtruth labels and features.


#--Below block of code stores half of the sample pixels corresponding to each class into training data and rest half to test data-
#--------------------------------------------A 50-50 split to training and test data----------------------------------------------
for i in range(16): # Run a loop for each class
    j = (int)(((df2.loc[df2[200]==i+1])[0].count())/2) # Count half the number of samples for each class
    temp = (df2.loc[df2[200]==i+1]) #Take all the samples of (i+1)th (i goes from 0 to 15) class
    temp = temp.sample(frac=1) #Randomly shuffle the samples taken in above line of code
    temp2 = temp[:j+2] #take half the samples for training
    temp3 = temp[j+2:] #take rest half of the samples for testing
    train_frame = train_frame.append(temp2) #Append half of the samples of each class to training data
    test_frame = test_frame.append(temp3) #Append rest half of the samples of each class to test data

#----------------------------------------------------Randomly shuffle the training and test data to mix it properly--------------
train_frame = train_frame.sample(frac=1) #Randomly shuffle training data
test_frame = test_frame.sample(frac=1) #Randomly suffle test data

train_dat = np.array(train_frame) #Convert training data to numpy arrray for numerical processing
test_dat = np.array(test_frame) #convert test data to numpy array for numerical processing


#----------------------------------------------Separate features and labels to X and Y matrices for training and testing---------
X_mat_train = (train_dat[:,0:200]).astype(np.float64) #Separate features only for taining
Y_mat_train = (train_dat[:,200]) #Separate corresponding labels for training

X_mat_test = (test_dat[:,0:200]).astype(np.float64) #Separate features from labels for testing
Y_mat_test = (test_dat[:,200]).astype(int)#Separate corresponding lables for testing

X_mat_test = np.insert(X_mat_test,[0],[1],axis=1) #Insert 1 before every training sample for multiplying with intecept hypothesis.


#----------------------------------------Declare variables to be used for gradient descent and other procesing below-------------
n_class = 16 #Total number of classes of data
n_sampels = len(X_mat_train[:,0]) #Number of training samples available
alpha = 0.08 #Learning rate
n_ite = 22000 #Number of iterations for gradient descent
G1 = np.ones((1,n_class)) #Vector of ones for summing by matrix multiplication purpose in gradient descent equation


X_mat_train = np.insert(X_mat_train,[0],[1],axis=1)#Insert a 1 before every training sample for multiplication with intercept----
theta_h = np.zeros((len(X_mat_train[0,:]),16))#Initialize a numpy array with zeros for storing hypothesis values 

one_hot_y_train = np.zeros((n_sampels,n_class)) #Declare empty array for storing one hot encoded data for class labels

#----------------------------------------------One hot encode training data labels-----------------------------------------------
for i in range(n_sampels):
    one_hot_y_train[i,((int)(Y_mat_train[i])-1)] = 1


#-----------------------------------Run gradient descent loop of n_ite number of iterations--------------------------------------
for i in range(n_ite):
    H2 = np.exp(np.matmul(X_mat_train,theta_h))#-----Vectorized calculation of matrix containg e^thetaj.xi
    theta_h = theta_h + (alpha/n_sampels)*(np.matmul(X_mat_train.T,(one_hot_y_train - (np.multiply(H2,np.matmul((np.reciprocal(np.matmul(H2,G1.T))),G1))))))
###############---Above line of code implements gradient descent equation in vectorized form


H = np.exp(np.matmul(X_mat_test,theta_h)) #Multiply test features with obtained hypothesis to get matrix with elements e^thetaj.xi

J = np.matmul(H,G1.T) #Obtain vector containg values summation of e^thetaj.xi over all the values of j

Y_predict = (np.argmax(H/J,axis=1)+1).astype(int) #Obtain the predicted labels for each test sample

print("Obtained percentage accuracy of classification is:")
print((np.mean(Y_mat_test==Y_predict))*100)




