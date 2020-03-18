import csv
import numpy as np
import matplotlib.pyplot as plt

with open('./data_tr.csv', 'r') as csvFile:             #Open the CSV file containing traffic data
 traffic_data = list(csv.reader(csvFile,delimiter=','))

mat_X = np.ones((135,18))        # Training Data
mat_Y = np.zeros(135)            # Output Variable
lambda_val = 0.01

training_data_len = int(((len(traffic_data)-1)*80)/100)         #Calculate the length of 80% of the training data
reporting_data_len = int(((len(traffic_data)-1)*20)/100)        #Calculate the length of 20% of the data to be used for validation

for i in range(len(traffic_data)-1):                            #Load the imported data from csv in the form of a matrix of input parameters
    for j in range(len(traffic_data[0])-1):
        mat_X[i][j+1] = float(traffic_data[i+1][j])

for i in range(len(traffic_data)-1):                            #Load data in matrix of output variables
    mat_Y[i] = float(traffic_data[i+1][len(traffic_data[0])-1])

mat_X_train = mat_X[0:training_data_len:1]                      #Load 80% training data in a matrix
mat_X_report = mat_X[training_data_len:training_data_len+reporting_data_len:1] #20% data to be used for reporting

mat_Y_train = mat_Y[0:training_data_len:1]                         #80% Output variable values for training
mat_Y_report = mat_Y[training_data_len:training_data_len+reporting_data_len:1] #20% Output variable values for validation


beta_hypo = np.zeros(18)        #Create matrix for Hypothesis

Ipp = np.identity(18)           #Create an 18x18 Identity matrix for calculations


temp_1 = np.linalg.inv(((np.matmul((np.transpose(mat_X_train)),mat_X_train)) +  lambda_val*Ipp))  #Intermediate expression
temp_2 = np.matmul(np.transpose(mat_X_train),mat_Y_train) #Intermediate expression for final calculation

beta_hypo = np.matmul(temp_1,temp_2)    #Calculate final hypothesis

report_var = np.matmul(beta_hypo,np.transpose(mat_X_report)) #Calculate output values 


x = ([range(len(mat_Y_report))])            # makeindex vector for plotting data
Index = np.reshape(x,(len(mat_Y_report),))

mat_Y_predicted = (np.matmul(beta_hypo,np.transpose(mat_X_report)))
y = np.reshape(mat_Y_predicted,(len(mat_Y_predicted),)) #Reshape for plotting

plt.plot(Index,y,label='Predicted')         #Plot the values of predicted and actual values 
plt.plot(Index,mat_Y_report,label='Actual')
plt.xlabel("Training example No.")
plt.ylabel("Output Value")
plt.title("Closed Form Solution")
plt.legend()
plt.show()

y_diff = mat_Y_predicted - mat_Y_report          #Evaluate mean square error
sum_of_squares = np.matmul(y_diff,np.transpose(y_diff))
rmse = np.sqrt(sum_of_squares/len(mat_Y_predicted))

print("Value of Root mean square error is :")       #Print root mean square error
print(rmse)