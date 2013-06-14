# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
#The first thing to do is to import the relevant packages
# that I will need for my script, 
#these include the Numpy (for maths and arrays)
#and csv for reading and writing csv files
#If i want to use something from this I need to call 
#csv.[function] or np.[function] first

import csv as csv 
import numpy as np
# Import the random forest package
from sklearn.ensemble import GradientBoostingClassifier

#Open up the csv file in to a Python object
train_file_object = csv.reader(open('./csv/data/train.csv', 'rb')) 
train_file_object.next()  #The next() command just skips the 
                                 #first line which is a header
rtrain_data=[]                          #Create a variable called 'data'
for row in train_file_object:      #Run through each row in the csv file
    rtrain_data.append(row)             #adding each row to the data variable
rtrain_data = np.array(rtrain_data) 	         #Then convert from a list to an array
			         #Be aware that each item is currently
                                 #a string in this format
                                 
test_file_object = csv.reader(open('./csv/data/test.csv', 'rb')) 
test_file_object.next()  #The next() command just skips the 
                                 #first line which is a header
rtest_data=[]                          #Create a variable called 'data'
for row in test_file_object:      #Run through each row in the csv file
    rtest_data.append(row)             #adding each row to the data variable
rtest_data = np.array(rtest_data) 	         #Then convert from a list to an array
			         #Be aware that each item is currently
                                 #a string in this format                                 
                                 
#data extending
#combine siblings and parch into "family" field
#add two columns to train and test dataset arrays

train_data = np.zeros((rtrain_data.shape[0],rtrain_data.shape[1]+2), dtype=rtrain_data.dtype)
train_data[:,:-2] = rtrain_data

train_data[(train_data[0::,5]!=0) | (train_data[0::,6]!=0),11] = 1
train_data[train_data[0::,11] == '',11] = 0

#same for test data
test_data =np.zeros((rtest_data.shape[0],rtest_data.shape[1]+2), dtype=rtest_data.dtype)
test_data[:,:-2] = rtest_data

test_data[(test_data[0::,4]!=0) | (test_data[0::,5]!=0),10] = 1
test_data[test_data[0::,10] == '',10] = 0

#derive title field from name
train_data[np.char.find(train_data[0::,2],"Mr") != -1,12]=1
train_data[np.char.find(train_data[0::,2],"Mrs")!= -1,12]=2
train_data[np.char.find(train_data[0::,2],"Miss")!= -1,12]=3                                 
train_data[np.char.find(train_data[0::,2],"Master")!=-1,12]=4
train_title_median = np.round(np.median(train_data[train_data[0::,12] != '',12].astype(np.float)))
train_data[train_data[0::,12] == '',12] = train_title_median

#same for test dataset
test_data[np.char.find(test_data[0::,1],"Mr")!= -1,11]=1
test_data[np.char.find(test_data[0::,1],"Mrs")!=-1,11]=2
test_data[np.char.find(test_data[0::,1],"Miss")!= -1,11]=3                                 
test_data[np.char.find(test_data[0::,1],"Master")!=-1,11]=4
test_title_median = np.round(np.median(test_data[test_data[0::,11] != '',11].astype(np.float)))
test_data[test_data[0::,11] == '',11] = test_title_median
                          
#data mapping
#Map male and female to 1 and 0, respectively
train_data[train_data[0::,3] == 'male', 3] = 1
train_data[train_data[0::,3] == 'female', 3] = 0

#same on test data
test_data[test_data[0::,2] == 'male', 2] = 1
test_data[test_data[0::,2] == 'female', 2] = 0

#convert Embark to  0, 1 or 2 (Cherbourg, Southamption and Queenstown)
train_data[train_data[0::,10] == 'C', 10] = 0
train_data[train_data[0::,10] == 'S', 10] = 1
train_data[train_data[0::,10] == 'Q', 10] = 2

#same on test data
test_data[test_data[0::,9] == 'C', 9] = 0
test_data[test_data[0::,9] == 'S', 9] = 1
test_data[test_data[0::,9] == 'Q', 9] = 2

#data filling
#estimate missing age values with median of existing ones
train_age_median = np.median(train_data[train_data[0::,4] != '',4].astype(np.float))
train_data[train_data[0::,4] == '',4] = train_age_median

#same on test data
test_age_median = np.median(test_data[test_data[0::,3] != '',3].astype(np.float))
test_data[test_data[0::,3] == '',3] = test_age_median

#estimate missing embarked values with the median of existing ones
#rounding is neccessary
train_embarked_median = np.round(np.median(train_data[train_data[0::,10] != '',10].astype(np.float)))
train_data[train_data[0::,10] == '',10] = train_embarked_median

#same on test data
test_embarked_median = np.round(np.median(test_data[test_data[0::,9] != '',9].astype(np.float)))
test_data[test_data[0::,9] == '',9] = test_embarked_median

#estimate fare using the class
for i in xrange(np.size(train_data[0::,0])):
    if train_data[i,8] == '':
        train_data[i,8] = np.median(train_data[(train_data[0::,8] != '') & (train_data[0::,0] == train_data[i,0]),8].astype(np.float))

#same on test data
for i in xrange(np.size(test_data[0::,0])):
    if test_data[i,7] == '':
        test_data[i,7] = np.median(test_data[(test_data[0::,7] != '') & (test_data[0::,0] == test_data[i,0]),7].astype(np.float))

#Remove name,ticket and cabin
train_data = np.delete(train_data,[2,7,9],1) # Remove the name data, cabin and ticket

#and... same for test data
test_data = np.delete(test_data,[1,6,8],1) # Remove the name data, cabin and ticket

#dummy_filename = "./csv/results/dummy.csv"
#np.savetxt(dummy_filename, train_data, delimiter=",")

# Create the random forest object which will include all the parameters
# for the fit


gb= GradientBoostingClassifier(n_estimators = 100)

# Fit the training data to the training output and create the decision
# trees
gb = gb.fit(train_data[0::,1::],train_data[0::,0])

# Take the same decision trees and run on the test data
output = gb.predict(test_data)
train_output = gb.predict(train_data[0::,1::])
print output

#write to csv
fdescriptor = open('./csv/results/gradientboostingbasedmodelpy.csv',"wb")
test=open('./csv/data/test.csv', 'rb')
open_file_object = csv.writer(fdescriptor)
test_file_object = csv.reader(test)
test_file_object.next()



i = 0
for row in test_file_object:
    row.insert(0,output[i].astype(np.uint8))
    open_file_object.writerow(row)
    i += 1


test.close()
fdescriptor.close()

#generate train result file in order to calculate accuracy

checkdescriptor = open('./csv/results/gradientboostingcheckpy.csv',"wb")
train=open('./csv/data/train.csv', 'rb')
open_check_object = csv.writer(checkdescriptor)
train_file_object = csv.reader(train)
train_file_object.next()

j = 0
for row in train_file_object:
    row.insert(0,train_output[j].astype(np.uint8))
    open_check_object.writerow(row)
    j += 1

train.close()
checkdescriptor.close()

print "EOP"