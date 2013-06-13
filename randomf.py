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

#Open up the csv file in to a Python object
train_file_object = csv.reader(open('./csv/data/train.csv', 'rb')) 
header = train_file_object.next()  #The next() command just skips the 
                                 #first line which is a header
data=[]                          #Create a variable called 'data'
for row in train_file_object:      #Run through each row in the csv file
    data.append(row)             #adding each row to the data variable
data = np.array(data) 	         #Then convert from a list to an array
			         #Be aware that each item is currently
                                 #a string in this format

embarkment_values = ['C','S','Q']
#create a map from your variable names to unique integers:
intmap = dict([(val, i) for i, val in enumerate(set(embarkment_values))]) 
#make the new array hold corresponding integers instead of strings:
print intmap

parsed_data = np.copy(data)
for k, v in intmap.iteritems():
    parsed_data[data==k] = v

#data[data[0::,9]] = intmap[data[0::,9]]

print parsed_data