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

fare_ceiling = 40
data[data[0::,8].astype(np.float) >= fare_ceiling, 8] = fare_ceiling-1.0
fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size
number_of_classes = 3 #There were 1st, 2nd and 3rd classes on board 
# Define the survival table
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))

for i in xrange(number_of_classes):       #search through each class
  for j in xrange(number_of_price_brackets):   #search through each price

    women_only_stats = data[                          #Which element           
                         (data[0::,3] == "female")    #is a female
                       &(data[0::,1].astype(np.float) #and was ith class
                             == i+1)                                       
                       &(data[0:,8].astype(np.float)  #was greater 
                            >= j*fare_bracket_size)   #than this bin              
                       &(data[0:,8].astype(np.float)  #and less than
                            < (j+1)*fare_bracket_size)#the next bin    
                          , 0]                        #in the 1st col                           
 						                                    									


    men_only_stats = data[                            #Which element           
                         (data[0::,3] != "female")    #is a male
                       &(data[0::,1].astype(np.float) #and was ith class
                             == i+1)                                       
                       &(data[0:,8].astype(np.float)  #was greater 
                            >= j*fare_bracket_size)   #than this bin              
                       &(data[0:,8].astype(np.float)  #and less than
                            < (j+1)*fare_bracket_size)#the next bin    
                          , 0] 
                          
survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float)) #Women stats
survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float)) #Men stats

survival_table[ survival_table != survival_table ] = 0.

survival_table[ survival_table < 0.5 ] = 0
survival_table[ survival_table >= 0.5 ] = 1 

test_file_obect = csv.reader(open('./csv/data/test.csv', 'rb'))
fname = "./csv/results/genderclasspricebasedmodelpy.csv"
open_file_object = csv.writer(open(fname, "wb"))
header = test_file_obect.next() 

for row in test_file_obect:                   #we are going to loop
                                              #through each passenger
                                              #in the test set                     
  for j in xrange(number_of_price_brackets):  #For each passenger we
                                              #loop thro each price bin
    try:                                      #Some passengers have no
                                              #price data so try to make
      row[7] = float(row[7])                  # a float
    except:                                   #If fails: no data, so 
      bin_fare = 3-float(row[0])              #bin the fare according class
      break                                   #Break from the bin loop
    if row[7] > fare_ceiling:              #If there is data see if
                                              #it is greater than fare
                                              #ceiling we set earlier
      bin_fare = number_of_price_brackets-1   #If so set to highest bin
      break                                   #And then break bin loop
    if row[7] >= j*fare_bracket_size\
       and row[7] < \
      (j+1)*fare_bracket_size:                #If passed these tests 
                                              #then loop through each 
                                              #bin 
      bin_fare = j                            #then assign index
      break                                   
  
  if row[2] == 'female':                             #If the passenger is female
        row.insert(0,                                   #at element 0, insert
             int(survival_table[0,float(row[0])-1, \    #the prediction from
             bin_fare])) #Insert the prediciton         #survival table
        open_file_object.writerow(row)                  #And write out row          
    else:
        row.insert(0,\
             int(survival_table[1,float(row[0])-1, \
             bin_fare]))                               
                                                       
        open_file_object.writerow(row)