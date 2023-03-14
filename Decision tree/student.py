#create decision tree using ID3
import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="File Name")
args = parser.parse_args()
#received values
file=args.data
#file="/Users/maria/OneDrive/Documents/ML/Assignment/Programming Assignment 2/decisiontree_data/car.csv"
data = pd.read_csv(file, header=None)
#adding attribute names
header = np.array([])
for i in range(len(data.columns)):
    header = np.append(header, f"att{i}")
data.columns= header

#function to calculate entropy
def entropy(target_data, target, target_classes):
    result = 0
    prop = 1
    for i in range(len(target_classes)):
        denom = len(target_data.iloc[:, 0])
        if denom != 0:
            prop = len(target_data[target_data[target] == target_classes[i]].iloc[:, 0]) / denom
        if prop != 0:
            result = result + -1*prop*(np.log(prop)/ np.log(len(target_classes)))
    return result

#recursive function for id3 algorithm
def id3_algo(data, target, target_classes, depth, prev_entropy, previous_attribute, majority):
    #each time when function is called, information gain array will be declared & initialized to empty array
    info_gain_attr = np.array([])
    info_gains = np.array([])
    
    columns = np.array([])
    #to load attributes without the previously selected attribute into columns array
    if previous_attribute:
        for i in range(len(data.columns) - 1):
            if data.columns[i] != previous_attribute:
                columns = np.append(columns, data.columns[i])
    else:
        for i in range(len(data.columns) - 1):
            columns = np.append(columns, data.columns[i])
            
    #calculating information gain        
    for i in range(len(columns)):
        info_gain_attr = np.append(info_gain_attr, columns[i])
        result = prev_entropy
        new_branch=columns[i]
        unique_attr = data[new_branch].unique()
        for i in range(len(unique_attr)):
            temp = data[data[new_branch] == unique_attr[i]]
            prop = len(temp.iloc[:, 0]) / len(data.iloc[:, 0])
            temp_entropy = entropy(temp, target, target_classes)
            result = result - prop*temp_entropy
        info_gains = np.append(info_gains,result)
        
    #finding greatest information gain and its index   
    place = 0
    for i in range(len(info_gains)):
        if info_gains[i] > info_gains[place]:
            place = i
    if (info_gains[place]) == 0.0:
        return
    
    attribute_selected = info_gain_attr[place]
    #values of the selected attribute
    attribute_branches = data[attribute_selected].unique()
    for i in range(len(attribute_branches)):
        #selecting from last
        branch = attribute_branches[len(attribute_branches) - (i+1)]
        filtered_data = data[data[attribute_selected] == branch]
        new_entropy = entropy(filtered_data, target=target, target_classes=target_classes)
        if new_entropy == 0.0:
            leaf_node = filtered_data[target].unique()[0]
            print(f"{depth},{attribute_selected}={branch},{new_entropy},{leaf_node}")
        else:
            leaf_node = "no_leaf"
            print(f"{depth},{attribute_selected}={branch},{new_entropy},{leaf_node}")
            id3_algo(filtered_data, target, target_classes, depth+1, new_entropy, attribute_selected, majority=majority)
    
#taking last column : to get classes      
target = data.columns[-1]
# finding classes by taking unique values
target_classes = data[target].unique()
#calculate entropy of whole data
entropy_value = entropy(target_data=data, target=target, target_classes=target_classes)

#manually printing first line : here depth=0
print(f"{0},root,{entropy_value},no_leaf")
depth = 1
#finding the majority class
majority_class = target_classes[1]
for i in range(len(target_classes)):
    if len(data[data[target] == target_classes[i]].iloc[:, 0]) > len(data[data[target] == majority_class].iloc[:, 0]):
        majority_class = target_classes[i]

id3_algo(data, target, target_classes, 1, entropy_value, previous_attribute=None, majority = majority_class)