from __future__ import division
import csv
from os import path
import random
from math import log


class Node:
    def __init__(self, left_child=None, right_child=None, threshold=None, features=None, majority=None):
        #print "Node created"
        self.left_child = left_child
        self.right_child = right_child
        self.threshold = threshold
        self.features = features
        self.majority = majority

InfoGain = raw_input("Enter Gini or Entropy")
print "Please enter the file path starting with a back slash in this format: /Users/KeerthiTejaNuthi/Downloads/iris.data.txt"
SystemPath = raw_input("Enter the system path of the dataset you want to classify")

#file_path = path.relpath("/Users/KeerthiTejaNuthi/Downloads/iris.data.txt")
file_path = path.relpath(SystemPath)
data_list = []
with open (file_path, 'rb') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        data_list.append(row)
for row in range(len(data_list)):
    for i in range(0,len(data_list[row])-1):
        data_list[row][i] = float(data_list[row][i])

#calculate entropy for a list typed data
def entropy(data_set):
    entrpy = 0.0
    length = len(data_set)
    diction = {}
    for data in data_set:
        diction.setdefault(data[-1], 0)
        diction[data[-1]] += 1
    for count in diction.values():
        prob = count/length
        entrpy -= prob * log(prob, 2)
    return entrpy

def entropy_split(data_set, feature):
    final_list = []
    children = {}
    total_length = len(data_set)
    entropy_data_list = entropy(data_set)
    feature_list = []
    for data in data_set:
        feature_list.append(data[feature])
    feature_list.sort()
    mean = [(feature_list[i] + feature_list[i+1])/2 for i in range(0,len(feature_list)-1)]
    for m in mean:
        small_child = []
        large_child = []
        for i in range(len(data_set)):
            if data_set[i][feature] >= m:
                large_child.append(data_set[i])
            else:
                small_child.append(data_set[i])
        sc = len(small_child)/total_length
        lc = len(large_child)/total_length
        entropy_children = (sc*entropy(small_child)) + (lc*entropy(large_child))
        final_entropy = entropy_data_list - entropy_children
        final_list.append((final_entropy,m))
        children[final_entropy] = [small_child,large_child]
    ent, t = max(final_list, key = lambda x:x[0])
    return children[ent][0], children[ent][1], ent, t

def gini_split(data_list, feature):
    gini = 0.0
    final_lchild = None
    final_rchild = None
    t = None
    #children = {}
    total_length = len(data_list)
    gini_data_list = entropy(data_list)
    feature_list = []
    for data in data_list:
        feature_list.append(data[feature])
    #print feature_list
    feature_list.sort()
    mean = [(feature_list[i] + feature_list[i+1])/2 for i in range(0,len(feature_list)-1)]
    final_list = []
    for m in mean:
        small_child = []
        large_child = []
        for i in range(len(data_list)):
            if data_list[i][feature] >= m:
                large_child.append(data_list[i])
            else:
                small_child.append(data_list[i])
        sc = len(small_child)/total_length
        lc = len(large_child)/total_length
        gini_children = (sc*entropy(small_child)) + (lc*entropy(large_child))
        final_gini = gini_data_list - gini_children
        final_list.append((final_gini,m))
        #children[final_gini] = [small_child,large_child]
        if final_gini > gini:
            gini =final_gini
            t = m
            final_lchild = small_child
            final_rchild = large_child
    #gini, t = max(final_list, key = lambda x:x[0])
    return final_lchild, final_rchild, gini, t
    #return children[gini][0], children[gini][1], gini, t
    #return small_child, large_child, gini, t

#Calculate gini of a list typed data
def gini_impurity(data_set):
    length = len(data_set)
    #print length
    total = 0
    diction = {}
    for data in data_set:
        diction.setdefault(data[-1],0)
        diction[data[-1]] += 1
    #print diction
    for key in diction:
        #print diction[key]
        total += pow((diction[key]/length),2)
    impure_gini = 1 - total
    return impure_gini


def find_split_attribute(train_set):
    if InfoGain == 'gini':
        gini_data_list = gini_impurity(train_set)
    else:
        gini_data_list = entropy(train_set)
    best_gain = 0
    split_feature = None
    split_thresh = None
    best_l_child = None
    best_r_child = None
    total_length = len(train_set)
    features = []
    #mark
    for i in range(0,len(train_set[0])-1):
        features.append(i)
    for feature in features:
        feature_list = []
        for data in train_set:
            feature_list.append(data[feature])
        #print feature_list
        feature_list.sort()
        #print feature_list
        mean = [(feature_list[i] + feature_list[i+1])/2 for i in range(0,len(feature_list)-1)]
        for m in mean:
            small_child = []
            large_child = []
            for data in train_set:
                if isinstance(m,int) or isinstance(m,float):
                    if data[feature] >= m:
                        large_child.append(data)
                    else:
                        small_child.append(data)
                else:
                    if data[feature] == m:
                        large_child.append(data)
                    else:
                        small_child.append(data)
            #print small_child
            sc = float(len(small_child))/total_length
            lc = float(len(large_child))/total_length
            #print sc,lc
            if InfoGain == 'gini':
                gini_children = (sc*gini_impurity(small_child)) + (lc*gini_impurity(large_child))
            else:
                gini_children = (sc*entropy(small_child)) + (lc*entropy(large_child))
            final_gain = gini_data_list - gini_children
            #print final_gain
            if final_gain > best_gain and len(small_child) > 0 and len(large_child) > 0:
                split_thresh = m
                split_feature = feature
                best_gain = final_gain
                best_l_child = large_child
                best_r_child = small_child
    return split_thresh, split_feature, best_gain, best_l_child, best_r_child

def create_tree(train_set):
    Tree = Node()
    length = len(train_set)
    if length == 0:
        return Tree
    t, split_attribute, info_gain, left_child, right_child = find_split_attribute(train_set)
    if info_gain > 0:
        l_child = create_tree(left_child)
        r_child = create_tree(right_child)
        return Node(left_child=l_child, right_child=r_child, threshold=t, features=split_attribute)
    else:
        temp = {}
        for data in train_set:
            temp.setdefault(data[-1], 0)
            temp[data[-1]] += 1
        return Node(majority=temp)

def check_feature(data, feature, th):
    if isinstance(th,int) or isinstance(th,float):
        if data[feature] < th:
            return False
        else:
            return True

def testing(struct, data):
    while struct.majority == None:
        if check_feature(data,struct.features,struct.threshold):
            struct = struct.left_child
        else:
            struct = struct.right_child
    return (struct.majority)

def training(tree, test_set):
    tr = 0
    fls = 0
    for data in test_set:
        if testing(tree, data).get(data[-1]):
            tr += 1
        else:
            fls += 1
    print "correctly predicted", tr
    print "wrong predicted", fls
    return tr/(tr + fls)

# 10 fold cross validation

for n in range(0,10):
    test_set = []
    train_set = []
    rand = range(len(data_list))
    random.shuffle(rand)
    for i in range(int(len(data_list)*0.80)):
        train_set.append(data_list[rand[i]])
    for i in range(int(len(data_list)*0.80), len(data_list)):
        test_set.append(data_list[rand[i]])
    decision_tree = create_tree(train_set)
    percentage = training(decision_tree, test_set)
    acc = percentage*100
    print "The accuracy of this model in ",n+1,"th fold is: %.2f%%" % acc
