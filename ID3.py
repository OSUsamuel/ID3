import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import json


CLASSES = {0: 'Iris Setosa', 1: 'Iris Versicolor', 2: 'Iris Virigincia'}


"""
Given a np.ndarray calculate the entropy
"""
def entropy(column: np.ndarray) -> float:
    length = len(column)
    _, counts = np.unique(column, return_counts=True)
    probs = counts/length
    log_probs = np.log(probs)
    return -np.sum(probs * log_probs)


"""
Calculates conditional entropy of datasets
i.e. for each unique value in given feature, find the conditional entropy
    of the target column given said feature value 
"""
def conditional_entropy(df, feature, target_col):
    length = len(df[feature])
    unique_vals, counts = np.unique(df[feature], return_counts=True)
    
    entropies = []
    for unique, count in zip(unique_vals, counts):
        Ht = entropy(df[df[feature] == unique]['target'])
        entropies.append(count/length*Ht)
    return sum(entropies)

"""
Calculates the information gain given entropy and conditional entropy
"""
def information_gain(entropy, conditional_entropy):
    return entropy - conditional_entropy


def id3(tree, subset):
    if(len(subset.columns) == 1):
        return {'leaf' : int(subset['target'].mode()[0])}

    S = entropy(subset['target'])
    IG = {}
    for col in subset.columns:
        if col != 'target':
            S_given_A = conditional_entropy(subset, col, 'target')
            IG[col] = information_gain(S, S_given_A)

# Calculates the best feature and makes it the root
    best_feature = max(IG, key=IG.get)

    tree['node'] = best_feature
    tree['children'] = {} 


    for label in subset[best_feature].unique():
        set = subset[subset[best_feature] == label]
        set = set.drop(columns = [best_feature])
        tree['children'][label] = id3({}, set)

    return tree


def inference(tree, value):
    if('leaf' in tree):
        return tree['leaf']

    for interval in tree['children'].keys():
        if value[tree['node']] in interval:
            return inference(tree['children'][interval], value) 

    best = min(tree['children'].keys(), key=lambda iv: min(abs(iv.left - value[tree['node']].mid), abs(iv.right - value[tree['node']].mid)))
    return inference(tree['children'][best], value)

if __name__ == "__main__":
    # Loads the dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sets up the dataframe
    df = pd.DataFrame(train_X, columns=iris.feature_names)

    bin_edges = {}
    for col in df.columns:
        df[col], edges = pd.cut(df[col], bins=[i/2 for i in range(0, 20, 1)], retbins=True)
        bin_edges[col]  = edges

    df['target'] = train_y

    tree = id3({}, df)

    test_df = pd.DataFrame(test_X, columns=iris.feature_names)
    test_df['target'] = test_y


    for col in test_df.columns:
        if col != 'target':
            clipped = np.clip(test_df[col], bin_edges[col][0], bin_edges[col][-1])
            test_df[col] = pd.cut(clipped, bins=bin_edges[col])


    print(test_df)

    predicted = np.ones(len(test_df)) * -1
    for index, row in test_df.drop(columns=['target']).iterrows():
        predicted[index] = inference(tree, row)

    print(predicted)
    print(test_df['target'].to_numpy() - predicted)
