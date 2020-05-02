'''
CAPP30122 W'20: Building decision trees

Raymond Eid & Marc Richardson
'''

import math
import sys
import pandas as pd


class Tree:
    '''
    Class for representing a tree node
    '''

    def __init__(self, data):
        '''
        Constructor

        Inputs:
            data: (pandas DataFrame) object holding training data information
        '''

        cols = data.columns

        self.data = data.copy()
        self.yes, self.no, yn = self.set_binary(cols)
        self.attributes = set(cols[:-1])
        self.yes_count = yn.get(self.yes, 0)
        self.no_count = yn.get(self.no, 0)
        self.label = str(self.yes) if self.yes_count > self.no_count \
                     else str(self.no)
        self.total_count = sum(yn.values())
        self.split_on = None
        self.edge = None
        self.children = []
        self.gini = 1 - (self.yes_count / self.total_count)**2 - (self.no_count\
                    /self.total_count)**2

    def set_binary(self, columns):
        '''
        Defines the nomenclature for the binary variables

        Inputs:
            columns: (pandas Index object) the Tree's attributes

        Returns:
            yes: (str or int) The naming for the affirmative
                prediction
            no: (str or int) The naming for the negative
                prediction
            yn: (dictionary) dictionary containing counts the target attribute
                at the current Tree node
        '''

        yn = dict(self.data[columns[-1]].value_counts())

        if 1 in yn.keys() or 0 in yn.keys():
            yes, no = (1, 0)
        else:
            yes, no = ("Yes", "No")

        return yes, no, yn


    def num_attributes(self):
        '''
        Returns the number of attributes at the Tree node
        '''

        return len(self.attributes)


    def equal_attr_division(self):
        '''
        Returns a boolean indicating if the Tree node's attributes all have
        only one distincy category split
        '''

        return all(self.data[self.attributes].nunique() == 1)


    def split_categories(self, attr):
        '''
        Creates a nested dictionary holding each of the given attributes'
        categories, which are also split on the target (binary) attribute

        Input:
            attr: (str) Name of the column to split on

        Returns:
            categories: (dict) contains the attribute's cateogry and
                        binary splits
        '''

        categories = {}
        col = self.data[attr]

        for cat in col.unique():
            categories[cat] = self.data[col == cat][self.data.columns[-1]].\
                              value_counts().to_dict()

        return categories


    def calc_gain_ratio(self, attr):
        '''
        Calculates the gain ratio for a given attribute

        Input:
            attr: (str) Name of the attribute to split on

        Returns:
            gain_ratio: (float) calculated from each attribute's cateogry splits
                        and the weighted gini value of each attribute category's
                        split on the target attribute
        '''

        weighted_avg = 0.
        split_info = 0.
        categories = self.split_categories(attr).values()

        if len(categories) == 1:
            return 0.
        for value in categories:
            total_count = sum(value.values())
            no = value.get(self.no, 0.) / total_count
            yes = value.get(self.yes, 0.) / total_count
            weight = total_count / self.total_count
            log_weight = math.log(weight)
            split_info += weight * log_weight
            weighted_gini = (1 - (no**2 + yes**2)) * weight
            weighted_avg += weighted_gini

        gain = self.gini - weighted_avg
        gain_ratio = gain / -split_info

        return gain_ratio


    def find_best_split(self):
        '''
        Find the best attribute to split the data on and returns that attribute
        and its gain ratio.

        Returns:
            best_attribute: (str) the attribute with the highest gain ratio
            best_gain_ratio: (float) the optimal gain ratio among all attributes
        '''

        best_attribute = None
        best_gain_ratio = 0.0

        for attribute in self.attributes:
            gain_ratio = self.calc_gain_ratio(attribute)
            if best_attribute is None or gain_ratio > best_gain_ratio:
                best_attribute = attribute
                best_gain_ratio = gain_ratio
            elif gain_ratio == best_gain_ratio:
                best_attribute = min(best_attribute, attribute)

        return best_attribute, best_gain_ratio


    def __repr__(self):
        '''
        Format Tree as a string
        '''

        s = ("Tree attributes \nColumns: {}\nConnecting edge: {}\nColumn split:"
             " {}\nNumber of Obs: {}\n")
        output = s.format(self.attributes, self.edge, self.split_on,
                          self.total_count)

        return output


def go(training_filename, testing_filename):
    '''
    Construct a decision tree using the training data and then apply
    it to the testing data

    Inputs:
      training_filename (str): the name of the file with the
        training data
      testing_filename (str): the name of the file with the testing
        data

    Returns: (list of strings or pandas series of strings): result of
              applying the decision tree to the testing data
    '''

    predictions = []
    train_data = pd.read_csv(training_filename, sep=",")
    test_data = pd.read_csv(testing_filename, sep=",")
    train_root = Tree(train_data)
    tree = build_tree(train_root)

    for i in range(len(test_data)):
        row = test_data.iloc[i]
        pred = traverse_tree(tree, row)
        predictions.append(pred)

    return predictions


def build_tree(tree):
    '''
    Given a root node, recursively builds a series of interconnected trees
    which comprise the decision tree model

    Inputs:
        tree: (a tree object)

    Output:
        tree: (a Tree object) a tree one level lower than the inputed tree
    '''

    if not tree.num_attributes():
        return tree

    if tree.yes_count == 0 or tree.no_count == 0:
        return tree

    if tree.equal_attr_division():
        return tree

    best_attr, best_gain = tree.find_best_split()

    if best_gain == 0:
        return tree

    tree.split_on = best_attr
    for cat in tree.split_categories(best_attr).keys():
        data = tree.data[tree.data[best_attr] == cat].copy()
        data.drop(columns=[best_attr], inplace=True)
        child = Tree(data)
        child = build_tree(child)
        child.edge = cat
        tree.children.append(child)

    return tree


def traverse_tree(tree, row):
    '''
    Runs a row from the testing dataset through the model tree to assign a
    predicted outcome for the observation

    Input:
        tree: (Tree object) The decision tree model built from the training
            data
        row: (Pandas series object) an observation from the training dataset

    Output:
        yn: (string) a string indicating an affirmative or negative predicted
            outcome
    '''

    if tree.split_on is None:
        return tree.label

    cat = row.loc[tree.split_on]
    if cat not in tree.data[tree.split_on].unique():
        return tree.label

    for child in tree.children:
        if cat == child.edge:
            yn = traverse_tree(child, row)

    return yn


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python3 {} <training filename> <testing filename>".format(
            sys.argv[0]))
        sys.exit(1)

    for result in go(sys.argv[1], sys.argv[2]):
        print(result)
