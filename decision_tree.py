'''
CAPP30122 W'20: Building decision trees

Raymond Eid & Marc Richardson
'''

import math
import sys
import pandas as pd
import textwrap

class Tree(object):

    def __init__(self, data):
        cols = data.columns
        yn = dict(data[data.columns[-1]].value_counts())
        if 1 in yn.keys() or 0 in yn.keys():
            yes, no = (1, 0)
        else:
            yes, no = ('Yes', 'No')

        self.yes_count = yn.get(yes, 0.)
        self.no_count = yn.get(no, 0.)
        self.total_count = sum(yn.values())
        self.label = str(yes) if self.yes_count > self.no_count else str(no)
        self.data = data.copy()
        self.split_attribute = None
        self.edge = None
        self.children = []
        self.attributes = set(data.columns[:-1])
        self.gini = 1 - ((self.yes_count / self.total_count)**2 + (self.no_count\
                            /self.total_count)**2)

    def num_attributes(self):
        '''
        '''
        return len(self.attributes)

    def equal_attr_division(self):
        '''
        '''
        return all(self.data[self.attributes].nunique() == 1)

    def split_categories(self, attr):
        '''
        '''
        categories = {}
        col = self.data[attr]
        for cat in col.unique():
            categories[cat] = self.data[col == cat][self.data.columns[-1]].\
                              value_counts().to_dict()

        return categories

    def calc_gain_ratio(self, attr):
        '''
        '''
        weighted_avg = 0.
        split_info = 0.
        categories = self.split_categories(attr).values()
        if attr == 'Outlook':
            print("split categories")
            print(self.split_categories(attr))
        for value in categories:
            total_count = sum(value.values())
            no = value.get(1, 0.) / total_count  # hard-coding no/yes
            yes = value.get(0, 0.) / total_count
            weight = total_count / self.total_count
            if weight == 1.0:
                continue
            log_weight = math.log(weight)
            split_info += weight * log_weight
            weighted_gini = (1 - (no**2 + yes**2)) * weight
            weighted_avg += weighted_gini
            if attr == 'Outlook':
                print()
                print('Outlook stats')
                print('Weight is', weight)
                print('Weighted Gini is', weighted_gini)
        gain = self.gini - weighted_avg
        gain_ratio = gain / -split_info
        if attr == 'Outlook':
            print("gain is", gain)
            print('gain_ratio', gain_ratio)
            print('split_info is', split_info)
            print('weighted_avg', weighted_avg)

        return gain_ratio

    def find_best_split(self):
        '''
        '''
        best_attribute = None
        best_gain_ratio = 0.0
        for attribute in self.attributes:
            gain_ratio = self.calc_gain_ratio(attribute)
            if best_attribute is None or gain_ratio > best_gain_ratio:
                best_attribute = attribute
                best_gain_ratio = gain_ratio
            elif gain_ratio == best_gain_ratio:
                print()
                print('two attributes have best gain ratio')
                print(best_attribute, attribute)
                best_attribute = min(best_attribute, attribute)
                print(best_attribute)
                print(self.data)
        print(best_attribute)

        return best_attribute, best_gain_ratio

    def __print_r(self, prefix, last, kformat, vformat, eformat, lformat, maxdepth):
        ''' Recursive helper method for print() '''
        if maxdepth is not None:
            if maxdepth == 0:
                return
            else:
                maxdepth -= 1

        if len(prefix) > 0:
            if last:
                lprefix1 = prefix[:-3] + u"  └──"
            else:
                lprefix1 = prefix[:-3] + u"  ├──"
        else:
            lprefix1 = u""

        if len(prefix) > 0:
            lprefix2 = prefix[:-3] + u"  │"
        else:
            lprefix2 = u""

        if last:
            lprefix3 = lprefix2[:-1] + "   "
        else:
            lprefix3 = lprefix2 + "  "


        if self.total_count is None:
            ltext = (kformat).format(self.label)
        else:
            ltext = (eformat + ": " + kformat + ": " + vformat + ": " + lformat).format(self.edge, 
                                                            self.split_attribute,
                                                               self.total_count,
                                                               self.label)

        ltextlines = textwrap.wrap(ltext, 80, initial_indent=lprefix1,
                                   subsequent_indent=lprefix3)

        print(lprefix2)
        print(u"\n".join(ltextlines))

        if self.children is None:
            return
        else:
            for i, st in enumerate(self.children):
                if i == len(self.children) - 1:
                    newprefix = prefix + u"   "
                    newlast = True
                else:
                    newprefix = prefix + u"  │"
                    newlast = False

                st.__print_r(newprefix, newlast, kformat, vformat, eformat, lformat, maxdepth)

    def print(self, kformat="{}", vformat="{}", eformat="{}", lformat="{}", maxdepth=None):
        '''
        Inputs: self: (the tree object)
                kformat: (format string) specifying format for label
                vformat: (format string) specifying format for label and count
                maxdepth: (integer) indicating number of levels to print.
                          None sets no limit
                verbose: (boolean) Prints verbose labels if True

        Returns:  no return value, but a tree is printed to screen
        '''
        self.__print_r(u"", False, kformat, vformat, eformat, lformat, maxdepth)


def go(training_filename, testing_filename):
    '''
    Construct a decision tree using the training data and then apply
    it to the testing data.

    Inputs:
      training_filename (string): the name of the file with the
        training data
      testing_filename (string): the name of the file with the testing
        data

    Returns (list of strings or pandas series of strings): result of
      applying the decision tree to the testing data.
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
    tree.print()

    return predictions


def build_tree(tree):
    '''
    '''

    # Base Case 1
    if not tree.num_attributes():
        return tree

    # Base Case 2 (make into attribute (bool or dict) or method)
    if tree.yes_count == 0 or tree.no_count == 0:
        return tree

    # Base Case 3
    if tree.equal_attr_division():
        return tree

    best_attr, best_gain = tree.find_best_split()

    # Base case 4 
    if best_gain == 0.:
        return tree

    # Recursive Case
    tree.split_attribute = best_attr
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
    '''

    print()
    print(tree.label)
    print(tree.split_attribute)
    print(tree.data)
    print()
    #base case
    if tree.split_attribute is None:
        return tree.label
    cat = row.loc[tree.split_attribute]
    if cat not in tree.data[tree.split_attribute].unique():
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
