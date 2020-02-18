#  Created by Luis Alejandro (alejand@umich.edu)
import numpy as np

def report_feature_ranking(rank, feature_names, print_count = 6):
    """
    Prints out the feature with its corresponding rank value

    Arguments:
        rank(nparray): a rank of each feature
        feature_names(list): a list of all feature names
        print_count(int): how many features to print. It picks half from the top, half form the bottom to print
    """
    
    indexes = rank.flatten().argsort()
    d = len(indexes)
    if print_count > d:
        print_count = d
    
    # prints top features
    top = int(np.ceil(print_count / 2))
    for i in range(1, top + 1):
        print('Feature ranked %d is (%s) with value %lf' % (i,feature_names[indexes[-i]],rank[indexes[-i]]))
    
    # prints the points if needed
    if d > print_count:
        print('.\n.\n.\n')
        
    # prints bottom features
    bottom = print_count - top
    for i in range(bottom-1,-1,-1):
        print('Feature ranked %d is (%s) with value %lf' % (d - i,feature_names[indexes[i]],rank[indexes[i]]))    