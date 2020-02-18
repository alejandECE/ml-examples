#  Created by Luis Alejandro (alejand@umich.edu)
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

# Utility function to report best scores in the cross-validation
def report_grid_search(results, n_top=3):
    """ Report the results from a grid search using GridSearchCV implementation

        Parameters: 
        results: The resulting object from GridSearchCV fit method
        n_top (int): Top ranks to report
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
# Utility function to report the test peformance
def report_classification(y_true,y_pred,avg='binary',title='Test'):
    """ Report the classification results using accuracy, f1 score, recall and precision

        Parameters: 
        y_pred: Vector of predicted outputs
        y_true: Vector of true outputs
        avg: Indicates what average mode (binary, micro, macro) to use
        title: Title shown in the output
    """
    
    print(title, '(Metrics): ')
    print('')
    print('Accuracy: ', '%.2f' % accuracy_score(y_true,y_pred))
    print('F1 Score: ', '%.2f' % f1_score(y_true,y_pred,average=avg))
    print('Recall: ', '%.2f' % recall_score(y_true,y_pred,average=avg))
    print('Precision: ', '%.2f' % precision_score(y_true,y_pred,average=avg))
    print('\nConfusion Matrix:\n', confusion_matrix(y_true,y_pred))