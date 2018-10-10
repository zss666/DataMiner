# Import, read, and split data
import pandas as pd
data = pd.read_csv('./data/data2.csv')
import numpy as np
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

# Fix random seed
np.random.seed(55)


from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# It is good to randomize the data before drawing Learning Curves
def randomize(X, Y):
    permutation = np.random.permutation(Y.shape[0])
    X2 = X[permutation,:]
    Y2 = Y[permutation]
    return X2, Y2

X2, y2 = randomize(X, y)
def draw_learning_curves(X, y, estimator, num_trainings):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X2, y2, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.plot(train_scores_mean, 'o-', color="g",
             label="Training score")
    plt.plot(test_scores_mean, 'o-', color="y",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()


### Imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


### Logistic Regression
estimator1 = LogisticRegression()
draw_learning_curves(X=None, y=None, estimator=estimator1, num_trainings=10)

### Decision Tree
estimator2 = GradientBoostingClassifier()
draw_learning_curves(X=None, y=None, estimator=estimator2, num_trainings=10)

### Support Vector Machine
estimator3 = SVC(kernel='rbf', gamma=1000)
draw_learning_curves(X=None, y=None, estimator=estimator3, num_trainings=10)
