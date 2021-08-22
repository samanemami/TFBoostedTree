""" Gradient Boosted Tree"""

# Author: Seyedsaman Emami

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)


import numpy as np
import pandas as pd
import os
import os.path
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin


class TFBT(BaseEstimator, ClassifierMixin):

    '''
    This model is based on Sklearn standards. 
    Therefore, it contains fit, score, and 
    predict probe. 
    I developed this class to implement the 
    GridsearchCV and pipeline.

    '''

    def __init__(self, *,
                 n_batches_per_layer=1,
                 label_vocabulary=None,
                 n_trees=1,
                 max_depth=5,
                 learning_rate=0.1,
                 max_step=100,
                 steps=100,
                 model_dir=None):
        self.n_batches_per_layer = n_batches_per_layer
        self.label_vocabulary = label_vocabulary
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_step = max_step
        self.steps = steps
        self.model_dir = model_dir

        '''
    Parameters
    -----------------
    n_batches_per_layer: if it is equal to 1,
            it wil consider entire dataset.

    label_vocabulary: if the labels are string,
            then add the label vector in string.

    n_trees: number of trees in model.

    max_depth: depth of tree.

    learning_rate: shrinkage parameter to be 
            used when a tree added to the
            model.

    features: Vector of features in String.

    step: Number of boosting iterations.

    model_dir: If (model_dir), then at each training 
    iteration, the algorithm will empty the checkpoint_path. 
    This will increase your disk space during the training process.
    If you have memory limitations, use a path.


    '''

    def _dataframe(self, X, y):
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        X = X.astype("int64")
        y = y.astype(str) if (self.label_vocabulary) else y.astype("int64")

        self.feature = []
        for i in range(len(X.columns)):
            self.feature.append(str(i))

        col_rename = {i: j for i, j in zip(X.columns, self.feature)}
        X = X.rename(columns=col_rename, inplace=False)

        return X, y

    def _make_input_fn(self, X, y, n_epochs=None, shuffle=True):
        def input_fn():
            NUM_EXAMPLES = len(y)
            dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
            if shuffle:
                dataset = dataset.shuffle(NUM_EXAMPLES)
            # For training, cycle thru dataset as many times as need (n_epochs=None).
            dataset = dataset.repeat(n_epochs)
            # In memory training doesn't use batching.
            dataset = dataset.batch(NUM_EXAMPLES)
            return dataset
        return input_fn

    def _accuracy(self, evaluate):
        item = list(evaluate.items())
        array = np.array(item)
        return (array[0, 1]).astype(np.float64)


class BoostedTreesClassifier(TFBT):

    def fit(self, X, y):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_samples,) 

        """

        X, y = self._dataframe(X, y)

        # Training and evaluation input functions.
        train_input_fn = self._make_input_fn(X, y)

        # feature selection
        num_columns = self.feature
        feature_columns = []
        n_classes = len(np.unique(y))

        for feature_name in num_columns:
            feature_columns.append(tf.feature_column.numeric_column(feature_name,
                                                                    dtype=tf.float32))
        self.est = tf.estimator.BoostedTreesClassifier(feature_columns,
                                                       n_batches_per_layer=self.n_batches_per_layer,
                                                       n_classes=n_classes,
                                                       n_trees=self.n_trees,
                                                       max_depth=self.max_depth,
                                                       learning_rate=self.learning_rate,
                                                       label_vocabulary=self.label_vocabulary,
                                                       model_dir=self.model_dir)

        self.est.train(train_input_fn, max_steps=None,
                       steps=self.steps)

        return self

    def score(self, X, y):
        """Returns the score on the given data
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_samples)

        Returns
        -------
        score : float
        """
        X, y = self._dataframe(X, y)

        eval_input_fn = self._make_input_fn(X, y,
                                            shuffle=False,
                                            n_epochs=1)

        accuracy = self._accuracy(self.est.evaluate
                                  (eval_input_fn,
                                   steps=self.steps))

        if (self.model_dir):
            for root, dirs, files in os.walk(self.model_dir):
                for file in files:
                    os.remove(os.path.join(root, file))

        return accuracy

    def predict_proba(self, X, y):
        X, y = self._dataframe(X, y)
        eval_input_fn = self._make_input_fn(X, y,
                                            shuffle=False,
                                            n_epochs=1)
        pred_dicts = list(self.est.predict(eval_input_fn))

        return pd.Series([pred['probabilities']
                          for pred in pred_dicts])


class BoostedTreesRegressor(TFBT):

    def fit(self, X, y):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        y : array-like of shape (n_samples,) 

        """

        X, y = self._dataframe(X, y)

        # Training and evaluation input functions.
        train_input_fn = self._make_input_fn(X, y)

        # feature selection
        num_columns = self.feature
        feature_columns = []

        for feature_name in num_columns:
            feature_columns.append(tf.feature_column.numeric_column(feature_name,
                                                                    dtype=tf.float32))
        self.est = tf.estimator.BoostedTreesRegressor(feature_columns,
                                                      n_batches_per_layer=self.n_batches_per_layer,
                                                      n_trees=self.n_trees,
                                                      max_depth=self.max_depth,
                                                      learning_rate=self.learning_rate,
                                                      model_dir=self.model_dir)

        self.est.train(train_input_fn, max_steps=None,
                       steps=self.steps)

        return self

    def _predict(self, X, y):
        X, y = self._dataframe(X, y)
        eval_input_fn = self._make_input_fn(X, y,
                                            shuffle=False,
                                            n_epochs=1)
        pred_ = [pred['predictions']
                 for pred in list(self.est.predict(eval_input_fn))]

        pred = []
        for i in range(len(pred_)):
            pred.append(pred_[i][0])
        return np.array(pred)

    def score(self, X_true, y_true):
        """Mean squared error regression loss
        ----------
        X_true: array-like of shape (n_samples, n_features)
        Input data

        y_true : array-like of shape (n_samples,)
        correct target values.

        Returns
        -------
        score : float or ndarray of floats
        """
        pred = self._predict(X_true, y_true)
        output_errors = np.average((y_true - pred) ** 2, axis=0)

        return np.sqrt(output_errors)
