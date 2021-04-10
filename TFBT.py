""" Gradient Boosted Tree"""

# Author: Seyedsaman Emami 

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)



import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin


class TFBT(BaseEstimator, ClassifierMixin):
    def __init__(self, *,
                 n_batches_per_layer=1,
                 label_vocabulary=None,
                 n_trees=1,
                 max_depth=5,
                 learning_rate=0.1,
                 features=None,
                 step=100):
        self.n_batches_per_layer = n_batches_per_layer
        self.label_vocabulary = label_vocabulary
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.features = features
        self.step = step
        
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
    
    '''

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

    def fit(self, X, y):
      
      """Run fit with all sets of parameters.
      
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            
        y : array-like of shape (n_samples,) 
        
        """

        X = X.astype("int64")

        if self.label_vocabulary is None:
            y = y.astype("int64")
        else:
            y = y.astype(str)

        # NUM_EXAMPLES = len(y)

        # Training and evaluation input functions.
        train_input_fn = self._make_input_fn(X, y)

        # feature selection
        num_columns = self.features
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
                                                       label_vocabulary=self.label_vocabulary)
        self.est.train(train_input_fn, max_steps=self.step)

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

        X = X.astype("int64")

        if self.label_vocabulary is None:
            y = y.astype("int64")
        else:
            y = y.astype(str)

        eval_input_fn = self._make_input_fn(X, y,
                                            shuffle=False,
                                            n_epochs=1)

        return self._accuracy(self.est.evaluate
                              (eval_input_fn, steps=1))
