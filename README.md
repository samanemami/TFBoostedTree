# TFBT.BoostedTree(Wrapper)
A [gradient boosting](TFBT.py) classifier based on the [tensorflow](https://github.com/tensorflow/estimator/blob/781c0d30c6bf100aa174591dd97cb70fc39d294d/tensorflow_estimator/python/estimator/canned/boosted_trees.py#L1933).
<br/>
A wrapper of the tf.BoostedTree of [tensorflow](https://github.com/tensorflow/estimator/blob/781c0d30c6bf100aa174591dd97cb70fc39d294d/tensorflow_estimator/python/estimator/canned/boosted_trees.py#L1933) based on the Sklearn standard.
This model, unlike the TF method, has fit and score methods. Therefore, one can implement [gridsearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) or [pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) as well.


# Title
TensorFlow Compact Boosted Tree Wrapper

## Citation 
If you use this script, please [cite](CITATION.cff) it as below.

```yaml
References:
    Author:
      - Seyedsaman Emami
```

### License
The package is licensed under the [GNU Lesser General Public License v2.1](https://github.com/GAA-UAM/GBNN/blob/main/LICENSE).

# Usage
import into python file. 

```python
import TFBT
import sklearn.datasets as dts
from sklearn.model_selection import train_test_split

X, y = dts.load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2)
    
model = model = TFBT(learning_rate=1, features=feature, max_depth=2)
model.fit(x_train, y_train)
model.score(x_test, y_test)
model.predict_proba(x_test, y_test)
```

One could also implement GridsearchCV to tune the hyper-parameters

```python
import TFBT
import sklearn.datasets as dts
from sklearn.model_selection import GridSearchCV, train_test_split

model = model = TFBT(learning_rate=1, features=feature, max_depth=2)

X, y = dts.load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2)

param = {"max_depth": [1, 2, 5, 10, 20],
         "learning_rate": [0.025, 0.5, 0.1, 1]}
         
grid = GridSearchCV(model, param)
grid.fit(x_train, y_train)
grid.score(x_test, y_test)

```

# Contributions
[Contributions](contributing.txt) to this model are welcome! . You can improve this project by creating an issue, 
reporting an improvement or a bug, forking and creating a pull request to the 
development branch. 

# Keywords
**`Boosted_tree`**, **`tensorflow`**

# Version 
2.3.1
<br/>
In the latest update, there is no need to convert data to pandas DataFrame, also, the model will extract the features as well.
Moreover, the predict_proba method has been added.
Also, by adding the path to the `model_dir` you can increase your disk space and reduce memory usage. If you set the `model_dir`, the wrapper will delete the saved logs to increase the disk space.

# Latest updates

03.sep.2021
<ul>
<li>In the last update, as I need the exported models in the wrapper, I added this feature from the TensorFlow library. </li>
<li>If you need more detail about the added method (export_saved_model), check the [reference](https://www.tensorflow.org/api_docs/python/tf/estimator/BoostedTreesClassifier). </li>
</ul>

12.sep.2021
<ul>
<li>Add a method to measure the training time </li>
</ul>