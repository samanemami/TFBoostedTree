# TFBT.BoostedTree
A [gradient boosting](TFBT.py) classifier based on the [tensorflow](https://github.com/tensorflow/estimator/blob/781c0d30c6bf100aa174591dd97cb70fc39d294d/tensorflow_estimator/python/estimator/canned/boosted_trees.py#L1933).
<br/>
A novel estimator based on the [tensorflow](https://github.com/tensorflow/estimator/blob/781c0d30c6bf100aa174591dd97cb70fc39d294d/tensorflow_estimator/python/estimator/canned/boosted_trees.py#L1933) and Sklearn standard.
This model, unlike the TF method, has fit and score methods. Therefore, one can implement [gridsearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) or [pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) as well.


# Title
Compact boosted tree

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
1.1.1
<br/>
In the latest update, there is no need to convert data to pandas DataFrame, also, the model will extract the features as well.
Moreover, the predict_proba method has been added.
