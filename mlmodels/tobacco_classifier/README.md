## Brief explanation of mlmodels modules

* Preprocessing.py, contain preprocessing functions for pandas dataFrame format, the to-be-process text should be in label data['text']
* TrainingProcess.py, ML training process of creating and saving the model
* TobaccoClassifier.py (depends on Preprocessing), main module used to classify tobacco tweets

To use the classify module, please import the module as follow
```
from mlmodels import *
```

There are two 2 model pickle files under the folder
* tobacco_cv.sav, count vectorize fiting model
* tobacco_model.sav, sklearn SVC classify model

There is also 1 jupyter notebook demo for the usage of classifier as references

* ClassifyDemo.ipynb
