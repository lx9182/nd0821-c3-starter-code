# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Using Random Forest Classifier algorithm, scikit-learn library has a built-in algorithm which is RandomForestClassifier. The model parameter was:
- random_state=42
- max_depth=16
- n_estimators=100

## Intended Use

Model is used to predict a person's income range based on some information like: age, workclass, education, marital status,... 

## Training Data

Census data from UCI library: https://archive.ics.uci.edu/dataset/20/census+income

## Evaluation Data

Evaluation Data is splitted from training dataset (20%)

## Metrics

Precision: 0.7423971377459749
Recall: 0.5460526315789473
F-beta: 0.6292645943896891

## Ethical Considerations

Census data from UCI should be credited when model is publicized

## Caveats and Recommendations
Make sure data is cleaned: remove space, handle missing information, drop unnecessary column.
