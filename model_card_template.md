# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A RandomForestClassifier was used from the sklearn library
- Date: 29/03/2022
#### Hyperparameters
- n_estimators = 100
- max_depth = 15

## Intended Use
This model is intended to classify the salary class given US census data as a part of Udacity MLOps online degree.

## Training Data
Census data provided by UCI 

## Evaluation Data
Census data provided by UCI 

## Metrics
Precision, Recall and F1 scores were used to evaluate this model.
#### Performance on validation set:
- Precision: 79.3
- Recall: 57.5
- F1: 66.7

## Ethical Considerations
The inferences produced by this model are directly related to the data provided by UCI. Concerns regarding race and social status biases were not considered in this project.

## Caveats and Recommendations
A meaningful hyper-parameter search was not attempted in this project. It is therefore highly recommeded to consider that as a next step.