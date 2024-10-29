# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

A Logical Regession model was used in this analysis.

## Intended Use
The intended use of this model was to analyze the Census Income Data from a 1994 Census dataset. The goal was to predict whether someone made over $50,000 per year based on the census data. The overall project was to teach us about how to deploy a Machine Learning pipeline. 

## Training Data
The data used came from a 1994 Census Income dataset.
There are 14 features in the dataset that are either Integers or Categorical. 

## Evaluation Data
The data was split between a trained dataset(80%) and a test dataset(20%).
I used hyperparameters to improve metric results.

## Metrics
The metrics used were Precision, Recall, and F1.
The results of these metrics when used on the test data were: Precision: 0.7523, Recall: 0.6130, and F1: 0.6756.

## Ethical Considerations
There is a chance of there being biases within the date. Further research would need to be done before this data could be used for anything other than educational purposes. 

## Caveats and Recommendations
There are possible biases within the data.