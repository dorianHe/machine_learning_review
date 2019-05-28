# EDA
## Building intuition about the data
* Get domain knowledge
* Check if the data is intuitive
* Understand how the data was generated

## Exploring anonymized data
* Try to decode the features
* Guess the feature types

helpful functions can be:
* df.dtypes
* df.info()
* x.value_counts()
* x.isnull()

## Visulizations
* Explore individual features
    * Histogram
    * Plot
    * Statistic
* Explore feature relations
    * Pairs
	* Scatter plot, scatter matrix
	* Corrplot
    * Groups
	* Corrplot + clustering
	* Plot (index vs feature statistics)
# Validation
## Validation strategies
* Holdout scheme
* K-Fold scheme
* LOO (Leave-One-Out) scheme

## Problem occuring during validation
* Causes of different scores and optimal parameters
    * Too little data
    * Too diverse and inconsistent data
We should do extensive validation.
* Average scores from different KFold splits.
* Tune model on one split, evaluate score on the other.

# Data leakages
