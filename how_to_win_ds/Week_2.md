# Exploratory Data Analysis
## Exploratory data analysis
* Get domain knowledge
* Check if the data is intuitive
* Understand how the data was generated

## EDA examples
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
* Holdout scheme (huge datasest & K-Fold scores on each fold are roughly the same)
* K-Fold scheme (Medium-sized dataset & scores on each fold differ noticeably)
* LOO (Leave-One-Out) scheme

## Problem occuring during validation
* Causes of different scores and optimal parameters
    * Too little data
    * Too diverse and inconsistent data
We should do extensive validation.
* Average scores from different KFold splits.
* Tune model on one split, evaluate score on the other.

* Indication of an expeceted leaderboard shuffle
    * Diferent public / private data or target distributions
    * Little amount of training or / and testing data
    * Most of the competitors have very similar scores
# Data leakages
