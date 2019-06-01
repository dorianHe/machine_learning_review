# Exploratory Data Analysis
## Exploratory data analysis

* With EDA we can: 1）get comfortable with the data and 2) find magic features. Do EDA first. Do not immediately dig into modelling.

* In order to build intuition about the data, we should:

	* Get domain knowledge – It helps to deeper understand the problem
	* Check if the data is intuitive – And agrees with domain knowledge
	* Understand how the data was generated – As it is crucial to set up a proper validation
	
* Two things to do with anonymized features:

	1. Try to decode the features: guess the true meaning of the feature
	2. Guess the feature types: each type needs its own preprocessing

* Explore individual features
	
	* Histogram
	* Plot(index vs. value)
	* Statistics

* Explore feature relations
	
	* Pairs
		* Scatter plot, scatter matrix
		* Corrplot
	* Groups
		* Corrplot + clustering
		* Plot (index vs feature statistics)
		
## Validation
* Notes on validation: 
	* Validation helps us evaluate a quality of the model
	* Validation helps us select the model which will perform best on the unseen data
	* Underfitting refers to not capturing enough patterns in the data
	* Generally, overfitting refers to
		* capturing noize
		* capturing patterns which do not generalize to test data
	* In competitions, overfitting refers to low model’s quality on test data, which was unexpected due to validation scores


* Validation strategies

	* There are three main validation strategies:
		1. Holdout
		2. KFold 
		3. LOO
	* Stratification preserve the same target distribution over different folds

# By Dorian
## EDA examples
* Try to decode the features
* Guess the feature types
* Check the notebook on this [link](https://jokkzswedqbueyrtgrujqn.coursera-apps.org/notebooks/readonly/reading_materials/EDA_Springleaf_screencast.ipynb).

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
