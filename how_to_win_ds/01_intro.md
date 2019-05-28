# Week 1
## Introduction & Recap
### Competition mechanics

* To prevent overfitting of test set, organizers of Kaggle competition do:
	
	* Spliting test for publich / private
	* Adding non-relevant samples to test set
	* Limiting number of submissions per day

* Things we should take care about during the competition is the target metric value.

### Recap of main ML algorithms
	
* There is no “silver bullet” algorithm
	
* Linear models split space into 2 subspaces
	
* Tree-based methods splits space into boxes
	
* k-NN methods heavy rely on how to measure points “closeness”
	
* Feed-forward NNs produce smooth non-linear decision boundary
	
* The most powerful methods are Gradient Boosted Decision Trees and Neural Networks. But you shouldn’t underestimate the others

* SVM is a linear model with special loss function. Even with "kernel trick", it's still linear in new, extended space.

* ExtraTrees classifier always tests random splits over fraction of features (in contrast to RandomForest, which tests all possible splits over fraction of features)
	
### Software/Hardware requirements

* Anaconda works out-of-box
	
* Proposed setup is not the only one, but most common
	
* Don’t overestimate role of hardware/software

## Feature Preprocessing and Generation with Respect to Models

### Feature Preprocessing and Generation with Respect to Models

* Feature preprocessing is necessary instrument you have to use to adapt data to your model. 
	
* Feature generation is a very powerful technique which can aid you significantly in competitions and sometimes provide you the required edge.
	
* Both feature preprocessing and feature generation depend on the model you are going to use.
	
* In General case, we should use preprocessing to scale all features to one scale, so that their initial impact on the model will be roughly similar.
	
* Numeric feature preprocessing (scaling and rank) is different for tree and non-tree models:
		
	* Tree-based models doesn’t depend on them
	* Non-tree-based models hugely depend on them
	
* Most often used preprocessings are:
		
	* MinMaxScaler - to [0,1]
	* StandardScaler - to mean==0, std==1
	* Rank - sets spaces between sorted values to be equal
	* np.log(1+x) and np.sqrt(1+x)
	
* Feature generation is powered by:
	
	* Prior knowledge
	* Exploratory data analysis
	
* For categorical features:

	* Values in ordinal features are sorted in some meaningful order
	* Label encoding maps categories to numbers
	* Frequency encoding maps categories to their frequencies
	* Label and Frequency encodings are often used for tree-based models
	* One-hot encoding is often used for non-tree-based models
	* Interactions of categorical features can help linear models and KNN
	
* For Datetime:

	* Periodicity
	* Time since row-independent/dependent event
	* Difference between dates
	
* For Coordinates:

	* Interesting places from train/test data or additional data
	* Centers of clusters
	* Aggregated statistics

* Handling missing values: 
	
	* The choice of method to fill NaN depends on the situation
	* Usual way to deal with missing values is to replace them with -999, mean or median
	* Missing values already can be replaced with something by organizers
	* Binary feature “isnull” can be beneficial
	* In general, avoid filling nans before feature generation
	* Xgboost can handle NaN
	
## Final Project Description
