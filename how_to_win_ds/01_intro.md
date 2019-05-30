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

* Feature extraction from text and images: the main goal of post-processing here is 
	1. to make samples more comparable on one side, using term frequency: 
		* tf = 1 / x.sum(axis=1) [:,None]
		* x = x * tf
	2.  to boost more important features while decreasing the scale of useless ones, using inverse document frequency:
		idf = np.log(x.shape[0] / (x > 0).sum(0)) 
		x = x * idf
		
* Pipeline of applying bag of words:
	
	1. Preprocessing: Lowercase, stemming, lemmatization, stopwords
	2. Bag of words: Ngrams can help to use local context
	3. Postprocessing: TFiDF
	
* BOW and w2v comparison
	* Bag of words
		* Very large vectors
		* Meaning of each value in vector is known
	* Word2vec
		* Relatively small vectors
		* Values in vector can be interpreted only in some cases
		* The words with similar meaning often have similar embeddings
	* Rule of thumb: usually, both Bag of Words and Word2vec approaches give quite different results and can be used together in your solution.
	
* Feature extraction from text and images
	1. Texts
		a. Preprocessing: 
			i. Lowercase, stemming, lemmarization, stopwords 
		b.Bag of words
			i. Huge vectors
			ii. Ngrams can help to use local context
			iii. TFiDF can be of use as postprocessing
		c. Word2vec
			i. Relatively small vectors
			ii. Pretrained models
	2. Images
		a. Features can be extracted from different layers
		b. Careful choosing of pretrained network can help
		c. Finetuning allows to refine pretrained models
		d. Data augmentation can improve the model

* Descriptors (i.e., outputs from inner layers) from later layers are better way to solve texts similar to one network was trained on. In contrary, descriptors from early layers have more text independent information. 

* Fine-tuning (i.e.,  process of pre-trained model tuning), especially for small data sets, is usually better than training standalone model on descriptors or a training network from scratch: 
	1. fine-tuning is better than training standalone model on descriptors because it allows to tune all networks parameters and thus extract more effective image representations.
	2. fine-tuning is better than training network from scratch if we have too little data, or if the text we are solving is similar to the text model was trained on.
		
## Final Project Description
