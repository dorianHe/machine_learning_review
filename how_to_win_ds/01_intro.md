# Intro and recap
## Take-aways
* To prevent overfitting of test set, organizers of Kaggle competition do:
	* Spliting test for publich / private
	* Adding non-relevant samples to test set
	* Limiting number of submissions per day

* Things we should take care about during the competition is the target metric value.

* SVM is a linear model with special loss function. Even with "kernel trick", it's still linear in new, extended space.

* ExtraTrees classifier always tests random splits over fraction of features (in contrast to RandomForest, which tests all possible splits over fraction of features)
