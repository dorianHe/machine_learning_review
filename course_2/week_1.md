# Week 1

Take-aways:

* with a higher amount of data, a larger proportion should be assigned to the training set.

* the dev and test set should come from the same distribution.

* for a NN with high bias, it is promising to try (with more parameters, i.e.):

   * to make the model deeper
    
   * to increase the number of units in the hidden layers
    
* For classifier with a low error rate in the training set but a high error in the test set (i.e. overfitting):
  
  * increase the regularization parameter
  
  * get more training data
  
* def.: weight decay is a regularization technique that results in gradient descent shrinking the weights on every iteration.

* With a increase of the regularization hyperparameter lambda, weights are pushed toward becoming smaller (i.e. close to $0$)

* Inverted dropout at test time means that you don't apply dropout nor keep the 1/keep_prob factor in the calculations during training.

* With a smaller dropout rate:

  * regularization effect is reduced
  
  * NN ends up with a lower training set error.

* Useful techniques for reducing variances (i.e. reducing overfitting):

  * L1 / L2 regularization
  
  * Dropout
  
  * Data augmentation
  
* Normalizing the inputs makes the cost function faster to optimize.
