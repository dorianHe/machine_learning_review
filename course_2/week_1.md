# Week 1

## Take-aways:

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

## Recaps

### Gradient check
Gradient check is basically for checking the correctness of your implementation of backpropagation by comparing the difference between the vectorized backpropagation value and numerical approximation of gradient. The explanation here is mainly based on 
Andrew Ng's [machine learning course](https://www.youtube.com/watch?v=P6EtCVrvYPU)

Assume our funtion is $J(\theta)$ (assume $\theta \in R$). The gradeint of function $J$ w.r.t $\theta$ is 
$$\frac{dJ}{d\theta} \approx \frac{J(\theta + \epsilon) - J(\theta - \epsilon)}{2\epsilon}$$
When $\epsilon$ is small, we can consider the approximation as equality. This is called [symmetric difference quotient](https://en.wikipedia.org/wiki/Symmetric_derivative).
Andrew also mentioned another way of approxmation, [Newton's difference quotient](https://en.wikipedia.org/wiki/Difference_quotient) (see below).
He claimed that the first approximation is better. I am not going to dig that deep why the first one is better here. So for now let's just remember this rule.
$$
\frac{dJ}{d\theta} \approx \frac{J(\theta + \epsilon) - J(\theta)}{\epsilon}
$$

In higher dimension space, assume $\theta \in n$, we just need to do this approximation for every $\theta_i$ in $\theta$.
$$
\begin{align}
\frac{\partial J}{\partial \theta_i} \approx \frac{J(\theta_i + \epsilon) - J(\theta_i - \epsilon)}{2\epsilon}
\end{align}
$$
### P-norm regularization
For details pls check another md file [here](https://github.com/dorianHe/machine_learning_review/blob/master/course_2/pnorm_regularization.md)

### Inverted dropout

...
