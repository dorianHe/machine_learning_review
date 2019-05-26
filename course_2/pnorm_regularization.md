# norm and regularization
## Definition of $p-$norm
The definition of norm: a norm is a function that assigns a strictly positive length or size to each vector in a vector space—except for the zero vector, which is assigned a length of zero. 

$$\|\|x\|\|\_p := (\sum_{i=1}^{n}|x_i|^p)^{\frac{1}{p}}$$

If we take $p=1$, we have Mahattan norm or $L_1$ norm, namely $||x||\_1 := \sum_{i=1}^{n}|x_i|$

If we take $p=2$, we have euclidean norm or $L_2$ norm, namely $||x||\_2 := (\sum_{i=1}^{n}|x\_i|^2)^{(1/2)} = \sqrt{\sum\_{i=1}^{n}x\_i^2}$

## The usage of norm in machine learning
### As loss function
It is quite obvious that the mean-square-error (MSE) is based on $p=2$ and mean-absolute-error (MAE) is based on $p=1$.

### As regularization
To fight against overfitting, regularization is introduced. Based on different types of norm, we have different types of regularization.
The regularization is added to the original loss function. For example, $\mathcal{L} = $MSE$(y, \hat{y}) + \alpha R $, 
where $w$ represent the weight in the model, $\alpha$ is a hyperparameter, and R is either $L_1$ or $L_2$ regularization, which is based on the definition of the norm.

$$ R = \|\|w\|\|\_1 := \sum_{i=1}^{n}|w_i|$$

$$ R = ||w||^2_2 := \sum_{i=1}^{n}|w_i|^2 = \sum_{i=1}^{n}w_i^2 $$

The general idea of using regularization is acutally based on joint loss. By introducing another component in the loss function, 
we have some certain measure of learned $w$. By changing the $\alpha$ value,
we change the penalty of learned $w$ in the total loss such that we more and less avoid overfitting problem. With $L_1$ regularization our model tries to make the $w$ sparse. With $L_2$ our model tries to reduce the magnitude of $w$.

#### Robustness of $L_1$ and $L_2$ regularization
There are many great blog posts about the differences between $L_1$ and $L_2$ regularizations. This part is mostly based on this [kaggle notebook](https://www.kaggle.com/residentmario/l1-norms-versus-l2-norms). 

The biggest difference between $L_1$ and $L_2$ regularization of learned weight $w$ is obviously the definition.

$$|\|w\|\|\_1 := \sum_{i=1}^{n}|w_i|$$

$$||w||^2_2 := \sum_{i=1}^{n}|w_i|^2 = \sum_{i=1}^{n}w_i^2 $$

Assume that there are two same models with same weight $w$, during the training process, both models face an outlier. 

According to the definition of $L_1$ and $L_2$ regularization, we know $||w||\_1 < ||w||^2\_2$, which means that $L_2$ regularization causes more penalty. And the weight $w$ in model with $L_2$ regularization is updated more than $w$ in model with $L_1$. This is not what we want. That's why we say $L_1$ regularization is more robust than $L_2$. 

Also based on the definition, $L_1$ introduces more sparsity.

#### Example visualization of $L_1$ and $L_2$ regularization
<img src="/images/L_1_regularization.png"  width="320" height="280"> <img src="/images/L_2_regularization.png"  width="320" height="260">
