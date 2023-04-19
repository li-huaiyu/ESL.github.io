[toc]



# Maching Learning Procedure

## Prep Methodology

- Breadth First studying plan. Acutally *write* codes.
- Iterate over material, instead of trying to learn everything of a single topic at the first try. 
- Iteratevely update your knowledge, notes, even this list

- Coursera: [U Mich Applied ML](https://www.coursera.org/learn/python-machine-learning/home/module/3), [U co](https://www.coursera.org/learn/introduction-to-machine-learning-supervised-learning?specialization=machine-learnin-theory-and-hands-on-practice-with-pythong-cu), [projects](https://www.coursera.org/projects/used-car-price-prediction-using-machine-learning-models)
- Kaggle: Pandas and ML
- ESL, [Codes](https://github.com/empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks/tree/master/examples) replicating plots in the book. NYU [course](https://github.com/davidrosenberg/mlcourse) by David Rosenberg. [Codes](https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition_Original) for ML in Algo Trading. Perdue course [slides](https://engineering.purdue.edu/ChanGroup/ECE595/files/)
- [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
- The YouTube channel [StatQuest](https://www.youtube.com/watch?v=3CC4N4z3GJc), Tübingen [course](https://www.youtube.com/watch?v=jFcYpBOeCOQ&list=PL05umP7R6ij2XCvrRzLokX6EoHWaGA2cC&index=1)
- 中文书，[Medium 中文站](https://easyaitech.medium.com/), [知乎](https://zhuanlan.zhihu.com/p/82054400)，[Visualize ML](https://github.com/Visualize-ML/Book7_Visualizations-for-Machine-Learning)

[Lecture 1](https://github.com/davidrosenberg/mlcourse/blob/gh-pages/Lectures/01.black-box-ML.pdf) of D. R., Jansen MLAT Chapter 6

- Define the problem and measure of success
  - Regression: predict returns
  - Binary classification: future price movements
  - multiclass problem: performance classes such as return quitiles.
  - Prediction vs inference (clear for linear models); 

- Collect, clean and validate data
- Feature selection and engineering
- split labeled data into train, validation, test.

- - Build / revise feature extraction methodology
  - choose ML algorithm
  - train model (learn parameters), tune hyperparamters (CV, etc.)
  - evaluate prediction functions on validation set
  - repeat
- Retrain model on train + val
- Evaluate performance on test set
- Retrain on train + val + test
- Deploy

Test performance $\approx$ deployment performance



# Basics of Statistical Learning Theory

[D. R. Slides](https://github.com/davidrosenberg/mlcourse/blob/gh-pages/Lectures/01b.intro-stat-learning-theory.pdf)

- Make decision $\to$ take actions $\to$ produce output $\to$ evaluation
- Observe input $x$ , take action $a$, observe outcome $y$, evaluate using $\ell(a,y)$, i.e. the loss function; outcome is oftern but not always indep of $a$. 

## The spaces

- Input space $\mathcal X$, action (predicted result) space $\mathcal A$, target space $\mathcal Y$.
- corresp. to feature data, predictions, truths
- **Prediction** function: $f: \mathcal X \to \mathcal A$. **Loss** function $\ell: \mathcal A \times \mathcal Y \to \mathbb R$. **Loss** evaluates a **single** action. 

## Statistical learning framework

***Assumption: actions don't affect outputs***

- Data generating distribution $P = \mathbb P_{\mathcal X \times \mathcal Y}$, draw i.i.d. pairs $(x,y)$: $\mathcal D_n = \{ (x_i,y_i)\big\}_{i = 1}^n $, define the **risk** as $R[f] = \mathbb E\Big[\ell\big( f(x) , y\big)\Big]$. **Bayes optimal prediction function** $$f^\star : \mathcal X \to \mathcal A \in \mathrm{argmin}_{f} R[f]$$, this minimum is **Bayes risk**. 
- BPF of regression with *L2 loss* is conditional $\mathbb E$: $f^\star(x) = \mathbb E[y|X = x]$
  multiclass-classification: $x \mapsto$ most probable class $f^\star(x) = \mathrm{argmin}_{g\in \mathcal G} \mathbb P[g|X = x]$

Risk **cannot be computed** (or no need of learning). **SLLN**: mean of iid samples a.s. conv. to $\mathbb E$. 

**Empirical risk**: mean of sample loss. $\ell \big(f(x_i), y_i\big)$ is a seq of iid rv's:
$$
\hat R_n[f] := n^{-1} \sum_{i=1}^n \ell \big(f(x_i), y_i\big) \overset{\mathrm{a.s.}}{\to} R[f]
$$


 Minimizer of ER is $\hat f\in \mathrm{argmin}_f \hat R_n[f]$, the **ER minimizer**.

- Typically **multiple** ERMs, not all make sense. Will overfit: fit training data perfectly but performs poorly on test data. 
- **Hypothesis space**: set of functions from which to choose PFs. Need regularity, complexity criterias... Linear functions, polynomials, locally constant functions on sets with lower-bounded areas. Hypothesis space $\mathcal F$, ERM in $\mathcal F$ is the fitted PF in this model.

## Linear Probabilistic models

Given $x$, preict probability distribution $p(y)$. Consider parametric families of distributions. 

- Generalized linear modeld: Linear regression (normal with fixed variance); logistic/probit regression (Bernoulli); Poisson regression (Poisson)
- Generalized additive models, gradient boosting machines
- Most neural network models in practice

### Bernoulli regression

Input space $\mathcal X = \mathbb R^p$, target space $\mathcal Y = \{0,1\}$. Sufficient to specify $\theta = \mathbb P(y=1)$. Construct **scores** from $x \in \mathbb R^p$. Linear method: $x \mapsto w^\mathsf T x + b$. **Transfer function** to map $\mathbb R^p$ into $[0,1]$. The prediction function if $x \mapsto f\big(w^\mathsf T x) = \theta$. 

## Gradient Descent (GD)

Need to find minimizer of $f$, step size $\eta$: $x \leftarrow x - \eta \nabla f(x)$. If $f$ is convex, $\eta$ smaller than $(\mathrm{Lip} \nabla f)^{-1}$ then converges $\big| f\big(x^{(k)}\big) - f(x^*) \big| \leq \frac{\big\| x^{(0)} - x^*\big\|^2}{2 \eta k}$. 

- set a stopping criterion, e.g. $\|\nabla f\| < \epsilon$.

For Linear regression:

- Loss = $\ell (\hat y, y) = (y-\hat y)^2$ and Hypothesis space $\mathcal F = \big\{ w^\mathsf T x:\ \beta \in \mathbb R^p\big\}$. The ER is $\hat R_n(w) = n^{-1} \sum_{i=1}^n \big(w^\mathsf T x_i - y_i)$.

Generally if $n$ is large, $\nabla \hat R_n(w) = n^{-1} \sum \nabla_w \ell \big( f ( x_i; w), y_i \big)$ takes $\mathcal O(n)$ time to compute. **Intuition**: don't have to be precise at every step, correct afterwards. Choose a **minibatch** instead of the whole batch $n$ points. Choose $m_{i}$ for $i = 1,\cdots N < n$. Since each point $x_i^{(k)}$ is equally likely to be chosen at step $k$, the minibatch batch $\nabla \ell$ average is an **unbiased estimator** of $\nabla R_n(w)$. 

- do not need independence of sampling these $N$ points.
- you can choose even $N = 1$
- make sure refer to the minibatch size of SGD when talking about it.

**Robbins-Monro conditions**, the stepsizes $\eta_k$ should be $\ell^2$ summable but not $\ell^1$ summable. $O (k^{-1})$ for example. But SGD works with fixed step size due to tolarance. Decrease if distance to minimum not reduced. 

There are other variance-reduction techniques, for example SAGA which is used by sklearn, see [this](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) documentation page. 

## Excess Risk Decomposition

Recall the class of prediction functions is restricted. There are two approximations. 

**Approximation error** : the risk of the best among the model, $R(f_\mathcal F) - R(f^\star)$ which is **not** random. $\mathcal F \uparrow$ then AE $\downarrow$.

**Estimation error**: the risk due to not having this suboptimal PF, $R\big(\hat f_n\big)-R(f_\mathcal F)$, $\hat f_n$ being the ERM. 

- Smaller $\mathcal F$, fixed num of data points $\Rightarrow$ EE $\downarrow$. Typically, more training data $\Rightarrow$ EE $\downarrow$.

- $\hat f_n$ is the minimizer of the *empirical* risk, the minimum of ER can be small while **excess risk**:= AE + EE be big (overfitting)

### Framework of Empicial Risk Minimization

[D. R. slides](https://github.com/davidrosenberg/mlcourse/blob/gh-pages/Lectures/02b.excess-risk-decomposition.pdf)

Input: loss function, hypothesis space; output: minimizer of empirical risk. Need to choose $\mathcal F$ (tune hyperparam etc); as we get more training data, enlarge $\mathcal F$. ERM $\hat f_n\in \mathcal F$. Get an **approximate ERM** $\tilde f_n$. 

The **opitmization error**: $R(\tilde f_n) - R(\hat f_n)$, may be *negative*, since $\hat f_n$ is the minimzer of $\hat R$ not $R$.

- In practice, the excess risk decomposes into OptErr+EstErr+ApprErr
  $$
  R\big(\tilde f_n\big) - R \big(f^\star\big) = R\big(\tilde f_n\big) - R\big(\hat f_n\big) + R \big(\hat f_n \big) - R \big(f_\mathcal F \big) + R \big(f_\mathcal F \big) - R\big(f^\star\big)
  $$

Usually impossible to find this decomp explicitly since no information of $\mathbb P_{\mathcal X \times\mathcal Y}$. Construct artificial example. See slides.





# Linear regression

- Linearity in coefficients $w$'s, the traning **features**, can be $X_1^2, X_2\sin X_3, \cdots$ or other base functions.

- The **dofs** of lasso and ridge; larger penalty parameter `alpha` usually means smaller dof. Formulae see ESL. For **ridge** $\mathrm{dof} = \sum_{j=1}^p\frac{d_j^2}{d_j^2+\lambda}$ ; dof of **lasso** given alpha under **homoskedastic** assumption = Expected number of active features $\mathbb E |\mathcal A|$. 

- outliers > nonlinearity > heteroscedasticity > non-normality

## Linear least squares regression

Linear models: $\mathcal F$ linear functions or affine functions on $\mathbb R^p$. With square empirical loss. 
$$
\hat w = \mathrm{argmin}_{w \in \mathbb R^p}
\big\| \mathbf X^\mathsf T w - \mathbf y\big\|_2^2
\equiv \mathrm{argmin}_{w \in \mathbb R^p} \frac{1}{n} 
\sum_{i=1}^n \big( w^\mathsf T x_i - y_i \big)^2
$$
 This can **overfit** for $p$ large compared to $n$ (NLP, sequencing etc.) Recall the minimizer of the quadratic risk $\mathrm{argmin}_{g} \mathbb E \big[ Y - g(X) \big]$ with $(X,Y) \sim f_{X,Y}$ is just $g = E[Y|X] = \int_\mathbb R f_{Y|X}(y|X) y \ \mathrm{d} y$ where $f_{Y|X}(y|x) f_X(x) = f_{X,Y}(x,y)$. The empirical risk minimization is to replace $f_{X,Y}$ by (in terms of cdf): $F_n(x,y) = N^{-1} \sum_{i=1}^N \mathbb 1 \big\{ X_i \leq x ,  y_i \leq y\big\}$. 

### Univariate linear regression

The least square estimator is the MLE of $p(y|x) \sim \mathcal N (b + wx, \sigma^2)$. $\hat w = \frac{\mathrm{cov} (x,y)}{\mathrm{var} x}$ where cov and var are empirical ones. $\hat b = \bar y_n - \hat w \bar x_n$. The MLE of $(b,w)$ matches the least square estimators obtained by ERM. Also the MLE of $\sigma^2$ is $n^{-1} \sum_{i=1}^N \epsilon_i^2$, where $\epsilon_i = y_i - b - w x_i$. Since $y|x$ is multivariate normal, $(\hat b,\hat w)$ is as well. In shorthands writing $\hat w$ for the pair, there is $\mathbb V \hat w = \big(\mathbf X^\mathsf T \mathbf X \big)^{-1} \sigma^2 $. In particular, 
$$
\hat b \sim  \mathcal N \bigg( w, \frac{\sigma^2 \sum_{i=1}^N x_i^2 / N}{\mathrm{var} x} \bigg),\ \hat w \sim \mathcal N \bigg( w, \frac{\sigma^2}{\mathrm{var} x} \bigg)
$$
Intuitive way to remember is $\mathbb V \hat w = \sigma^2 \mathrm{var} x $.

### Weighted Least Squares

$p(y|x) \sim \mathcal N \big( x^\mathsf T w , \sigma(x)^2\big)$, that is $\mathbf y = \mathbf X w + \boldsymbol \epsilon$, let $\mathbf \Omega(\mathbf X) = \mathbb V \boldsymbol \epsilon = \mathrm{diag} \begin{bmatrix} \sigma(x_1)^2 & \cdots & \sigma(x_N)^2\end{bmatrix} $. Then the MLE is the minimizer of the NLL: $\hat w = \mathrm{argmin} \big(\mathbf y - \mathbf X w )^\mathsf T \mathbf \Omega^{-1} \big(\mathbf y - \mathbf X w )  $ thus 
$$
\hat w = \big(\mathbf X^\mathsf T \mathbf \Omega^{-1} \mathbf X\big)^{-1} \mathbf X^\mathsf T \mathbf \Omega^{-1} \mathbf y
$$


### Gauss-Markov Theorem

Let $\hat w (\mathbf y) = \big(\mathbf X^\mathsf T \mathbf X\big)^{-1} \mathbf X^\mathsf T \mathbf y$ be the OLS estimator **given fixed $\mathbf X$**. Then if $\hat w_1$ is any other linear (in $\mathbf y$) unbiased estimator of $w$, then any linear combination of these satisfies $\mathbb V \big(v^\mathsf T \hat w\big) \leq \mathbb V \big(v^\mathsf T \hat w'\big)$.  Equivalently, $\mathbb V \hat w \preceq \mathbb V \hat w_1$.

**Proof** Let $\hat w_1 = \mathbf W_1^\mathsf T \mathbf y$, and $\mathbf W_1 = \mathbf W_0 + \mathbf G$ then for any *true* $w$, there is $\mathbb E \hat w_1 = \mathbb E \Big[ \big(\mathbf X^\mathsf T \mathbf X\big)^{-1} \mathbf X^\mathsf T + \mathbf G^\mathsf T \Big] \big(\mathbf X w + \boldsymbol \epsilon\big) = w + \mathbf G^\mathsf T \mathbf X w$. Thus since $w$ is arbitrary, $\mathbf G^\mathsf T \mathbf X = \mathbf 0_{p\times p}$. But the variance is $\mathbb V \hat w_1 = \mathbf W_1^\mathsf T \sigma^2 \mathbf W_1 = \mathbb V \hat w + \sigma^2 \mathbf G^\mathsf T \mathbf G \succeq \mathbb V \hat w$.

### Distribution of errors

The sum of the squares of errors is called the **residual sum of squares** (RSS): $\mathrm{RSS} = \sum_{i=1}^N \hat \epsilon_i^2$. This quantity satisfies the $\chi^2_{N-p-1}$ distribution scaled by $\sigma^2$, so an unbiased estimator of the mean squared error is $\mathrm{MSE} = \frac{\mathrm{RSS}}{N-p-1}$. In fact, $\hat {\boldsymbol \epsilon} = \mathbf Y - \mathbf X \big(\mathbf X^\mathsf T \mathbf X \big)^{-1} \mathbf X^\mathsf T \mathbf Y = \Big[ \mathbf I_N - \mathbf X \big(\mathbf X^\mathsf T \mathbf X \big)^{-1} \mathbf X^\mathsf T\Big] \boldsymbol \epsilon$. 

Note that $\mathbf X \big(\mathbf X^\mathsf T \mathbf X \big)^{-1} \mathbf X^\mathsf T$ is an orthogonal projector of rank $p+1$ so $\hat{\boldsymbol \epsilon}$ is a linear combination of $N-p-1$ orthogonal Gaussian variables. The RSS is the sum of their squares thus is $\sigma^2$ times an rv of $\chi^2_{N-p-1}$. The uncorrelatedness of $\hat w$ and $\hat {\boldsymbol \epsilon}$ is due to $\mathrm{Cov} \big( \hat w , \hat{\boldsymbol \epsilon}\big) = \mathbb E \big(\hat w-w\big) \hat {\boldsymbol \epsilon} = 0$. Therefore this gives rise to the Student-t test since $t_\nu = \frac{Z}{\sqrt{\chi^2_\nu / \nu}}$. 

Since $\hat \sigma^2 / \sigma^2 \sim \chi^2_{N-p-1}$, in general, $\hat w_j \sim \mathcal N \big( w_j, \sigma^2 v_j)$. where $v_j$ is the $j$^th^ diagonal element of $\big(\mathbf X^\mathsf T \mathbf X \big)^{-1}$, thus 
$$
\frac{\hat w_j - w_j}{\sqrt{\hat \sigma^2 v_j} } = \frac{\big(\hat w_j - w_j\big) / \sigma \sqrt{v_j}}{\sqrt{\hat \sigma^2 / \sigma^2}} = \frac{Z}{\sqrt{\chi^2_{N-p-1}/(N-p-1)}}\sim \chi^2_{N-p-1}
$$
and we can conduct t-test with H0: $w_j=0$, and this statistic is the z-score $z_j$ of $w_j$.

Moreover, we can conduct an F-test with H0: $w_{j} = 0$ for $j = p_0+1, \cdots, p$. Then 
$$
\begin{aligned}
\hat {\boldsymbol \epsilon} & = \mathbf Y - \mathbf X \big(\mathbf X^\mathsf T \mathbf X \big)^{-1} \mathbf X^\mathsf T \mathbf Y 
= \Big[ \mathbf I_N - \mathbf P_1 - \mathbf X_0 \big(\mathbf X_0^\mathsf T \mathbf X_0 \big)^{-1} \mathbf X_0^\mathsf T\Big] \big(\mathbf X_0 \beta_0 + \boldsymbol \epsilon\big)
\\
& = \Big[ \mathbf I_N - \mathbf X \big(\mathbf X^\mathsf T \mathbf X \big)^{-1} \mathbf X^\mathsf T\Big] \boldsymbol \epsilon
\end{aligned}
$$
but $\hat{\boldsymbol \epsilon}_0 = \Big[ \mathbf I_N - \mathbf X_0 \big(\mathbf X_0^\mathsf T \mathbf X_0 \big)^{-1} \mathbf X_0^\mathsf T\Big] \boldsymbol \epsilon$, so $\mathrm{RSS}_0 - \mathrm{RSS} = \sigma^2 (p-p_0) \chi^2_{p-p_0}$, while $\mathrm{RSS} = \sigma^2 \big(N - p - 1\big) \chi^2_{N-p-1}$, whence with the definition for F-statistics, 
$$
\frac{\big(\mathrm{RSS}_0 - \mathrm{RSS}\big) / (p-p_0)}{\mathrm{RSS}/(N-p-1)} \sim F_{p-p_0,N-p-1}
$$
If $p-p_0=1$, this is the square of the z-score $z_j$.

## Tikhonov and Ivanov Regularization

**Nested sequences** of sp w/ incr cplx, $\mathcal F_d \subset \mathcal F$, polyn of deg $d$ in the sp of all polyn. **Complexity measures** $\Omega: \mathcal F \to [0,\infty)$ for decision functions: #var, depth of decision trees, deg of polyn, # of non0 coefs, **lasso** ($\ell^1$) or **ridge** ($\ell^2$) cplx $\sum_{j=1}^p |w_j|$ etc. 

Hyp sps can be nested by cplx means, $\mathcal F_r := \big\{f\in\mathcal F| \Omega[f] \leq r\}$. 

- **Ivanov** regularization: $\hat f \in \mathrm{argmin}_{f\in \mathcal F_r} n^{-1} \sum_{i=1}^n \ell \big( f(x_i),y_i\big)$, bounding *complexity*.
- **Tikhonov** regularization: penalization factor, $\hat f \in \mathrm{argmin}_{f\in \mathcal F} 
  n^{-1} \sum_{i=1}^n \ell\big( f(x_i) , y_i\big) + \lambda \Omega[f]$. **Unconstrained** minimization.
- Typically these are equiv. Minimizing $L[f]$ in $\mathcal F_r$, or over $\mathcal F$ but mnimizing $L[f] + \lambda \Omega[f]$, where $L[f]$ is the ER or other **performance metrics**.

Params $r$, $\lambda$ are chosen by e.g. validation data or **cross-validation**.



## Ridge regression and lasso regression

- Intuition: If data has high multicollinearity, restrict the sizes of coefs. PF is $\hat f(x) = x^\mathsf T \hat w$, $\mathrm{Lip}\hat f = \big\|\hat w\big\|_2$, 

The complexity measure $\Omega[f] = \| w\|_2^2 = \sum_{j=1}^p w_i^2$, or $\Omega[f] = \| w\|_1 = \sum_{j=1}^p |w_i|$. 

- Lasso induces feature **sparcity**: reduce time and space costs, interpretability, may give better prediction and as feature selection for traing slower nonlinear models.
- For ridge and lasso, Ivanov = Tikhonov.

The contours for the **quadratic** ER of $w$ are ellipsoids with axis in the singular directions of $\mathbf X$:
$$
\big\{\hat R_n(w) - \hat R_n(\hat w) = c\big\} = \big\{ (w-\hat w)^\mathsf T \mathbf X^\mathsf T \mathbf X (w-\hat w) = nc \big\}
$$
See slides for regions of $\hat w$ in which lasso gives sparse coefs. $\ell_q$ for $0\leq q<1$ is even sparser but the balls are not convex.

### Ridge

Assume the design matrix $\mathbf X$ is fixed, then with the OLS model $\hat w$ is the maximum likelihood operator under assumption $p\big(\mathbf y| X , w \big) \sim \mathcal N \big(\mathbf X w , \sigma^2 \mathbf I\big)$. Assume now there is prior $w \sim \mathcal N (0, \tau^2 I)$ then the posterior $p (w | \mathbf y,\mathbf X) \propto p(\mathbf y|\mathbf X, w) p(w) = \exp\Big[- \frac{|w|^2}{2\tau^2}\Big] \exp \Big[ \frac{-|\mathbf y-\mathbf X w |^2}{2\sigma^2}\Big]$, need to minimize $|\mathbf Y - \mathbf X w |^2 + \frac{\sigma^2}{\tau^2} |w|^2$, which is exacly ridge. 

Now let $\alpha = \sigma^2 / \tau^2$, taking derivative wrt $w$, $ - \mathbf X^\mathsf T \mathbf y + \mathbf X^\mathsf T \mathbf X w + \alpha w = 0$ namely $w = \big( \mathbf X^\mathsf T \mathbf X + \alpha I\big)^{-1} \mathbf X^\mathsf T \mathbf y$. If the design matrix has orthogonal columns then $w_j$ is shrunk with $d_j^2/(d_j^2+\alpha)$ factor.

### Lasso

Assume the prior for $w$ is given by $p(w|\tau) = (\tau/2)^p e^{-\tau |w|_1}$. The MAP is given by minimizing $2\sigma^2 \tau |w|_1 + |\mathbf y - \mathbf X w|^2$, $w_j = \mathrm{sgn}\big(w_j^\mathrm{OLS}\big)\big( |w_j^\mathrm{OLS}| - \alpha)_+$ where $\alpha = \sigma^2 \tau$.

### Lasso is a quadratic programming problem

 Unconstrained (Tikhonov) form of lasso problem 
$$
\min_{a, b\geq 0} L(a,b)+ \lambda \Omega(a, b) = \min_{a_i, b_i \geq 0} \sum_{i=1}^n \big[ \sum_{j=1}^p (a_j-b_j) x_{ij} - y_i]^2 + \lambda \sum_{j=1}^p(a_j+b_j)
$$
for any $j$, keeping $a_j-b_j$ constant, $\lambda \geq 0$ necessarily $a_jb_j = 0$ otherwise cannot minimize; $a^\star$ and $b^\star$ are positive and negative parts of $w^\star$. Objective function is convex. Let $L(a,b;\lambda)$ be the to-be-minimized function above. Use SGD (with minibatches) as if unconstrained by nonnegativity, then take positive part after each step.

### Lasso with coordinate descent method (shooting method)

Randomly or cyclically choose a coordinate $w_j$ to minimize. Use the original form of Tikhonov-regularized lasso empirical loss function in $\mathbb R^p$, in each step $w_j \leftarrow \mathrm{argmin}_{w_{j}} L ( \cdots, w_j, \cdots)$. See slides for closed-form solution. *(I basically employed this in Aquatic's data exercise...)*

Variation; write $\tilde w_j$, $j = 1,2,\cdots, 2p$ in place of $a_j,b_j$. Each step, $\tilde w_j \leftarrow \tilde w_j - \min \big\{ \tilde w_j , \partial_j L(\tilde w)\}$. Might not shrink to zero at once.

### Elastic net for the case where there are highly correlated data

Lasso is not ideal when some features are highly correlated: selects one of them arbitrarily. Want to 

- Select features 
- speard coefs evenly among highly correlated features (reducing estimation risk, i.e. variance of predictions).

# Linear Classification

## Logistic regression

**Logistic** regression is a class of **discriminative** classification (whereas LDA is **generative**), and an instance of GLM. 

### Binary logistic regression

The parameter is $\theta = (w,b)$. Define $\sigma(a)=\frac{1}{1+e^{-a}}$, then the model is given by $\mathbb P(y = 1 | x; \theta) = \sigma(a)$, $a = \log \Big(\frac p{1-p}\Big)$ = the log odd = the **logit** or **pre-activation**. The label $\mathcal Y = \{\pm1\}$ gives rise to $\mathbb P(Y = y| x, \theta) = \sigma(ya)$ since $\sigma(a) + \sigma(-a) = 1$. The **probability** of $y=1$ given $\theta$ of $x_i$ is $\mu_i = p(x_i|w) = \sigma\big(w^\mathsf T x\big)$, where $b$ is absorbed in $w$.

The linear **decision boundary** (DB) is given by $a = 0$, where $a = w^\mathsf T x + b$, which is an (affine) hyperplane. If the prediction data can be separated by a hyperplane call it **linearly separable**. The **direction** of $w$ decides the hyperplane, and the **magnitude** the **confidence** of the predictions. 

For data that are not linearly separable, we can add more transformed features. For example $\phi(x) = \begin{bmatrix} 1 & x_1 & x_2 \end{bmatrix}$ and $w = \begin{bmatrix} -R^2 & 1 & 1\end{bmatrix}$, the DB is given by $x_1^2 + x_2^2 - R^2 = 0$. Using neural networks one may decide $\phi(x)$ and $w$ at the same time.

#### Maximum likelihood formulation

Parameter $\theta = (w,b)$ from MLE. The **negative log likelihood** (NLL) is given by 
$$
\ell(w) = \mathrm{NLL}(w) = - N^{-1}\sum_{i=1}^N \log \big[\mu_i^{y_i} (1-\mu_i)^{1-y_i}] = N^{-1} \sum_{i=1}^N\mathbb H_\mathrm{ce} (y_i,\mu_i)
$$
where $\mathbb H_\mathrm{ce}(p,q) = - p\log q - (1-p) \log(1-q)$ is the **cross entropy**.

Now $\nabla_{w} \mu_i = \sigma'(a_i) \nabla_w a_i = \mu_i (1-\mu_i) x_i$, where $a_n = w^\mathsf T x_n$ and $\sigma' = \sigma(1-\sigma)$, and $\nabla_w \ell(w) = N^{-1} \sum_{i=1}^N (\mu_i - y_i) x_i$. 

> **Important remark** if the data is perfectly linearly separable, this will correspond to the case where $w$ in  $a = w^\mathsf T x + b$ goes to infinity. Then some regularization is needed.

The entries of Hessian is $\partial_{jk}^2 \ell(w) = N^{-1} \sum_{i=1}^N \mu_i(1-\mu_i) x_{ij} x_{ik} = N^{-1} \mathbf X^\mathsf T \mathbf S \mathbf X$, $\mathbf S =\mathrm{diag} \Big[ \mu_1(1-\mu_1) , \cdots, \mu_N (1-\mu_N) \Big]$, and the Hessian is positive-definite. However it is possible that the $\mu_i$'s are close to 0 or 1. Need regularization in this case. 

The **score** equations for the *exact* weights are $\sum_{i=1}^N \big(p(y_i|w) - y_i\big) x_i = 0$, these are $p+1$ nonlinear equations in $w$. The 0-th equation is $\sum_{i=1}^N y_i = \sum_{i=1}^N p(y_i|w)$, which means with exact weights, the empirical conditional expectaion of counts of class 1 is the count of observed class 1.

#### Numerically minimize NLL

If $N$ is large, we sample with minibatch of size $B_t$ at each iterative step $t$. Choosing $B_t = 1$, the process is $w_{t+1} \leftarrow w_t - \eta_t \partial_{i} \ell (w_t)$ with $\eta_t$ adjusted to avoid overshooting.

> Gradient descent is a first order optimization method, which means it only uses first order gradients to navigate through the loss landscape. This can be slow, especially when some directions of space point steeply downhill, whereas other have a shallower gradient. (PML)

Now we introduce the newton method for finding $w$. Update $w \leftarrow w - \eta \big( \nabla_w \nabla_w^\mathsf T \mathbf \ell(w)\big)^{-1} \nabla_w \ell(w)$, where $g = \nabla_w \ell(w)= \sum_{i=1}^N \big( p(x_i|w) - y_i\big) x_i$, and the Hessian $\nabla_w \nabla_w^\mathsf T \mathbf \ell(w) = H = N^{-1} \mathbf X^\mathsf T \mathbf S \mathbf X $. So $w = w_0 - H^{-1} g = \big( \mathbf X^\mathsf T \mathbf S \mathbf X  \big)^{-1} \mathbf X^\mathsf T \mathbf S \mathbf z$, where $\mathbf z := \mathbf X w + \mathbf S^{-1} \big[ \mathbf y - \mathbf p(\mathbf X|w)\big]$ is the **working response**. This is the **iteratively reweighted least squares** algorithm. 

We can understand the **IRLS** algorithm in this way. The weight $w$ is fitted so that the weighted square loss is minimized:
$$
w = \underset{w\in \mathbb R^{p+1} }{\mathrm{argmin}} \sum_{i=1}^N p(x_i|w_0) \big(1- p(x_i|w_0)\big) \big(z_i - w^\mathsf T x_i \big)^2
$$
The contribution to the $i$^th^ term comes from two parts:

1. The weight, which is **smaller** if non of the classes dominates, and **larger** if either $p(x_i|w_0) \to 1$ or $0$.
2. The error in scores $z_i - w^\mathsf T x_i = - (w-w_0)^\mathsf T x_i + \frac{y_i- p(x_i|w_0)} {p(x_i|w_0) \big(1- p(x_i|w_0)\big) }$. If $p(x_i|w_0)$ is close to 0, then a misclassification will result in $p(x_i|w_0)^{-1}$ which is large, and ; the same happens if it is close to 1, and $w-w_0$ is more affected. 

It is reminiscent to the AdaBoost algorithm. 

#### Regularization and Standardization

Using higher orders of feature transformations may result in overfitting. Using a Gaussian Prior for $w \sim \mathcal N (0, \lambda^{-1} I_{p+1})$ and the MAP estimator is the minimizer of $\ell(w) + \lambda \|w\|_2^2$. This is also called **weight decay**. Now the **penalized NLL** is $\mathrm{PNLL} = \mathrm{NLL} +\lambda w^{\mathsf T} w$, $\nabla \mathrm{PNLL} = g + 2 \lambda w$, $\nabla \nabla^\mathsf T \mathrm{PNLL} = H + 2 \lambda I_{p+1}$. This (and some gradient descent methods) needs **standardization**. 

# Model validation and selection

Data leakage [tutorial](https://www.kaggle.com/code/alexisbcook/data-leakage) on Kaggle. **Target leakage**: when predictors contain data that will not be available at the time of predictions. For example, whether someone has taken a drug targeting some disease should not be predicting whether she has this disease, since taking this drug usually happens **after** onset of the disease. 

Do not include features in historical data generated after the target is determined. That is, do not peek forward in time.

 **Train-test contamination** happens when the test data info affects training or preprocessing.

> If your validation is based on a simple train-test split, exclude the validation data from any type of *fitting*, including the fitting of preprocessing steps.

## Loss functions 

See [this](https://heartbeat.comet.ml/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0) post on L1, L2, Huber, Log-cosh, and quantile loss functions. Details TBD. 

**RMS of log of error** is appropriate when the target is subject to exponential growth, but one can log-transform the target and then use RMSE.

**R2** score is one minus the ratio of the RSS of the fitted linear model, and the RSS of the dummy model where the prediction is the mean. 



## Examples from classification, Michigan [Applied ML course](https://www.coursera.org/learn/python-machine-learning/lecture/90kLk/confusion-matrices-basic-evaluation-metrics)

`GridSearchCV` uses grid search for different hyperparams for model tuning. The split is done in the training set -> training+validation. K-Fold Validation are done for different **tuned** models to compare theri performance.

Compare models to the `DummyRegressor` regression model with different `strategies` . See [this](https://www.coursera.org/learn/python-machine-learning/lecture/BE2l9/model-evaluation-selection) video Null/Baseline metric/accuracy/score for sanity check. Data imbalance of negative labels.

**Types I and II errors**: FP/FN. `sklearn.metrics.confusion_matrix`. 
$$
\begin{bmatrix} \mathrm{TP} & \mathrm{FP}\\
\mathrm{FN} & \mathrm{TN} \end{bmatrix}
$$


[ [ TN=400, FP=7 ] 

  [ FN=17,   TP=26] ] 

**Accuracy** = $\frac{\text{correct}}{N}$ (affected by PN distribution), dummy may have seemingly high scores.

**Recall** = **sensitivity**= True positive rates = $\mathbb P \big(\text{test T and T} | \text{T} ) = 1 - \mathbb P \big( \text{ test F and T} | \text{T})$.

**Precision** = $\mathbb P\big(\text{test T and T}| \text{test T})$ 

**Specificity** = $\mathbb P\big(\text{test F and F}|\text{F})$= 1 - FPR

**F_beta** score = $\frac{1}{2} \big( \text{recall}^{-1} + \text{precision}^{-1}\big)^{-1}$. 

General format of `sklearn.metrics` `scores` is `some_score(y_test, y_predicted)`.

**Competing precision and recall**! Decision boundary close to 1, higher precision, close to 0, higher recall. **Recall-precision plot** (看起来大概是是一个第一象限的圆弧). 

[ROC](https://www.coursera.org/learn/python-machine-learning/lecture/8v6DL/precision-recall-and-roc-curves) (Receiver operating characteristic) plots TPR vs FPR. Parametrized by decision boundary parameter. The diagonal where TPR=FPR is the base case since a poor classifier will benefit from relabling the predictions. **Area under the curve** is the area under the ROC, AUC is not sensitive to class imbalances.

## Cross-Validation

Reading learning curves. **Learning curves** for a **given** model shows the trend of train and test errors (e.g. RMSE) agianst the **train size**

- Underfit: high error for both testing and training, steady by train size.
- Overfit: low training error, high test error, the latter decreases steadily with train size.
- Just right: similar and not-too-high training and testing errors.
- The learning curve shows also whether the model would benefit from increasing training size, and if either bias or varaince is driving the estimated generalization error.

Need an unbiased estimator of generalization error. A key assumption in CV for nontemporal data is that the data are drawn in an iid way. For time series data, the splits needs to respect the time order to avoid **look-ahead** bias.

The part of the dataset used to perform CV to tune the hyperparameters, are not suitable for estimating the generalization error. Instead we split the data into three parts: train-validation for model selection, and the test/holdout set for estimating the generalization error.

For classification problems we need to stratify the data to keep the ratio of different class lables similar across folds. 

More folds lead to more correlated train sets, the generalization error is estimated with lower bias and higher variance as a result. 

For time-series data, use `TimeSeriesSplit` from `model_validation`. Or use the [timeseriescv](https://github.com/sam31415/timeseriescv) package.

# SVM and Kernel Methods

In the **nonparametric** setting, with observation $\{x_i, y_i\}_{i=1}^N$, need to predict $f(x)$ as a weighted combination of $f(x_i)$'s. The weights are $\mathcal K(x,x_i)$, measuring the proximity of $x$ to $x_i$. A **Mercer** or **positive-definite** kernel is a *generator* of PD **Gram matrices**: $\forall N, \forall \{x_i\}_{i=1}^N, \Big[ \mathcal K(x_i,x_j)\Big]_{i,j=1}^N \succ 0$. The **Gaussian** kernel is $\mathcal K = e^{-|x-x'|^2 / 2 \ell^2}$ where $\ell$ is the **bandwidth** parameter.

## Support Vector Machines

Consider predictors with the following form $f(x)  = \sum_{i=1}^N \alpha_i \mathcal K(x,x_i)$. By adding suitable constraints the $\alpha_i$'s can be made sparse.The nonzero $x_i$'s are called **support vectors**. 

### Hard margin/linearly separable case

First assume the **data set is linearly separable**. Binary classifier $h(x) = \mathrm{sgn}f(x)$ where $f(x)= w^\mathsf T x + w_0$ and decision boundary is an (affine) hyperplane. Need **margin** to be large. $f(x)$ is a coordinate in the direction of $w$ and vanishes on the DB, $\mathrm{dist}(x,\mathrm{DB}) = f(x) / |w|$. Now if two classes of points are linearly separable, their **convex hulls** are separated with a positive distance. 

Goal: construct a linear margin large enough. Scale the coordinate system (choosing $w$ and $w_0$), s.t. the minimum of absolute values of coordinates is 1. To maximize the margin, is equivalent to minimize $w$ since $w$ is the rate of change. Let $\tilde y \in \{\pm1\}$ code the classes, the optimization problem is:
$$
\frac{1}{2}  \underset{w,w_0} \min {|w|^2},\ \text{s.t. }
 \underset{i=1,2\cdots,N}{\max}1- \tilde y_i \big( w^\mathsf T x_i + w_0) \leq 0
$$
It's important to scale the input first. This is a standard **convex** optimization problem enjoying the **strong duality** property. See the [section](##convex-optimization) for details. Now the **primal problem** is to look for
$$
p^\star = \inf_{w,w_0} \sup_{\alpha \succeq 0} 
\mathcal L(w,w_0,\alpha) = \inf_{w,w_0} \sup_{\alpha \succeq 0} \frac12 w^\mathsf T w - \sum_{i=1}^N \alpha_i \big[ \tilde y_i (w^\mathsf T x_i + w_0 ) - 1\big]
$$
Due to strong duality this is equivalent to finding $d^\star = p^\star$ where 
$$
d^\star =  \sup_{\alpha \succeq 0}  \inf_{w,w_0}
\mathcal L(w,w_0,\alpha) =  \sup_{\alpha \succeq 0}  \inf_{w,w_0}  \frac12 w^\mathsf T w - \sum_{i=1}^N \alpha_i \big[ \tilde y_i (w^\mathsf T x_i + w_0 ) - 1\big]
$$
which can be attained by some $\hat \alpha$. The Lagrangian is minimized for fixed $\alpha$ satisfying $w = \sum_{i=1}^N \alpha_i \tilde y_i x_i$, $0 = \sum_{i=1}^N \alpha_i \tilde y_i$, thus need to maximize 
$$
\inf_{w,w_0} \mathcal L(w,w_0,\alpha) = - \frac12 \sum_{i,j=1}\alpha_i \tilde y_i \big( x_i^\mathsf T x_j \big) \tilde y_j \alpha_j + \sum_{i=1}^N\alpha_i
$$
 for $\alpha \succeq 0$ and $\boldsymbol \alpha ^\mathsf T \mathbf y = 0$. If the data is **linearly separable** there is a solution which has the **complementary slackness**: $\alpha_i \big( \tilde y_i f(x_i) - 1 \big) = 0$. The $x_i$'s for which $\alpha_i \neq 0$ are called the **support vectors**. 

- To solve for $w_0$, we note that for support vectors $\tilde y_i = w^\mathsf T x_i + w_0$. Average over the SV's to get $\hat w_0 = |\mathcal S|^{-1} \sum_{i\in \mathcal S} \big(\tilde y_i - \hat w^\mathsf T x_i\big) = \tilde y_\mathcal S - \hat w^\mathsf T x_\mathcal S$.

- With $\hat w = \sum_{i=1}^N \hat \alpha_i \tilde y_i x_i = \sum_{i\in \mathcal S} \hat \alpha_i \tilde y_i x_i$, we have $f(x; \hat w, \hat w_0) = \sum_{i\in\mathcal S}\hat \alpha_i \tilde y_i x_i^\mathsf T x + \hat w_0$. 

- Another more transparent way to write the prediction: 
  $$
  f(x;\hat w, \hat w_0) = y_\mathcal S + \hat w^\mathsf T (x-x_\mathcal S)
  $$
  where $y_\mathcal S$, $x_\mathcal S$ stands for the class and coordinate of an arbitrary support vector, or their average.

### Soft margin classifiers

The original form of the problem to maximize margin $M$ ruires no point is inside the margin. This does not work for the general case when the data is not linearly separable. Now introduce the **slack variables** $\xi_1,\cdots, \xi_N$ and the problem becomes:

- Maximize $M$, under constraint $|w| = 1$, and $w_0$ arbitrary, subject to $\min_{i=1,\cdots, N} y_i \big( w^\mathsf T x_i + w_0\big) \geq M (1-\xi_i)$. 

This allows some points to fall in the margin. If $\xi_i > 1$, then $x_i$ is **misclassified**. Scale $w$ and we have:

- minimize $|\beta|$ subject to $\min_{i=1,\cdots, N} y_i \big( w^\mathsf T x_i + w_0\big) \geq 1-\xi_i$, $\xi \succeq 0$, $\sum \xi \leq $constant.

The Lagrangian to minimize becomes 
$$
\mathcal L(w, w_0, \boldsymbol \xi, \boldsymbol \alpha, \boldsymbol \mu) = \frac{|w|^2}{2} + \sum_{i=1}^N (C- \mu_i) \xi_i -\alpha_i \big[ y_i (w^\mathsf T x_i + w_0) - (1-\xi_i)\big]
$$
Again $w = \sum_{i=1}^N \alpha_i \tilde y_i x_i$, $0 = \sum_{i=1}^N \alpha_i \tilde y_i$, and $\alpha_i + \mu_i = C$, and we need to maximize 
$$
\inf_{w,w_0,\boldsymbol \xi} \mathcal L(w, w_0, \boldsymbol \xi, \boldsymbol \alpha, \boldsymbol \mu)= - \frac12 \sum_{i,j=1}\alpha_i \tilde y_i \big( x_i^\mathsf T x_j \big) \tilde y_j \alpha_j + \sum_{i=1}^N\alpha_i
$$
subject to $\alpha_i + \mu_i = C$, $\alpha_i,\mu_i \geq 0$. 

The strong duality again yields the complementary slackness conditions 

- $\alpha_i \big [ y_i (w^\mathsf T x_i + w_0\big) - (1-\xi_i)\big] = (C-\alpha_i) \xi_i =0$ and the factors are all $\geq 0$.

The weight $\hat w$  is represented by those $x_i$ with **nonvanishing** $\hat \alpha_i$. There are two cases

- If such $x_i$ is correctly classified as if in the case of a **hard margin**, $\xi_i = 0$ and thus, $y_i(w^\mathsf T x_i + w_0) = 1$, which means $x_i$ is exactly on their **correct** margins. In this case $0<\alpha_i < C$.
- If $\xi_i > 0$ then $x_i$ is within the transition area, and on the wrong side of **its margin**. In this case $\alpha_i = C$ and $y_i(\hat w^\mathsf T x_i + \hat w_0) < 1$.

These are the **support vectors** for a soft margin.

### Kernel trick

Intuitively, if the dimension of the feature space is large enough, any binary labeling of a fixed number of observations can be linearly separable. Transform the features $\phi: \mathbb R^{p} \to \mathbb R^{p'}$, e.g., $\phi(x_1,x_2) = \begin{bmatrix} x_1 & x_2& x_1^2 + x_2^2 \end{bmatrix}$. The transformed feature may be "better" seperable with the same labelings. The dual problem becomes maximizing $- \frac12 \sum_{i,j=1}\alpha_i \tilde y_i \mathcal K(x_i,x_j) \tilde y_j \alpha_j + \sum_{i=1}^N\alpha_i$ over $\alpha$ (still $N$ components since feature transformations do not add new datapoints.) The **kernel** is $\mathcal K(x_i,x_j) = \phi(x_i)^\mathsf T \phi(x_j)$ which is positive definite. 

Alternatively, given a continuous symmetric PD kernel function $\mathcal K$ on $[0,1]^2$, and let $\{e_i(x)\}_{i=1}^\infty$ be any basis (trig, polynoms, etc.) of $L^2[0,1]$, there exists nonnegative $\lambda_i$, such that $\sum_{i=1}^\infty \lambda_i e_i(x)e_i(x') = \mathcal K(x,x')$ in $\mathcal C[0,1]^2$. See this Wikipedia [page](https://www.wikiwand.com/en/Mercer%27s_theorem). Sometimes there are only finite number of nonzero $\lambda_i$'s. If the range of $\mathcal K$ is of countably infinite dimensions, $\phi(x)$ is of $\infty$ length. 

In this case we will often need to regularize with $C$ and the resulting in a smoother decision boundary. Prediction is given by the sign of $\hat f(x) = \sum_{i\in\mathcal S} \hat \alpha_i \tilde y_i \mathcal K(x_i, x) + \hat w_0 = y_\mathcal S + \hat w \big(\phi(x) - \phi(x_\mathcal S) \big)$ only involving the support vectors in the higher dimensional transformed feature space.

> The RBF kernel performs well for data arising from the mixture of Gaussians. (ESL)

Some common kernels:

- $d$^th^ degree polynomials $\big[1 + x^\mathsf T x'\big]^d$;
- RBF: $\exp \big(-\gamma | x- x'|^2\big)$;
- Newral network: $\tanh \big(\kappa_1 x^\mathsf T x' + \kappa_2 \big)$.

Note that we can approximate **any** function $f(x)$ if some completeness requirements are met... So can learn any decision boundary. This backed by the theory of **reproducing kernel Hilbert spaces** (RKHS).

### SVM as penalization method.

There is an equivalent formulation of the primal problem for the soft margin case. See this documentation [page](https://scikit-learn.org/stable/modules/svm.html#svc) of `sklearn` also ESL:

- Minimize $\frac{|w|^2}{2} + C\sum_{i=1}^N \Big[ 1 - y_i \big(w^\mathsf T \phi(x_i) + w_0 \big)\Big]_+$ where $\phi(x)$ is the feature transform, giving kernel $\mathcal K(x,x') = \phi(x)^\mathsf T \phi(x')$. The l oss function here is called the **hinge loss**. Here depending on $\phi(x)$, the length of $w$ can be large even $\infty$. But the final $w$ minimizer will be a linear combination of $\big[\phi(x_i)\big]_{i=1}^N$, due to the **representer theorem**.

- If $C$ is **large**, then $\xi_i > 0$ is disfavored, since thi swill make $\alpha_i=C$ large: less tolerant to misclassifications. In the linearly seperable case, tuning $C \to \infty$ will make all support vectors fall on the correct margin, the classifier degenerates to that with a hard margin.
  Alternatively, $C$ large will favor the hinge loss to vanish.
- On the other hand, if $C$ is **small**, more tolerant to $\xi_i > 0$.

There are other choices of loss functions, we list them with the hinge loss

- Hinge loss: does not care about the correctly classified points, hence sparser estimates (PML). Howerver it is not differentiable at 1 so cannot apply gradient methods directly.
- **Binomial deviance**: $\log \big[ 1+ e^{-\tilde y f(x)}\big]$. Smooth and robust to outliers but "always wants to be smaller" since never zero. Also unstable for perfectly separable case (see logistic regression) without penalization.
- Squared error. Easier to interpret but penalizes outliers, and also penalizes "too correctly" classified data
- **Huberized** square hinge loss. Similar to Huber loss but is zero for $\tilde y f(x) \geq 1$, sparsity, robust to outliers, as well as differentiable (Gradient methods applicable).

# Trees basics

A **decision tree** partitions the feature space into blocks with **assigned** values or classes; prediction made according to which block a new observation belongs. A tree has a **root node**, **leaves/ternimal nodes**, and **internal/split nodes**. Typically, the trees are binary, split single features at each internal node, split continuous variable like $x_j \leq s$, and split discrete variables into two groups. 

TBD: other trees

## Binary regression trees, Classification and regression trees (CART)

### Growing a regression tree

The prediction function $f(x) = \sum_{m=1}^M c_m \mathbb 1[x \in R_m]$. At each step, need to choose the optimal **splitting variable** and the optimal **splitting point**. To prevent overfit, generate and then prune. CART uses the square loss and the minimizer is the mean of data in each region:
$$
(x_j, s) = \mathrm{argmin}_{j, s} \Big[\min_{c_1} \sum_{x_i\in R_1(j,s)} \big(y_i - c_1\big)^2 +
\min_{c_2} \sum_{x_i\in R_2(j,s)} \big(y_i - c_1\big)^2\Big]
$$
Split-point $s \in \Big\{\frac{x_{j}^{(i)}+x_{j}^{(i+1)}}{2},\ i=1,2,\cdots, n-1 \Big\}$. For square loss, the minimizers in each of the regions are $\hat c_{1,2} = \mathrm{ave} \big[y_i: x_i \in R_{1,2}(j,s)\big]$.

> For each splitting variable, the determination of the split point $s$ can be done very quickly and hence by scanning through all of the inputs, determination of the best pair $(j,s)$ is possible. 

If the number of points are large, say $N = 10^9$, scan through quantiles, for example, $\alpha = 1/100, 2/100, \cdots, 99/100$, instead will result in approximate best splits.

The **stopping criterion** or **complexity control** methods can be:

- Total number of leaves exceeds a bound
- number of elts in a leaf drops below a bound
- number of elts in a new leaf will drop below a bound
- imporvement of sum of ER of the newly split nodes over that of the to-be-split node drops below a bound.
- for classification trees if a node contains only elts with the same label

Note: Only at the **leaves** are $c_m$'s assigned to the region, choose $\hat c_m$ at these steps as the output of $f: x\in R_m \mapsto \hat c_m$.



### Pruning a regression tree

**Regularize** (cf. lasso) the ER with a **cost complexity parameter** `ccp_alpha` resulting in $C_\alpha(T) = \hat R(T) + \alpha|T| $ which. Let $T(\alpha)$ be the unique smallest subtree that minimizes $C_\alpha$. It can be obtained by the following algorithm (see Breiman et al. 1984 *CART* Chapter 10 (10.15)-(10.17) and 李航《统计学习方法》Chapter 5):

- $k=0$, $T = T_0$; let $\alpha_0 \leftarrow +\infty$;
  - **while** $T_k$ is not a single root do:
    		$T_{k+1} \leftarrow$ smallest tree such that $g(t, T_k) = \frac{\hat R(t) - \hat R(T_{k,t})}{\mathrm{redcution\ in\ node\ count}} = g_k$ is minimized;
      		$\alpha_{k+1} \leftarrow \min g_k$; $k\leftarrow k+1$;
    end **while**;
    $K \leftarrow k$;
- $\bigcup_{\alpha\in\mathbb R}\big\{ \mathrm{argmin}_{T\subset T_0}C_\alpha(T) \big\} \subset \{T_0, T_1,\cdots, T_k\}$
   (*the two sets seem to actually coincide*)
- Now the regularized risk is minimized for all hyperparameters, use cross-validation to find the best $\hat T_{\hat \alpha}$ among them (and the best $\hat \alpha$).

Implemented by `sklearn.tree.DecisionTreeRegressor.cost_complexity_pruning_path()`. See this [example](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py). After getting the path of $\alpha$'s, set `ccp_alpha` in constructor of the DTR; cross validate.

## Classification Trees

Multiclass classification problem, with $\mathcal Y = \{1,2,\cdots,K\}$. In $R_m$ with $N_m$ observations, for class $k$, $\hat p_{mk} = N_m^{-1} \sum_{x_i\in R_m} \mathbb 1(y_i = k)$; **predicted classification** $\hat k(m)=\mathrm{argmax}_{k\in\mathcal Y} \hat p_{mk}$, distribution $\big(\hat p_{mk}\big)_{k=1}^K$.

### Node impurity measures

These are to be contrued as pertaining to **each point** in $R_m$. $Q(R_m)$, **node impurities** in these boxes:

- **misclassification error**: $1 - \hat p_{mk}$. Only takes into account the most probable class, despite info gain.
- **Gini index**: $\sum_{k=1}^K \hat p_{mk}\big(1-\hat p_{mk}\big)$
- **entropy** or **deviance**: $-\sum_{k=1}^K \hat p_{mk} \log \hat p_{mk}$= information gain.

Each split in $R = R_L \cup R_R$ should minimize $N_L Q(R_L)+N_RQ(R_R)$.

## General considerations

Missing features (TBD), Linear relations, interpretability.

Nonlinear feature discovery

# Ensemble methods

## Basics

**Parallel ensembles**, each model has high complexity but low bias, combined to reduce variance (bagging, random forests). **Sequential ensembles**, generate models sequentially (boosting).

**Standard error** is the SD of the sampling distr of a *statistic*; e.g. SD of a population but *SE* of the estimator on average, $\mathrm{se}(\bar X_n)$. Variation of mean of **iidrv's** scale as $1/\sqrt{n}$. If we have large samples then sample $B$ indep training sets to get $\hat f = B^{-1}\sum_{b=1}^B \hat f_b$, $\mathbb E\hat f = \mathbb E \hat f_b$ but variance reduced by $\sqrt{B}$. But lacking samples.

**Bootstrap sampling**

Also see the section on [Bootstrap](###bootstrap). A **bootstrap sample** is a sample of size $n$ drawn **w/ replacement** from data. Each $x_i$ has $(1-n^{-1})^{n} \approx e^{-1} \approx 0.368$ not to be selected in **a single** bootstrap sample. These **out of bag** samples are used to assess models.

Draw bootstrap samples $D^1, \cdots, D^B$ from data, somehow combine PF to get $\hat f_{\mathrm{avg}}$. **Bagging for regression**: just take average of the predictions. **Empirically reduces variance**, same bias. Define for each datapoint $x_i$ its **out-of-bag** observations $S_i = \{x_i \not\in D^b\}$ and **OOB prediction** $\hat f_\mathrm{OOB} \sum_{b \in S_i} \hat f_b(x_i)$ to estimate test error. Compare with CV. 

Since we have as many samples from $F_{n,X,Y}$ as possible, can "predict" it well. But there are biases for prediction on $F_{X,Y}$. Note bias is the $\mathbb E$ of errors, low bias and high variance can occur at the same time. Variance refers to how unstable the fit is across random training sets.

 **Intuition**: when base PF has *high var and low bias*, e.g. decision trees.

## Random forests

Bootstrap performs best when the **samples** are iid. Randomly select features to **decorrelate**. Complexity of a forest depends on $m$, the number of features from which the split is done at **each node**. Increasing the trees does not introduce **more** variance (but $\exists$ correlation of trees, bias of the underlying model, restriction of sampling space). Use **cross validation** to find optimal $m$.

### Algorithm

**Algorithm to create a random forest** (ESL Chap 15)

- Input: training data $\mathbf Z$, some single tree stopping criteria, # of feature selected at each split $m$;
  Output: an **ensemble** of trees $\{T_b\}_{b=1}^B$ called a **random forest**;

- **for** $b\leftarrow 1$ to $B$ do:
  construct_trees:

  - Bootstrap sample $\mathbf Z^*$ of size $N$ from **training** data;

  - **randomly** select $m$ of $p$ variables

  - **if** not stopping criteria:

    - split the best $(j,s)$ where $j \in \{p_1, \cdots, p_m\}$ and $s$ minimizes the sum of impurities or loss

    end **if**;

  construct_trees returns $T_b$;

  end **for**;

- To make new prediction, average $f_{\mathrm{rf}}^B(x) = B^{-1} \sum_{b=1}^B T_b(x)$ for regression and majority vote for classification.

Trees are full-grown, have small biases, but large variances. Each tree is an **rv**, generally correlate. RF reduces the correlation by randomization, without increasing **individual tree variance** too much. 

Parametrize the trees in an RF as $T(x; \Theta_b)$, where $\Theta_b = \Theta_b(m,\mathrm{params};\mathbf Z) = \{R_{j_b}, \gamma_{j_b}\}_{j_b = 1}^{J_b}$ where $m$ and params control the **individual** tree growing processes.

> Highly nonlinear estimators like trees benefits from this randomization the most (ESL Chap 15). $\rho$ is typically lower than 0.05. Bagging does **not** change linear estimates, where pariwise correlation $\sim$ 50%. (ESL)

### Random forest analysis

- Number of trees in the forest needs to be large
- Small $m$ likely hurts performance when fraction of relevant variables is small.

The RF forest will not be generalizing more poorly **as a result of** large $B$. Given training data, and fix $m$, given a single test point $x$, $\hat f_\mathrm{rf}(x) = \lim \hat f_\mathrm{rf}^B(x)$ stablilizes. This value is **conditioned** on training data $\mathbf Z$, and also the number of selected feature $m$ and params to control tree growths.

> Segal (2004) demonstrates small gains in performance by controlling the depths of the individual trees grown in random forests. Our experience is that using full-grown trees seldom costs much, and results in one less tuning parameter. (ESL)

#### Additional randomization

The limit of RF prediction **given** a dataset $\mathbf Z$ as $B \to \infty$ is (by either W/SLLN since the random bagging processes are independent)
$$
\big[\hat f_\mathrm{rf}(x)\big](\mathbf Z) = \lim \hat f_\mathrm{rf}^B(x) = \mathbb E_{\Theta|\mathbf Z}
 \Big(T \big [x; \Theta(m,\mathrm{parames} ; \mathbf Z) \big] | x, \mathbf Z \Big)
$$

- Since the trees are trained from empirical distribution $\hat F_{N, \mathbf Z, \mathbf Y}$, the empirical distribution, larger trees may result in lower bias; but needs large $B$ to reduce variance. 
- If $B$ is sufficiently large, $\hat f_\mathrm{rm}(x)$ makes as good a prediction on $F_{N,\mathbf Z, \mathbf Y}$ as possible, *if assuming no bias for trees large enough*. But $\hat F_{N,\mathbf Z, \mathbf Y} \overset{\mathbb P,\mathrm {uf.}}{\longrightarrow} F_{\mathbf Z, \mathbf Y}$ (Glivenko-Cantelli) so intrinsic error.

Fix $x$ now, and $\mathbf Z$ is a random vector with fixed length $N$. These predictions $\big[\hat f_\mathrm{rf}(x)\big](\mathbf Z) $ are iidrvs since the samples $\mathbf Z$ are iidrvs. Important result: identically distributed $T, T_i$, $\mathrm{corr}(T_i, T_{j\neq i}) = \rho$, then as $N\to \infty$,
$$
\mathbb E\bar T_N = \mathbb ET,\ \mathbb V \bar T_N = \rho \sigma^2 + \frac{1-\rho}{N}\sigma^2 \to \rho \sigma^2
$$
The individual tree prediction is a random variable of which the sample space is the pair $\mathrm{grProc},\mathbf Z$. 

The variation is thus given by $\rho(x)\sigma(x)^2$ where $T=T \big[x; \Theta(m,\mathrm{parames} ; \mathbf Z)\big]$ is the prediction of any tree grown with the hyperparametrized process, as a random variable of $\omega = \mathbf Z$, at a fixed point $x$. So

- $\rho(x) = \mathrm{corr}_{\Theta,\mathbf Z} \Big[ T\big(x,\Theta (\mathbf Z_i), \mathbf Z_i \big), T\big(x,\Theta (\mathbf Z_j), \mathbf Z_j\big) \Big]$
- $\sigma(x)^2 = \mathbb V_{\Theta,\mathbf Z} T\big(x,\Theta(\mathbf Z),\mathbf Z\big)$

There seems to be some hand-waving in ESL, since this $\rho(x)\sigma(x)^2$ variation is obtained if the "averegand" each has variance $\sigma(x)^2$. However the averaging is conditioned on $\mathbf Z$, The variance comes entirely from the process of drawing samples. SO WHAT IS AVERAGED OVER?

**Single tree variance** for RF algorithm (15.9) $\mathbb V_{\Theta,\mathbf Z}\big(x,\Theta(\mathbf Z), \mathbf Z\big)
= \mathbb V_{\mathbf Z} \mathbb E_{\Theta|\mathbf Z} T \big(x,\Theta(\mathbf Z), \mathbf Z\big) + \mathbb E_{\mathbf Z} \mathbb V_{\Theta|\mathbf Z} T \big(x,\Theta(\mathbf Z), \mathbf Z\big)$:
$$
\begin{aligned}
& \mathbb V_{\Theta,\mathbf Z}\big(\Theta, \mathbf Z\big) = \mathbb E \big[T(\Theta, \mathbf Z )^2 \big] - \mathbb E \big[T(\Theta,\mathbf Z)\big] ^ 2\\
= & \mathbb E \mathbb E \bigg[\Big(T(\Theta,\mathbf Z) 
- \mathbb E\big[T(\Theta, \mathbf Z) | \mathbf Z\big] 
+ \mathbb E\big[T(\Theta, \mathbf Z) | \mathbf Z\big]\Big)^2 \Big| \mathbf Z \bigg] - \Big(\mathbb E \mathbb E \big[T(\Theta,\mathbf Z) | \mathbf Z\big] \Big)^2 \\
= & \mathbb E \mathbb E \Big[ \big( T(\Theta, \mathbf Z) - f(\mathbf Z) \big)^2 | \mathbf Z \Big] + (\text{cross-term = 0}) +\mathbb E\big(f(\mathbf Z)^2\big) - \big(\mathbb E f(\mathbf Z)\big)^2\\
= & \mathbb E_\mathbf Z \mathbb V_{\Theta|\mathbf Z} T (\Theta,\mathbf Z) + \mathbb V_\mathbf Z \mathbb E_{\Theta | \mathbf Z} T (\Theta, \mathbf Z) = (\text{within-$\mathbf Z$ var}) + \mathbb V_\mathbf Z \hat f_\mathrm{rf}
\end{aligned}
$$
we wrote $f(\mathbf Z)= \hat f_\mathrm{rf}(x)= \mathbb E \big[ T(\Theta, \mathbf Z) | \mathbf Z\big]$ since it is measureable wrt $\sigma(\mathbf Z)$, and $\Theta = \Theta (\mathbf Z)$ requires some larger $\sigma$-algebra which may depend on individual values of $\mathbf Z = \mathbf z$. Moreover, the second term is exactly the variance of $\hat f_\mathrm{rf}(x)$. 

- The bootstrapping process reduces the first term, the second term is small ($\rho < 0.1$ [ESL]).
- The **STV** has two competing effects; first term is **intrasample** variance, is a result of randomization, which is decreased as $m \uparrow$, and the whole RF reduces this part by averaging; the second term is  **intersample** variance, which increases as $m\uparrow$. Typically the STV does not change much as $m$ changes. 
- The total MSE of RF is decided by two factors. As $m\uparrow$, the bias is lowered, but variance can increase due to **correlation**. The **decorrelation** happens as $B \to \infty$ which is assumed already done.

**Bias** of the RF is the same as single trees (averaged over different samples $\mathbf Z$ as well as the random tree growth process). 

### Feature importance

In a random forest, **Feature importance** is calculated by

-  for each tree in RF, calculate the *decrease in impurity* due to each of the variables (for classification at least)
- Average over the trees in RF
- Add standard error if necessary. 

Accessed by calling `RandomForestClassifier.feature_importances_`. 





## Boosting

See this [paper](http://rob.schapire.net/papers/Schapire99.pdf) and these [slides](https://yosinski.com/mlss12/media/slides/MLSS-2012-Schapire-Theory-and-Applications-of-Boosting.pdf), both by Shapire, for more theoretical details.

### Forward Stagewise Additive Modeling (FSAM)

- **FSAM** is an iterative optimization algorithm for fitting adaptive basis function models, also called **generalized additive models** (GAM).

Create a sequence of PFs $f_0,f_1,\cdots, f_m, \cdots$ incrementally improves some objective function value. In the framework of **ERM**, the objective function is the empirical risk; the optimization at each step is constrained s.t. $f_m - f_{m-1} = \beta_m b_m$ where $b_m \in \mathcal H$ (Assumed to be a linear space). $b_m$ is usually a basis function.

**General Algorithm for ERM FSAM**:

- Input: data $\mathcal D = \{ (x_i, y_i)\}_{i=1}^n$, linear hypothesis space $\mathcal H$, maximal step $M$, loss function $\ell$;
  Output: boosted PF $f_M$.
- $f_0(x) \leftarrow 0$;
- for $m\leftarrow1$ to $M$ do:
  - Construct ER: $\mathrm{ER}_m(\beta, b; \mathcal D) = n^{-1} \sum_{i=1}^n \ell \Big( y_i , f_{m-1} (x_i) + \beta b(x_i)\Big) $;
  - minimize or approx minimize $ER_m$:
    - $\big( \beta_m, b_m\big) \in \underset{\beta\in \mathbb R, b\in \mathcal H}{\mathrm{argmin}} \frac{1}{n} \sum_{i=1}^n \ell \Big( y_i , f_{m-1} (x_i) + \beta b(x_i)\Big)$
  - $f_m \leftarrow f_{m-1} + \beta_m b_m$;
- return $f_M$.

For basis functions like stumps suppress dependency on $\beta$. The $i-th$ **residual** fitted at the $m$-th stage is $r_{im}=y_i - f_{m-1}(x_i)$.

In ESL the basis function is written as $b(x;\gamma)$. $\gamma$ parametrizes the Hypothesis space.

**Remarks**: For OLR, $\mathrm{argmin}_{w \in \mathbb R^p} \frac{1}{n} 
\sum_{i=1}^n \big( w^\mathsf T x_i - y_i \big)^2$ is the minimization problem. If there is only one elt in $\mathcal H$, then GAM is just linear regression models; or polynomial models with transformed features. Possible transformed number will explode as $p^d$ with $d$-th order polynomials. W/in the ERM framework, the actual problem for minimization is $\underset{\{\beta_m,\gamma_m\}_{m=1}^M}{\mathrm{argmin}} \sum \ell \Big[ y_i, \sum_{m=1}^M \beta_m b(x_i;\gamma_m ) \big) \Big]$ which is hard. Therefore FSAM is a **greedy** substitution.

### Explicit examples: $L^2$ and AdaBoost

#### $L^2$ Boost with regression stumps

A **regression stump** is a simple function of the feature space partitioned into two linear halfspaces.

Suppressing $\beta_m$and the explicit dependence on $\mathcal D$. The loss function at each stage $m$ is $\mathrm{ER}_m (b) = \frac{1}{2n} \sum_{i=1}^n \big[ r_{im} - b(x_i)\big]^2$. The minimization problem is equivalent to the following:
$$
c_1,c_2 = \underset{c_1, c_2, s \in \mathbb R, j = 1,2,\cdots p}
{\mathrm{argmin}} \frac{1}{2n} \sum_{x_j \leq s} \big[r_{im} - c_1]^2 + \frac
1{2n} \sum_{x_{ij} > s} \big[r_{im} - c_2]^2
$$
Where for each region $c_1 =\underset{x_{ij} \leq s}{\mathrm{ave} }\big( y_i - f_{m-1} (x_{i})\big)$, $c_2 =\underset{x_{ij} > s}{\mathrm{ave} }\big( y_i - f_{m-1} (x_{i})\big)$.

#### Linear regression

Assume labels are centered. The ER is $\sum_{i=1}^N \Big( y_i - \sum_{m=1}^M w_{p_m} x_{ip_m} \Big)^2$. Here $p_m = \gamma_m$ and $w_{p_m} = \beta_m$.  Note that the $p_m$'s may repeat for different $m$'s.

>At each step the algorithm identifies the variable most correlated with the current residual. It then computes the simple linear regression coefficient of the residual on this chosen variable, and then adds it to the current co- efficient for that variable. (ESL 3.3.3)

#### FSAM is a series of SLR problems of transformed features

At each step, the FSAM attempts to minimize $\sum_{i=1}^n \big( r_{im} - \beta b(x_i; \gamma)\big)^2$ which **for each $\gamma$** is the SLR problem of regressing $\big(r_{im}\big)_{i=1}^n$ on $\big(b(x_i;\gamma)\big)_{i=1}^n$ and the result is the real number $\beta_m$.

#### AdaBoost with exponential loss

The setting for **classification probems** (see this [section](##the-spaces)): Outcome space $\mathcal Y = \{ -1,1\}$, action space $\mathcal A = \mathbb R$, PF/**score function** $f:\mathcal X \to\mathcal A$, margin for example $(x,y)$ isd $m = y f(x)$, the larger the better. So loss function should decrease as $m\uparrow$. $\mathcal H =$ set of classifiers taking values $\pm1$. **Exponential loss** $\ell(y,f(x)) = e^{-yf(x)}$, puts very large weight on incorrectly classified samples. Logistic loss performs better when there is **intrinsic** randomness, when **Bayes error rate** is high. 

**AdaBoost.M1 Algorithm (original)**

- Input: data $\mathcal D = \{ (x_i, y_i)\}_{i=1}^n$, linear hypothesis space $\mathcal H$ of weak classifiers, maximal step $M$;
  Output: boosted classifier $G_M$.
- Weights $w_{i1} \leftarrow n^{-1}$ for $i\leftarrow 1$ to $n$;
- for $m \leftarrow 1$ to $M$:
  - Fit weak classifier $G_m(x)$ to training data with $w_{im}$;
  - weighted error rate of $G_m$: $\mathrm{err}_m = \sum_{i=1}^n w_{im} \mathbb 1 \big(y_i \neq G_m(x_i) \big)$. Can be realized by weighted sampling with replacement; 
  - **the amount-of-say** $\alpha_m = - \mathrm{logit} ( \mathrm{err}_m ) = \log \Big(\frac{1-\mathrm{err}_m}{\mathrm{err}_m}\Big)$. The better $G_m$ performs the more weight it is assigned.
  - $w_{i,m+1} \leftarrow w_{im} e^{\alpha_m \mathbb 1 \big(y_i \neq G_m (x_i) \big)}$, for $i \leftarrow 1$ to $n$, then **normalize**, updated weights. 
    Correctly classified points are not affected; incorrectly classified points are emphasized; the better $G_m$ performs (the larger $\alpha_m$), the more are misclassified points emphasized.
- return $G(x) = \mathrm{sgn} \Big [ \sum_{m=1}^M \alpha_m G_m (x) \Big]$.

AdaBoost is FSAM with exponential loss. Suppress the $m$ index keeping in mind this is for the $m$-th step process. Assume we already have at the end of last step a classifier, and constructed a new weak classifier $G(x):\mathcal D \to \{\pm1\}$ by some method, need to find a $\beta$ to minimize $\mathcal J[f + \beta G] = \sum_{i=1}^n \exp \Big[ - y_i \big( f(x_i) + \beta G(x_i) \big) \Big]$. Define weight **proportional** to $w_i \sim e^{-y_i f(x_i)}$, so need to minimize
$$
\sum_{i=1}^N w_i e^{-\beta y_i G(x_i)} = e^{-\beta} \underset {y_i = G(x_i)}{\sum} w_i + e^{\beta} \underset {y_j \neq G(x_j)}{\sum} w_j = \big(e^\beta - e^{-\beta} \big) \sum_{i=1}^N w_i \mathbb 1 (y_i \neq G(x_i) ) + e^{-\beta}
$$
So $\beta = \frac12 \log \frac{1-\mathrm{err}_m}{\mathrm{err}_m}$. The updted classifier $F(x)= f(x) + \beta G(x)$ and at the next step the weights are proportional to $w_i' \sim w_i e^{\beta_m \mathbb 1 (y_i \neq G(x_i))}$.

- The result of the $f(x)$ classifier after the $(m-1)$-st step affects $\beta$ only by the weights; plays similar role as does the residual $(r_{im})_{i=1}^N$ for the square loss
- The minimization problem is simple in these forms and $\beta_m$ can be obtained in closed forms once $G_m(x)$ is known.

### Loss functions, robustness, applicability (TBD)

TBD ESL 10.5,6,7

### Boosting Trees

A boosting tree is an ensemble tree **built iteratively by FSAM**: $f_M = \sum_{m=1}^M T(x;\Theta_m)$, where $\Theta_m = \{R_j,\gamma_j\}_{j=1}^J$ is the leaf boxes and assigned values. Nontrivial intersections of pairs of boxes are again boxes, so at each step, we have a simple function $f_m(x)$ and a minimization problem $\hat \Theta_m = \mathrm{argmin}_\Theta \sum_{i=1}^n \ell \big( y_i, f_{m-1} (x_i) + T(x_i;\Theta)\big)$.

(I think it is true that *a sum of regression trees can be realized by another probably deeper regression tree* with maximum of the minimum depth, 
	that is, the maximum depth needed to construct such a new tree but not splitting regions in the feature space that is in some smallest (under set partial order) intersection of the leaves of the original ensemble, 
	as the sum of (individual depths - 1), see [this](https://stats.stackexchange.com/questions/336276/is-the-sum-of-two-decision-trees-equivalent-to-a-single-decision-tree) thread on Stack Exchange)

So if we do not restrict the max leaf count of the $m$-th tree, we can make the residual as small as possible. However $J$ is not very small...? 

>For squared-error loss, the solution to (10.29) is no harder than for a single tree. It is simply the regression tree that best predicts the current residuals $y_i-f_{m-1}(x_i)$ , and $\hat \gamma_{jm}$ is the mean of these residuals in each corresponding region.

*I am a bit suspicious of this claim.* If $J$ is not restricted, then yes, one can perfectly fit a regression tree, with $L^2$ loss at least. However $J$ is restricted but moderately small finding the tree fitted to minimize the ER at a certain step, is still as hard as fitting an optimal single tree. Best one can do is by a greedy approach.

- AdaBoost and L2-regression tree boosting have closed-form solutions, at least for the coefs $\beta_m$, if weak classifiers $G_m(x)$ or a stump regressor $F_m(x)$ is already constructed.
- Exponential loss and $L^2$-loss is susceptible to outliers and tend to overfit for noisy data.
- Huber loss, deviance are less prone to do so
- Trade closed-form solutions for robustness, and use **gradient boosting** to construct approximate minimizer $\tilde \Theta_m$. Recall the excessive risk = approx risk (using regression trees to approximate true dependence) + estimation risk (finite number of datapoints, restriction to **GAM**s) + optimization risk (FSAM + gradient boosting for optimization subproblems). 

### Gradient boosting method (GBM)

Note: Gradient boosting trees are also called **multiple additive regression trees**.

Recall $\underset{\{\Theta_m\}_{m=1}^M}{\mathrm{argmin}} \sum_{i=1}^N \ell \Big[ y_i, \sum_{m=1}^M T^{(J)}(x_i;\Theta_m) \big) \Big]$ is the full problem. Then simplify to recursively $\underset{\Theta_m}{\mathrm{argmin}} \underset{\{\Theta_k\}_{k=1}^{m-1}}{\min}  \sum_{i=1}^N \ell \Big[ y_i, \sum_{k=1}^m  T^{(J)}(x_i;\Theta_k ) \big) \Big]$, where $f_{m-1}(x) = \sum_{k=1}^{m-1}  T^{(J)}(x_i;\Theta_k ) $. The problem is thus minimize $\mathcal J[f_m] =  \sum_{i=1}^N \ell \big[ y_i, f_{m-1}(x_i) + T^{(J)} (x_i; \Theta_m) \big]$, now fit the different $T^{(J)}(x_i; \Theta_m)$ to the **pseudoresiduals** $(r_{im})_{i=1}^N = -\nabla_{f(x_i)_{i=1}^N} \mathcal J[f_{m-1}(x_i)_{i=1}^N] = - \partial_{f_{m-1}(x_i)} \ell (y_i, f_{m-1}(x_i))$. The full algorithm for growing gradient boosting trees is as follows:

- Input: dataset $\mathcal D = \{ (x_i,y_i)\}_{i=i}^N$, max leaf count $J$, total number of trees in the ensemble $M$, loss function $\ell(y,\hat y)$.
  Output: gradient boosted tree $\hat f(x) = f_M(x)$
- Initialize $f_0(x) = \mathrm{argmin}_\gamma \sum_{i=1}^N \ell(y_i, \gamma)$;
- for $m \leftarrow 1$ to $M$ do:
  - compute pseudoresiduals $r_{im} = - \partial_{f_{m-1}(x_i)} \ell (y_i, f_{m-1}(x_i))$ for $i = 1,2,\cdots,N$;
  - Fit a regression tree $T^{(J)} (x; \Theta_m)$ to $( r_{im} )_{i=1}^N$;
  - (This step is different from different sources) ESL suggests the last step only determined the different regions $ (R_{jm})_{j=1}^{J}$ but $\gamma_{jm} = \mathrm{argmin}_\gamma \sum_{x_i \in R_{jm}} \ell \big(y_i, f_{m-1}(x_i) + \gamma)$; other sources, such as the David Rosenberg slides, consider these $\gamma_{jm}$ the same for different $j$'s and absorbs them into the **shrinkage** paramter or **step size**.
- return $f_M$.

**Regularization**

- Try more robust loss functions: Huber, quantile for regression, deviance for classification, etc.
- The usual cost-complexity pruning results in trees too large, as they assume the individual trees are the last tree. Tune $J$ by CV, train-validation-test split, etc. More recently the choices of $J$ tend to be 10 to 100. Too large $J$ produces higher correlations that may not exist in the original variables $X_j$, $j=1,2,\cdots,p$. 
- **Shrinkage**. Incoporate a small **step-size** $\nu$ and in accordance a max number of subtrees $M$, the latter can be chosen by **early-stopping**. 
- Row (observation) or column (feature) **subsamplings**, apply **stochastic gradient boosting** similar to minibatch SGD.
- Prune individual trees
- Drop-out method

### Interpretation

See ESL 10.7 and 10.13

Ensembles of decision trees lose some of their interpretability. Use **partial dependence plots** (see [this](https://slds-lmu.github.io/iml_methods_limitations/pdp-correlated.html) page for the case against using PDP in the case of correlated features) and **relative importance plots**.Also [this](https://christophm.github.io/interpretable-ml-book/ale.html) page for why conditional expectation plots on a single variable might be not as good to isolate its effect when it is correlated with other features. 

## XGBoost

The general form of a functional to be optimized in supervised machine learning has form $J(\theta) = \ell(\theta) + \Omega(\theta)$ where $\theta$ stands for generic parameters. At stage $t$, 

# Neural Networks

[Universal approximation](http://neuralnetworksanddeeplearning.com/chap4.html) by NNs. **Feature transformations** can enrich the models in linear regression. The weight $w$ with some (say polynomial) feature transformation $\phi(x,\theta)$ with $f(x,\theta)= \phi(x,\theta_2)^\mathsf T w + b$, where $\theta = (\theta_1, \theta_2)$ and $\theta_1 = (w,b)$. We can continue: $f(x,\theta) = \Phi\Big(\phi(x,\theta_2)^\mathsf T w + b, \theta_4 \Big)^\mathsf T W + B$, where $\theta = (\theta_1,\cdots, \theta_4)$ and $\theta_3 = (W, B)$, etc. This can be done recursively resulting in an example of **deep neural networks**, called **feedforward neural network** or **multilayer perceptrons**.  

## Multi-layer perceptrons

Perdue course [slides](https://engineering.purdue.edu/ChanGroup/ECE595/files/Lecture18_MLP.pdf). PML. [Book](http://neuralnetworksanddeeplearning.com) by Michael Nielson.

 An MLP assumes the input is $x \in \mathbb R^D$, and $\mathbf X \in \mathbb R^{N\times D}$ called strutrued/tabular data. A **perceptron** is the mapping for the form $f(x,\theta) = \mathbb 1 \big(x^\mathsf T w + b \geq 0\big)$ which is a "deterministic" version of Logistic regressions. 

Single perceptrons are limited in their use. But we can compose them. The **XOR problem** with $x_{1,2} \in \{0,1\}$ and need to construct $x_1$ XOR $x_2$. $h_1 = \mathrm{sgn} \big( x_1 + x_2 - 1.5\big)$ computes the logical AND. $h_2 = \mathrm{sgn} \big(x_1 + x_2 - 0.5)$ computes the logical OR. While $y = \mathrm{sgn}\big(-x_1 + x_2 - 0.5\big)$ computes the logical (NOT $x_1$) OR $x_2$. Then the XOR = $y(h_1,h_2)$. In fact, MLPs can represent any logical function. But need to **automate** the weight assignments.

An **activation** function can be thought of a differentiable version of the Heaviside step function. The $k$^th^ unit of hidden layer $l$ outputs 
$$
a_{k}^{l} = \sigma_l \bigg( b_{k}^{l} + \sum_{j=1}^{K_{l-1}} w^{l}_{kj} a_{j}^{l-1}\bigg)
$$
where $a_{j}^{l-1}$ is the $j$^th^ unit output of hidden layer $l-1$. $w^{l}_{kj}$ is the weight assigned to connect $j$^th^ unit on $l-1$^st^ layer to $k$^th^ unit on $l$^th^ layer. $b_{k}^{l}$ is the **bias** vector. 

Note that the book by Nielson uses different notation than PML. We will be following the former and occasionally refer to the latter.

### Backpropagation

**Backpropagation** is an algorithm to effectively calculate the gradient of loss function of an NN. The vectorized form of the expression for activation in $l$^th^ layer is $a^l = \sigma\big(w^l a^{l-1} + b^l \big)$, where $w^l$ is the **weight matrix** and $b^l$ is the **bias vector**. $z^l = w^l a^{l-1} + b^l$ is called the **preactivation**, the weighted input fed in the $l$^th^ layer of the network.

The goal of backprop is to ompute the partial derivatives $\partial \ell / \partial w$ and $\partial \ell / \partial b$ of the loss function. The "standardized" loss is given by $\ell (\mathbf X)= (2n)^{-1} \sum_{i=1}^N \big| y(x_i) - a^L (x_i) \big|_2^2$ where $N$ is the total number of training examples. We **assume** the loss function can be **decomposed** into the sum of indiviual loss functions. Backprop will make us able to compute $\partial \ell_i /\partial w$ etc. Also **assume** the loss function can be written as a function of the activations of the **output** layer, $a^L$. $y(x_i)$ are treated as parameters once the training points are fixed.

The **Hadamard product** is just the elementwise product of two vectors. $s\odot t = \begin{bmatrix} s_1 t_1 & \cdots & s_n t_n \end{bmatrix}$.

The **error** $\delta^l_j$ of the $j$^th^ neuron on the $l$^th^ layer, defined by $\delta^l_j = \partial \ell / \partial z^l_j$, reminiscent of the "pseudoresiduals" in the [section](###Gradient boosting method (GBM)) on gradient boosting method. 

> Backpropagation will give us a way of computing $\delta^l$ for every layer, and then relating those errors to the quantities of real interest, $\frac{\partial \ell}{\partial w^l_{jk}}$ and $\frac {\partial \ell}{\partial b^l_j}$.

#### BP1, the error $\delta^L$ on the output layer.

On the output layer, the error/"pseudoresidual" with respect to the preactivation is:
$$
\label{BP1}
\delta_j^L = \frac{\partial \ell(a^L)}{\partial a^L_j} \sigma'\big(z^L_j\big)
$$
which is equivalent to $\delta^L = \nabla_a \ell \odot \sigma'(z^L)$. For nonstandardized quadratic cost this is  $\delta^L = (a^L - y) \odot \sigma'(z^L)$. This is **BP1**

#### BP2, the recursion relation of $\delta^l$

On the $L-1$^st^ level, we have 
$$
\begin{aligned}
\delta_j^{L-1} & = \frac{\partial \ell (a^L)}{\partial z^{L-1}_j} = \sum_{k=1}^{K_L}\frac{\partial \ell (a^L)}{\partial a^L_k} \sigma'(z^L_k) \frac{\partial z^L_k}{\partial a^{L-1}_j} \sigma'(z^{L-1}_j) \\
& = \sum_{k=1}^{K_L}\frac{\partial \ell (a^L)}{\partial a^L_k} \sigma'(z^L_k) w^L_{kj} \sigma'(z^{L-1}_j) \\
& = \Big[ \big( w^{L}\big)^\mathsf T \delta^L\Big]_j \sigma'(z^{L-1}_j)
\end{aligned}
$$
Therefore $\delta^{L-1} = \Big[ \big(w^L\big)^\mathsf T \delta^{L} \Big] \odot \sigma'(z^{L-1})$. Using mathematical induction, we have 
$$
\delta^l = \Big[ \big(w^{l+1}\big) ^\mathsf T \delta^{l+1} \Big] \odot \sigma'\big(z^l\big)
$$
This is **BP2**. This is the origin of the *back*- in *back*-propagation. We can calculate recursively $\delta^L$, $\delta^{L-1}$, etc. down to $\delta^l$.

#### BP3, loss function partial derivative wrt bias $b^l$

Note that the preactivation $z^l = w^{l}a^{l-1} + b^l$ so immediately we have 
$$
\frac{\partial \ell } {\partial b^l}  = \delta^l
$$

#### BP4, loss function partial derivative wrt weight $w^l_{jk}$

$$
\frac{\partial \ell}{\partial w^l_{jk} } = \frac{\partial \ell}{\partial z^l_j}\frac{\partial z^l_j}{\partial w^l_{jk}} = a^{l-1}_k \delta^l_j
$$

which can also be written as $\partial \ell / \partial w = a_\mathrm{in} \delta_\mathrm{out}$. This has a consequence that 

> [W]eights output from low-activation neurons learn slowly.

- From BP1 + BP4: weights in the output neuron learns slowly if the last layer neurons are underactivated or saturated.
- BP2 + BP4: a weight on $l$^th^ layer will learn slowly if neurons on this layer is either underactivated or saturated. Unless the next layer has either a huge weight and moderately well-activated.

#### Alternative formulations for BP1 and BP2

Let $\Sigma'(z^l)$ denote the diagonal matrix consisting of $\sigma'(z^l_j)$ on the diagonal. BP1 can be written as $\delta^L = \Sigma'(z^L)\nabla_a \ell$. BP2 can be written recursively as $\delta^l = \Sigma'(z^l) \big(w^{l+1}\big)^\mathsf T \cdots \Sigma'(z^{L-1}) \big(w^{L}\big)^\mathsf T \Sigma'(z^L) \nabla_a \ell$. 

> We can think of backpropagation as a way of computing the gradient of the cost function by systematically applying the chain rule from multi-variable calculus. Thas's all there really is to backpropagation - the rest is details.

#### The backpropagation algorithm

- Input $\mathbf X$, consisting of $x_1, \cdots, x_p$, the activation of the input layer $a^0$.

- **Feedforward**: compute the preactivations $z^l = w^l a^{l-1} + b^l$ and the activation $a^l = \sigma(z^l)$.

- Output error: $\delta^L = \nabla_a \ell \odot \sigma'(z^L)$

- **Backpropagate** the error: recursively compute $\delta^l = \Big[ \big(w^{l+1} \big)^\mathsf T \delta^{l+1} \Big]\odot \sigma'\big(z^l\big)$.

- Output: the gradient of the lost function:
  $$
  \begin{aligned}
  \frac{\partial \ell }{\partial w^l_{jk}} & \leftarrow  a^{l-1}_k \delta^l_j \\
  \frac{\partial \ell}{\partial b^l_j} & \leftarrow \delta^l_j
  \end{aligned}
  $$

#### Remarks on backpropagation

The backprop algorithm computes the gradient of the loss wrt a single input $\big(x_1, \cdots, x_p\big)$, and the loss function is **not standardized**. That is $\delta^{x,l}$ for each $x$. We choose a minibatch at each step of $m$ samples, then compute the average of these weight movements. At step/**epoch** $k$, for example

- $w^l_{\mathrm{step}\ k} = w^l_{\mathrm{step}\ k-1} - \frac{\eta}{m} \sum_{x \in \text{minibatch} } \delta^{x,l} \big(a^{x,l-1} \big)^\mathsf T$, note that $w^l$ is a matrix and that from BP4 $\partial \ell / \partial w^l_{jk} = \delta^l_j a^{l-1}_k$.
- $b^l_{\mathrm{step}\ k} = b^l_{\mathrm{step}\ k-1} - \frac{\eta} {m} \sum_{x\in \mathrm{mb}} \delta^{x,l}$.

The backpropagation is faster in the sense that we traverse the network only twice. In comparison consider the naive way where we write $\ell = \ell(w^1, \cdots, w^L)$ as the function of weights. Each time we numerically differentiate, we increment $w^l$ by $\epsilon$ and do a forward-calculation. This amounts to do at least $L K^2$ forward passes, each with $LK^2$ multiplications.

However with backpropagation, We do $L K^2$ multiplications in the forward pass, and do $LK^2$ in the backpropagate pass. Then we do $LK^2$ multiplications to calculate the weights. This is much faster since the total multiplication is $LK^2$ instead of $L^2K^4$, especially when the inner layers consist of a lot of neurons.

Consider a change in the weight $w^l_{jk}$. Then the next activation will change by $\Delta a^l_j \approx \frac{\partial a^l_j}{\partial w^l_{jk}} \Delta w^l_{jk}$. In the $l+1$^st^ layer, for some neuron $q$, $\Delta a^{l+1}_q \approx \frac{\partial a^{l+1}_q}{\partial a^l_j} \frac{\partial a^l_j}{\partial w^l_{jk}} \Delta w^l_{jk}$. Carrying on we have a chain expressing $\partial \ell / \partial w^l_{jk}$, which can be written out as $\partial C / \partial w^l \sim M^L M^{L-1} \cdots M^{l}$ and we backpropagate to get these matrix multiplications.

## Better Backpropagation techniques

### Cross-entropy loss function

The sigmoid function is flat when the neuron is saturated. For an algorithm with a single neuron, $\ell = (y-a)^2/2$, and $a = \sigma(z)$. If $z$ is large the learning rate input $\times \sigma'(z)$ is small. We may amend the cost function to be of the  **cross-entropy** type.

# preprocessing

Mutual information. `sklearn.feature_selection.mutual_info_regression` computing the mutual information b/w all features and a continuous outcome.

## Principal component analysis

------



