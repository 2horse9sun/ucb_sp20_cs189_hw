## 1 Honor Code

*I certify that all solutions are entirely my own and that I have not looked at anyone else’s solution. I have given credit to all external sources I consulted.*

## 2 Gaussian Classification

### 2.1

$$
P(Y=C_i|X=x)=\frac{P(X=x|Y=C_i)P(Y=C_i)}{P(X=x)}
$$

$$
f_{C_i}(x)=\frac{1}{\sqrt{2\pi}\sigma}exp\{-\frac{||x-\mu_i||^2}{2\sigma^2}\}
$$

$$
Q_{C_i}(x)=\ln(\sqrt{2\pi}f_{C_i}(x)\pi_{C_i})=-\frac{||x-\mu_i||^2}{2\sigma^2}-\ln\sigma-\ln2
$$

$$
Q_{C_1}(x)>Q_{C_2}(x)
$$



Bayes decision rule:
$$
\begin{equation}
r^*(x)=
\left\{
             \begin{array}{lr}
            C_1, & x<\frac{\mu_1+\mu_2}{2} \\
            C_2 & o.w.
             \end{array}
\right.
\end{equation}
$$


### 2.2

$$
P_e=\frac{1}{2}\int_{-\infty}^{\frac{\mu_1+\mu_2}{2}}f_{C_2}(x)+\frac{1}{2}\int_{\frac{\mu_1+\mu_2}{2}}^{\infty}f_{C_1}(x)=\frac{1}{\sqrt{2\pi}}\int_{\frac{\mu_2-\mu_1}{2\sigma}}^{\infty}e^{-z^2/2}dz
$$

## 3 Isocontours of Normal Distributions

### 3.1













Supposed that:
$$
f(x)=\frac{1}{(\sqrt{2\pi})^d\sqrt{\Sigma_I}}exp\{-\frac{1}{2}x^T\Sigma_I^{-1}x\},\Sigma_I=I
$$
Apply the transform:
$$
t=Ax=RSx,x=A^{-1}t
$$

$$
t'=t+\mu
$$



in which, $S$ is a scale matrix, $R$ is a rotation matrix. We get:
$$
f(t')\sim exp\{-\frac{1}{2}(t'-\mu)^T(AA^T)^{-1}(t'-\mu)\}
$$
Obviously, the covariance matrix:
$$
\Sigma=AA^T
$$
Apply the eigendecomposition:
$$
\Sigma=U\Lambda U^T=(U\Lambda^{1/2})(U\Lambda^{1/2})^T
$$
We can get:
$$
A=U\Lambda^{1/2}=RS
$$

$$
R=U
$$

$$
S=\Lambda^{1/2}
$$

Now, let's apply the inverse transform:

(1) centering:
$$
t=t'-\mu
$$
(2) decorrelating:
$$
R^{-1}t=U^{-1}t=U^Tt
$$
(3) sphering:
$$
x=S^{-1}U^Tt=\Lambda^{-1/2}U^Tt
$$








$$
f(X_i)=\frac{1}{(\sqrt{2\pi})^d\sqrt{|\Sigma|}}\exp\{-\frac{1}{2}(X_i-\mu)^T\Sigma^{-1}(X_i-\mu)\}
$$
Likelihood:
$$
\mathcal{L}(\mu,\sigma;X_1,\cdots,X_n)=\prod_{i=1}^nf(X_i)
$$
Log likelihood:
$$
\mathcal{l}(\mu,\sigma;X_1,\cdots,X_n)=\sum_{i=1}^n\ln f(X_i)=\sum_{i=1}^n[-\frac{1}{2}(X_i-\mu)^T\Sigma^{-1}(X_i-\mu)-d\ln\sqrt{2\pi}-\ln\sqrt{|\Sigma|}]
$$
Take the derivative with respect to $\mu$ and set it to zero:
$$
\nabla_{\mu}\mathcal{l}=\sum_{i=1}^n\Sigma^{-1}(X_i-\mu)=0
$$

$$
\hat{\mu}=\frac{1}{n}\sum_{i=1}^nX_i
$$

To find $\hat{\sigma}_i$:
$$
f(X_i)=\frac{1}{(\sqrt{2\pi})^d\prod_{j=1}^d\sigma_j}\exp\{-\frac{1}{2}(X_i-\mu)^T\Sigma^{-1}(X_i-\mu)\}
$$

$$
\mathcal{l}(\mu,\sigma;X_1,\cdots,X_n)=\sum_{i=1}^n[-\frac{1}{2}\sum_{j=1}^d\frac{1}{\sigma_j^2}||X_i-\mu||^2-d\ln\sqrt{2\pi}-\ln\prod_{j=1}^d\sigma_j]
$$

$$
\frac{\partial\mathcal{l}}{\partial \sigma_i}=\sum_{i=1}^n\frac{||X_i-\mu||^2-\sigma_i^2}{\sigma_i^3}=0
$$

$$
\sigma_i^2=\frac{1}{n}\sum_{i=1}^n||X_i-\mu||^2=\frac{1}{n}\sum_{i=1}^n||X_i-\hat{\mu}||^2
$$






$$
\nabla_{\mu}\mathcal{l}=\sum_{i=1}^nA^T\Sigma^{-1}(X_i-A\mu)=0
$$

$$
\hat{\mu}=\frac{1}{n}\sum_{i=1}^nA^{-1}X_i
$$

