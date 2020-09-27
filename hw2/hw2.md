## 1 Identities with Expectation

### 1.1

When $n=k$:
$$
E[X^{k}]=\frac{k!}{\lambda^{k}}=\int_{0}^{\infty}x^{k}f(x)dx=\lambda\int_{0}^{\infty}x^{k}e^{-\lambda x}dx
$$
When $n=k+1$:
$$
\begin{equation}
\begin{split}
E[X^{k+1}]&=\int_{0}^{\infty}x^{k+1}f(x)dx \\
&=\lambda\int_{0}^{\infty}x^{k+1}e^{-\lambda x}dx\\
&=-\int_{0}^{\infty}x^{k+1}de^{-\lambda x}\\
&=-[x^{k+1}e^{-\lambda x}|_{0}^{\infty}-\int_{0}^{\infty}(k+1)x^{k}e^{-\lambda x}dx]
&=\frac{(k+1)!}{\lambda^{k+1}}
\end{split}
\end{equation}
$$

### 1.2

$$
\begin{equation}
\begin{split}
E[X]&=\int_{0}^{\infty}xf(x)dx \\
&=\int_{0}^{\infty}xdF(x)\\
&=-\int_{0}^{\infty}xd(1-F(x))\\
&=-x(1-F(x))|_{0}^{\infty}+\int_{0}^{\infty}(1-F(x))dx\\
&=\int_{0}^{\infty}P(X\geq t)dt
\end{split}
\end{equation}
$$

### 1.3

$$
\begin{equation}
\begin{split}
E[X]&=E[X\mathbf{1}\{X=0\}+E[X\mathbf{1}\{X>0\}\\
&=0+E[X\mathbf{1}\{X>0\}\\
&\leq \sqrt{E[X^2]E[(\mathbf{1}\{X>0\})^2]} &(Cauchy–Schwarz)\\
&=\sqrt{E[X^2]P(X>0)}&(P(A)=E[\mathbf{1}\{A\}])\\
\end{split}
\end{equation}
$$

### 1.4

$$
\begin{equation}
\begin{split}
E[t-X]&\leq E[(t-X)\mathbf{1}\{t-X>0\}]\\
&\leq \sqrt{E[t^2+X^2]E[(\mathbf{1}\{t-X>0\})^2]}\\
&=\sqrt{(t^2+E[X^2])(1-P(X\geq t))}
\end{split}
\end{equation}
$$



## 2 Probability Potpourri

### 2.1

$$
\begin{equation}
\begin{split}
x^{T}\Sigma x&=x^{T}E[(Z-\mu)(Z-\mu)^{T}]x\\
&=E[x^{T}(Z-\mu)(Z-\mu)^{T}x]\\
&=E[[(Z-\mu)^{T}x]^{T}[(Z-\mu)^{T}x]]\\
&=E[||(Z-\mu)^{T}x||^2]\\
&\geq 0
\end{split}
\end{equation}
$$

### 2.2

Define:
$$
H:hit
$$

$$
W:windy
$$

Given:
$$
P(H|W)=0.4
$$

$$
P(H|\overline{W})=0.7
$$

$$
P(W)=0.3
$$

(i)
$$
P(H|W)=0.4
$$
(ii)
$$
P(H)=P(HW)+P(H\overline{W})=p(W)P(H|W)+P(\overline{W})P(H|\overline{W})
$$
(iii)
$$
2P(H)(1-P(H))
$$
(iv)
$$
P(\overline{W}|\overline{H})=\frac{P(\overline{W})P(\overline{H}|\overline{W})}{p(W)P(\overline{H}|W)+P(\overline{W})P(\overline{H}|\overline{W})}
$$

### 2.3

$$
E[score]=\int_{0}^{\frac{1}{\sqrt{3}}}4f(x)dx+\int_{\frac{1}{\sqrt{3}}}^{1}3f(x)dx+\int_{1}^{\sqrt{3}}2f(x)dx
$$

### 2.4

$$
\begin{equation}
\begin{split}
P(X=k|X+Y=n)&=\frac{P(X=k)P(Y=n-k)}{P(X+Y=n)}\\
&=\frac{\frac{\lambda ^{k}}{k!}e^{-\lambda}\frac{\mu^{n-k}}{(n-k)!}e^{-\lambda}}{\frac{(\lambda+\mu)^{n}}{n!}e^{-\lambda}}\\
&=C_{n}^{k}\frac{\lambda^{k}\mu^{n-k}}{(\lambda+\mu)^{k}}e^{-\lambda} \\
\end{split}
\end{equation}
$$

## 3 Properties of Gaussians









## 7 Theory of Hard-Margin Support Vector Machines

### Dual Optimization Problem

Consider the optimization problem below:
$$
\min f(x)
$$

$$
s.t.\quad g_{k}(x)\leq 0
$$

Lagrange multiplier:
$$
\mathcal{L}(x,u)=f(x)+\sum_{k=1}^{n}\lambda_{k}g_{k}(x),\quad \lambda_{k}\geq0,g_{k}(x)\leq 0
$$
Obviously:
$$
\max_{\lambda}\mathcal{L}(x,\lambda)=f(x)
$$

$$
\min_{x}f(x)=\min_{x}\max_{\lambda}\mathcal{L}(x,\lambda)
$$

Then:
$$
\begin{equation}
\begin{split}
\max_{\lambda}\min_{x}\mathcal{L}(x,\lambda)&=\max_{\lambda}\min_{x}f(x)+\max_{\lambda}\min_{x}\lambda g(x) \\
&=\min_{x}f(x)+\max_{\lambda}\min_{x}\lambda g(x)\\
\end{split}
\end{equation}
$$
We can prove that:
$$
\begin{equation}
\min_{x}\lambda g(x)=
\left\{
             \begin{array}{lr}
            0, & \lambda=0 \quad or\quad g(x)=0 \\
            -\infty, & \lambda>0 \quad and\quad g(x)<0
             \end{array}
\right.
\end{equation}
$$
Therefore, when $\lambda=0 $ or $ g(x)=0$:
$$
\min_{x}f(x)=\min_{x}\max_{\lambda}\mathcal{L}(x,\lambda)=\max_{\lambda}\min_{x}\mathcal{L}(x,\lambda)
$$

### Karush-Kuhn-Tucker (KKT) conditions

Consider the standard constraint optimization problem:
$$
\min f(\mathbf{x})
$$

$$
\begin{equation}
\begin{split}
s.t.\quad&g_{j}(\mathbf{x})=0 \\
&h_{k}(\mathbf{x})\leq0\\
\end{split}
\end{equation}
$$

Define Lagrange multiplier:
$$
\mathcal{L}(\mathbf{x},\lambda,\mu)=f(x)+\sum_{j=1}^{m}\lambda_{j}g_{j}(\mathbf{x})+\sum_{k=1}^{p}\mu_{k}h_{k}(\mathbf{x})
$$
KKT conditions:
$$
\begin{equation}
\left\{
             \begin{array}{lr}
            \nabla_{\mathbf{x}}\mathcal{L}=0 \\
            g_{j}(\mathbf{x})=0 \\
            h_{k}(\mathbf{x})\leq 0\\
            \mu_{k}\geq 0\\
            \mu_{k}h_{k}(\mathbf{x})=0
             \end{array}
\right.
\end{equation}
$$

### Hard-Margin Support Vector Machines

Equivalent optimization problem rephrased by dual optimization problem:
$$
\max_{\lambda_{i}\geq 0}\min_{w,\alpha}\|w\|^2-\sum_{i=1}^n\lambda_{i}[y_{i}(X_{i}\cdot w+\alpha)-1]
$$
To solve the inner optimization problem, define Lagrange multiplier:
$$
\mathcal{L}(w,\alpha,\lambda)=w^Tw-\sum_{i=1}^n\lambda_{i}[y_{i}(w^TX_{i}+\alpha)-1]
$$

$$
\frac{\partial\mathcal{L}}{\partial w}=2w-\sum_{i=1}^n\lambda_{i}y_{i}X_{i}=0
$$

$$
\frac{\partial\mathcal{L}}{\partial \alpha}=-\sum_{i=1}^n\lambda_{i}y_{i}=0
$$

Finally, we get:
$$
\max_{\lambda_{i}\geq 0}\sum_{i=1}^n\lambda_{i}-\frac{1}{4}\sum_{i=1}^n\sum_{=1}^n\lambda_{i}\lambda_{j}y_{i}y_{j}X_{i}X_{j},\quad s.t.\sum_{i=1}^n\lambda_{i}y_{i}=0
$$
