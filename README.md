# Computational Neurology

This repository deals with computational models with relevance to neurological syndromes. 

#### Contents
1. Internal clocks
2.  

#### Appendices
1. Active Filtering
2. Message Passing
3. Related Models

## 1 - Internal Clocks
This demo is based on the idea that we can make use of internal clocks to synchronise our behaviour to events in the world. This is of particular relevance for conditions such as Parkinson's disease or Lewy Body Dementia in which everything from movement to thought can be slowed. The demo depends upon an Active Bayesian Filtering scheme (c.f., <a href="https://en.wikipedia.org/wiki/Generalized_filtering">Generalised Filtering</a>) and simulates timed movements between two targets. 

<img src="Metronomes/Graphics/Animation Default.gif"/>

See <a href="Metronomes/README.md">this linked documentation</a> for details and variations on these simulations.

## A1 - Active Filtering
Active filtering refers to the use of a generalised filtering scheme that can be derived from an online application of variational Bayes, under a local Laplace assumption, to dynamical systems expressed in generalised coordinates of motion. Generalised coordinates effectively represent a timeseries in terms of position, velocity, acceleration, and subsequent orders of motion. The active part comes from allowing the filtering scheme to interact with the data-generating process. Effectively, this means equipping the sensory receptors that communicate data to the filter with reflex arcs, such that any deviation from predicted data can be 'corrected' through low level reflexes that bring data in line with our predictions. Crucially, this means both inference and action optimise the same objective function (the ELBO or variational free energy) - often used in machine learning to arrive at an approximation to a marginal likelihood. Please see <a href = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8320263/"> Parr et al, 2021</a> and associated supplementary material for an overview of this scheme and its application to simple movement problems in neurology.

The generative model upon which generalised filtering is based takes the following form:

```math
\begin{align}
\dot{x} = f(x) + \omega_x \\
y = g(x) + \omega_y
\end{align}
```
The $\omega$ terms are each assumed to be normally distributed with expectation values of zero. For the generalised filtering model, the $x$ and $y$ variables are expressed in terms of a Taylor series expansion of their trajectories. The coefficients of these expansions are simply successive temporal derivatives - i.e., $\tilde{x} = [x, x', x'', x''', ...]^T$. Using the symbol $D$ to represent a matrix that shifts every element of a vector up by one, we can re-express the pair of equations above in terms of 'generalised coordinates' as:

```math
\begin{align}
D\tilde{x} = \tilde{f}(\tilde{x}) + \tilde{\omega}_x \\
\tilde{y} = \tilde{g}(\tilde{x}) + \tilde{\omega}_y
\end{align}
```
These equations can be expressed in terms of probability distributions:
```math
\begin{align}
p(\tilde{x}) = N(D \cdot \tilde{f}(\tilde{x}),\tilde{\Pi}_x) \\
p(\tilde{y}|\tilde{x}) = N(\tilde{g}(\tilde{x}),\tilde{\Pi}_y)
\end{align}
```

[When complete, this section will include further detail on this scheme.]


Generation of action depends upon optimisation of the free energy with respect to the data. This requires an expression of the gradients of the data with respect to action. Assuming for the purposes of what follows that the generative process is determinstic, and expressing this with bold functions, we have for the observation model (omitting function arguments for simplicity):

```math
\begin{align}
y^{[i]} = \mathbf{g}^{[i]} \Rightarrow \partial_a \mathbf{g}^{[i]}  = \partial_a x^{[i]} \partial_{x^{[i]}}  \mathbf{g}^{[i]} \\
 \mathbf{g}^{[i]} \approx \partial_{x^{[0]}} \mathbf{g}^{[0]} x^{[i]} \\
\Rightarrow \\
\partial_a \mathbf{g}^{[i]}  \approx \partial_a x^{[i]} \partial_{x^{[0]}}  \mathbf{g}^{[0]}

\end{align}
```
This can be supplemented with the relevant gradients from the dynamic model:

```math
\begin{align}
x^{[i]} = \mathbf{f}^{[i-1]} \Rightarrow \partial_a x^{[i]} = \partial_a \mathbf{f}^{[i-1]} \\
\mathbf{f}^{[i-1]} \approx \partial_{x^{[0]}} \mathbf{f}^{[0]} x^{[i-1]} = \partial_{x^{[0]}} \mathbf{f}^{[0]} \mathbf{f}^{[i-2]} \\
\Rightarrow \\
\partial_a x^{[i]} \approx \partial_{x^{[0]}} \mathbf{f}^{[0]} \partial_a \mathbf{f}^{[i-2]} = (\partial_{x^{[0]}} \mathbf{f}^{[0]})^2 \partial_a \mathbf{f}^{[i-3]} = (\partial_{x^{[0]}} \mathbf{f}^{[0]})^{i-1} \partial_a \mathbf{f}^{[0]} \\
\Rightarrow \\
\partial_a y^{[i]} \approx \partial_{x^{[0]}}  \mathbf{g}^{[0]} (\partial_{x^{[0]}} \mathbf{f}^{[0]})^{i-1} \partial_a \mathbf{f}^{[0]}

\end{align}
```

## A2 - Message Passing

## A3 - Related Models
