# Computational Neurology

This repository deals with computational models with relevance to neurological syndromes. 

#### Contents
1. Internal clocks
2.  

#### Appendices
1. Active Filtering
2. Related models

## 1 - Internal Clocks
This demo is based on the idea that we can make use of internal clocks to synchronise our behaviour to events in the world. This is of particular relevance for conditions such as Parkinson's disease or Lewy Body Dementia in which everything from movement to thought can be slowed. The demo depends upon an Active Bayesian Filtering scheme (c.f., <a href="https://en.wikipedia.org/wiki/Generalized_filtering">Generalised Filtering</a>) and simulates timed movements between two targets. 

<img src="Metronomes/Graphics/Animation_defaults.gif"/>

See <a href="Metronomes/README.md">this linked documentation</a> for details and variations on these simulations.

## A1 - Active Filtering
Active filtering refers to the use of a generalised filtering scheme that can be derived from an online application of variational Bayes, under a local Laplace assumption, to dynamical systems expressed in generalised coordinates of motion. Generalised coordinates effectively represent a timeseries in terms of position, velocity, acceleration, and subsequent orders of motion. The active part comes from allowing the filtering scheme to interact with the data-generating process. Effectively, this means equipping the sensory receptors that communicate data to the filter with reflex arcs, such that any deviation from predicted data can be 'corrected' through low level reflexes that bring data in line with our predictions. Crucially, this means both inference and action optimise the same objective function (the ELBO or variational free energy) - often used in machine learning to arrive at an approximation to a marginal likelihood. Please see <a href = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8320263/"> Parr et al, 2021</a> and associated supplementary material for an overview of this scheme and its application to simple movement problems in neurology.

[When complete, this section will include further detail on this scheme.]

## A2 - Related Models
