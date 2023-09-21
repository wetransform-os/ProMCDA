---
title: 'ProMCDA: A Python package for Probabilistic Multi Criteria Decision Analysis'
tags:
  - Python
  - multi-criteria optimization
  - decision support
  - probabilistic approach
  - sensitivity analysis
  - Monte Carlo sampling
authors:
  - name: Flaminia Catalli
    orcid: 0000-0003-0515-5282
    equal-contrib: true
    affiliation: 1 
  - name: Matteo Spada
    orcid: 0000-0001-9265-9491
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
affiliations:
 - name: wetransform GmbH, Germany
   index: 1
 - name: Zürich University of Applied Sciences - School of Engineering - INE Institute of Sustainable Development, Switzerland
   index: 2
date: 18 September 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

In the context of decision management and support, Multi Criteria Decision Analysis (MCDA) 
is important because it helps decision-makers handle multifactorial, complex decision 
problems by providing a structured and systematic scoring process. MCDA methods are widely used 
in problems in which a set of alternatives should be ranked according to a set of criteria. A criterion 
can be classified as being of benefit (e.g., to be maximized), or of cost (e.g., to be minimized). 
By considering MCDA, the decision-makers are allowed to explore different options' trade-offs and potential impacts.
The introduction of a probabilistic approach, where uncertainties are taken into account,
leads to more robust and well-informed decisions, accompanied by a thorough sensitivity analysis. # TODO: add references

# Statement of need

`ProMCDA` is  a Python package for MCDA with a probabilistic approach. 
The tool offers a full variability and sensitivity analysis of the ranking results.
The variability of the MCDA scores are caused by the different pairs of normalization/aggregation functions 
(# TODO: add reference to www.mcdaindex.net, and others) that can be used in the process of evaluation.
The uncertainty instead is caused by either the standard deviation related to the average criteria values 
(# TODO: add reference to R-code related work) or the randomness that might be associated to their weights 
(# TODO: add reference to SMAA). `ProMCDA` is unique for combining all those different sources of dispersion and offering 
a systematic analysis.    

This tool was designed to be used by both researchers and practitioners in operations research.
The approach has a wide spectrum of possible applications that ranges from sustainability, to health-care, and risk 
assessment, to mention a few. `ProMCDA` has been developed as core methodology for engineering a decision support system 
for forest management ([FutureForest](https://future-forest.eu/)). However, the tool is generic and can be used in any other 
domain pertaining multi-criteria decision.

# Overview
MCDA is an algorithm that allows to aggregate information coming from different indicators (i.e. criteria) into one score.
The evaluation process behind MCDA is based on two main steps of data manipulation:
- data normalization, for working with data values on the same scale;
- data aggregation, for estimating a single composite indicator from all criteria.

`ProMCDA` receives all needed input information by mean of a configuration file in JSON format (for more details see 
the [README](https://github.com/wetransform-os/ProMCDA/blob/main/README.md)).
The alternatives are displayed in an input matrix (in CSV file format) as rows, and described by the different values of the criteria in the columns.
The variability analysis comes by comparing the different scores associated with the alternatives, which are estimated
by using different combination of normalization and aggregation functions. `ProMCDA` implements 4 different normalization and 4 
different aggregation functions, as reported in \autoref{fig:normalization} and \autoref{fig:aggregation}. However, the user can decide to
run `ProMCDA` with one specific pair of normalization and aggregation, and therefore switch-off the variability investigation.
The user can also decide to run `ProMCDA` with or without a sensitivity analysis. The sensitivity analysis is potentially triggered by 
associating randomness either with the weights or with the indicators. This means that either the weight or the indicator values are 
randomly sampled by mean of a Monte-Carlo method. We do not allow randomness for both weights and indicators because we want the 
results to be as easy to interpret as possible. To mix the uncertainty that comes from the weights and the indicators would mean 
not being able to distinguish between the effect of one or the other. Randomness on the weights can be associated on one weight 
at a time or on all weights at the same time. In the first case, the objective is to be able to analyse the effect of each 
individual indicator on the scores; in the second case, it is to have an overview of the uncertainty potentially linked to the 
weights all. Weights are sampled by default from a uniform distribution [0-1]. On the other hand, if the user decides to analyse indicator-related sensitivity, 
there are two possible scenarios. One is the case where the indicators are observables, thus not associated with an intrinsic standard deviation. 
In this scenario, the user must provide the mean value of each indicator for each alternative, the standard deviation of interest, and the marginal 
distribution (i.e. a probability density function, pdf) that best describes the distribution of the specific indicator. Then `ProMCDA` randomly samples 
n-values of each indicator per alternative from the given pdf. If, on the other hand, the indicators come from simulations and their values are intrinsically
linked to a standard deviation, this means that it is not necessary to use Monte-Carlo sampling but that the user himself will provide the n-values of each 
indicator for each alternative. However, this is only true for simulations involving stochasticity; if the simulations are deterministic, we are again in 
the first scenario and a Monte-Carlo sampling is required and performed within `ProMCDA`. `ProMCDA` can sample values from 5 different pdfs: exact, uniform, normal, lognormal, and Poisson.
In all possible cases (i.e. a simple MCDA; MCDA with variability analysis for the different normalization/aggregation functions in use; MCDA with sensitivity 
investigation related either to randomness on the weights or on the indicators), `ProMCDA` will output a CSV file with the scores/average scores and their plots. 
For a quicker overview of the functionalities of `ProMCDA`, refer to \autoref{fig:promcda_overview}

![Normalization functions implemented in `ProMCDA`.\label{fig:normalization}](normalization_table.png)
![Aggregation functions implemented in `ProMCDA`. The sum of the weights is normalized to 1, `@Langhans et al.:2014`..\label{fig:aggregation}](aggregation_table.png)
![Overview on the functionalities of `ProMCDA`.\label{fig:promcda_overview}](ProMCDA_overview.png)

# Citations
For a quick reference, the following citation commands can be used:
- `@Pearson:2017`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"


# Acknowledgements
TBD

# References