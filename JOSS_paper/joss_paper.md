---
title: "ProMCDA: A Python package for Probabilistic Multi-Criteria Decision Analysis"
tags:
- Python
- "multi-criteria optimization"
- decision support
- probabilistic approach
- sensitivity analysis
- Monte Carlo sampling
date: "06 November 2023"
output:
  html_document:
    df_print: paged
authors:
- name: Flaminia Catalli
  orcid: "0000-0003-0515-5282"
  equal-contrib: yes
  affiliation: 1
- name: Matteo Spada
  orcid: "0000-0001-9265-9491"
  equal-contrib: yes
  affiliation: 2
bibliography: paper.bib
affiliations:
- name: wetransform GmbH, Germany
  index: 1
- name: Zurich University of Applied Sciences, School of Engineering, INE Institute
    of Sustainable Development, Switzerland
  index: 2
editor_options:
  markdown:
    wrap: 72
---

# Summary

Multi-Criteria Decision Analysis (MCDA) is a formal process to assist
decision makers (DMs) in structuring their decision problems and to
provide them with tools and methods leading to recommendations on the
decisions at stake (@roy_decision_1996). The recommendations are based
on a comprehensive identification of the alternatives considered and the
selection of criteria/subcriteria/etc. to evaluate them, which are
aggregated taking into account the preferences of the DMs
(@bouyssou_problem_2006). In the literature, there is a wide range of
MCDA methods used to integrate information and either classify
alternatives into preference classes or rank them from best to worst
(@cinelli_proper_2022). In the context of ranking and benchmarking
alternatives across complex concepts, composite indicators (CIs) are the
most widely used synthetic measures (@greco_methodological_2019).
Indeed, they have been applied, for example, in the context of
environmental quality (@otoiu_proposing_2018), resilience of energy
supply (@gasser_comprehensive_2020), sustainability
(@volkart_interdisciplinary_2016), global competitiveness
(@klaus_schwab_global_2018), etc. However, the uncertainty of the
criteria, the choice of methods (normalization/aggregation) to construct
CIs, etc. have been shown to influence the final ranking of alternatives
(e.g. @cinelli_mcda_2020).

The `ProMCDA` Python module proposed here allows a DM to explore the
sensitivity and robustness of the CIs results in a user-friendly way. In
other words, it allows the user to assess either the sensitivity related
to the choice of normalization and/or aggregation method, but also to
account for uncertainty in the criteria and weights.

# Statement of need

There are already dedicated tools for CIs in the literature. In *R*,
there is an existing package called *COINr*, which allows the user to
develop CIs by including all common operations, from criteria selection,
data treatment, normalization and aggregation, and sensitivity analysis
(@becker_coinr_2022). There are also other packages in R, such as
compind, that focus on weighting and aggregation (@fusco_spatial_2018).
In *MATLAB*, there are some packages dedicated to specific parts of CI
development, such as the *CIAO* tool (@linden_framework_2021). The
Python module *Decisi-o-Rama* (@chacon-hurtado_decisi-o-rama_2021)
focuses on the implementation of the Multi-Attribute Utility Theory
(MAUT) to normalize criteria, considering a hierarchical criteria
structure and uncertain criteria, and to aggregate the results using
different aggregation methods. Finally, the web tool called *MCDA Index
Tool* allows sensitivity analysis based on different combinations of
normalization functions and aggregation methods ([MCDA Index
Tool](https://www.mcdaindextool.net)).

`ProMCDA` is a Python module for performing CIs MCDA considering a full
probabilistic approach. The tool provides sensitivity and robustness
analysis of the ranking results. The sensitivity of the MCDA scores is
caused by the different pairs of normalization/aggregation functions
(@cinelli_mcda_2020) that can be used in the evaluation process. The
uncertainty is instead caused by either the variability associated with
the criteria values (@stewart_dealing_2016) or the randomness that may
be associated with their weights (@lahdelma_smaa_1998). `ProMCDA` is
unique in combining all these different sources of variation and
providing a systematic analysis.

The tool is designed to be used by both researchers and practitioners in
operations research. The approach has a wide range of potential
applications, ranging from sustainability to healthcare and risk
assessment, to name but a few. `ProMCDA` has been developed as a core
methodology for the development of a decision support system for forest
management ([FutureForest](https://future-forest.eu/)). However, the
tool is generic and can be used in any other domain involving
multi-criteria decision making.

# Overview

`ProMCDA` is a module consisting of a set of functions that allow CIs to be 
constructed considering the uncertainty associated with the criteria, the 
weights and the combination of normalization/aggregation methods. 
The evaluation process behind `ProMCDA` is based on two main steps of data 
manipulation: 

- data normalisation, to work with data values on the same scale; 

- data aggregation, to estimate a single composite indicator from all criteria.

`ProMCDA` receives all the necessary input information via a configuration file 
in JSON format (for more details see the 
[README](https://github.com/wetransform-os/ProMCDA/blob/main/README.md)). 
The alternatives are represented in an input matrix (in CSV file format) as rows 
and described by the different values of the criteria in the columns. 
The sensitivity analysis is performed by comparing the different scores 
associated with the alternatives, which are obtained by using a different 
combination of normalization and aggregation functions.
`ProMCDA` implements 4 different normalization and 4 different aggregation 
functions, as described in \autoref{fig:normalisation} and 
\autoref{fig:aggregation} respectively. However, the user can decide to run 
`ProMCDA` with a specific pair of normalization and aggregation functions, 
and thus the sensitivity analysis.

The user can also decide to run `ProMCDA` with or without robustness analysis. 
The robustness analysis is potentially triggered by adding randomness to either 
the weights or the criteria. This means that either the weights or the criteria 
values are randomly sampled using a Monte Carlo method. In `ProMCDA` randomness 
is not allowed for both weights and criteria in order to make the results as 
transparent as possible. In fact, mixing uncertainty from both weights and 
criteria would lead to a lack of distinction between the effect of one or the 
other. Randomness in the weights can be applied to one weight at a time or to 
all weights at the same time. In the first case, the aim is to be able to 
analyse the effect of each individual criteria on the scores; in the second 
case, it is to have an overview of the uncertainty potentially associated with 
all the weights. In both cases, by default, the weights are sampled from a 
uniform distribution [0-1]. On the other hand, if the user decides to analyse 
the robustness of the criteria, he/she has to provide the parameters defining 
the marginal distribution (i.e. a probability density function, pdf) that best 
describes the criteria, rather than the criteria values. This means that if a 
criteria is characterized by a pdf described by 2 parameters, it should be 
allocated two columns in the input CSV file. In `ProMCDA` 4 different pdfs 
describing the criteria uncertainty are considered:

-   uniform, which is described by 2 parameters, i.e., minimum and
    maximum

-   normal, which is described by 2 parameters, i.e., mean and standard
    deviation

-   lognormal, which is described by 2 parameters, i.e., log(mean) and
    log(standard deviation)

-   Poisson, which is described by 1 parameter, i.e., the rate.

Once the pdfs for each criteria is selected and the input parameters are
in place in the input CSV file, `ProMCDA` randomly samples n-values of
each criteria per alternative from the given pdf to assess the score and
ranking of alternatives considering robustness at the criteria level.

Once the pdfs for each criteria are selected and the input parameters are 
in the input CSV file, `ProMCDA` randomly samples n-values of each criterion 
per alternative from the given pdf to evaluate the score and ranking of the 
alternatives, taking into account robustness at the criteria level.

Finally, in all possible cases (i.e. a simple MCDA; MCDA with
sensitivity analysis for the different normalization/aggregation
functions used; MCDA with robustness investigation related either to
randomness on the weights or on the indicators), `ProMCDA` will output a
CSV file with the scores/average scores and their plots. For a quick
overview of the functionality of `ProMCDA`, refer to
\autoref{fig:promcda_overview}

![Normalization functions implemented in
ProMCDA.](normalization_table.png) \# From Matteo: We need to crosscheck
this table with the published one, since they should not be equal (==
plagiarism) ![Aggregation functions implemented in ProMCDA. The sum of
the weights is normalized to 1, \@Langhans et
al.:2014..](aggregation_table.png) \# From Matteo: We need to crosscheck
this table with the published one, since they should not be equal (==
plagiarism) ![Overview on the functionalities of
ProMCDA.](ProMCDA_overview.png)

# Acknowledgements

TBD

# References
