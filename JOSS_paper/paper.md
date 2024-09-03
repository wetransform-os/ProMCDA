---
title: "ProMCDA: A Python package for Probabilistic Multi-Criteria Decision Analysis"
tags:
- Python
- Multi-criteria Decision Analysis (MCDA)
- decision support
- probabilistic approach
- sensitivity analysis
- Monte Carlo sampling
date: "15 May 2024"
output:
  pdf_document: 
    fig_caption: yes
  html_document: 
    fig_caption: yes
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
  chunk_output_type: inline
---

# Summary

Multi-Criteria Decision Analysis (MCDA) is a formal process used to assist decision-makers in structuring complex decision problems and providing recommendations based on a comprehensive evaluation of alternatives. This evaluation is conducted by selecting relevant criteria and subcriteria, which are then aggregated according to the preferences of the decision-makers to produce a ranking or classification of the alternatives (@roy_decision_1996; @bouyssou_problem_2006). A wide range of MCDA methods are available in the literature for integrating information to classify alternatives into preference classes or rank them from best to worst (@cinelli_proper_2022). Among these, composite indicators (CIs) are commonly used synthetic measures for ranking and benchmarking alternatives across complex concepts (@greco_methodological_2019). Examples of CI applications include environmental quality assessment (@otoiu_proposing_2018), resilience of energy supply (@gasser_comprehensive_2020), sustainability (@volkart_interdisciplinary_2016), and global competitiveness (@klaus_schwab_global_2018).

However, the final ranking of alternatives in MCDA can be influenced by various factors such as uncertainty in the criteria, the choice of weights assigned to them, and the selection of methods for normalization and aggregation to construct CIs (@cinelli_mcda_2020). To address these challenges, the `ProMCDA` Python module has been developed to allow decision-makers to explore the sensitivity and robustness of CI results in a user-friendly manner. This tool facilitates sensitivity analysis related to the choice of normalization and aggregation methods and accounts for uncertainty in criteria and weights, providing a systematic approach to understanding the impact of these factors on decision outcomes.

# Statement of need

Several MCDA tools are available in the literature. For example, the Python library `pymcdm` (@kizielewicz2023pymcdm) provides a broad collection of different MCDA methods, including those commonly used to construct CIs. The `pyDecision` library (@pereira_enhancing_2024) offers a large collection of MCDA methods and allows users to compare outcomes of different methods interactively, thanks to integration with ChatGPT. In R, the package `COINr` enables users to develop CIs with all standard operations, including criteria selection, data treatment, normalization, aggregation, and sensitivity analysis (@becker_coinr_2022). Other packages, such as `compind`, focus specifically on weighting and aggregation (@fusco_spatial_2018), while MATLAB tools like CIAO (@linden_framework_2021) offer specialized capabilities for parts of CI development.

The Python module `Decisi-o-Rama` (@chacon-hurtado_decisi-o-rama_2021) focuses on implementing Multi-Attribute Utility Theory (MAUT) to normalize criteria, considering a hierarchical criteria structure and uncertain criteria, and aggregate the results using different aggregation methods. Additionally, the web-based [MCDA Index Tool](https://www.mcdaindextool.net) supports sensitivity analysis based on various combinations of normalization functions and aggregation methods.

While these tools provide valuable functionalities, `ProMCDA` differentiates itself by adopting a fully probabilistic approach to perform MCDA for CIs, providing sensitivity and robustness analysis of the ranking results. The sensitivity of the MCDA scores arises from the use of various combinations of normalization/aggregation functions (@cinelli_mcda_2020) that can be used in the evaluation process. Meanwhile, uncertainty stems from the variability associated with the criteria values (@stewart_dealing_2016) or the randomness that may be associated with their weights (@lahdelma_smaa_1998). `ProMCDA` is unique in combining all these different sources of variability and providing a systematic analysis.

The tool is designed for use by both researchers and practitioners in operations research. Its approach offers a broad range of potential applications, including sustainability, healthcare, and risk assessment, among others. `ProMCDA` has been developed as a core methodology for the development of a decision support system for forest management ([FutureForest](https://future-forest.eu/)). However, the tool is versatile and can be used in any other domain involving multi-criteria decision-making.

# Overview

`ProMCDA` is a Python module that allows users to construct CIs while considering uncertainties associated with criteria, weights, and the choice of normalization and aggregation methods. The module's evaluation process is divided into two main steps:
- **Data Normalization:** Ensuring all data values are on the same scale.
- **Data Aggregation:** Estimating a single composite indicator from all criteria.


`ProMCDA` receives all necessary input information via a configuration file in JSON format (for more details, see the [README](https://github.com/wetransform-os/ProMCDA/blob/main/README.md)). The alternatives are represented as rows in an input matrix (CSV file format), with criteria values in columns. The tool offers the flexibility to conduct sensitivity analysis by comparing the different scores associated with alternatives using various combinations of normalization and aggregation functions. `ProMCDA` currently implements four normalization and four aggregation functions, as described in [Table 1](#Table 1) and [Table 2](#Table 2), respectively. However, the user can run `ProMCDA` with a specific pair of normalization and aggregation functions, thus switching off the sensitivity analysis. <br />

<a name="Table 1"></a>*Table 1: Normalization functions used in `ProMCDA`.*
\begin{center} 
\includegraphics[width=300px]{Table1.png}
\end{center} 

<br />

<a name="Table 2"></a>*Table 2: Aggregation functions used in `ProMCDA`.The sum of the weights is normalized to 1 as in @langhans_method_2014.*
\begin{center} 
\includegraphics[width=300px]{Table2.png}
\end{center} 

The user can bypass both the sensitivity and robustness analysis when running `ProMCDA`.

**Sensitivity Analysis:** `ProMCDA` provides a default sensitivity analysis based on the predefined normalization and aggregation pairs. However, users can specify the pair of functions they want to use and switch this analysis off.

**Robustness Analysis:** `ProMCDA` also allows for robustness analysis by introducing randomness to either the weights or the criteria in order to make the results as transparent as possible and avoid a lack of distinction between the effect of one or the other. Randomly sampling the weights or the criteria values is done using a Monte Carlo method.

The randomness in the weights can be applied to one weight at a time or to all weights simultaneously. In both cases, by default, the weights are sampled from a uniform distribution [0-1]. If the user decides to analyse the robustness of the criteria, they have to provide the parameters defining the marginal distribution (i.e., a probability density function, pdf) that best describes the criteria rather than the criteria values. This means that if a pdf described by 2 parameters characterizes a criterion, two columns should be allocated in the input CSV file for it.
In `ProMCDA` 4 different pdfs describing the criteria uncertainty are considered:

-   *uniform*, which is described by 2 parameters, i.e., minimum and maximum

-   *normal*, which is described by 2 parameters, i.e., mean and standard deviation

-   *lognormal*, which is described by 2 parameters, i.e., log(mean) and log(standard deviation)

-   *Poisson*, which is described by 1 parameter, i.e., the rate.

Once the pdf for each criterion is selected and the input parameters are in place in the input CSV file, `ProMCDA` randomly samples n-values of each criterion per alternative from the given pdf and assesses the score and ranking of the alternatives by considering robustness at the criteria level. The number of samples is defined in the configuration file by the user.

Once the pdfs for each criterion are selected and the input parameters are in the input CSV file, `ProMCDA` randomly samples n-values of each criterion per alternative from the given pdf to evaluate the alternatives' scores and rankings, taking into account robustness at the criteria level.

Finally, in all possible cases (i.e., a simple MCDA, MCDA with sensitivity analysis for the different normalization/aggregation functions used, MCDA with robustness investigation related either to randomness on the weights or on the indicators), `ProMCDA` will output a CSV file with the scores/average scores and their plots. For a quick overview of the functionality of `ProMCDA`, refer to  [Table 3](#Table 3). For more details, refer to the [README](https://github.com/wetransform-os/ProMCDA/blob/main/README.md).

<a name="Table 3"></a>*Table 3: Overview on the functionalities of ProMCDA.*
\begin{center} 
\includegraphics[width=300px]{Table3.png}
\end{center} 

# Acknowledgements

Flaminia Catalli was supported by the Future Forest II project funded by the Bundesministerium für Umwelt, Naturschutz, nukleare Sicherheit und Verbraucherschutz (Germany) grant Nr. 67KI21002A. The authors would like to thank Kapil Agnihotri for thorough code revisions, Thorsten Reitz, and the whole Future Forest II team for productive discussions on a problem for which we have found a robust and transparent solution over time.

# References
