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
can be classified as being of benefit (e.g., to be maximized), or of cost,(e.g., to be minimized). 
By considering MCDA, the decision-makers are allowed to explore different options' trade-offs and potential impacts.
The introduction of a probabilistic approach, where uncertainties are taken into account,
leads to more robust and well-informed decisions accompanied by a thorough sensitivity analysis. # TODO: add references

# Statement of need

`ProMCDA` is  a Python package for Multi Criteria Decision Analysis with a probabilistic approach. 
The tool offers a full variability and sensitivity analysis of the ranking results.
The variability of the MCDA scores are caused by the different pairs of normalization/aggregation functions 
(# TODO: add reference to www.mcdaindex.net) that can be used in the process of evaluation.
The uncertainty instead is caused by either the standard deviation related to the average criteria values 
(# TODO: add reference to R-code related work) or the randomness that might be associated to their weights 
(# TODO: add reference to SMAA). ProMCDA is unique for combining all those different source of dispersion and offering 
a systematic analysis.    

`ProMCDA` was designed to be used by both researchers and practitioners in operations research.
The approach has a wide spectrum of possible applications that range from sustainability, to health-care, and risk 
assessment, to mention a few. `ProMCDA` has been developed as core methodology for engineering a decision support system 
for forest management (# TODO: add reference to FutureForest). However, the tool is generic and can be used in any other 
domain pertaining multi-criteria decision.

# Overview


# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References