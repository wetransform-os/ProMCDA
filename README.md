<div align="left">
<img src="https://raw.githubusercontent.com/wetransform-os/ProMCDA/release/logo/ProMCDA_logo.png">
</div>

# Probabilistic Multi Criteria Decision Analysis

<!-- [![PyPi version](https://img.shields.io/pypi/v/promcda?color=blue)](https://pypi.org/project/promcda) -->

![PyPI](https://img.shields.io/pypi/v/ProMCDA?label=pypi%20package)
[![pytest](https://github.com/wetransform-os/ProMCDA/actions/workflows/python-app-tests.yml/badge.svg)](https://github.com/wetransform-os/ProMCDA/actions/workflows/python-app-tests.yml)
![License](https://img.shields.io/badge/license-EPL%202.0-blue)
[![status](https://joss.theoj.org/papers/cd66aa1ed9ff89b5519d977f4a16379d/status.svg)](https://joss.theoj.org/papers/cd66aa1ed9ff89b5519d977f4a16379d)
![PyPI - Downloads](https://img.shields.io/pypi/dm/ProMCDA)
[![Downloads](https://static.pepy.tech/badge/ProMCDA)](https://pepy.tech/project/ProMCDA)

A tool to estimate scores of alternatives and their uncertainties based on a Multi Criteria Decision Analysis (MCDA) approach.

### Table of Contents
- [Contributing](#contributing)
- [MCDA quick overview and applications](#mcda-quick-overview-and-applications)
- [Requirements to use ProMCDA](#requirements-to-use-promcda)
- [How to use ProMCDA](#how-to-use-promcda)
- [The output of ProMCDA](#the-output-of-promcda)
- [Input matrix](#input-matrix)
- [Running the unit tests](#running-the-unit-tests)
- [Toy example](#toy-example)
- [A high level summary](#a-high-level-summary)
- [General information and references](#general-information-and-references)
- [Releases](#releases)
- [Get in touch](#get-in-touch)

### Contributing
We welcome contributions from the community! Before contributing, please read our 
[Contribution Guidelines](./CONTRIBUTING.md) to learn about our development process, coding standards, and how to submit 
pull requests.

### MCDA quick overview and applications
A MCDA approach is a systematic framework for making decisions in situations where multiple criteria need to be 
considered. It ranks a set of alternatives considering the multiple criteria and the user preferences.
It can be applied in various domains and contexts. Here are some possible usages of an MCDA approach:

- **Environmental impact assessment**: assess the environmental consequences of various projects or policies by considering 
  multiple environmental criteria.
- **Project selection**: choose the most suitable project from a set of alternatives by considering multiple criteria such 
  as cost, risk, and strategic alignment.
- **Healthcare decision-making**: decide on the allocation of healthcare resources, such as funding for medical treatments 
  or the purchase of medical equipment, while considering criteria like cost-effectiveness, patient outcomes, 
  and ethical considerations. It applies to other decision-making problems too.
- **Investment portfolio optimization**: construct an investment portfolio that balances criteria like risk, return, 
  and diversification.
- **Personnel recruitment**: evaluate job candidates based on criteria like qualifications, experience, and cultural fit.
- **Location analysis**: choose the optimal location for a new facility or business by considering factors like cost, 
  accessibility, and market potential.
- **Public policy evaluation**: assess the impact of proposed policies on various criteria, such as economic growth, 
  social welfare, and environmental sustainability.
- **Product development**: prioritize features or attributes of a new product by considering criteria like cost, market demand, 
  and technical feasibility.
- **Disaster risk management**: identify high-risk areas and prioritize disaster preparedness and mitigation measures based 
  on criteria like vulnerability, exposure, and potential impact.
- **Energy planning**: make decisions about energy resource allocation and investments in renewable energy sources, 
  taking into account criteria like cost, environmental impact, and reliability.
- **Transportation planning**: optimize transportation routes, modes, and infrastructure investments while considering 
  criteria like cost, time efficiency, and environmental impact.
- **Water resource management**: optimize water allocation for various uses, including agriculture, industry, 
  and municipal supply, while considering criteria like sustainability and equity.
- **Urban planning**: decide on urban development projects and land-use planning based on criteria such as social equity, 
  environmental impact, and economic development.

These are just a few examples of how MCDA can be applied across a wide range of fields to support decision-making processes 
that involve multiple, often conflicting, criteria. The specific application of MCDA will depend on the context 
and the goals of the decision-maker. The final goal is to facilitate a process of informative decision-making that is transparent,
reliable and explainable.

Before using ```ProMCDA```, we suggest you first to get familiar with the main steps and concepts of an MCDA method: 
the normalization of the criteria values; setting the polarities and weights of the indicators; and the aggregation of information into a composite indicator (CI). 
In MCDA context an *alternative* is one possible course of action available; an *indicator*, or *criterion*, is a parameter 
that describes the alternatives. The variability of the MCDA scores are caused by:

- the sensitivity of the algorithm to the different pairs of normalization/aggregation functions (--> sensitivity analysis);
- the randomness that can be associated to the weights (--> robustness analysis weights);
- the uncertainty associated with the indicators (--> robustness analysis indicators).

Here we define:
- the **sensitivity analysis** as the one aimed at capturing the output score stability to the different initial modes;
- the **robustness analysis** as the one aimed at capturing the effect of any changes in the inputs (their uncertainties) on the output scores.

The tool can be also used as a simple (i.e. deterministic) MCDA ranking tool with no robustness/sensitivity analysis (see below for instructions).

### Requirements to use ProMCDA

It’s advisable to install any packages within a virtual environment to manage dependencies effectively and avoid conflicts. 
To set up and activate a virtual environment:

On Windows:
```bash
conda create --name <choose-a-name-like-Promcda> python=3.9
activate.bat <choose-a-name-like-Promcda>
pip install -r requirements.txt
```
On Mac and Linux:
```bash
conda create --name <choose-a-name-like-Promcda> python=3.9
source activate <choose-a-name-like-Promcda>
pip install -r requirements.txt
```
The Python version should de 3.9 or higher.

### How to use ProMCDA
This section provides a comprehensive guide on utilizing ProMCDA for Probabilistic Multi-Criteria Decision Analysis. 
ProMCDA is designed to help decision-makers explore the sensitivity and robustness of Composite Indicators (CIs) in a user-friendly manner. 
You can use ```ProMCDA``` as a Python library. With Python and pip set up, you can install ProMCDA using the following command:

```bash
pip install ProMCDA
```

This command fetches and installs the ProMCDA package along with its dependencies from the Python Package Index (PyPI).

After installation, confirm that ProMCDA is correctly installed by opening a Python interpreter or a Jupyter notebook
and attempting to import the package:
```python
import promcda
print(promcda.__version__)
```

Once ```ProMCDA``` is installed, you can start using it in your Python projects or Jupyter Notebooks. 
In `demo_in_notebook` you can find a Jupyter notebook that shows how to use the library in Python with a few examples.

In particular, the notebook contains:
- Two examples of setups for instatiating the ProMCDA object: one with a dataset without uncertainties and one with a dataset with uncertainties.
- A mock dataset is created in each setup representing three alternatives evaluated against two criteria.
- A ProMCDA object is created with the data.
- The data is normalized using the ```normalize``` method.
- The data is aggregated using the ```aggregate``` method.
- The ranks are evaluated using the ```evaluate_ranks``` method.
	
```ProMCDA``` has the following functionalities:
- **Case of non robustness on the indicators**
  - normalization with a specific method or any methods (i.e., sensitivity analysis on the normalization);
  - aggregation with a specific method or any methods (i.e., sensitivity analysis on the aggregation);
  - any combinations of the two above (partial or full sensitivity analysis);
  - case of robustness on the weights, i.e., all weights are randomly sampled from a uniform distribution [0,1];
  - case of robustness on the weights, i.e., one weight at time is randomly sampled from a uniform distribution [0,1];
  - the two cases above can be combined with the sensitivity analysis on the normalization and/or aggregation.
- **Case of robustness on the indicators** (indicators are associated with uncertainties)
  - normalization with a specific method or any methods (i.e., sensitivity analysis on the normalization);
  - aggregation with a specific method or any methods (i.e., sensitivity analysis on the aggregation);
  - any combinations of the two above (partial or full sensitivity analysis);
  - if the robustness analysis is performed, the weights cannot be randomly sampled.

```ProMCDA``` has two *required* **input parameters**: 
- the *input matrix* (with or without uncertainties): a Pandas DataFrame containing the alternatives and their indicators;
- the *polarities* assigned to the indicators: a tuple of "+" - which means the higher the value of the indicator the better for the CI
  evaluation; or of "-" - which means the lower the value of the indicator the better for the CI evaluation. 

The *optional* parameters are the:
- *weights*: a list specifying the weights assigned to each indicator for aggregation. If not provided, equal weighting is assumed for all indicators.
- *robustness_weights*: a boolean (default False); if enabled (True), the analysis will incorporate robustness checks concerning the weights, assessing how variations in weights impact the results.
- *robustness_single_weights*: a boolean (default False); when True, the analysis will perform robustness checks on individual weights, evaluating the sensitivity of the outcome to changes in each weight separately.
- the *robustness_indicators*: a boolean (default False); if enabled (True), the analysis will include robustness assessments related to the indicators, examining how their uncertainties affect the overall evaluation.
- the *marginal_distributions*: a tuple of strings specifying the probability distribution functions (PDFs) to be used for modeling the uncertainty of each indicator. The available PDFs are defined in the PDFType Enum class. 
- *num_runs*: an integer (default=10000) that determines the number of simulation runs to be executed during the probabilistic analysis. A higher number of runs can increase the accuracy of the results.
- *num_cores*: an integer (default=1) that specifies the number of CPU cores to utilize for parallel processing. 
- *random_seed*: an integer (default=43) that sets the seed for random number generation to ensure reproducibility of results. 

The PDFType Enum includes the following options:
- PDFType.EXACT: represents an exact value with no uncertainty.
- PDFType.UNIFORM: represents a uniform distribution.
- PDFType.NORMAL: represents a normal (Gaussian) distribution. ￼
- PDFType.LOGNORMAL: represents a log-normal distribution.
- PDFType.POISSON: represents a Poisson distribution.

The method ```normalize``` has one optional input parameters that is the normalization function; if not provided, the function will apply all the available normalization functions.
The available normalization functions are defined in the NormalizationFunctions Enum class:
- minmax normalization.
- target normalization.
- standardized normalization.
- rank normalization.

The method ```aggregate``` has one optional input parameters that is the aggregation function; if not provided, the function will apply all the available aggregation functions.
The available aggregation functions are defined in the AggregationFunctions Enum class:
- weighted-sum aggregation.
- geometric aggregation.
- harmonic aggregation.
- minimum aggregation.

The method ```evaluate_ranks``` is a static method, which compute percentile ranks from scores.

The method ```run```executes the full ProMCDA process, either with or without uncertainties on the weights or indicators.
It combines the normalization, aggregation, and evaluation of ranks steps into a single function call. The two optional input parameters are
the normalization and aggregation functions; if not provided, the function will apply all the available normalization and aggregation functions.

### The output of ProMCDA
The results of ```normalize```, ```aggregate```, and ```evaluate_ranks``` are exposed as Pandas DataFrames only for a simple run (i.e., without robustness analysis):

- simple normalization: Pandas DataFrame with normalized indicators, each column representing an indicator combined with a normalization method;
- simple aggregation: Pandas DataFrame with aggregated values, each column representing a normalization combined with an aggregation method;
- simple evaluation of ranks: Pandas DataFrame with ranks of the alternatives based on the aggregated values, each column representing the rank associated to a normalization combined with an aggregation method.

In case a robustness analysis is performed, the results can be accessed via one of ```get_normalized_values_with_robustness```, 
```get_aggregated_values_with_robustness_weights```, ```get_aggregated_values_with_robustness_one_weight```, 
and ```get_normalized_values_with_robustness_indicators``` methods, which return respectively:

- normalization with robustness on indicators: Pandas DataFrame with normalized indicators x num_runs;
- aggregation with robustness on weights: Tuple (means, normalized_means, stds); each objet is a Pandas DataFrame;
- aggregation with robustness one weight at time: dictionary with keys referring to means, normalized_means, and stds of the aggregated scores; each dictionary contains another dictionary with keys referring to the indicator whose weight has been perturbed;
- aggregation with robustness on indicators: Tuple (means, normalized_means, stds); each objet is a Pandas DataFrame.

The output of ```run``` adapts to the type of analysis performed.

Refer to the last note in [General information and references](#general-information-and-references)for more details about the use of means 
and normalized means in association with standard deviations.
    
For more details about the normalization and aggregation functions, please refer to the paper cited in the badge section.
By configuring these parameters appropriately, you can tailor the ProMCDA analysis to your specific needs, enabling comprehensive and customized multi-criteria decision analyses.

### Input matrix
The input matrix is a Pandas DataFrame containing the alternatives and their indicators. 
Each row of the DataFrame represents an alternative, while each column represents an indicator. 
The first column should contain the names of the alternatives, and the subsequent columns should contain the values of the indicators.

The input matrix can be provided in two formats: with uncertainties or without uncertainties.
- The **input matrix without uncertainties** is used when the indicators are deterministic values.
- The **input matrix with uncertainties** is used when the indicators are associated with uncertainties, represented by probability distributions.
- The input matrix can also contain negative values, which will be rescaled to the range [0,1] if necessary.

If the values of one or more indicators are all the same, the indicators are dropped from the input matrix because they contain no information.
Consequently, the relative weights and polarities are dropped.

Examples of input matrix:

- *input matrix without uncertainties* for the indicators (see an example here: `tests/resources/input_matrix_without_uncert.csv`);
- *input matrix with uncertainties* for the indicators (see an example here: `tests/resources/input_matrix_with_uncert.csv`).

The example input matrix with uncertainties is designed for an example where the PDFs describing the indicators are respectively: 
uniform; exact; normal (or lognormal); exact; and normal (or lognormal). Please modify it for your specific needs.

If the input matrix without uncertainties has any values of any indicators that are negative, those values are rescaled
between [0,1]. This is needed because some normalization functions (as for example *target*) cannot handle negative values 
properly.

The input matrix with uncertainties has the following characteristics:

- if an indicator is described by an *exact* probability density function (PDF), one needs only a column with its values;
- if an indicator is described by a *uniform* PDF, one needs two columns with the lowest and highest values (in this order);
- if an indicator is described by a *normal* PDF, one needs two columns with the mean and standard deviation values (in this order);
- if an indicator is described by a *lognormal* PDF, one needs two columns with the log(mean) and log(standard deviation) values (in this order);
- if an indicator is described by a *Poisson* PDF, one needs only one colum with the rate.

If any mean value of any indicator is equal or smaller of its standard deviation - in case of a normal or lognormal PDF - 
```ProMCDA``` breaks to ask you if you need to investigate your data further before applying MCDA. This means that your data shows
a high variability regarding some indicators. If you want to continue anyway, negative sampled data will be rescaled 
into [0,1], as in the case without uncertainty.

### Running the unit tests
```bash
python3 -m pytest -s tests/unit_tests -vv
```
### Toy example
```ProMCDA``` contains a toy example, a simple case to test run the package. 
The toy example helps you identify the best car models (i.e., the alternatives) you can buy based on a few indicators 
(e.g., Model, Fuel Efficiency, Safety Rating, Price, Cargo Space, Acceleration, Warranty).
The input matrix for the toy example is in `input_files/toy_example/car_data.csv`. 
You can import the input matrix in a Python environment as a Pandas dataframe and perform normalization and aggregation with the ProMCDA library. 
The directory `toy_example/toy_example_utilities` contains also a Jupyter notebook to allow you to modify the input matrix easily. 
The chosen example is very simple and not suitable for a robustness analysis test. Running the robustness analysis requires an input matrix 
containing information on the uncertainties of the criteria as described above, and it is not under the scope of the toy example. 

### A high level summary
If no robustness analysis is selected, then:
- the indicator values are normalized by mean of all the possible normalization methods (or by the selected one);
- the normalized indicators are aggregated by mean of all the possible aggregation methods (or by the selected one), 
  by considering their assigned weights.

If the weights are randomly sampled (robustness analysis of the weights), then:
- all weights or one weight at time are randomly sampled from a uniform distribution [0,1];
- the weights are normalized so that their sum is always equal to 1;
- if all weights are sampled together, MCDA calculations receive N-inputs (N being the number of `num_runs`; 
  if the weights are sampled one at time, MCDA will receive (*n-inputs x num_weights*) inputs;
- iterations 1,2,3 of the first condition follow.

If the robustness analysis is selected on the indicators, then:
- for each indicator, the parameters (e.g., mean and standard deviation) describing the marginal distribution under interest are extracted from the input matrix;
- for each N, and for each indicator, a value is sampled from the relative assigned marginal distribution: therefore, one of N input matrix is created;
- normalizations and aggregations are performed as in points 1,2 of the first case: a list of all the results is created in the output directory;
- mean and standard deviation of all the results are estimated across (monte_carlo_runs x pairs of combinations);  
- in this case, no randomness on the weights is allowed.

### General information and references
The aggregation functions are implemented by following [*Langhans et al.*, 2014](https://www.sciencedirect.com/science/article/abs/pii/S1470160X14002167)

The normalization functions *minmax*, *target* and *standardized* can produce negative or zero values, therefore a shift to positive values
is implemented so that they can be used also together with the aggregation functions *geometric* and *harmonic* (which require positive values).

The code implements 4 normalization and 4 aggregation functions. However, not all combinations are 
meaningful or mathematically acceptable. For more details refer to Table 6 in 
[*Gasser et al.*, 2020](https://www.sciencedirect.com/science/article/pii/S1470160X19307241)

The standard deviation of rescaled (i.e. normalized) scores with any form of randomness are not saved nor plotted because they cannot bring a statistically meaningful information.
In fact, when one calculates the standard deviation after rescaling between (0,1), the denominator used in the standard deviation formula becomes smaller. 
This results in a higher relative standard deviation compared to the mean.
However, the higher relative standard deviation is not indicating a greater spread in the data but rather a consequence of the rescaling operation and 
the chosen denominator in the standard deviation calculation.

### Releases
The ProMCDA library is continuously being improved and updated.
Refer to the [CHANGELOG](CHANGELOG.md) for a detailed list of changes and updates in each version.

### Get in touch
We hope you enjoy exploring and utilizing ProMCDA. If you have any questions or need assistance, please don't hesitate to contact us.


