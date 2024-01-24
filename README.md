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

### MCDA quick overview and applications
A MCDA approach is a systematic framework for making decisions in situations where multiple criteria or objectives need to be 
considered. It can be applied in various domains and contexts. Here are some possible usages of an MCDA approach:

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
and the goals of the decision-maker.

Before using ```ProMCDA```, we suggest you first to get familiar with the main steps and concepts of an MCDA method: 
the normalization of the criteria values; their polarities and weights; and the aggregation of information into a composite indicator. 
In MCDA context an *alternative* is one possible course of action available; an *indicator*, or *criterion*, is a parameter 
that describes the alternatives. The variability of the MCDA scores are caused by:

- the sensitivity of the algorithm to the different pairs of normalization/aggregation functions (--> sensitivity analysis);
- the randomness that can be associated to the weights (--> robustness analysis);
- the uncertainty associated with the indicators (--> robustness analysis).

Here we define:
- the **sensitivity analysis** as the one aimed at capturing the output score stability to the different initial modes;
- the **robustness analysis** as the one aimed at capturing the effect of any changes in the inputs (their uncertainties) on the output scores.

The tool can be also used as a simple (i.e. deterministic) MCDA ranking tool with no robustness/sensitivity analysis (see below for instructions).

### Input information needed in the configuration file
A configuration file is needed to run```ProMCDA```.
The configuration file collects all the input information to run ```ProMCDA``` in your specific study case.
You find a configuration.json file as an example in this directory: please modify it for your needs. In the following, 
the entries of the configuration file are described.

***Path to the input matrix***, a table where rows represent the alternatives and columns represent the indicators.
Be sure that the column containing the names of the alternatives is set as the index column, e.g. by:
```bash
input_matrix = input_matrix.set_index('Alternatives').
```
Be sure that there are no duplicates among the rows. If the values of one or more indicators are all the same, 
the indicators are dropped from the input matrix because they contain no information.
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
```ProMCDA``` breaks to ask you if you need to investigate your data further before applying MCDA. In fact, your data shows
a high variability regarding some indicators. If you want to continue anyway, negative sampled data will be rescaled 
into [0,1], as in the case without uncertainty.


***List of polarities*** for each indicator, "+" (the higher the value of the indicator the better for the evaluation) 
or "-" (the lower the value of the indicator the better).

The configuration file can trigger a run with or without ***sensitivity analysis***; this is set in the `sensitivity_on` parameter (*yes* or *no*); 
if *no* is selected, then the pair normalization/aggregation should be given in `normalization` and `aggregation`. If *yes*
is selected, then the normalization/aggregation pair is disregarded.

Similarly, a run with or without uncertainty on the indicators or on the weights (i.e. with ***robustness analysis***) 
can be triggered by setting the `robustness_on` parameter to *yes* or *no*. If `robustness_on` is set to *yes*, then 
the uncertainties might be on the indicators (`on_indicators`) or on the weights (`on_single_weights` or `on_all_weights`). 
In the first case (`on_single_weights=yes`) one weight at time is randomly sampled from a uniform distribution; 
in the second case (`on_all_weights=yes`) all weights are simultaneously sampled from a normal distribution. 
If there is no uncertainty associated to the weights, then the user should provide a ***list of weights*** for the indicators. 
The sum of the weights should always be equal to 1 or the values will be normalised internally. 
Depending on the different options, information not needed are disregard. Sensitivity and robustness analysis can be run
together. If robustness analysis is selected, it can run either on the weights or on the indicators, but not on both simultaneously.

If robustness analysis is selected, a last block of information is needed:
the ***Number of Monte Carlo runs***, "N" (default is 0, then no robustness is considered; N should be a sufficient big number, 
e.g., larger or equal than 1000). The ***number of cores*** used for the parallelization; and a 
***List of marginal distributions*** for each indicator; the available distributions are: 
  - exact, **"exact"**,
  - uniform distribution, **"uniform"**
  - normal distribution, **"norm"**
  - lognormal distribution, **"lnorm"**
  - Poisson distribution, **"poisson"**

### Output

The user gives the ***path to output file*** (e.g. `path/output_file.csv`). In the output file the scores (normalised or rough) 
and the ranks relative to the alternatives can be found in the form of CSV tables. If the weights are iteratively sampled, 
multiple tables are saved in a PICKLE file as an object ```dictionary```. Plots of the scores are saved in PNG images. The configuration.json file
is saved in the output directory too; the information stored in the configuration settings are useful in case 
multiple tests are stored and need to be reviewed.

All file names of the output objects are followed by a time stamp that group them with the specific test setting.

To retrieve a PICKLE file in Python one can:

```pythongroup them 
import pickle

# Replace 'your_file.pickle' with the path to your PICKLE file
file_path = 'your_file.pickle'

# Load the PICKLE file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Now, 'data' contains the object or data stored in the PICKLE file
```
Note that in case of a robustness analysis, the error bars in the plots are only shown for non-normalized scores. This is
because when one calculates the standard deviation after rescaling, the denominator used in the standard deviation formula 
becomes smaller. This results in a higher relative standard deviation compared to the mean that is solely an artificial 
effect.
  

### Requirements
On Windows:
```bash
conda create --name <choose-a-name-like-Promcda> python=3.6
activate.bat <choose-a-name-like-Promcda>
pip install -r requirements.txt
```
On Mac and Linux:
```bash
conda create --name <choose-a-name-like-Promcda> python=3.6
source activate <choose-a-name-like-Promcda>
pip install -r requirements.txt
```

### Running the code (from root dir)
On Windows:
```bash
activate.bat <your-env>
python3 -m mcda.mcda_run -c configuration.json
```
On Mac and Linux:
```bash
source activate <your-env>
python3 -m mcda.mcda_run -c configuration.json
```
where an example of configuration file can be found in `./configuration.json`.

### Running the tests
```bash
python3 -m pytest -s tests/unit_tests -vv
```

### Code overview: a high-level summary
If no robustness analysis is selected, then:
- the indicator values are normalized by mean of all the possible normalization methods (or by the selected one);
- the normalized indicators are aggregated by mean of all the possible aggregation methods (or by the selected one), 
  by considering their assigned weights;
- the resulting scores of all the combinations normalization/aggregation (or the selected ones only) are provided in form 
  of a csv table and plots in png format in the output directory.

If the weights are randomly sampled (robustness analysis of the weights), then:
- all weights or one weight at time are randomly sampled from a uniform distribution [0,1];
- the weights are normalized so that their sum is always equal to 1;
- if all weights are sampled together, MCDA calculations receive N-inputs (N being the number of `monte_carlo_runs`; 
  if the weights are sampled one at time, MCDA will receive (*n-inputs x num_weights*) inputs;
- iterations 1,2,3 of the first condition follow;
- the results of all the combinations normalization/aggregation (or the one selected) are provided in the form of mean and standard deviation over all the runs 
  (if the weights are iteratively sampled, this applies for *num_indicators-times*).

If the robustness analysis regards the indicators, then:
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

The standard deviation of rescaled scores with any form of randomness are not saved nor plotted because they cannot bring a statistically meaningful information.
In fact, when one calculates the standard deviation after rescaling between (0,1), the denominator used in the standard deviation formula becomes smaller. 
This results in a higher relative standard deviation compared to the mean.
However, the higher relative standard deviation is not indicating a greater spread in the data but rather a consequence of the rescaling operation and 
the chosen denominator in the standard deviation calculation.


