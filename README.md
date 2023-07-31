# MCDTool
A tool to estimate ranks of alternatives and their uncertainties based on the Multi Criteria Decision Analysis approach.
The variability of the MCDA scores are caused by:
- the uncertainty associated with the indicators
- the sensitivity of the algorithm to the different pairs of norm/agg functions
- the randomness that can be associated to the weights (also as single source of randomness)

The tool can be used also as a simple MCDA ranking tool with no variability (see also below).


### Input information needed in the configuration file
The following input information are contained in the `configuration.json` file:
- input matrix, a table where rows represent the alternatives and the columns represent the indicators. Be sure that there is not an index column but that the column with the alternative names is set as index by:
```bash
input_matrix = input_matrix.set_index('Alternatives').
```
Be also sure that there are no duplicates among the rows. If the values of one or more indicators are all the same, the indicators are dropped from the input matrix because they do not carry any information.
  - input matrix without uncertainties for the indicators (see an example here: `tests/resources/input_matrix_without_uncert.csv`)
  - input matrix with uncertainties for the indicators (see an example here: `tests/resources/input_matrix_with_uncert.csv`)
The input matrix with uncertainties has for each indicator a column with the mean values and a column with the standard deviation; 
if the marginal distribution relative to the indicator is 'exact', then the standard deviation column contains only 0.
- list with names of marginal distributions for each indicator (see an example here: `tests/resources/tobefilled`); the available distributions are 
  - exact, "exact",
  - uniform distribution, "uniform"
  - normal distribution, "norm
  - lognormal distribution, "lnorm"
  - Poisson distribution, "poisson"
- list of polarities for each indicator, "+" or "-"
- number of Monte Carlo runs, "N" (default is 0, no variability is considered; N should be a sufficient big number, e.g. larger or equal than 1000)
- the number of cores used for the parallelization, "numCores"
- list of weights for each indicator 
    - should the weights be randomly sampled by mean of a Monte Carlo sampling? (*yes* or *no*)
    - if *no*, a list of weights should be given as input
    - if *yes*
        - the number of samples should be given as input 
        - should the process be *iterative*?
            - *yes*, i.e. randomness is associated to one weight at time
            - *no*, i.e. randomness is associated to all the weights at the same time
    - the sum of the weights should always be equal to 1 or the values will be normalised 
    - depending on the different options, the other information are disregard
- output file (e.g. `path/output_file.csv`).

In case the variability of results is of no interest, then:
- the input matrix should be the one without uncertainties associated to the indicators
- the marginal distribution associated to the indicators should all be of the kind "exact"
- cores=1
- N=1

The configuration file can trigger a run with or without uncertainty on the indicators. This is implicitly set in the 
`marginal_distribution_for_each_indicator` parameter: if the marginal distributions are all *exact*, then the run is without uncertainty;
if instead the marginal distributions are also other than *exact*, it means that the indicator values can be randomly sampled from those PDFs.
- `configuration_without_uncertainty.jon`
  - no uncertainty relative to the indicators is considered, however one can
    - see the variability caused by using different pairs of normalization/aggregation functions
    - turn on the variability due to added randomness to the weights (optional)
- `configuration_w_uncertainty.jon`
    - uncertainty relative to the indicators is considered, together with
    - the variability caused by using different pairs of normalization/aggregation functions

### Requirements
```bash
conda create --name <choose-a-name-like-mcda> python=3.6
source activate <choose-a-name-like-mcda>
pip install -r requirements.txt
```

### Running the code (from root dir)
```bash
source activate <your-env>
python3 -m mcda.mcda_run -c configuration_w_uncertainty.json
```
where an example of configuration file can be found in `mcda/configuration_w_uncertainty.json` or `mcda/configuration_without_uncertainty.json`.

### Running the tests
```bash
python3 -m pytest -s tests/unit_tests/test_mcda_run.py -vv
```

### What does the code do: overview
If N=0 and the input matrix has no uncertainties associated to the indicators, then:
- the indicator values are normalized by mean of all the possible normalization methods 
- the normalized indicators are aggregated by mean of all the possible aggregation methods, by considering their assigned weights
- the results of all the combinations normalization/aggregation are provided

Else if N=0 and the input matrix has no uncertainties associated to the indicators, but the weights are randomly sampled, then:
- all weights (*iterative="no"*) or one weight at time (*iterative="yes"*) are randomly sampled from a uniform distribution [0,1]
- the weights are normalized
- if all weights are sampled together, MCDA receives n-inputs; if the weights are sampled one at time, MCD will receive n-inputs x num_weights times
- iterations 1,2 of the first condition follow
- the results of all the combinations normalization/aggregation are provided in the form of mean and std over all the runs (if the weights are iteratively sampled, this applies for num_indicators-times)

If else, N>1 and the input matrix has uncertainties associate to some or all the indicators, then:
- for each indicator, the mean and standard deviation (std) are extracted from the input matrix
- for each N, and for each indicator, a value is sampled from the relative assigned marginal distribution: one of N input matrix is created
- normalizations and aggregations are performed as in points 1,2,3 of the case N=1: a list of all the results is created
- mean and std of all the results are estimated across (N x pairs of combinations) 
- in this case, no randomness on the weights is allowed


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


