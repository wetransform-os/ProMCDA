# MCDTool
A tool to estimate ranks of alternatives and their uncertainties based on the Multi Criteria Decision Analysis approach.
The variability of the MCDA scores are caused by:
- the uncertainty associated with the indicators
- the sensitivity of the algorithm to the different pairs of norm/agg functions
- the randomness that can be associated to the weights (TODO: implement)

The tool can be used also as a simple MCDA ranking tool with no variability (see also below).

### Input information needed
The following input information are contained in the `configuration.json` file:
- input matrix, a table where rows represent the alternatives and the columns represent the indicators
  - input matrix without uncertainties for the indicators (see an example here: `tests/resources/input_matrix_without_uncert.csv`)
  - input matrix with uncertainties for the indicators (see an example here: `tests/resources/input_matrix_with_uncert.csv`)
- list with names of marginal distributions for each indicator (see an example here: `tests/resources/tobefilled`); the available distributions are 
  - exact, "exact",
  - uniform distribution, "uniform"
  - normal distribution, "norm
  - lognormal distribution, "lnorm"
  - Poisson distribution, "pois"
  - negative Binomial distribution, "ng"
  - beta distribution, "beta"
- list of polarities for each indicator, "+" or "-"
- number of Monte Carlo runs, "N" (default is 0, no variability is considered; N should be a sufficient big number, e.g. larger or equal than 1000)
- the number of cores used for the parallelization, "numCores"
- list of weights for each indicator - the sum should always be equal to 1 or the values will be corrected
- output file (e.g. `path/output_file.csv`).

In case the variability of results is of no interest, then:
- the input matrix should be the one without uncertainties associated to the indicators
- the marginal distribution associated to the indicators should all be of the kind "exact"
- cores=1
- N=1

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
If N=1 and the input matrix has no uncertainties associated to the indicators, then:
- the indicator values are normalized by mean of all the possible normalization methods 
- the normalized indicators are aggregated by mean of all the possible aggregation methods, by considering their assigned weights
- results of all the combinations normalization/aggregation are provided
If else, N>1 and the input matrix has uncertaities associate to some or all the indicators, then:
- for each indicator, the meand and standard deviation (std) are extracted from the input matrix
- for each N, and for each indicator, a value is sampled from the relative assigned marginal distribution: one of N input matrix is created
- normalizations and aggregations are performed as in points 1,2,3 of the case N=1: a list of all the results is created
- mean and std of all the results are estimated across (N x pairs of combinations) 


### General information and references
The normalization functions are implemented by following [*Langhans et al.*, 2014](https://www.sciencedirect.com/science/article/abs/pii/S1470160X14002167)

The normalization functions *minmax*, t*arget* and *standardized* can produce negative or zero values, therefore a shift to positive values
is implemented so that they can be used also together with the aggregation functions *geometric* and *harmonic* (which require positive values). 

The code implements 4 normalization and 4 aggregation functions. However, not all combinations are 
meaningful or mathematically acceptable. For more details refer to Table 6 in 
[*Gasser et al.*, 2020](https://www.sciencedirect.com/science/article/pii/S1470160X19307241)


