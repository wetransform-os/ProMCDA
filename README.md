# MCDTool
A tool to estimate ranks of alternatives and their uncertainties based on the Multi Criteria Decision Analysis approach.

### Input information needed
The following input information are contained in the `configuration.json` file:
- input matrix (e.g. `path/input_matrix.csv`), a table where rows represent the alternatives and the columns represent the indicators
  - input matrix without uncertainties for the indicators (see an example here: `tests/resources/tobefilled`)
  - input matrix with uncertainties for the indicators (see an example here: `tests/resources/tobefilled`)
- list with names of marginal distributions for each indicator (see an example here: `tests/resources/tobefilled`); the available distributions are 
  - exact, "exact",
  - uniform distribution, "uniform"
  - normal distribution, "norm
  - lognormal distribution, "lnorm"
  - Poisson distribution, "pois"
  - negative Binomial distribution, "ng"
  - beta distribution, "beta"
- list of polarities for each indicator, "+" or "-"
- number of Monte Carlo runs, "N" (default is 1)
- the number of cores used for the parallelization, "numCores"
- list of weights for each indicator - if no list is given, then the weights are sampled between 0 and 1 with at each Monte Carlo run
- output file (e.g. `path/output_file.csv`).

### Requirements
```bash
conda create --name <choose-a-name-like-mcda> python=3.6
source activate <choose-a-name-like-mcda>
pip install -r requirements.txt
```

### Running the code
```bash
source activate <your-env>
python3 -m tobefilled -c configuration.json
```
where an example of `configuration.json` can be found in `test/resources/tobefilled.json`.

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
