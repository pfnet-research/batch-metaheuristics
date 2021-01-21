# batchopt

![license:MIT](https://github.com/pfnet/i20_meit/blob/release/license.svg)

This is a repository of black-box-optimization using meta-heuristics.

## Usage

Quick Start:

```sh
git clone git@github.com:pfnet-research/batch_metaheuristics.git
cd batch_metaheuristics
pip install ./batchopt
python example.py
```

Output (Black box optimization involves random elements, so the results will be different each time):

```
> Epoch: 1, Best fit: 884.7695325180321
> Epoch: 2, Best fit: -63.122234265670585
> Epoch: 3, Best fit: -85.04995732445039
> Epoch: 4, Best fit: -85.04995732445039
> Epoch: 5, Best fit: -111.70023680346456
> Epoch: 6, Best fit: -127.10859058276085
> Epoch: 7, Best fit: -131.96113684754317
> Epoch: 8, Best fit: -135.051712626202
> Epoch: 9, Best fit: -137.46424504437974
> Epoch: 10, Best fit: -138.53373567844406
```

Entire code of `example.py` is like this:

```python
from batchopt.benchfunctions import batch_styblinski_tang
from batchopt.optimizers.whale_optimization_algorithm import WhaleOptimizationAlgorithm

opt = WhaleOptimizationAlgorithm(
    domain_range=[(-20, 20), (-20, 20), (-20, 20), (-20, 20)],
    pop_size=50,
    epoch=10,
    log=True,
)
objective_function = batch_styblinski_tang
# must be function 2-dimension-array to 1 dimension array

opt.optimize(objective_function)
```

This example uses `WhaleOptimizationAlgorithm` as optimizer. You can use other 20+ algorithms.
Optimization algorithms are initialized with the following arguments:

- `domain_range`: List of domain of the objective function.
- `pop_size`: Number of candidate points to generate at one time.
- `epoch`: Number of times to repeat the optimization process.
- `log`: Whether to display logs.

Once you have created an instance of `Optimizer`, optimize with the `optimize` method.
The argument of the optimize method is the objective function.
**The Objective function needs to be a mapping from a 2D-array to a 1D-array.**
