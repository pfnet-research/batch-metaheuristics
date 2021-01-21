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
