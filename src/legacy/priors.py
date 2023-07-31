import pyro.distributions as dist

class NormalPrior():
    def __init__(self, weight_loc=0., weight_scale=1., bias_loc=0., bias_scale=10.):
        self.prior_dist = dist.Normal
        self.weight_loc = weight_loc
        self.weight_scale = weight_scale
        self.bias_loc = bias_loc
        self.bias_scale = bias_scale

prior_types = {
    "normal": NormalPrior,
}