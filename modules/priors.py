import pyro.distributions as dist

class NormalPrior():
    def __init__(self, weight_loc=0., weight_scale=1., bias_loc=0., bias_scale=1., sigma_concentration=1., sigma_rate=1.):
        self.prior_dist = dist.Normal
        self.weight_loc = weight_loc
        self.weight_scale = weight_scale
        self.bias_loc = bias_loc
        self.bias_scale = bias_scale
        self.sigma_dist = dist.Gamma
        self.sigma_concentration = sigma_concentration
        self.sigma_rate = sigma_rate

class NormalPriorUniform():
    def __init__(self, weight_loc=0., weight_scale=1., bias_loc=0., bias_scale=1., sigma_concentration=1., sigma_rate=1.):
        self.prior_dist = dist.Normal
        self.weight_loc = weight_loc
        self.weight_scale = weight_scale
        self.bias_loc = bias_loc
        self.bias_scale = bias_scale
        self.sigma_dist = dist.Uniform
        self.sigma_concentration = sigma_concentration
        self.sigma_rate = sigma_rate

class NormalPriorGamma():
    def __init__(self, weight_loc=0., weight_scale=1., bias_loc=0., bias_scale=1., sigma_concentration=1., sigma_rate=1.):
        self.prior_dist = dist.Normal
        self.weight_loc = weight_loc
        self.weight_scale = weight_scale
        self.bias_loc = bias_loc
        self.bias_scale = bias_scale
        self.sigma_dist = dist.Gamma
        self.sigma_concentration = sigma_concentration
        self.sigma_rate = sigma_rate

prior_types = {
    "normal": NormalPrior,
    "normal_uniform": NormalPriorUniform,
    "normal_gamma": NormalPriorGamma
}