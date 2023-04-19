import torch
import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.contrib.easyguide.easyguide import EasyGuide
from pyro.nn import PyroSample, PyroModule, PyroParam
from pyro.infer.autoguide import AutoDiagonalNormal, AutoDelta, AutoGaussian, AutoNormal
from pyro.infer.autoguide.initialization import init_to_feasible, init_to_median

def getAutoDiagonalNormal(model, device="cpu"):
    def init_func(*args):
        return init_to_median(*args).to(device)
    
    return AutoDiagonalNormal(model, init_loc_fn=init_func)

def getAutoDelta(model, device="cpu"):
    def init_func(*args):
        return init_to_median(*args).to(device)
    
    return AutoDelta(model, init_loc_fn=init_func)

def getAutoGaussian(model, device="cpu"):
    def init_func(*args):
        return init_to_feasible(*args).to(device)
    
    return AutoGaussian(model, init_loc_fn=init_func)

def getAutoNormal(model, device="cpu"):
    def init_func(*args):
        return init_to_median(*args).to(device)
    
    return AutoNormal(model, init_loc_fn=init_func)

def getAutoNormalGamma(model, device="cpu"):
    def init_func(*args):
        return init_to_feasible(*args).to(device)
    
    return AutoNormalGamma(model, init_loc_fn=init_func)

class OwnGuide(PyroModule):
    def __init__(self, model, device="cpu"):
        super().__init__()
        self.model = model
        self.sample_input = torch.randn(1, model.in_features, device=device)

        with poutine.trace() as tr:
            model(self.sample_input)

        self.sites = [(site["name"], site["value"]) for site in tr.trace.nodes.values()]
        print(tr.trace.nodes.keys())

        #Register sigma loc
        self._register_parameter("sigma_loc", PyroParam(torch.tensor(1.0, device=device), constraint=constraints.interval(0., 10.)))

        #Register weight and bias loc and scale parameters
        for name, val in self.sites:
            if "weight" in name:
                self._register_parameter(f"{name}_loc", PyroParam(torch.full_like(val, 0.0, device=device)))
                self._register_parameter(f"{name}_scale", PyroParam(torch.full_like(val, 1.0, device=device), constraint=constraints.positive))
            elif "bias" in name:
                self._register_parameter(f"{name}_loc", PyroParam(torch.full_like(val, 0.0, device=device)))
                self._register_parameter(f"{name}_scale", PyroParam(torch.full_like(val, 10.0, device=device), constraint=constraints.positive))

    def _register_parameter(self, name, param):
        att_name = str.replace(name, ".", "_")
        setattr(self, att_name, param)

    def _get_param(self, name):
        att_name = str.replace(name, ".", "_")
        return getattr(self, att_name)

    def forward(self, x, y=None):
        pyro.sample("sigma", dist.Delta(self.sigma_loc))
        for name, val in self.sites:
            if "weight" in name:
                pyro.sample(name, dist.Normal(self._get_param(f"{name}_loc"), self._get_param(f"{name}_scale")).to_event(2))
            elif "bias" in name:
                pyro.sample(name, dist.Normal(self._get_param(f"{name}_loc"), self._get_param(f"{name}_scale")).to_event(1))

class OwnGuideTest(PyroModule):
    def __init__(self, model, device="cpu"):
        super().__init__()
        self.model = model

        #Register sigma loc
        self._register_parameter("sigma_loc", PyroParam(torch.tensor(1.0, device=device), constraint=constraints.interval(0., 10.)))

        self._register_parameter("weight1_loc", PyroParam(torch.full((32, 1), 0.0, device=device)))
        self._register_parameter("weight1_scale", PyroParam(torch.full((32, 1), 1.0, device=device), constraint=constraints.positive))

        self._register_parameter("bias1_loc", PyroParam(torch.full((32,), 0.0, device=device)))
        self._register_parameter("bias1_scale", PyroParam(torch.full((32,), 10.0, device=device), constraint=constraints.positive))

        self._register_parameter("weight2_loc", PyroParam(torch.full((1, 32), 0.0, device=device)))
        self._register_parameter("weight2_scale", PyroParam(torch.full((1, 32), 1.0, device=device), constraint=constraints.positive))

        self._register_parameter("bias2_loc", PyroParam(torch.full((1,), 0.0, device=device)))
        self._register_parameter("bias2_scale", PyroParam(torch.full((1,), 10.0, device=device), constraint=constraints.positive))

    def _register_parameter(self, name, param):
        att_name = str.replace(name, ".", "_")
        setattr(self, att_name, param)

    def _get_param(self, name):
        att_name = str.replace(name, ".", "_")
        return getattr(self, att_name)

    def forward(self, x, y=None):
        pyro.sample("sigma", dist.Delta(self.sigma_loc))
        
        pyro.sample("model.fc.0.0.linear.weight", dist.Normal(self._get_param("weight1_loc"), self._get_param("weight1_scale")).to_event(2))
        pyro.sample("model.fc.0.0.linear.bias", dist.Normal(self._get_param("bias1_loc"), self._get_param("bias1_scale")).to_event(1))

        pyro.sample("model.fc.1.linear.weight", dist.Normal(self._get_param("weight2_loc"), self._get_param("weight2_scale")).to_event(2))
        pyro.sample("model.fc.1.linear.bias", dist.Normal(self._get_param("bias2_loc"), self._get_param("bias2_scale")).to_event(1))


class OwnEasyGuide(PyroModule):
    def __init__(self, model, device="cpu"):
        super().__init__()
        self.model = model

        #Register sigma loc
        self.sigma_loc = PyroParam(torch.tensor(1.0, device=device), constraint=constraints.interval(0., 10.))
        self.sigma_rate = PyroParam(torch.tensor(1.0, device=device), constraint=constraints.positive)

        self.weight1_loc = PyroParam(torch.full((32, 1), 0.0, device=device))
        self.weight1_scale = PyroParam(torch.full((32, 1), 1.0, device=device), constraint=constraints.positive)

        self.bias1_loc = PyroParam(torch.full((32,), 0.0, device=device))
        self.bias1_scale = PyroParam(torch.full((32,), 10.0, device=device), constraint=constraints.positive)

        self.weight2_loc = PyroParam(torch.full((32, 32), 0.0, device=device))
        self.weight2_scale = PyroParam(torch.full((32, 32), 1.0, device=device), constraint=constraints.positive)

        self.bias2_loc = PyroParam(torch.full((32,), 0.0, device=device))
        self.bias2_scale = PyroParam(torch.full((32,), 10.0, device=device), constraint=constraints.positive)

        self.weight3_loc = PyroParam(torch.full((1, 32), 0.0, device=device))
        self.weight3_scale = PyroParam(torch.full((1, 32), 1.0, device=device), constraint=constraints.positive)

        self.bias3_loc = PyroParam(torch.full((1,), 0.0, device=device))
        self.bias3_scale = PyroParam(torch.full((1,), 10.0, device=device), constraint=constraints.positive)

    def forward(self, x, y=None):
        pyro.sample("OwnEasyGuideTest.ownguide.model.fc.0.0.linear.weight", dist.Normal(self.weight1_loc, self.weight1_scale).to_event(2))
        pyro.sample("OwnEasyGuideTest.ownguide.model.fc.0.0.linear.bias", dist.Normal(self.bias1_loc, self.bias1_scale).to_event(1))

        pyro.sample("OwnEasyGuideTest.ownguide.model.fc.1.0.linear.weight", dist.Normal(self.weight2_loc, self.weight2_scale).to_event(2))
        pyro.sample("OwnEasyGuideTest.ownguide.model.fc.1.0.linear.bias", dist.Normal(self.bias2_loc, self.bias2_scale).to_event(1))

        pyro.sample("OwnEasyGuideTest.ownguide.model.fc.2.linear.weight", dist.Normal(self.weight3_loc, self.weight3_scale).to_event(2))
        pyro.sample("OwnEasyGuideTest.ownguide.model.fc.2.linear.bias", dist.Normal(self.bias3_loc, self.bias3_scale).to_event(1))

        pyro.sample("sigma", dist.Gamma(self.sigma_loc, self.sigma_rate))


class OwnEasyGuideTest(EasyGuide):
    def __init__(self, *args, device="cpu"):
        super().__init__(*args)
        self.device = device
        self.ownguide = OwnEasyGuide(self.model, device=device)
    def guide(self, x, y=None):
        self.ownguide(x, y)



guide_types = {
    "autodiagonalnormal": getAutoDiagonalNormal,
    "autodelta": getAutoDelta,
    "autogaussian": getAutoGaussian,
    "autoNormal": getAutoNormal,
    "ownguide": OwnGuide,
    "ownguidetest": OwnGuideTest,
    "owneasyguidetest": OwnEasyGuideTest,
}