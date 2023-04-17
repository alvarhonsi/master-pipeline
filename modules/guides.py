import torch
import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.nn import PyroSample, PyroModule, PyroParam
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer.autoguide.initialization import init_to_feasible, init_to_median

def getAutoDiagonalNormal(model, device="cpu"):
    def init_func(*args):
        return init_to_median(*args).to(device)
    
    return AutoDiagonalNormal(model, init_loc_fn=init_func)

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



guide_types = {
    "autodiagonalnormal": getAutoDiagonalNormal,
    "ownguide": OwnGuide,
}