from pyro.infer import Trace_ELBO, TraceMeanField_ELBO

loss_types = {
    "trace_elbo": Trace_ELBO,
    "trace_mean_field_elbo": TraceMeanField_ELBO,
}