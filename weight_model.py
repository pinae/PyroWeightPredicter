from torch import tensor
from torch.distributions.constraints import positive
import pyro
from pyro.distributions import Normal

#pyro.set_rng_seed(101)

prior_weight = pyro.param("prior_weight", tensor(60.0))
weight_variance = pyro.param("weight_variance", tensor(1.0), constraint=positive)
weight = pyro.sample("weight", Normal(prior_weight, weight_variance))
prior_bmr = pyro.param("prior_bmr", tensor(1600.0))
bmr_variance = pyro.param("bmr_variance", tensor(50.0), constraint=positive)
logging_variance = pyro.param("logging_variance", tensor(250.0), constraint=positive)
bmr = pyro.sample("bmr", Normal(prior_bmr, bmr_variance))
cal_weight_fac = pyro.param("cal_weight_fac", tensor(1/2000.0))
consumed_calories = pyro.sample("consumed_calories", Normal(prior_bmr, logging_variance))
time_step = 1.0  # Time is in days
posterior_weight = weight + time_step * (consumed_calories - bmr) * cal_weight_fac
print(posterior_weight)
