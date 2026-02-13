# %%
from enterprise.pulsar import Pulsar
from enterprise.signals.gp_signals import MarginalizingTimingModel
from enterprise.signals.white_signals import MeasurementNoise
from enterprise_extensions.blocks import red_noise_block, dm_noise_block
from enterprise.signals.signal_base import PTA

import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

# %%
psr = Pulsar("sim5.par", "sim5.tim")

# %%
def get_pta_obj(nharm_arn, nharm_dmn):
    tm = MarginalizingTimingModel()
    wn = MeasurementNoise(efac=1) # Put selection here
    model = tm + wn
    
    if nharm_arn > 0:
        model += red_noise_block(components=nharm_arn)
    
    if nharm_dmn > 0:
        model += dm_noise_block(components=nharm_dmn)

    return PTA([model(psr)])

# %%
pta = get_pta_obj(7, 5)

# %%
x0 = np.array([p.sample() for p in pta.params])

# %%
float(pta.get_lnlikelihood(x0))

# %%
def find_maxlike_params(nharm_arn, nharm_dmn):
    pta = get_pta_obj(nharm_arn, nharm_dmn)

    if len(pta.param_names) == 0:
        return np.array([]), pta.get_lnlikelihood([])

    try:
        mlnlike = lambda params: -pta.get_lnlikelihood(params)
        x0 = np.array([p.sample() for p in pta.params])

        res = minimize(mlnlike, x0, method="Nelder-Mead")
        assert res.success
        return res.x, -res.fun
    except Exception:
        return find_maxlike_params(nharm_arn, nharm_dmn)

# %%
def calc_aic(nharm_arn, nharm_dmn):
    xmax, lnLhat = find_maxlike_params(nharm_arn, nharm_dmn)
    q = len(xmax)
    return 2*q - 2*lnLhat

# %%
calc_aic(7, 5)

# %%
nharms_arn = np.arange(46)
nharms_dmn = np.arange(46)

aics = np.zeros((46, 46))
for nharm_arn in nharms_arn:
    for nharm_dmn in nharms_arn:
        print(nharm_arn, nharm_dmn)
        aics[nharm_arn, nharm_dmn] = calc_aic(nharm_arn, nharm_dmn)


# %%
plt.imshow(np.log(aics - np.min(aics) + 1))


# %%
np.unravel_index(np.argmin(aics), aics.shape)

# %%
