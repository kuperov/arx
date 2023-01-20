from typing import Any, Callable, Dict

import jax.numpy as jnp
from chex import Array, PRNGKey
import numpy as np
import os
import jax
import chex

from arx import sarx, arx

EFFECT_SIZES = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])

# \beta_*^{easy}
EASY_EFFECTS = jnp.array([1.0, 2.0, 1.0])
# \beta_*^{hard}
HARD_EFFECTS = jnp.array([1.0, 0.5, 1.0])
# \sigma^2_*
NOISE_VARIANCE = 1.0

DATA_LENGTHS = [
    2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 
    110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 
    250, 300, 400, 500, 750, 1000, 1500, 2000]
# alpha values for plotting (finer grid)
ALPHA_RANGE = jnp.linspace(0.0, 1.0, 21)
# alpha values for small multiple plots
ALPHA_SM = [0.0, 0.5, 0.75, 1.0]

HALOS = jnp.arange(31)

DIMENSIONS = jnp.arange(31)

QUANTILES = jnp.array([0.01, 0.99])

ZFILE = os.path.join(os.path.dirname(__file__), "Z.csv")


def make_Z(q: int, T: int, prng_key: chex.PRNGKey):
    """Construct exogenous covariate matrix"""
    if q > 1:
        Z = jnp.hstack(
            [
                jnp.ones((T, 1)),
                jax.random.normal(key=prng_key, shape=(T, q - 1)),
            ]
        )
    else:
        Z = jnp.ones((T, 1))
    return Z


def save_Z(max_len, max_q, seed):
    """Construct and save Z for use across experiments
    
    Args:
        q: number of columns, including 1s in first column
        max_len: maximum size T (ie number of rows)
        max_q: maximum aerodynamic^H^H^H number of columns
        seed: random number generator seed
    """
    prng_key = jax.random.PRNGKey(seed)
    Z = make_Z(max_q, max_len, prng_key)
    np.savetxt(ZFILE, Z, delimiter=",")


def load_Z(q, T):
    """Load Z from a csv file
    
    Args:
        q: number of columns, including a vector of ones
        T: number of rows
    """
    Z = np.loadtxt(ZFILE, delimiter=",")
    return Z[:T, :q]


class ExperimentInstance:
    """An instance of an experiment"""

    def __init__(self, experiment, dgp, mA, mB):
        self.experiment = experiment
        self.dgp = dgp
        self.mA = mA
        self.mB = mB

    def optimal_params_mA(self):
        return self.mA.optimal_params(self.dgp)

    def optimal_params_mB(self):
        return self.mB.optimal_params(self.dgp)


class Experiment:
    """An experiment shown in the paper"""

    def __init__(
        self,
        name: str,
        model_params: Dict[str, Any],
        mA_p_q,
        mB_p_q,
        make_phi: Callable[[float], Array],
    ) -> None:
        self.name = name
        self.model_params = model_params
        self.make_phi = make_phi
        self.mA_p_q = mA_p_q
        self.mB_p_q = mB_p_q

    def make_simplified_instance(
        self, alpha, prng_key: PRNGKey, model_params: dict
    ) -> ExperimentInstance:
        phi = self.make_phi(alpha)
        this_params = self.model_params.copy()
        this_params.update(model_params)
        if "Z" not in this_params:
            this_params["Z"] = load_Z(this_params["q"], this_params["T"])
        del this_params["q"]
        dgp = sarx.SARX(phi_star=phi, **this_params)  # type: ignore
        mA = dgp.misspecify(p=self.mA_p_q[0], q=self.mA_p_q[1])
        mB = dgp.misspecify(p=self.mB_p_q[0], q=self.mB_p_q[1])
        return ExperimentInstance(self, dgp, mA, mB)

    def make_full_instance(
        self, alpha, prng_key: PRNGKey, model_params: dict
    ) -> ExperimentInstance:
        phi = self.make_phi(alpha)
        this_params = self.model_params.copy()
        this_params.update(model_params)
        if "Z" not in this_params:
            this_params["Z"] = load_Z(this_params["q"], this_params["T"])
        del this_params["q"]
        dgp = arx.ARX(phi_star=phi, **this_params)  # type: ignore
        mA = dgp.misspecify(p=self.mA_p_q[0], q=self.mA_p_q[1])
        mB = dgp.misspecify(p=self.mB_p_q[0], q=self.mB_p_q[1])
        return ExperimentInstance(self, dgp, mA, mB)


def make_full_experiment(number: int, variant: str) -> Experiment:
    """Make an experiment with a given name and variant"""
    param = EXPERIMENTS[number]
    beta_star = {"easy": EASY_EFFECTS, "hard": HARD_EFFECTS}[variant]
    e = Experiment(
        name=f"ex{number}-{variant}",
        model_params=dict(
            beta_star=beta_star,
            sigsq_star=NOISE_VARIANCE,
            q=3,
            mu_beta0=beta_star,
            T=param["T"],
        ),
        make_phi=param["make_phi"],
        mA_p_q=param["mA_p_q"],
        mB_p_q=param["mB_p_q"],
    )
    return e


def make_simplified_experiment(number: int, variant: str) -> Experiment:
    """Make an experiment with a given name and variant"""
    param = EXPERIMENTS[number]
    beta_star = {"easy": EASY_EFFECTS, "hard": HARD_EFFECTS}[variant]
    e = Experiment(
        name=f"ex{number}-{variant}",
        model_params=dict(
            beta_star=beta_star,
            sigsq_star=NOISE_VARIANCE,
            q=3,
            mu_beta0=beta_star,
            T=param["T"],
        ),
        make_phi=param["make_phi"],
        mA_p_q=param["mA_p_q"],
        mB_p_q=param["mB_p_q"],
    )
    return e


EXPERIMENTS = {
    1: dict(
        q=3,
        make_phi=lambda alpha: alpha * jnp.array([0.75, 0.2]),
        mA_p_q=(1, 2),
        mB_p_q=(1, 1),
        T=100,
    ),
    2: dict(
        q=3,
        make_phi=lambda alpha: alpha * jnp.array([0.95]),
        mA_p_q=(1, 3),
        mB_p_q=(1, 2),
        T=100,
    ),
    3: dict(
        q=3,
        make_phi=lambda alpha: alpha * jnp.array([0.95]),
        mA_p_q=(1, 2),
        mB_p_q=(1, 1),
        T=100,
    ),
    4: dict(
        q=3,
        make_phi=lambda alpha: alpha * jnp.array([0.95]),
        mA_p_q=(0, 2),
        mB_p_q=(0, 1),
        T=100,
    ),
    5: dict(
        q=3,
        make_phi=lambda alpha: alpha * jnp.array([0.95]),
        mA_p_q=(1, 2),
        mB_p_q=(0, 2),
        T=100,
    ),
    6: dict(
        q=3,
        make_phi=lambda alpha: alpha * jnp.array([0.75, 0.20]),
        mA_p_q=(1, 3),
        mB_p_q=(1, 1),
        T=100,
    ),
}
