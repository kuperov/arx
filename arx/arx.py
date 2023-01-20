"""Full ARX(p,q) model, using quadrature for inference.

This limited first version can only do inference for p=1,
but can simulate from any ARX(p,q) dgp.
"""

f"This script needs python 3.x"


from jax.config import config

from arx.cv import CVScheme

config.update("jax_enable_x64", True)

from collections import namedtuple
from typing import Callable, Tuple, Dict

import arviz as az
import blackjax
import jax
import jax.numpy as jnp
from chex import Array, PRNGKey, assert_rank, assert_shape, dataclass
from jax.experimental import sparse
from jax.numpy import exp, log
from jax.numpy.linalg import det, slogdet, solve
from jax.scipy.linalg import solve_triangular, inv
from jax.scipy.optimize import minimize
from jax.scipy.special import gammaln, logsumexp
from jax.scipy.stats import beta, multivariate_normal
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd
from enum import Enum

class Evaluation(Enum):
    POINTWISE = 1
    JOINT = 2


Theta = namedtuple("Theta", ["phi", "beta", "sigsq"])

# parameter space transformations for MCMC
sigsq_tfm = tfb.Exp()
phi_tfm = tfb.Sigmoid(low=-1, high=1)


def make_L(phi: Array, T: int):
    assert_rank(phi, 1)  # vector
    L = jnp.eye(T)
    for i in range(len(phi)):
        L += jnp.diag(jnp.repeat(-phi[i], T - 1 - i), k=-i - 1)
    return L


def make_L_ar1(phi: Array, T: int):
    assert_rank(phi, 0)  # scalar
    return jnp.eye(T) + jnp.diag(-phi * jnp.ones(T - 1), k=-1)


def nig_logpdf(x: Array, mu: Array, Sigma: Array, a: float, b: float) -> float:
    """Normal-inverse-gamma log density"""
    assert_rank(x, 1)
    assert_rank(mu, 1)
    assert_rank(Sigma, 2)
    assert_shape(Sigma, (len(x), len(x)))
    p = len(x)
    e = x - mu
    return (
        gammaln(0.5 * p + a)
        - gammaln(a)
        + a * log(b)
        - 0.5 * p * log(2 * jnp.pi)
        - 0.5 * slogdet(Sigma)[1]
        - (0.5 * p + a) * log(b + 0.5 * jnp.dot(e, solve(Sigma, e)))
    )


# https://blackjax-devs.github.io/blackjax/examples/change_of_variable_hmc.html#arviz-plots
def arviz_trace_from_states(position, info, burn_in=0):
    """Create Arviz trace from states

    Args:
        position: position member of blackjax states
        info: multichain info object
    """
    if isinstance(position, jnp.DeviceArray):  # if states.position is array of samples
        position = dict(samples=position)
    else:
        try:
            position = position._asdict()
        except AttributeError:
            pass

    samples = {}
    for param in position.keys():
        ndims = len(position[param].shape)
        if ndims >= 2:
            samples[param] = jnp.swapaxes(position[param], 0, 1)[
                :, burn_in:
            ]  # swap n_samples and n_chains
            divergence = jnp.swapaxes(info.is_divergent[burn_in:], 0, 1)

        if ndims == 1:
            divergence = info.is_divergent
            samples[param] = position[param]

    trace_posterior = az.convert_to_inference_data(samples)
    trace_sample_stats = az.convert_to_inference_data(
        {"diverging": divergence}, group="sample_stats"
    )
    trace = az.concat(trace_posterior, trace_sample_stats)
    return trace


CVPosteriorSet = namedtuple("CVPosteriorSet", ["elpdhat", "contribs", "posts"])


class ARX:
    """Full ARX(p,q) model, using quadrature for inference."""

    def __init__(
        self,
        T: int,
        phi_star: Array,
        sigsq_star: float,
        beta_star: Array,
        mu_beta0: Array = None,
        sigma_beta0: Array = None,
        a_phi0: float = 1.0,
        b_phi0: float = 1.0,
        a_sigsq0: float = 1.0,
        b_sigsq0: float = 1.0,
        Z: Array = None,
    ):
        self.T = T
        self.phi_star = phi_star
        self.sigsq_star = sigsq_star
        self.beta_star = beta_star
        self.q = Z.shape[1]
        self.mu_beta0 = mu_beta0
        if self.mu_beta0 is None:
            self.mu_beta0 = jnp.zeros(self.q)
        self.sigma_beta0 = sigma_beta0
        if self.sigma_beta0 is None:
            self.sigma_beta0 = jnp.eye(self.q)
        self.a_phi0 = a_phi0
        self.b_phi0 = b_phi0
        self.a_sigsq0 = a_sigsq0
        self.b_sigsq0 = b_sigsq0
        self.Z = Z
        self.sigma_beta0_inv = jnp.linalg.inv(self.sigma_beta0)
        self.p = len(self.phi_star)
        assert_rank(self.phi_star, 1)
        assert_rank(self.beta_star, 1)
        assert_rank(self.Z, 2)
        assert_shape(self.Z, (self.T, len(self.beta_star)))

    def simulate(self, key: PRNGKey = None) -> Array:
        L = make_L(self.phi_star, self.T)
        mstar = solve_triangular(L, self.Z @ self.beta_star, lower=True)
        eps = jax.random.normal(key, shape=(self.T,))
        return mstar + jnp.sqrt(self.sigsq_star) * solve_triangular(L, eps, lower=True)

    def misspecify(self, **args) -> "ARX":
        # let's not futz with T because we don't want to adjust the shape of Z
        assert "T" not in args, "Can't change T in misspecify()"
        args["T"] = self.T
        for param in [
            "phi_star",
            "sigsq_star",
            "beta_star",
            "Z",
            "q",
            "mu_beta0",
            "sigma_beta0",
            "a_phi0",
            "b_phi0",
            "a_sigsq0",
            "b_sigsq0",
        ]:
            if param not in args:
                args[param] = getattr(self, param)
        if "q" in args:
            assert args["q"] <= self.q, "Can only misspecify to a lower q"
            args["Z"] = self.Z[:, : args["q"]]
            args["beta_star"] = self.beta_star[
                : args["q"],
            ]  # won't actually be used
            args["sigma_beta0"] = self.sigma_beta0[: args["q"], : args["q"]]
            args["mu_beta0"] = self.mu_beta0[: args["q"]]
            del args["q"]
        if "p" in args:
            args["phi_star"] = args["phi_star"][
                : args["p"],
            ]
            del args["p"]
        return ARX(**args)  # type: ignore

    def optimal_params(self, dgp: "ARX") -> Tuple[Array, Tuple[Array, Array, Array]]:
        """Compute optimal params for a misspecified model.

        This is the KLD minimizing parameters with respect to the DGP.

        Returns (min kld, (phi, beta, sigsq))
        """

        def D(x):
            phi = x[: self.p]
            beta = x[self.p : (self.p + self.q)]
            sigsq = x[-1]
            Lk = make_L(phi, self.T)
            Lstar = make_L(dgp.phi_star, self.T)
            Lstar_inv = jnp.linalg.inv(Lstar)
            A = Lk @ Lstar_inv
            m = self.Z @ beta - A @ dgp.Z @ dgp.beta_star
            return 0.5 * (
                dgp.sigsq_star / sigsq * jnp.trace(A @ A.T)
                - self.T
                + self.T * (jnp.log(sigsq) - jnp.log(dgp.sigsq_star))
                + m.T @ m / sigsq
            )

        # starting point (x0)
        x0 = jnp.concatenate([jnp.zeros(self.p), jnp.ones(self.q), jnp.array([1.0])])

        from jax.scipy.optimize import minimize

        f = lambda theta: D(theta)
        theta_min = minimize(f, x0, method="BFGS")
        phi = theta_min.x[: dgp.p]
        beta = theta_min.x[dgp.p : (dgp.p + dgp.q)]
        sigsq = theta_min.x[-1]
        return (theta_min.fun, (phi, beta, sigsq))

    def phi_prior_logpdf(self, phi) -> float:
        return beta.logpdf(x=0.5 * (phi + 1.0), a=self.a_phi0, b=self.b_phi0)

    def beta_prior_logpdf(self, beta, sigsq):
        return multivariate_normal.logpdf(
            beta, mean=self.mu_beta0, cov=sigsq * self.sigma_beta0
        )

    def sigsq_prior_logpdf(self, sigsq):
        return (
            self.a_sigsq0 * log(self.b_sigsq0)
            - gammaln(self.a_sigsq0)
            - (self.a_sigsq0 + 1) * log(sigsq)
            - self.b_sigsq0 / sigsq
        )

    def llhood(self, y, phi, beta, sigsq) -> float:
        """Full-data likelihood"""
        L = make_L(phi, self.T)
        mu = L @ y - self.Z @ beta
        return -0.5 * self.T * log(2 * jnp.pi * sigsq) - 0.5 * jnp.sum(mu**2) / sigsq

    def dgp_llhood(self, y: Array) -> float:
        """Return the log likelihood using stored parameters.

        Args:
            y: data array
        """
        return self.llhood(y, self.phi_star, self.beta_star, self.sigsq_star)

    def log_p_y_phi(self, phi: float, y: Array) -> float:
        """Computes log p(phi, y)

        Args:
           phi:     univariate AR(1) autoregressive coefficient
           y:       data vector
        """
        L = make_L_ar1(phi, self.T)
        Ly = L @ y
        c1 = Ly.T @ Ly + self.mu_beta0.T @ self.sigma_beta0_inv @ self.mu_beta0
        Sigma_beta_inv = self.Z.T @ self.Z + self.sigma_beta0_inv
        c3 = self.sigma_beta0_inv @ self.mu_beta0 + self.Z.T @ L @ y
        log_c4 = (
            self.phi_prior_logpdf(phi)
            + self.a_sigsq0 * log(self.b_sigsq0)
            - 0.5 * (self.T) * log(2 * jnp.pi)
            - 0.5 * log(det(self.sigma_beta0))
            - gammaln(self.a_phi0)
        )
        nu = 0.5 * self.T + self.a_sigsq0
        log_dens = (
            log_c4
            - 0.5 * log(det(Sigma_beta_inv))  # note _inv
            + gammaln(nu)
            - nu * log(self.b_sigsq0 + 0.5 * (c1 - c3.T @ solve(Sigma_beta_inv, c3)))
        )
        return log_dens

    def p_y_phi(self, phi: float, y: Array, ln_coef: float = 0) -> float:
        """Computes p(phi, y)*exp(ln_coef)

        Args:
           phi:     univariate AR(1) autoregressive coefficient
           y:       data vector
           ln_coef: log of coefficient to scale density

        The coefficient is there to deal with roundoff error. See paper draft
        for the formula (without coefficient).
        """
        return exp(self.log_p_y_phi(phi, y) + ln_coef)

    def log_p_y(self, y: Array) -> float:
        """Compute log evidence p(y) by adaptive quadrature.

        Args:
            y:            data vector

        This function uses adaptive quadrature from QUADPACK via scipy starting
        with a Laplace estimate of log p(y).
        """
        assert_shape(y, (self.T,))
        # estimate log evidence Z using laplace's method
        phi_ols = self.ols(y)[
            0:1,
        ]

        f = jax.jit(lambda phi: -self.log_p_y_phi(phi[0], y))
        mode = minimize(f, x0=phi_ols, method="BFGS")
        assert -1 < mode.x < 1, "mode of p(y, phi) out of bounds"
        hessian = jax.hessian(f)(mode.x)
        ln_Z_laplace = (
            0.5 * jnp.log(2 * jnp.pi)
            - 0.5 * jnp.log(jnp.linalg.det(hessian))
            - mode.fun
        )
        # integrate to improve log evidence estimate, now in levels not logs
        integral = quad(self.p_y_phi, a=-1.0, b=1.0, args=(y, -ln_Z_laplace))[0]
        assert not integral < 0, "Panic: negative integral"
        # use accurate log for value close to 1?
        return log(integral) + ln_Z_laplace

    def ols(self, y):
        """Estimate phi and beta using OLS"""
        X = jnp.hstack(
            [
                jnp.expand_dims(y[0:-1], 1),
                self.Z[
                    1:,
                ],
            ]
        )
        return solve(X.T @ X, X.T @ y[1:])

    def post_dens(self, y: Array) -> Tuple[Callable, float]:
        """Construct multivariate posterior density function given y.

        Args:
            y:       data vector

        This is a higher-order function that returns a Callable
        and the log evidence.
        """
        log_p_y = self.log_p_y(y)

        def post(phi, beta, sigsq):
            log_joint = (
                self.beta_prior_logpdf(beta=beta, sigsq=sigsq)
                + self.phi_prior_logpdf(phi=phi)
                + self.sigsq_prior_logpdf(sigsq=sigsq)
                + self.llhood(y, phi, beta, sigsq)
                - log_p_y
            )
            return exp(log_joint)

        return post, log_p_y

    def post(self, y: Array, log_p_y_guess=None) -> "CVPosterior":
        """Construct posterior object given y.

        Args:
            y:       data vector
        """
        return CVPosterior(
            self,
            y=y,
            S_test=jnp.eye(self.T),
            S_train=jnp.eye(self.T),
            log_p_y_guess=log_p_y_guess,
        )

    def fd_post(self, y: Array) -> "FullDataPosterior":
        """Full data posterior, given y

        The FullDataPosterior is a spcial case of the CVPosterior that uses
        the full dataset and requires fewer matrix inversions when computing
        log likelihoods.
        """
        post = FullDataPosterior(self, y)
        return post

    def fd_nuts_post(
        self, y: Array, prng_key, ndraws=2000, nchains=4, nwarmup=1000
    ) -> "FullDataMCMCPosterior":
        """Construct posterior object given y using NUTS.

        Args:
            y:        data vector
            prng_key: random number generator key
            ndraws:   number of nuts draws after warmup, per chain
            nchains:  number of independent chains
            nwarmup:  number of window adaption iterations to run
        """
        post = FullDataMCMCPosterior(self, y=y)
        post.nuts(ndraws=ndraws, nchains=nchains, nwarmup=nwarmup, prng_key=prng_key)
        return post

    def cv(self, scheme: CVScheme, y: Array) -> CVPosteriorSet:
        """Compute cross-validation elpd.

        Args:
            scheme:  cross-validation scheme
            y:       data vector

        Note: we maintain a running moving average of the elpd estimate to use as normalizing
        guesses for the next fold. This is a bit of a hack, but it seems to work well.
        """
        assert_shape(y, (self.T,))
        elpd_contribs, cv_posts = [], []
        full_post = self.post(y)
        log_p_y = full_post.log_p_y
        log_p_y_tilde_g = log_p_y
        for i in range(scheme.n_folds()):
            fold_post = CVPosterior(
                self, y, S_test=scheme.S_test(i), S_train=scheme.S_train(i), log_p_y_guess=log_p_y
            )
            log_p_y = 0.25 * log_p_y + 0.75 * fold_post.log_p_y
            elpd_contribs.append(
                fold_post.log_p_y_tilde(y, log_p_y_tilde_guess=log_p_y_tilde_g)
            )
            log_p_y_tilde_g = 0.25 * log_p_y_tilde_g + 0.75 * elpd_contribs[-1]  # update guess moving average
            cv_posts.append(fold_post)
        arr_contribs = jnp.array(elpd_contribs)
        return CVPosteriorSet(
            elpdhat=arr_contribs.sum(),
            contribs=arr_contribs,
            posts=cv_posts,
        )

    def cv_nuts(self, scheme: CVScheme, y: Array, prng_key, ndraws=2000, nchains=4, nwarmup=1000) -> Dict[str, CVPosteriorSet]:
        """Compute cross-validation elpd using NUTS.
        
        This method is intended for CV methods with small numbers of folds,
        like K-fold CV.

        Args:
            scheme:     cross-validation scheme
            y:          data vector
        """
        assert_shape(y, (self.T,))
        elppd_contribs, eljpd_contribs, cv_posts = [], [], []
        for i in range(scheme.n_folds()):
            fold_post = MCMCCVPosterior(self, y=y, S_train=scheme.S_train(i), S_test=scheme.S_test(i))
            fold_post.nuts(prng_key=prng_key, ndraws=ndraws, nchains=nchains, nwarmup=nwarmup)
            elppd_contribs.append(fold_post.log_p_y(evaluation=Evaluation.POINTWISE))
            eljpd_contribs.append(fold_post.log_p_y(evaluation=Evaluation.JOINT))
            cv_posts.append(fold_post)
        elppd_contribs_a = jnp.array(elppd_contribs)
        pw = CVPosteriorSet(
            elpdhat=elppd_contribs_a.sum(),
            contribs=elppd_contribs,
            posts=cv_posts,
        )
        eljpd_contribs_a = jnp.array(eljpd_contribs)
        j = CVPosteriorSet(
            elpdhat=eljpd_contribs_a.sum(),
            contribs=eljpd_contribs_a,
            posts=cv_posts,
        )
        return {"pw": pw, "joint": j}
    
    def cv_laplace(self, scheme: CVScheme, y: Array) -> float:
        """Compute cross-validation elpd using laplace approximation.

        Args:
            scheme:  cross-validation scheme
            y:       data vector
        """
        assert_shape(y, (self.T,))
        elpd_contribs = []
        for i in range(scheme.n_folds()):
            fold_post = CVPosterior(self, y, scheme.S_test(i), scheme.S_train(i))
            elpd_contribs.append(fold_post.log_p_y_tilde_laplace(y))
        return jnp.sum(jnp.array(elpd_contribs))


class FullDataPosterior:
    """ARX Posterior for full data

    This is a separate class from CVPosterior because we can avoid inverting L in the
    likelihood when S_train is the identity.
    """

    def __init__(self, m: ARX, y):
        self.m = m
        self.T = m.T
        self.y = y
        self.Sigma_beta = jnp.linalg.inv(self.m.sigma_beta0_inv + self.m.Z.T @ self.m.Z)

    def log_y_tilde_phi(self, y_tilde: Array, phi: float, S_test: Array) -> float:
        """Posterior predictive log density for y_tilde, conditional on phi"""
        # posterior mean and variance - full data
        L = make_L_ar1(phi, self.T)
        LinvZ = solve_triangular(L, self.m.Z, lower=True)
        Linv = solve_triangular(L, jnp.eye(self.T), lower=True)
        StLinv = S_test @ Linv
        SW_phiS = StLinv @ StLinv.T
        mu_beta = self.Sigma_beta @ (
            self.m.Z.T @ L @ self.y + self.m.sigma_beta0_inv @ self.m.mu_beta0
        )
        # gaussian predictive parameters, marginalized down to elements selected by S_test
        mkphi_ff = S_test @ LinvZ @ mu_beta
        Vk_ff = SW_phiS + S_test @ LinvZ @ self.Sigma_beta @ LinvZ.T @ S_test.T
        cond_lpdf = nig_logpdf(
            S_test @ y_tilde, mkphi_ff, Vk_ff, self.m.a_sigsq0, self.m.b_sigsq0
        )
        return cond_lpdf + self.m.phi_prior_logpdf(phi)

    def log_p_y_tilde(self, y_tilde, log_guess):
        I_T = jnp.eye(self.T)

        def f(phi):
            return exp(self.log_y_tilde_phi(y_tilde, phi, I_T) - log_guess)

        integral = quad(f, a=-1, b=1)[0]
        return log(integral) + log_guess

    def log_p_y_tilde_pw(self, y_tilde, log_guess):
        """Compute pointwise predictive density.

        Returns total predictive AND array of shape (T,) contributions

        log_guess should be on the order of individual contributions
        """

        def elpd_contrib(i):
            # 1xT selection matrix, so S y_tilde is objective
            S_test = jnp.expand_dims(1.0 * (jnp.arange(self.T) == i), axis=0)

            def f(phi):
                return exp(self.log_y_tilde_phi(y_tilde, phi, S_test) - log_guess)

            integral = quad(f, a=-1, b=1)[0]
            return log(integral) + log_guess

        contribs = jnp.array(list(map(elpd_contrib, jnp.arange(self.T))))
        return jnp.sum(contribs), contribs

    def elpd_mc_quad(self, dgp, log_p_y_tilde_guess, prng_key, mc_reps=500):
        """Return a Monte Carlo estimate of the elpd using quadrature.

        Args:
            y:       data vector
            dgp:     model for generating data replicates
            mc_reps: number of samples to draw from dgp

        This is an "oracle" version of the elpd, in that it uses the true
        model to generate data replicates.
        """
        key, self.prng_key = jax.random.split(prng_key)
        keys = jax.random.split(key, mc_reps)
        # can't parallelize via vmap because log_p_y_tilde uses scipy quad
        elpd_hats = []
        log_guess = log_p_y_tilde_guess
        for (i, key) in enumerate(keys):
            y_tilde = dgp.simulate(key)
            elpd = self.log_p_y_tilde(y_tilde, log_guess=log_guess)
            log_guess = 0.25 * log_guess + 0.75 * elpd  # update guess
            elpd_hats.append(elpd)
        # use mean here not logmeanexp because we want the expectation of the
        # log predictive density
        return {"mean": jnp.mean(jnp.array(elpd_hats)), "elpd_hats": jnp.stack(elpd_hats)}


class FullDataMCMCPosterior:
    """MCMC Posterior for full data

    This is a separate class from CVPosterior because we can avoid inverting L in the
    likelihood when S_train is the identity.
    """

    def __init__(self, m: ARX, y):
        self.m = m
        self.T = m.T
        self.y = y
        self.Sigma_beta = jnp.linalg.inv(self.m.sigma_beta0_inv + self.m.Z.T @ self.m.Z)

    def log_joint(self, theta: Theta):
        """Log joint p(y, phi, beta, sigsq) for MCMC."""
        phi = phi_tfm.forward(theta.phi)
        phi_ldj = phi_tfm.forward_log_det_jacobian(theta.phi)
        sigsq = sigsq_tfm.forward(theta.sigsq)
        sigsq_ldj = sigsq_tfm.forward_log_det_jacobian(theta.sigsq)
        beta = theta.beta
        L = make_L_ar1(phi, self.T)
        e = L @ self.y - self.m.Z @ beta
        return (
            # transform log det jacobians
            +phi_ldj
            + sigsq_ldj
            # likelihood
            - 0.5 * self.T * jnp.log(sigsq)
            - 0.5 * jnp.dot(e, e) / sigsq
            # priors
            + self.m.phi_prior_logpdf(phi)
            + self.m.beta_prior_logpdf(beta, sigsq)
            + self.m.sigsq_prior_logpdf(sigsq)
        )

    def inference_loop(self, rng_key, kernel, initial_states, num_samples, num_chains):
        """MCMC inference loop with multiple chains"""

        @jax.jit
        def one_step(states, rng_key):
            keys = jax.random.split(rng_key, num_chains)
            states, infos = jax.vmap(kernel)(keys, states)
            return states, (states, infos)

        keys = jax.random.split(rng_key, num_samples)
        _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)
        return (states, infos)

    def nuts(self, prng_key: PRNGKey, ndraws=2000, nchains=4, nwarmup=1000):
        init_key, warmup_key, sampling_key = jax.random.split(prng_key, 3)
        # use stan's window adaption algo to choose step size and mass matrix
        warmup = blackjax.window_adaptation(blackjax.nuts, self.log_joint, nwarmup)
        def warmed_up_states(seed, param):
            state, _, _ = warmup.run(seed, param)
            return state
        warmup_ipos = self.initial_positions(nchains, init_key)
        warmup_keys = jax.random.split(warmup_key, nchains)
        initial_states = jax.vmap(warmed_up_states)(warmup_keys, warmup_ipos)
        # construct kernel using warmup, which we need to run again
        _, kernel, _ = warmup.run(warmup_key, self.init_param_fn(init_key))
        states, info = self.inference_loop(
            sampling_key, kernel, initial_states, ndraws, nchains
        )
        # wait until program is done on device
        states.position.phi.block_until_ready()
        # transform states back to constrained space
        position = Theta(
            phi=jax.vmap(phi_tfm.forward)(states.position.phi),
            sigsq=jax.vmap(sigsq_tfm.forward)(states.position.sigsq),
            beta=states.position.beta,
        )
        self.position = position
        self.info = info
        return self.arviz()

    def initial_positions(self, n_chains, rng_key):
        keys = jax.random.split(rng_key, n_chains)
        return jax.vmap(self.init_param_fn)(keys)

    def init_param_fn(self, key):
        """Initial parameters in unconstrained space."""
        k1, k2, k3 = jax.random.split(key, 3)
        return Theta(
            sigsq=tfd.Normal(0, 1).sample(seed=k1),
            beta=tfd.MultivariateNormalDiag(
                loc=jnp.zeros(self.m.q), scale_diag=jnp.ones(self.m.q)
            ).sample(seed=k2),
            phi=tfd.Normal(0, 1).sample(seed=k3),
        )

    def arviz(self):
        return arviz_trace_from_states(self.position, self.info)

    def estimate_elpd(self, dgp, draws_az, prng_key=None, nreps=500):
        """Estimate the expected log predictive density using MCMC.

        Args:
            dgp: True DGP
            draws_az: Arviz InferenceData object
            prng_key: rng state for generating ytilde replicates
            nreps: number of ytilde replicates to generate

        PRNGKey should not be in the range of keys used for data generation.
        """
        y_tilde_keys = jax.random.split(prng_key, nreps)
        y_tildes = jax.vmap(dgp.simulate)(y_tilde_keys)
        draws = az.extract(draws_az, combined=True, num_samples=nreps)

        M = draws.dims["sample"]
        phi = jnp.array(draws["phi"])
        beta = jnp.array(draws["beta"])
        sigsq = jnp.array(draws["sigsq"])

        def elpd_for_draw(self, phi, beta, sigsq):
            L = make_L_ar1(phi, self.T)
            Linv = jnp.linalg.inv(L)
            Wphi = Linv @ Linv.T
            Zbeta = self.m.Z @ beta
            pred_mean = jnp.linalg.solve(L, Zbeta)

            def eljpd(y_tilde):
                return (
                    -0.5 * self.m.T * jnp.log(2 * jnp.pi * sigsq)
                    # log(det(Wphi)) == 0
                    - 0.5 * jnp.sum((L @ y_tilde - Zbeta) ** 2) / sigsq
                )

            def elppd_point(y_tilde):
                return (
                    # vectors of length T
                    -0.5 * jnp.log(2 * jnp.pi * sigsq * jnp.diag(Wphi))
                    - 0.5 * ((y_tilde - pred_mean) ** 2) / (sigsq * jnp.diag(Wphi))
                )

            draw_eljpd = jax.vmap(eljpd)(y_tildes)
            draw_rep_pw_contribs = jax.vmap(elppd_point)(y_tildes)
            return {"eljpd": draw_eljpd, "elppd_point": draw_rep_pw_contribs}

        estimates = jax.vmap(elpd_for_draw, in_axes=(None, 0, 1, 0))(
            self, phi, beta, sigsq
        )
        eljpds = logsumexp(estimates["eljpd"], axis=0) - jnp.log(M)
        elppd_point_contribs = logsumexp(estimates["elppd_point"], axis=0) - jnp.log(M)
        # add T pointwise contributions to get elppds
        elppds = jnp.sum(elppd_point_contribs, axis=1)
        # average over y_tilde reps to estimate elpd
        return {
            "mean_eljpd": jnp.mean(eljpds),
            "std_eljpd": jnp.std(eljpds),
            "mean_elppd": jnp.mean(elppds),
            "std_elppd": jnp.std(elppds),
        }


class CVPosterior:
    """ARX Posterior for a CV fold

    All computations here are specialized versions of the above that deal with
    a subset of the data.
    """

    def __init__(
        self,
        m: ARX,
        y: Array,
        S_test: Array = None,
        S_train: Array = None,
        log_p_y_guess=None,
    ):
        self.m: ARX = m
        self.T = m.T
        self.S_test = S_test
        self.S_train = S_train
        self.y = y
        self.n = S_train.shape[0]
        self.v = S_test.shape[0]
        assert_shape(y, (self.m.T,))
        assert_rank(S_test, 2)
        assert_rank(S_train, 2)
        assert S_test.shape[1] == self.T
        assert S_train.shape[1] == self.T
        if log_p_y_guess is None:
            self.log_p_y = self.log_p_y_laplace()
        else:
            self.log_p_y = log_p_y_guess
        self.log_p_y = self.log_p_y_quad(guess=self.log_p_y)

    def log_p_y_quad(self, guess) -> float:
        """Compute log evidence p(y) by adaptive quadrature.

        Args:
            A guess for the log evidence. Can come from laplace approximation or
            something else.

        This function uses adaptive quadrature from QUADPACK via scipy starting
        with a Laplace estimate of log p(y).
        """
        g = jax.jit(lambda phi: jnp.exp(self.log_p_y_phi(phi) - guess))
        integral = quad(g, a=-1.0, b=1.0)[0]
        assert not integral < 0, "Panic: negative likelihood"
        return log(integral) + guess

    def log_p_y_laplace(self) -> float:
        """Approximate log evidence p(y) by Laplace approximation."""
        f = jax.jit(lambda phi: -self.log_p_y_phi(phi))
        mode = minimize_scalar(f, bounds=[-1, 1], method="bounded")
        hessian = jax.hessian(f)(mode.x)
        return 0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(hessian) - mode.fun

    def log_p_y_phi(self, phi: float) -> float:
        """Computes log p(phi, y)

        Args:
           phi:     univariate AR(1) autoregressive coefficient
        """
        L = make_L_ar1(phi, self.T)
        Linv = solve_triangular(L, jnp.eye(self.T), lower=True)
        StrainLinv = self.S_train @ Linv
        Sigma_y_phi = (
            StrainLinv
            @ (jnp.eye(self.T) + self.m.Z @ self.m.sigma_beta0 @ self.m.Z.T)
            @ StrainLinv.T
        )
        mu_y_phi = StrainLinv @ self.m.Z @ self.m.mu_beta0
        log_p_y_given_phi = nig_logpdf(
            self.S_train @ self.y,
            mu_y_phi,
            Sigma_y_phi,
            self.m.a_sigsq0,
            self.m.b_sigsq0,
        )
        return log_p_y_given_phi + self.m.phi_prior_logpdf(phi)

    # log joint density log p(y_tilde, phi | y)
    def log_p_phi_ytilde(self, phi, y_tilde):
        L = make_L_ar1(phi, self.T)
        Linv = solve_triangular(L, jnp.eye(self.T), lower=True)
        LinvZ = solve_triangular(L, self.m.Z, lower=True)
        Sy = jnp.hstack([self.S_train @ self.y, self.S_test @ y_tilde])
        mu = jnp.vstack([self.S_train, self.S_test]) @ LinvZ @ self.m.mu_beta0
        ZSZ = self.m.Z @ self.m.sigma_beta0 @ self.m.Z.T
        StrainLinv = self.S_train @ Linv
        StestLinv = self.S_test @ Linv
        IT = jnp.eye(self.T)
        Sigma = jnp.block(
            [
                [
                    StrainLinv @ (IT + ZSZ) @ StrainLinv.T,
                    StrainLinv @ ZSZ @ StestLinv.T,
                ],
                [
                    StestLinv @ ZSZ @ StrainLinv.T,
                    StestLinv @ (IT + ZSZ) @ StestLinv.T,
                ],
            ]
        )
        log_p_y_given_phi = nig_logpdf(
            Sy,
            mu,
            Sigma,
            self.m.a_sigsq0,
            self.m.b_sigsq0,
        )
        return log_p_y_given_phi + self.m.phi_prior_logpdf(phi) - self.log_p_y

    def marg_post_phi(self, phi: float) -> float:
        """Phi marginal"""
        return exp(self.log_p_y_phi(phi) - self.log_p_y)

    def log_p_y_tilde(self, y_tilde: Array, log_p_y_tilde_guess) -> float:
        """Compute predictive log density log p(y_tilde | y)

        Args:
            y_tilde:    full length test data vector w/ shape (T,)
        """
        f = jax.jit(
            lambda phi: exp(self.log_p_phi_ytilde(phi, y_tilde) - log_p_y_tilde_guess)
        )
        pred = quad(f, a=-1, b=1)[0]
        return jnp.log(pred) + log_p_y_tilde_guess

    def log_p_y_tilde_laplace(self, y_tilde: Array) -> float:
        """Approximate log p(y_tilde | y) by Laplace approximation.

        Args:
            y_tilde:    full length test data vector w/ shape (T,)
        """
        f = jax.jit(lambda phi: -self.log_p_phi_ytilde(phi, y_tilde))
        mode = minimize_scalar(f, bounds=[-1, 1], method="bounded")
        assert -1 < mode.x < 1, f"mode of p(y, phi) ({mode.x}) out of bounds"
        hessian = jax.hessian(f)(mode.x)
        return 0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(hessian) - mode.fun

    def elpd_mc_quad(self, dgp, log_p_y_tilde_guess, n_samples=500):
        """Return a Monte Carlo estimate of the elpd using quadrature.

        Args:
            y:         data vector
            dgp:       model for generating data replicates
            n_samples: number of samples to draw from dgp

        This is an "oracle" version of the elpd, in that it uses the true
        model to generate data replicates.
        """
        key, self.prng_key = jax.random.split(self.prng_key)
        keys = jax.random.split(key, n_samples)
        y_tildes = jax.vmap(dgp.simulate)(keys)
        # can't parallelize via vmap because log_p_y_tilde uses scipy quad
        elpd_hats = list(
            map(lambda yt: self.log_p_y_tilde(yt, log_p_y_tilde_guess), y_tildes)
        )
        return {"mean": jnp.mean(elpd_hats), "elpd_hats": elpd_hats}

    def elpd_mc_laplace(self, dgp, prng_key, n_samples=1000):
        """Return a Monte Carlo estimate of the elpd using Laplace approximation.

        Args:
            y:         data vector
            dgp:       model for generating data replicates
            n_samples: number of samples to draw from dgp

        This is an "oracle" version of the elpd, in that it uses the true
        model to generate data replicates.
        """
        keys = jax.random.split(prng_key, n_samples)
        y_tildes = jax.vmap(dgp.simulate)(keys)
        elpd_hats = jax.vmap(lambda yt: self.log_p_y_tilde_laplace(yt))(y_tildes)
        return jnp.mean(elpd_hats)


class MCMCCVPosterior:
    """MCMC variant of CV posterior.

    This version uses nuts to estimate the posterior. Note, mcmc must be
    run after object creation. This is to facilitate testing etc.
    """

    def __init__(self, m, y, S_train, S_test):
        self.m = m
        self.T = m.T
        self.q = m.q
        self.y = y
        self.S_train = S_train
        self.S_test = S_test
        self.n = S_train.sum()

    def log_joint(self, theta: Theta):
        """Log joint p(y, phi, beta, sigsq) for MCMC.
        
        This version of the log joint is reduced to just the training set 
        by S_train.
        """
        # transform unconstrained theta to constrained parameter space
        phi = phi_tfm.forward(theta.phi)
        phi_ldj = phi_tfm.forward_log_det_jacobian(theta.phi)
        sigsq = sigsq_tfm.forward(theta.sigsq)
        sigsq_ldj = sigsq_tfm.forward_log_det_jacobian(theta.sigsq)
        beta = theta.beta
        L = make_L(jnp.array([phi]), self.T)
        Linv = solve_triangular(L, jnp.eye(self.T), lower=True)
        Se = self.S_train @ (self.y - Linv @ self.m.Z @ beta)
        SWSt = self.S_train @ (Linv @ Linv.T) @ self.S_train.T
        return (
            # transform log det jacobians
            phi_ldj
            + sigsq_ldj
            # likelihood
            - 0.5 * self.n * jnp.log(2 * jnp.pi)
            - 0.5 * self.n * jnp.log(sigsq)
            - 0.5 * jnp.linalg.slogdet(SWSt)[1]
            - 0.5 * jnp.dot(Se, solve(SWSt, Se)) / sigsq
            # priors
            + self.m.phi_prior_logpdf(phi)
            + self.m.beta_prior_logpdf(beta, sigsq)
            + self.m.sigsq_prior_logpdf(sigsq)
        )

    def inference_loop(self, rng_key, kernel, initial_states, num_samples, num_chains):
        """MCMC inference loop with multiple chains"""

        @jax.jit
        def one_step(states, rng_key):
            keys = jax.random.split(rng_key, num_chains)
            states, infos = jax.vmap(kernel)(keys, states)
            return states, (states, infos)

        keys = jax.random.split(rng_key, num_samples)
        _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)
        return (states, infos)

    def nuts(self, prng_key, ndraws=2000, nchains=4, nwarmup=1000):
        init_key, warmup_key, sampling_key = jax.random.split(prng_key, 3)
        # use stan's window adaption algo to choose step size and mass matrix
        warmup = blackjax.window_adaptation(blackjax.nuts, self.log_joint, nwarmup)
        initial_states = jax.vmap(lambda key, param: warmup.run(key, param)[0])(
            jax.random.split(warmup_key, nchains),
            self.initial_positions(nchains, init_key)
        )
        # construct kernel using warmup, which we need to run again
        _, kernel, _ = warmup.run(warmup_key, self.init_param_fn(init_key))
        states, info = self.inference_loop(
            sampling_key, kernel, initial_states, ndraws, nchains
        )
        # wait until program is done on device
        states.position.phi.block_until_ready()
        # transform states back to constrained space
        position = Theta(
            phi=jax.vmap(phi_tfm.forward)(states.position.phi),
            sigsq=jax.vmap(sigsq_tfm.forward)(states.position.sigsq),
            beta=states.position.beta,
        )
        self.position = position
        self.info = info
        return self.arviz()

    def initial_positions(self, n_chains, rng_key):
        keys = jax.random.split(rng_key, n_chains)
        return jax.vmap(self.init_param_fn)(keys)

    def init_param_fn(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        return Theta(
            sigsq=tfd.Normal(0, 1).sample(seed=k1),
            beta=tfd.MultivariateNormalDiag(
                loc=jnp.zeros(self.q), scale_diag=jnp.ones(self.q)
            ).sample(seed=k2),
            phi=tfd.Normal(0, 1).sample(seed=k3),
        )

    def arviz(self):
        return arviz_trace_from_states(self.position, self.info)

    def log_p_y(self, evaluation: Evaluation = Evaluation.JOINT):
        """Log predictive density p(y* | y)
        
        Note that self.position has already been transformed back to constrained parameters.
        """
        # joint version returns a scalar per fold
        def lpred_joint(phi, beta, sigsq):
            L = make_L(jnp.array([phi]), self.T)
            StestLinv = self.S_test @ inv(L)
            m = StestLinv @ self.m.Z @ beta
            cov = sigsq * StestLinv @ StestLinv.T
            return tfd.MultivariateNormalFullCovariance(loc=m, covariance_matrix=cov).log_prob(self.S_test @ self.y)
        # pointwise version needs to return a vector of contributions,
        # to be combined with logsumexp coordinatewise
        def lpred_pw(phi, beta, sigsq):
            L = make_L(jnp.array([phi]), self.T)
            StestLinv = self.S_test @ inv(L)
            m = StestLinv @ self.m.Z @ beta
            sigs = jnp.sqrt(jnp.diag(sigsq * StestLinv @ StestLinv.T))
            return tfd.Normal(loc=m, scale=sigs).log_prob(self.S_test @ self.y)
        lpred = lpred_joint if evaluation == Evaluation.JOINT else lpred_pw
        phis = jnp.reshape(self.position.phi, (-1))
        sigsqs = jnp.reshape(self.position.sigsq, (-1))
        betas = jnp.reshape(self.position.beta, (-1, self.q))
        contribs = jax.vmap(lpred)(phis, betas, sigsqs)
        M = phis.shape[0]
        unnorm_contrib = logsumexp(contribs, axis=0) - jnp.log(M)
        return unnorm_contrib.sum()
