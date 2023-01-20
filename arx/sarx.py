from typing import Tuple

import chex
import jax
from jax import numpy as jnp
from jax.numpy.linalg import inv, slogdet, solve
from jax.scipy.linalg import solve_triangular
from jax.scipy.stats import multivariate_normal
from scipy import optimize

from arx.arx import make_L
from arx.cv import CVScheme


class Poly:
    """Second-degree polynomial in y.

    Contains the coefficients A, b, and c, as well as methods for computing
    their moments. See Lemma 3 in the paper.
    """

    def __init__(self, A, b, c, dgp: "SARX") -> None:
        self.A = A  # TxT
        self.b = b  # Tx1
        self.c = c  # scalar
        self.T = A.shape[0]
        self.Vstar = dgp.L_inv @ dgp.L_inv.T
        self.sigsq = dgp.sigsq_star
        self.mstar = dgp.L_star_invZ @ dgp.beta_star
        self.dgp = dgp

    def mean(self) -> float:
        Atilde = 0.5 * (self.A + self.A.T)
        return (
            self.sigsq * jnp.trace(Atilde @ self.Vstar)
            + self.c
            + self.b.T @ self.mstar
            + self.mstar.T @ Atilde @ self.mstar
        )

    def var(self) -> float:
        Atilde = 0.5 * (self.A + self.A.T)
        return (
            2.0 * self.sigsq**2 * jnp.trace(Atilde @ self.Vstar @ Atilde @ self.Vstar)
            + self.sigsq * self.b.T @ self.Vstar @ self.b
            + 4.0 * self.sigsq * self.b.T @ self.Vstar @ Atilde @ self.mstar
            + 4.0
            * self.sigsq
            * self.mstar.T
            @ Atilde
            @ self.Vstar
            @ Atilde
            @ self.mstar
        )

    def std(self) -> float:
        return self.var() ** 0.5

    def simulate(self, rng_key) -> float:
        y = self.dgp.simulate(rng_key)
        return y.T @ self.A @ y + self.b.T @ y + self.c

    def sim_moments(self, n, rng_key) -> Tuple[float, float]:
        """Simulate n draws of the polynomial and compute its mean and sd."""
        rng_keys = jax.random.split(rng_key, n)
        zs = jax.vmap(self.simulate)(rng_keys)
        return float(jnp.mean(zs)), float(jnp.std(zs))

    def sim_quantiles_neg_share(self, qs, n, rng_key) -> Tuple[chex.Array, float]:
        """Simulate quantiles and negative share drawn from this quadratic expression

        This is useful in experiments.
        """
        zs = jax.vmap(self.simulate)(jax.random.split(rng_key, n))
        quantiles = jnp.quantile(a=zs, q=qs)
        neg_share = jnp.mean(zs < 0)
        return quantiles, neg_share

    def __str__(self) -> str:
        return f"T = {self.T} mean = {self.mean() :.2f} sd = {self.var() ** 0.5 :.2f}"

    def __repr__(self) -> str:
        return str(self)

    def __sub__(self, x: "Poly") -> "Poly":
        """Subtract two polynomials."""
        return Poly(self.A - x.A, self.b - x.b, self.c - x.c, self.dgp)

    def __add__(self, x: "Poly") -> "Poly":
        """Add two polynomials."""
        return Poly(self.A + x.A, self.b + x.b, self.c + x.c, self.dgp)


class SARX:
    """Simplified ARX model class."""

    def __init__(
        self,
        T,
        phi_star,
        sigsq_star,
        beta_star,
        Z,
        mu_beta0,
        sigma_beta0=None,
        dgp=None,
    ):
        self.T = T
        self.phi_star = phi_star
        self.sigsq_star = sigsq_star
        self.beta_star = beta_star
        self.q = Z.shape[1]
        self.Z = Z
        self.L_star = make_L(phi_star, T)
        self.L_star_invZ = solve_triangular(self.L_star, self.Z, lower=True)
        self.L_inv = solve_triangular(self.L_star, jnp.eye(self.T), lower=True)
        self.mu_beta0 = mu_beta0
        self.sigma_beta0 = sigma_beta0
        if self.sigma_beta0 is None:
            self.sigma_beta0 = jnp.eye(self.q)
        self.sigma_beta0_inv = jnp.linalg.inv(self.sigma_beta0)
        self.dgp = dgp or self
        self._opt_sigsq_phi = False

    def misspecify(
        self,
        phi_star=None,
        sigsq_star=None,
        beta_star=None,
        p=None,
        q=None,
        mu_beta0=None,
        sigma_beta0=None,
    ):
        phi_star = phi_star or self.phi_star
        sigsq_star = sigsq_star or self.sigsq_star
        beta_star = beta_star or self.beta_star
        mu_beta0 = mu_beta0 or self.mu_beta0
        sigma_beta0 = sigma_beta0 or self.sigma_beta0
        Z = self.Z
        if p is not None:
            if p > len(phi_star):
                phi_star = jnp.hstack([phi_star, jnp.zeros(p - len(phi_star))])
            else:
                phi_star = phi_star[:p]
        if q:
            mu_beta0 = mu_beta0[:q]
            sigma_beta0 = sigma_beta0[:q, :q]
            beta_star = beta_star[:q]
            Z = Z[:, :q]
        return SARX(
            T=self.T,
            phi_star=phi_star,
            sigsq_star=sigsq_star,
            beta_star=beta_star,
            Z=Z,
            mu_beta0=mu_beta0,
            sigma_beta0=sigma_beta0,
            dgp=self.dgp,
        )

    def full_data_post(self, y) -> "SARXPost":
        return FullDataSARXPost(self, y)

    def post(self, y, Strain) -> "SARXPost":
        return PartialDataSARXPost(self, y, Strain)

    def simulate(self, prng_key) -> chex.Array:
        eps = jax.random.normal(prng_key, (self.T,))
        return (
            self.L_star_invZ @ self.beta_star
            + jnp.sqrt(self.sigsq_star) * self.L_inv @ eps
        )

    def eljpd(self, Stest: chex.Array) -> Poly:
        """Compute polynomial associated with eljpd of model.

        This version uses the full training data, which keeps calculations
        a little simpler.
        """
        phi_hat, sigsq_hat = self.memoize_opt_sigsq_phi()
        L = make_L(phi_hat, self.T)
        Wphi = jnp.linalg.inv(L.T @ L)
        LinvZ = solve_triangular(L, self.Z, lower=True)
        # simplified full-data version of sigma_beta
        sigma_beta = jnp.linalg.inv(self.sigma_beta0_inv + self.Z.T @ self.Z)
        Dk = LinvZ @ sigma_beta @ self.Z.T @ L
        mstar = self.dgp.L_star_invZ @ self.dgp.beta_star
        ek = LinvZ @ sigma_beta @ self.sigma_beta0_inv @ self.mu_beta0
        Vk = Wphi + LinvZ @ sigma_beta @ LinvZ.T
        v = jnp.sum(Stest)
        prec = jnp.linalg.inv(Stest @ Vk @ Stest.T)
        Ak = -0.5 * Dk.T @ Stest.T @ prec @ Stest @ Dk / sigsq_hat
        bk = -Dk.T @ Stest.T @ prec @ Stest @ (ek - mstar) / sigsq_hat
        ck = -0.5 * (
            v * jnp.log(2 * jnp.pi * sigsq_hat)
            - jnp.linalg.slogdet(prec)[1]
            + self.dgp.sigsq_star / sigsq_hat * jnp.trace(prec @ Stest @ Wphi @ Stest.T)
            + (ek - mstar).T @ Stest.T @ prec @ Stest @ (ek - mstar) / sigsq_hat
        )
        return Poly(Ak, bk, ck, self.dgp)

    def memoized_opt_sigsq_phi(self):
        """Memoize copy of sigsq and phi so we only have to do this once per sarx object."""
        if not self.sigsq_phi_memoized:
            self.phi_hat, self.sigsq_hat = opt_sigsq_phi(self, self.dgp)
            self.sigsq_phi_memoized = True
        return self.phi_hat, self.sigsq_hat

    def eljpdhat_cv(self, cv_scheme: CVScheme, pointwise: bool = False) -> Poly:
        """Compute (parameters of) distribution of the CV estimate of eljpdhat
        
        Args:
            cv_scheme: Cross-validation scheme to use
            pointwise: If True, return pointwise elppdhat instead of eljpdhat
        
        Returns Poly object representing the distribution of the CV estimate of eljpdhat
        """
        phi_hat, sigsq_hat = self.memoize_opt_sigsq_phi()
        L = make_L(phi_hat, self.T)
        Wphi = inv(L.T @ L)
        LinvZ = solve_triangular(L, self.Z, lower=True)
        I = jnp.eye(self.T)
        N = cv_scheme.n_folds()
        Ak, bk, ck = 0.0, 0.0, 0.0
        for i in range(N):
            Strain, Stest = cv_scheme.S_train(i), cv_scheme.S_test(i)
            sigma_beta = inv(
                self.sigma_beta0_inv
                + LinvZ.T @ Strain.T @ solve(Strain @ Wphi @ Strain.T, Strain @ LinvZ)
            )
            Dki = (
                LinvZ
                @ sigma_beta
                @ LinvZ.T
                @ Strain.T
                @ solve(Strain @ Wphi @ Strain.T, Strain)
            )
            eki = LinvZ @ sigma_beta @ self.sigma_beta0_inv @ self.mu_beta0
            Vk = Wphi + LinvZ @ sigma_beta @ LinvZ.T  # TxT
            if pointwise:
                # "diagonalize" Vk
                Vk = Vk * jnp.eye(self.T)
            v = jnp.sum(Stest)
            prec = inv(Stest @ Vk @ Stest.T)
            Ak += (
                -0.5
                * self.T
                / (N * v * sigsq_hat)
                * (I - Dki).T
                @ Stest.T
                @ prec
                @ Stest
                @ (I - Dki)
            )
            bk += (
                self.T
                / (N * v * sigsq_hat)
                * (I - Dki).T
                @ Stest.T
                @ prec
                @ Stest
                @ eki
            )
            ck += -0.5 * (
                self.T / N * jnp.log(2 * jnp.pi * sigsq_hat)
                + self.T / (N * v) * slogdet(Stest @ Vk @ Stest.T)[1]
                + self.T / (N * v * sigsq_hat) * eki.T @ Stest.T @ prec @ Stest @ eki
            )
        return Poly(Ak, bk, ck, self.dgp)

    def eljpd_cv_benchmark(self, cv_scheme: CVScheme) -> Poly:
        """Compute eljpd using same structure as CV scheme.

        Excludes the same training observations that the CV scheme does
        but uses fresh data rather than reusing y.
        """
        phi_hat, sigsq_hat = self.memoize_opt_sigsq_phi()
        L = make_L(phi_hat, self.T)
        Wphi = inv(L.T @ L)
        LinvZ = solve_triangular(L, self.Z, lower=True)
        mstar = self.dgp.L_star_invZ @ self.dgp.beta_star
        N = cv_scheme.n_folds()
        # we can't use vmap because some of the Stests have different shapes
        # means, stds = [], []
        Ak, bk, ck = 0.0, 0.0, 0.0
        for i in range(N):
            Strain, Stest = cv_scheme.S_train(i), cv_scheme.S_test(i)
            v = jnp.sum(Stest)
            # simplified full-data version of sigma_beta
            cov = Strain.T @ inv(Strain @ Wphi @ Strain.T) @ Strain
            sigma_beta = inv(
                self.sigma_beta0_inv
                + LinvZ.T @ Strain.T @ solve(Strain @ Wphi @ Strain.T, Strain @ LinvZ)
            )
            SteDk = Stest @ LinvZ @ sigma_beta @ LinvZ.T @ cov
            Steekm = Stest @ (
                LinvZ @ sigma_beta @ self.sigma_beta0_inv @ self.mu_beta0 - mstar
            )
            Vk = Wphi + LinvZ @ sigma_beta @ LinvZ.T
            prec = inv(Stest @ Vk @ Stest.T)
            Aki = -0.5 * SteDk.T @ prec @ SteDk / sigsq_hat
            bki = -SteDk.T @ prec @ Steekm / sigsq_hat
            cki = -0.5 * (
                v * jnp.log(2 * jnp.pi * sigsq_hat)
                + slogdet(Stest @ Vk @ Stest.T)[1]
                + self.dgp.sigsq_star
                * jnp.trace(prec @ Stest @ Wphi @ Stest.T)
                / sigsq_hat
                + Steekm.T @ prec @ Steekm / sigsq_hat
            )
            Ak += Aki / v
            bk += bki / v
            ck += cki / v
        # normalize to "sum length"
        Ak *= 1.0 * self.T / N
        bk *= 1.0 * self.T / N
        ck *= 1.0 * self.T / N
        return Poly(Ak, bk, ck, self.dgp)  # , means, stds

    def eljpd_cv_error(self, cv_scheme: CVScheme) -> Poly:
        """Compute error associated with data reuse in CV scheme"""
        return self.eljpdhat_cv(cv_scheme) - self.eljpd_cv_benchmark(cv_scheme)

    def memoize_opt_sigsq_phi(self) -> Tuple[chex.Array, chex.Array]:
        if not self._opt_sigsq_phi:
            self._opt_sigsq_phi = True
            self.phi_hat, self.sigsq_hat = opt_sigsq_phi(self, self.dgp)
        return self.phi_hat, self.sigsq_hat

    def cv(self, y: chex.Array, cv_scheme: CVScheme) -> float:
        """Estimate eljpd using CV.

        This is plan cross-validation, conditioned on a single y draw.

        Args:
            y:         T-vector of observations
            cv_scheme: CV scheme
        """
        N = cv_scheme.n_folds()
        contribs = []
        for i in range(N):
            # get training and test data
            Strain, Stest = cv_scheme.S_train(i), cv_scheme.S_test(i)
            v = jnp.sum(Stest)
            post = self.post(y, Strain)
            dens = post.log_pred_dens_subset(y, Stest) / v
            contribs.append(dens)
        return jnp.sum(jnp.array(contribs) * self.T / N)


def opt_sigsq_phi(m: SARX, dgp: SARX) -> Tuple[chex.Array, chex.Array]:
    """Find oracle plug-in sigsq and phi for a given model, with respect to its DGP.

    This minimizes the full process KL divergence between model predictive and dgp.
    """
    m_star = dgp.L_star_invZ @ dgp.beta_star
    I_T = jnp.eye(dgp.T)
    # The following are *dgp* parameters. These stay constant
    V_star = jnp.linalg.inv(dgp.L_star.T @ dgp.L_star)
    log_det_V_star = jnp.linalg.slogdet(V_star)[1]

    def EKLD(x):
        # The following are the *model* parameters we're optimizing over
        phi, sigsq = x[:-1], x[-1]
        L = make_L(phi, m.T)
        LinvZ = solve_triangular(L, m.Z, lower=True)
        Wphi = jnp.linalg.inv(L.T @ L)
        sigma_beta = jnp.linalg.inv(
            m.Z.T @ m.Z + m.sigma_beta0_inv
        )  # simplified version
        Dk = LinvZ @ sigma_beta @ m.Z.T @ L
        ek = LinvZ @ sigma_beta @ m.sigma_beta0_inv @ m.mu_beta0
        Vk = Wphi + LinvZ @ sigma_beta @ LinvZ.T
        mtilde = (Dk - I_T) @ m_star + ek
        return 0.5 * (
            m.T * jnp.log(sigsq)
            - m.T * jnp.log(dgp.sigsq_star)
            + jnp.linalg.slogdet(Vk)[1]
            - log_det_V_star
            - m.T
            + dgp.sigsq_star
            / sigsq
            * jnp.trace(jnp.linalg.solve(Vk, (V_star + Dk @ V_star @ Dk.T)))
            + mtilde @ jnp.linalg.solve(Vk, mtilde) / sigsq
        )

    x0 = jnp.concatenate([m.phi_star, jnp.array([m.sigsq_star])])
    bounds = [(-1.0, 1.0)] * len(m.phi_star) + [(1e-6, None)]
    f = jax.jit(EKLD)
    res = optimize.minimize(f, x0=x0, method="L-BFGS-B", bounds=bounds)
    phi_hat, sigsq_hat = res.x[:-1], res.x[-1]
    # differences should never be too wild
    # thresholds are a bit arbitrary, intended to stop unintended bad cases
    assert jnp.linalg.norm(phi_hat, ord=1) < 1.1
    assert jnp.linalg.norm(phi_hat - m.phi_star, ord=1) < 0.7
    # assert jnp.abs(sigsq_hat - m.sigsq_star) < 10
    return phi_hat, sigsq_hat


class SARXPost:
    def __init__(self, m, y, mk, sigsq_hat, phi_hat, Vk, mu_beta, sigma_beta) -> None:
        self.m = m
        self.y = y
        self.mk = mk
        self.sigsq_hat = sigsq_hat
        self.phi_hat = phi_hat
        self.Vk = Vk
        self.mu_beta = mu_beta
        self.sigma_beta = sigma_beta

    def log_pred_dens(self, ytilde):
        """Predictive density

        Args:
            ytilde: replicate data to evaluate predictive density at
        """
        return multivariate_normal.logpdf(
            ytilde, mean=self.mk, cov=self.sigsq_hat * self.Vk
        )

    def log_pred_dens_subset(self, ytilde, Stest):
        """Predictive density for subset of data

        Args:
            ytilde: full-length replicate data to evaluate predictive density at
            Stest: selection matrix for subset of data
        """
        sy, sm, sV = Stest @ ytilde, Stest @ self.mk, Stest @ self.Vk @ Stest.T
        return multivariate_normal.logpdf(sy, mean=sm, cov=self.sigsq_hat * sV)

    def eljpd(self, dgp: SARX) -> float:
        """Expected joint log predictive density

        This is $\\int dgp(ytilde) * log p(ytilde | mk, sigsq_hat * Vk) dytilde.$
        """
        mstar = dgp.L_star_invZ @ dgp.beta_star
        V_star = inv(dgp.L_star.T @ dgp.L_star)
        return -0.5 * (
            jnp.log(2 * jnp.pi) * self.m.T
            + jnp.linalg.slogdet(self.sigsq_hat * self.Vk)[1]
            + dgp.sigsq_star / self.sigsq_hat * jnp.trace(solve(self.Vk, V_star))
            + (self.mk - mstar).T @ solve(self.Vk, self.mk - mstar) / self.sigsq_hat
        )

    def __str__(self):
        return (
            f"Posterior: T = {self.m.T}, sigsq_hat = {self.sigsq_hat:.3f}, "
            f"phi_hat = {self.phi_hat}, mu_beta = {self.mu_beta}, sigma_beta = {self.sigma_beta}"
        )

    def __repr__(self) -> str:
        return self.__str__()


class PartialDataSARXPost(SARXPost):
    """SARX posterior class with data subsetting"""

    def __init__(self, m: SARX, y: chex.Array, Strain: chex.Array) -> None:
        self.Strain = Strain
        phi_hat, sigsq_hat = m.memoize_opt_sigsq_phi()
        L = make_L(phi_hat, m.T)
        LinvZ = solve_triangular(L, m.Z, lower=True)
        Wphi = inv(L.T @ L)
        # this is where we differ from the full-data case
        prec = Strain @ Wphi @ Strain.T
        sigma_beta = inv(
            m.sigma_beta0_inv + LinvZ.T @ Strain.T @ solve(prec, Strain @ LinvZ)
        )
        mu_beta = sigma_beta @ (
            LinvZ.T @ Strain.T @ solve(prec, Strain @ y)
            + m.sigma_beta0_inv @ m.mu_beta0
        )
        # whole data predictive mean & covariance, which will be subset later
        mk = LinvZ @ mu_beta
        Vk = Wphi + LinvZ @ sigma_beta @ LinvZ.T
        super().__init__(m, y, mk, sigsq_hat, phi_hat, Vk, mu_beta, sigma_beta)


class FullDataSARXPost(SARXPost):
    """SARX posterior class without data subsetting"""

    def __init__(self, m: SARX, y: chex.Array) -> None:
        phi_hat, sigsq_hat = m.memoize_opt_sigsq_phi()
        L = make_L(phi_hat, m.T)
        LinvZ = solve_triangular(L, m.Z, lower=True)
        sigma_beta = jnp.linalg.inv(m.sigma_beta0_inv + m.Z.T @ m.Z)
        mu_beta = sigma_beta @ (m.Z.T @ L @ y + m.sigma_beta0_inv @ m.mu_beta0)
        mk = LinvZ @ mu_beta
        Wphi = jnp.linalg.inv(L.T @ L)
        Vk = Wphi + LinvZ @ sigma_beta @ LinvZ.T
        super().__init__(m, y, mk, sigsq_hat, phi_hat, Vk, mu_beta, sigma_beta)
