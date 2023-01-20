from jax.config import config

config.update("jax_enable_x64", True)

import unittest

import cv
import jax
import jax.numpy as jnp
from chex import assert_equal, assert_shape, assert_tree_all_finite
from scipy.integrate import quad

import arx


class TestArx(unittest.TestCase):
    def setUp(self) -> None:
        key = jax.random.PRNGKey(123)
        self.m0 = arx.ARX(
            T=100,
            phi_star=jnp.array([0.9]),
            sigsq_star=1.0,
            beta_star=jnp.array([1.0, 0.5, 0.5]),
            q=3,
            prng_key=key,
        )  # type: ignore
        self.y = self.m0.simulate()

    def test_misspecify(self):
        assert_shape(self.y, (100,))
        m1 = self.m0.misspecify(phi_star=jnp.array([0.8]))
        assert_equal(m1.T, 100)
        assert_equal(m1.sigsq_star, 1.0)
        assert_equal(m1.phi_star, jnp.array([0.8]))
        y = m1.simulate()
        assert_shape(y, (100,))
        m2 = m1.misspecify(phi_star=jnp.array([0.8]), q=2, sigsq_star=2.0)
        assert_equal(m2.sigsq_star, 2.0)
        assert_equal(m2.phi_star, jnp.array([0.8]))
        assert_shape(m2.mu_beta0, (2,))
        assert_shape(m2.sigma_beta0, (2, 2))
        assert_shape(m2.sigma_beta0_inv, (2, 2))
        y = m2.simulate()
        assert_shape(y, (100,))

    def test_p_y_phi(self):
        for phi in [0.005, 0.5, 0.995]:
            dens = self.m0.p_y_phi(phi=0.5, y=self.y)
            assert not jnp.isnan(dens), f"nan result for phi={phi}"
            assert dens >= 0, f"negative density for phi={phi}"

    def test_post_and_pred(self):
        postf, log_py = self.m0.post_dens(self.y)
        assert not jnp.isnan(log_py)
        assert not jnp.isnan(
            postf(phi=jnp.array([0.5]), beta=jnp.array([1.0, 1, 1]), sigsq=1.0)
        )
        y_tilde = self.m0.simulate()
        post = self.m0.post(self.y)
        p_y_tilde = post.log_p_y_tilde(y_tilde)
        assert not jnp.isnan(p_y_tilde)

    def test_marg_post_phi(self):
        m0 = arx.ARX(
            T=100,
            phi_star=jnp.array([0.9]),
            sigsq_star=1.0,
            beta_star=jnp.array([1.0, 0.5, 0.5]),
            q=3,
            prng_key=jax.random.PRNGKey(123),
        )  # type: ignore
        y = m0.simulate()
        m1 = m0.misspecify(p=1, q=2)
        m1post = m1.post(y)
        # if we get this far the integration worked, but is it valid density?
        p = quad(m1post.marg_post_phi, -1, 1)[0]
        assert jnp.abs(p - 1.0) < 1e-5

    def test_ols(self):
        m0 = arx.ARX(
            T=100,
            phi_star=jnp.array([0.9]),
            sigsq_star=1.0,
            beta_star=jnp.array([1.0, 0.5, 0.5]),
            q=3,
            prng_key=jax.random.PRNGKey(123),
        )  # type: ignore
        y = m0.simulate()
        beta_hat = m0.ols(y)
        assert_shape(beta_hat, (4,))
        assert_tree_all_finite(beta_hat)

    def test_post(self):
        m0 = arx.ARX(
            T=100,
            phi_star=jnp.array([0.9]),
            sigsq_star=1.0,
            beta_star=jnp.array([1.0, 0.5, 0.5]),
            q=3,
            prng_key=jax.random.PRNGKey(123),
        )  # type: ignore
        y = m0.simulate()
        post = m0.post(y)
        y_tilde = m0.simulate()
        assert not jnp.isnan(post.log_p_y_tilde(y_tilde))


class TestPred(unittest.TestCase):

    # joint and pointwise should be the same when phi = 0 exactly, although
    # the model doesn't know that's the dgp parameter, so we'll specify a
    # strong prior concentrated around phi = 0 and hopefully the two results
    # are relatively similar
    def test_j_pw_indep(self):
        m0 = arx.ARX(
            T=100,
            phi_star=jnp.array([0.0]),
            sigsq_star=1.0,
            beta_star=jnp.array([1.0, 0.5, 0.5]),
            q=3,
            prng_key=jax.random.PRNGKey(123),
            a_sigsq0=100,
            b_sigsq0=100,
        )  # type: ignore
        y = m0.simulate()
        post = m0.post(y)
        y_tilde = m0.simulate()
        p_j = post.log_p_pred_joint(y_tilde)
        p_pw = post.log_p_pred_pointwise(y_tilde)
        assert not jnp.isnan(p_j)
        assert not jnp.isnan(p_pw)
        assert jnp.abs((p_j - p_pw) / p_pw) < 0.01

    # everything should still work when phi is close to 1
    def test_j_pw_corr(self):
        m0 = arx.ARX(
            T=100,
            phi_star=jnp.array([0.95]),
            sigsq_star=1.0,
            beta_star=jnp.array([1.0, 0.5, 0.5]),
            q=3,
            prng_key=jax.random.PRNGKey(123),
        )  # type: ignore
        y = m0.simulate()
        post = m0.post(y)
        y_tilde = m0.simulate()
        p_j = post.log_p_pred_joint(y_tilde)
        p_pw = post.log_p_pred_pointwise(y_tilde)
        assert not jnp.isnan(p_j)
        assert not jnp.isnan(p_pw)

    def test_fast_fd_predictive(self):
        m0 = arx.ARX(
            T=100,
            phi_star=jnp.array([0.95]),
            sigsq_star=1.0,
            beta_star=jnp.array([1.0, 0.5, 0.5]),
            q=3,
            prng_key=jax.random.PRNGKey(123),
        )  # type: ignore
        y = m0.simulate()
        post = m0.post(
            y
        )  # full data CV predictive, can compute p(y_tilde|y) the slow way
        y_tilde = m0.simulate()
        fdpost = arx.FullDataPosterior(m0, y)
        log_guess = -140.0
        fast_est = fdpost.log_p_y_tilde(y_tilde, log_guess)
        I_T = jnp.eye(m0.T)

        def joint_dens(phi):
            return jnp.exp(fdpost.log_y_tilde_phi(y_tilde, phi, I_T) - log_guess)

        integral = quad(joint_dens, -1, 1)[0]
        slow_est = jnp.log(integral) + log_guess
        assert jnp.abs((fast_est - slow_est) / slow_est) < 0.01


class TestPost(unittest.TestCase):
    def test_log_p_y(self):
        m0 = arx.ARX(
            T=100,
            phi_star=jnp.array([0.9]),
            sigsq_star=1.0,
            beta_star=jnp.array([1.0, 0.5, 0.5]),
            q=3,
            prng_key=jax.random.PRNGKey(123),
        )  # type: ignore
        y = m0.simulate()
        # check a few sensible values of phi
        for phi in [-0.995, -0.5, 0, 0.005, 0.5, 0.995]:
            log_p_y_phi = m0.log_p_y_phi(phi, y)
            assert jnp.isfinite(log_p_y_phi)
            assert log_p_y_phi <= 0
        log_p_y = m0.log_p_y(y)
        assert jnp.isfinite(log_p_y)
        assert log_p_y <= 0

    def test_fold_post(self):
        # actually a full posterior test, using CV fold machinery
        m0 = arx.ARX(
            T=100,
            phi_star=jnp.array([0.5]),
            sigsq_star=1.0,
            beta_star=jnp.array([1.0, 0.5, 0.5]),
            q=3,
            prng_key=jax.random.PRNGKey(123),
        )  # type: ignore
        y = m0.simulate()
        post = m0.post(y)
        full_fold_post = arx.CVPosterior(
            m=m0, y=y, S_test=jnp.eye(m0.T), S_train=jnp.eye(m0.T)
        )
        # check p(y, phi) a few sensible values of phi
        for phi in [-0.995, -0.5, 0, 0.005, 0.5, 0.995]:
            log_p_y_phi = full_fold_post.log_p_y_phi(phi)
            assert jnp.isfinite(log_p_y_phi)
            assert log_p_y_phi <= 0
        assert jnp.isfinite(full_fold_post.log_p_y) and jnp.isfinite(post.log_p_y)
        assert jnp.isclose(full_fold_post.log_p_y, post.log_p_y, atol=1e-5)

    def test_loo_fold(self):
        # one loo fold (not scrunch obvs)
        m0 = arx.ARX(
            T=100,
            phi_star=jnp.array([0.5]),
            sigsq_star=1.0,
            beta_star=jnp.array([1.0, 0.5, 0.5]),
            q=3,
            prng_key=jax.random.PRNGKey(123),
        )  # type: ignore
        y = m0.simulate()
        loo = cv.LOOCVScheme(T=m0.T)
        lf_post = arx.CVPosterior(
            m=m0, y=y, S_test=loo.S_test(37), S_train=loo.S_train(37)
        )
        assert_equal(lf_post.v, 1)
        assert_equal(lf_post.n, 99)
        assert lf_post.log_p_y < 0
        y_tilde = m0.simulate()
        lpdens = lf_post.log_p_y_tilde(y_tilde)
        lpdens_l = lf_post.log_p_y_tilde_laplace(y_tilde)
        assert lpdens < 0 and lpdens_l < 0
        # relative log difference <1%
        assert jnp.abs((lpdens_l - lpdens) / lpdens) < 1e-2

    def test_loo(self):
        m0 = arx.ARX(
            T=20,
            phi_star=jnp.array([0.5]),
            sigsq_star=1.0,
            beta_star=jnp.array([1.0, 0.5, 0.5]),
            q=3,
            prng_key=jax.random.PRNGKey(123),
        )  # type: ignore
        y = m0.simulate()
        ffold = cv.KFoldCVScheme(T=m0.T, k=5)
        eljpd = m0.cv(y=y, scheme=ffold)
        assert -45 < eljpd < -25  # expect about -35
        eljpd_l = m0.cv_laplace(y=y, scheme=ffold)
        assert -45 < eljpd < -25  # expect about -35
        # tolerate rel log difference < 1%
        assert jnp.abs((eljpd_l - eljpd) / eljpd) < 1e-2


class TestMCMCPosterior(unittest.TestCase):
    def test_mcmc_posterior(self):
        m0 = arx.ARX(
            T=100,
            phi_star=jnp.array([0.5]),
            sigsq_star=1.0,
            beta_star=jnp.array([1.0, 0.5, 0.5]),
            q=3,
            prng_key=jax.random.PRNGKey(123),
        )  # type: ignore
        y = m0.simulate()
        key = jax.random.PRNGKey(123)
        post = m0.fd_nuts_post(y, key, ndraws=200, nwarmup=200, nchains=1)
        self.assertIsInstance(post, arx.MCMCPosterior)


if __name__ == "__main__":
    unittest.main()
