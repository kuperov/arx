import unittest

import cv
import jax.numpy as jnp
from sarx import *


class TestSARX(unittest.TestCase):
    def setUp(self) -> None:
        self.T = 100
        self.phi_star = jnp.array([0.4])
        self.sigsq_star = 1.5
        self.beta_star = jnp.array([1.0, 2.0, 0.5])
        self.Z = make_Z(q=3, T=self.T, prng_key=jax.random.PRNGKey(0))
        self.sarx = SARX(
            self.T,
            phi_star=self.phi_star,
            sigsq_star=self.sigsq_star,
            beta_star=self.beta_star,
            Z=self.Z,
            mu_beta0=jnp.zeros(3),
            sigma_beta0=jnp.eye(3),
        )
        return super().setUp()

    def testMisspecify(self):
        m1 = self.sarx.misspecify(q=2)
        self.assertEqual(m1.q, 2)
        self.assertEqual(m1.Z.shape, (self.T, 2))
        self.assertEqual(m1.mu_beta0.shape, (2,))

    def testOptSigsqPhi(self):
        # Won't be the same, but should be closeish
        phi_hat, sigsq_hat = opt_sigsq_phi(m=self.sarx, dgp=self.sarx)
        self.assertTrue(jnp.linalg.norm(phi_hat - self.sarx.phi_star) < 0.01)
        self.assertTrue(jnp.abs(sigsq_hat - self.sarx.sigsq_star) < 0.01)
        # Should be a little less close
        m1 = self.sarx.misspecify(q=2)
        phi_hat, sigsq_hat = opt_sigsq_phi(m=m1, dgp=self.sarx)
        phi_abserr = jnp.linalg.norm(phi_hat - self.sarx.phi_star)
        sigsq_abserr = jnp.abs(sigsq_hat - self.sarx.sigsq_star)
        self.assertTrue(phi_abserr < 0.5)
        self.assertTrue(sigsq_abserr < 1.0)

    def testSimulate(self):
        keys = jax.random.split(jax.random.PRNGKey(0), 5)
        y = self.sarx.simulate(keys[0])  # debugger entry point
        ys = jax.vmap(self.sarx.simulate)(keys)
        self.assertEqual(ys.shape, (5, 100))

    def testPolyArithmetic(self):
        A1 = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 100))
        A2 = jax.random.normal(jax.random.PRNGKey(1), shape=(100, 100))
        b1 = jax.random.normal(jax.random.PRNGKey(0), shape=(100,))
        b2 = jax.random.normal(jax.random.PRNGKey(1), shape=(100,))
        c1, c2 = jnp.array(3), jnp.array(4.0)
        p1 = Poly(A1, b1, c1, self.sarx)
        p2 = Poly(A2, b2, c2, self.sarx)
        p3 = p1 + p2
        self.assertTrue(jnp.allclose(p3.A, A1 + A2))
        self.assertTrue(jnp.allclose(p3.b, b1 + b2))
        self.assertTrue(jnp.allclose(p3.c, c1 + c2))
        p4 = p1 - p2
        self.assertTrue(jnp.allclose(p4.A, A1 - A2))
        self.assertTrue(jnp.allclose(p4.b, b1 - b2))
        self.assertTrue(jnp.allclose(p4.c, c1 - c2))

    def testPolyMoments(self):
        """Check polynomial moments by simulation"""
        p = self.sarx.eljpd(jnp.eye(self.sarx.T))
        A, b, c = p.A, p.b, p.c
        # theoretical values
        m, sd = p.mean(), jnp.sqrt(p.var())
        # check with simulation
        keys = jax.random.split(jax.random.PRNGKey(0), 1000)
        ys = jax.vmap(self.sarx.simulate)(keys)
        vals = jax.vmap(lambda y: y.T @ A @ y + y.T @ b + c)(ys)
        mhat, sdhat = jnp.mean(vals), jnp.std(vals)
        # 5% rel error should be doable
        self.assertTrue(jnp.abs((m - mhat) / m) < 0.01)
        self.assertTrue(jnp.abs((sd - sdhat) / sd) < 0.05)
        # check class's simulation function
        mhat2, sdhat2 = p.sim_moments(1000, jax.random.PRNGKey(0))
        # 5% rel error should be doable
        self.assertTrue(jnp.abs((m - mhat2) / m) < 0.01)
        self.assertTrue(jnp.abs((sd - sdhat2) / sd) < 0.05)

    def testEljpd(self):
        # full-data eljpd
        elpd_p = self.sarx.eljpd(jnp.eye(self.sarx.T))
        self.assertIsInstance(elpd_p, Poly)
        elpd = elpd_p.mean()
        # rough eljpd range to expect
        self.assertTrue(-180.0 < elpd < -120.0)
        # eljpd from simulation
        reps = 1000
        post_key, rep_key = jax.random.split(jax.random.PRNGKey(0))
        ys = jax.vmap(self.sarx.simulate)(jax.random.split(post_key, reps))
        ytildes = jax.vmap(self.sarx.simulate)(jax.random.split(rep_key, reps))

        def rep_contrib(y, ytilde):
            # posterior predictive distribution for a single replicate
            post = self.sarx.full_data_post(y)
            return post.log_pred_dens(ytilde)

        elpd_contribs = jax.vmap(rep_contrib)(ys, ytildes)
        elpd_hat = jnp.mean(elpd_contribs)
        self.assertTrue(jnp.abs((elpd_hat - elpd) / elpd) < 0.001)

    def testAnalyticalEljpd(self):
        # test full-data eljpd conditional on a single y,
        # this time for a misspecified model vs true dgp
        m1 = self.sarx.misspecify(p=1, q=2)
        elpd_p = m1.eljpd(jnp.eye(self.sarx.T))
        self.assertIsInstance(elpd_p, Poly)
        elpd = elpd_p.mean()
        # rough eljpd range to expect
        self.assertTrue(-180.0 < elpd < -120.0)
        # eljpd from simulation
        reps = 500
        ys = jax.vmap(self.sarx.simulate)(jax.random.split(jax.random.PRNGKey(0), reps))

        def rep_eljpd(y):
            post = m1.full_data_post(y)
            return post.eljpd(self.sarx)

        elpd_contribs = jax.vmap(rep_eljpd)(ys)
        elpd_hat = jnp.mean(elpd_contribs)
        self.assertTrue(jnp.abs((elpd_hat - elpd) / elpd) < 0.001)

    def testCV(self):
        # smoke tests for CV functions
        m = self.sarx.misspecify(q=2)
        loo = cv.LOOCVScheme(self.sarx.T)
        elpdhat_loo = m.eljpdhat_cv(loo)
        self.assertIsInstance(elpdhat_loo, Poly)
        mean = elpdhat_loo.mean()
        std = elpdhat_loo.std()
        # we'll check values by simulation in the next test
        self.assertTrue(jnp.isfinite(mean), jnp.isfinite(std))

    def testCVMomentsBySimulation(self):
        # scheme = cv.LOOCVScheme(self.sarx.T)
        scheme = cv.HVBlockCVScheme(self.sarx.T, 5, 10)
        cvp = self.sarx.eljpdhat_cv(scheme)
        cvp_mean = cvp.mean()
        # check CV algo by simulation
        reps = 500
        N = scheme.n_folds()
        # we just generate one set of ys, there's no independent test data
        ykeys = jax.random.split(jax.random.PRNGKey(1), reps)
        ys = jax.vmap(self.sarx.simulate)(ykeys)
        contribs = []
        for i in range(N):
            # get training and test data
            Strain, Stest = scheme.S_train(i), scheme.S_test(i)
            v = jnp.sum(Stest)
            # compute benchmark for each replicate
            def rep_contrib(y, ytilde):
                post = self.sarx.post(y, Strain)
                return post.log_pred_dens_subset(ytilde, Stest) / v

            elpd_contribs = jax.vmap(rep_contrib)(ys, ys)
            elpd_contrib = jnp.mean(elpd_contribs)
            contribs.append(elpd_contrib)
        norm_contribs = jnp.array(contribs) * self.sarx.T / N
        cv_mean_hat = jnp.sum(norm_contribs)
        self.assertTrue(
            jnp.abs((cvp_mean - cv_mean_hat) / cvp_mean) < 0.01,
            f"{cvp_mean} != {cv_mean_hat}",
        )

    def testJointBenchmarkDummy(self):
        # most basic test for benchmark imaginable: 1 fold, full data, full test
        class DummyCVScheme(cv.CVScheme):
            def __init__(self, T):
                self.T = T
                self.n = 1
                self.indices = [(0, T)]

            def n_folds(self) -> int:
                return 1

            def test_mask(self, i: int) -> chex.Array:
                return jnp.ones(self.T, dtype=bool)

            def train_mask(self, i: int) -> chex.Array:
                return jnp.ones(self.T, dtype=bool)

        dummy = DummyCVScheme(self.T)
        bm = self.sarx.eljpd_cv_benchmark(dummy)
        e = self.sarx.eljpd(jnp.eye(self.T))
        self.assertAlmostEqual(bm.mean(), e.mean())
        self.assertAlmostEqual(bm.std(), e.std())

    def testBenchmarkCVMeasureBySimulation(self):
        # scheme = cv.LOOCVScheme(self.sarx.T)
        scheme = cv.HVBlockCVScheme(self.sarx.T, 5, 10)
        elpd_bm = self.sarx.eljpd_cv_benchmark(scheme)
        m_bm = elpd_bm.mean()
        # check benchmark by simulation
        reps = 400
        N = scheme.n_folds()
        ykeys = jax.random.split(jax.random.PRNGKey(1), reps)
        ytildekeys = jax.random.split(jax.random.PRNGKey(2), reps)
        ys = jax.vmap(self.sarx.simulate)(ykeys)
        ytildes = jax.vmap(self.sarx.simulate)(ytildekeys)
        contribs = []
        for i in range(N):
            # get training and test data
            Strain, Stest = scheme.S_train(i), scheme.S_test(i)
            v = jnp.sum(Stest)
            # compute benchmark for each replicate
            def rep_contrib(y, ytilde):
                post = self.sarx.post(y, Strain)
                return post.log_pred_dens_subset(ytilde, Stest) / v

            elpd_contribs = jax.vmap(rep_contrib)(ys, ytildes)
            elpd_contrib = jnp.mean(elpd_contribs)
            contribs.append(elpd_contrib)
        norm_contribs = jnp.array(contribs) * self.sarx.T / N
        m_bm_hat = jnp.sum(norm_contribs)
        self.assertTrue(
            jnp.abs((m_bm - m_bm_hat) / m_bm) < 0.01, f"{m_bm} != {m_bm_hat}"
        )


class TestSupportFunctions(unittest.TestCase):
    def test_make_Z(self):
        Z = make_Z(q=3, T=100, prng_key=jax.random.PRNGKey(0))
        self.assertEqual(Z.shape, (100, 3))
        Z = make_Z(q=1, T=100, prng_key=jax.random.PRNGKey(0))
        self.assertEqual(Z.shape, (100, 1))
