import os
import unittest

import experiments as ex
import jax
import sarx_experiments as sx

import arx


class TestExperimentInstance(unittest.TestCase):
    def setUp(self) -> None:
        self.ex1 = ex.make_full_experiment(1, "hard", simplified=False)

    def test_experiment_instance(self) -> None:
        key = jax.random.PRNGKey(0)
        inst = self.ex1.make_instance(alpha=0.5, T=110, prng_key=key)
        self.assertEqual(inst.dgp.T, 110)
        self.assertIsInstance(inst.dgp, arx.ARX)
        self.assertIsInstance(inst.mA, arx.ARX)
        self.assertIsInstance(inst.mB, arx.ARX)
        for m in [inst.dgp, inst.mA, inst.mB]:
            self.assertEqual(m.T, 110)
        self.assertEqual(inst.dgp.q, 3)
        self.assertEqual(inst.mA.q, 2)
        self.assertEqual(inst.mB.q, 1)

    def test_full_bayes_experiment(self) -> None:
        # just a smoke test
        TGT = "/tmp/ex1C.csv"
        if os.path.isfile(TGT):
            os.remove(TGT)
        ex.full_bayes(
            1,
            "easy",
            n_posts=2,
            T=10,
            mc_reps=10,
            filename=TGT,
            seed=0,
            n_warmup=100,
            n_chains=2,
        )
        assert os.path.isfile(TGT)
        os.remove(TGT)


class TestSARXExperiments(unittest.TestCase):
    def testExcludedEffect(self):
        filename = "/tmp/excluded_effect.csv"
        if os.path.isfile(filename):
            os.remove(filename)
        sx.by_excluded_effect(filename=filename, ex_no=1, variant="easy", T=20, seed=0)
        assert os.path.isfile(filename)
        os.remove(filename)

    def testIncludedEffect(self):
        filename = "/tmp/included_effect.csv"
        if os.path.isfile(filename):
            os.remove(filename)
        sx.by_included_effect(filename=filename, ex_no=1, variant="easy", T=20, seed=0)
        assert os.path.isfile(filename)
        os.remove(filename)

    def testDataLength(self):
        filename = "/tmp/data_length.csv"
        if os.path.isfile(filename):
            os.remove(filename)
        sx.by_data_length(filename=filename, ex_no=1, variant="easy", T=20, seed=0)
        assert os.path.isfile(filename)
        os.remove(filename)

    def testByHalo(self):
        filename = "/tmp/by_halo.csv"
        if os.path.isfile(filename):
            os.remove(filename)
        sx.by_halo(filename=filename, ex_no=1, variant="easy", T=20, seed=0)
        assert os.path.isfile(filename)
        os.remove(filename)

    def testByDimension(self):
        filename = "/tmp/by_dimension.csv"
        if os.path.isfile(filename):
            os.remove(filename)
        sx.by_dimension(filename=filename, ex_no=1, variant="easy", T=20, seed=0)
        assert os.path.isfile(filename)
        os.remove(filename)
