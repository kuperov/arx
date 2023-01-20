
import chex
import jax.numpy as jnp


class CVScheme:
    """Generic CV scheme class

    Methods:
        name: name of the scheme suitable for plots and output
        n_folds: number of folds, always numbered from 0
        test_mask: boolean mask for test data for fold i
        train_mask: boolean mask for train data for fold i
        S_test: selection matrix for testing data for fold i
        S_train: selection matrix for training data for fold i
    """

    def __init__(self, T: int) -> None:
        super().__init__()
        self.T = T

    def name(self) -> str:
        raise NotImplementedError()

    def n_folds(self) -> int:
        raise NotImplementedError()

    def test_mask(self, i: int) -> chex.Array:
        raise NotImplementedError()

    def train_mask(self, i: int) -> chex.Array:
        raise NotImplementedError()

    def S_test(self, i: int) -> chex.Array:
        """Selection matrix for test data for this scheme

        Args:
            i:  fold number, 0-based
        """
        mask = self.test_mask(i)
        chex.assert_shape(mask, (self.T,))
        chex.assert_type(mask, jnp.bool_)
        S = jnp.diag(1.0 * mask)
        return S[mask, :]

    def S_train(self, i: int) -> chex.Array:
        """Selection matrix for training data for this scheme

        Args:
            i:  fold number, 0-based
        """
        mask = self.train_mask(i)
        chex.assert_shape(mask, (self.T,))
        chex.assert_type(mask, jnp.bool_)
        S = jnp.diag(1.0 * mask)
        return S[mask, :]


class LOOCVScheme(CVScheme):
    """Leave-one-out CV scheme"""

    def name(self) -> str:
        return "LOO"

    def n_folds(self) -> int:
        return self.T

    def train_mask(self, i: int) -> chex.Array:
        return jnp.arange(self.T) != i  # type: ignore

    def test_mask(self, i: int) -> chex.Array:
        return jnp.arange(self.T) == i  # type: ignore


class KFoldCVScheme(CVScheme):
    """K-fold CV scheme"""

    def __init__(self, T: int, k: int) -> None:
        self.k = k
        self.block_size = T // k
        super().__init__(T)

    def name(self) -> str:
        return f"{self.k}-fold"

    def n_folds(self) -> int:
        return self.k

    def train_mask(self, i: int) -> chex.Array:
        return jnp.arange(self.T) // self.block_size != i

    def test_mask(self, i: int) -> chex.Array:
        return jnp.arange(self.T) // self.block_size == i


class PointwiseKFoldCVScheme(CVScheme):
    """Pointwise K-fold scheme.
    
    This is an (inefficient) scheme for evaluating K-fold pointwise. It's like
    LOO but it uses the training sets from K-fold.

    Best use with lengths T that are multiples of the block size
    """
    
    def __init__(self, T: int, k: int) -> None:
        self.k = k
        self.block_size = T // k
        super().__init__(T)

    def name(self) -> str:
        return f"Pointwise {self.k}-fold"
    
    def n_folds(self) -> int:
        return self.T

    def train_mask(self, t: int) -> chex.Array:
        # The k-fold training set: missing the whole block
        block = t // self.block_size
        return jnp.arange(self.T) // self.block_size != block
    
    def test_mask(self, t: int) -> chex.Array:
        # The LOO testing set: just one variate
        return jnp.arange(self.T) == t


class HVBlockCVScheme(CVScheme):
    """H-block and HV-block CV schemes"""

    def __init__(self, T: int, h: int, v: int) -> None:
        super().__init__(T)
        self.h = h
        self.v = v

    @classmethod
    def from_delta(cls, T: int, delta: float) -> "HVBlockCVScheme":
        """Create HV-block scheme from delta hyperparameter"""
        h = jnp.floor(T**delta)
        v = jnp.floor(min(0.1 * T, (T - T ^ delta - 2 * h - 1) / 2))
        return cls(T, h, v)

    def name(self) -> str:
        return f"hv-block (h={self.h}, v={self.v})" if self.v > 0 else "h-block(h={self.h})"

    def n_folds(self) -> int:
        return self.T

    def train_mask(self, i: int) -> chex.Array:
        idxs = jnp.arange(self.T)
        return jnp.logical_or(
            idxs < i - self.v - self.h,
            idxs > i + self.v + self.h)

    def test_mask(self, i: int) -> chex.Array:
        idxs = jnp.arange(self.T)
        return jnp.logical_and(
            idxs >= i - self.v,
            idxs <= i + self.v)


class LFOCVScheme(CVScheme):
    """LFO CV scheme

    Attrs:
        T: number of time points
        h: size of the halo
        v: size of the validation block
        w: size of the initial margin
    """
    def __init__(self, T: int, h: int, v: int, w: int) -> None:
        super().__init__(T)
        self.h = h
        self.v = v
        self.w = w

    def name(self) -> str:
        return f"LFO (h={self.h}, v={self.v}, w={self.w})"

    def n_folds(self) -> int:
        return self.T - self.w

    def train_mask(self, i: int) -> chex.Array:
        idxs = jnp.arange(self.T)
        return (idxs < i + self.w - self.v - self.h)

    def test_mask(self, i: int) -> chex.Array:
        idxs = jnp.arange(self.T)
        return jnp.logical_and(
            idxs >= i + self.w - self.v,
            idxs <= i + self.w + self.v)
