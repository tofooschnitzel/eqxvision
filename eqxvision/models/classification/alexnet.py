from typing import Any, Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array

from ...utils import load_torch_weights


class AlexNet(eqx.Module):
    """A simple port of `torchvision.models.alexnet`"""

    features: eqx.Module
    avgpool: eqx.Module
    classifier: eqx.Module

    def __init__(
        self,
        num_classes: int = 1000,
        dropout: float = 0.5,
        *,
        key: Optional["jax.random.PRNGKey"] = None
    ) -> None:
        """**Arguments:**

        - `num_classes`: Number of classes in the classification task.
                        Also controls the final output shape `(num_classes,)`. Defaults to `1000`
        - `dropout`: Parameter used for the `equinox.nn.Dropout` layers. Defaults to `0.5`
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        """
        super().__init__()
        if key is None:
            key = jrandom.PRNGKey(0)
        keys = jrandom.split(key, 8)

        self.features = nn.Sequential(
            [
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, key=keys[0]),
                nn.Lambda(jnn.relu),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2, key=keys[1]),
                nn.Lambda(jnn.relu),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1, key=keys[2]),
                nn.Lambda(jnn.relu),
                nn.Conv2d(384, 256, kernel_size=3, padding=1, key=keys[3]),
                nn.Lambda(jnn.relu),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, key=keys[4]),
                nn.Lambda(jnn.relu),
                nn.MaxPool2d(kernel_size=3, stride=2),
            ]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            [
                nn.Dropout(p=dropout),
                nn.Linear(256 * 6 * 6, 4096, key=keys[5]),
                nn.Lambda(jnn.relu),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 4096, key=keys[6]),
                nn.Lambda(jnn.relu),
                nn.Linear(4096, num_classes, key=keys[7]),
            ]
        )

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array with `3` channels
        - `key`: Required parameter
        """
        if key is None:
            raise RuntimeError("The model requires a PRNGKey.")
        keys = jrandom.split(key, 2)
        x = self.features(x, key=keys[0])
        x = self.avgpool(x)
        x = jnp.ravel(x)
        x = self.classifier(x, key=keys[1])
        return x

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value


def alexnet(torch_weights: str = None, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    [One weird trick...](https://arxiv.org/abs/1404.5997)` paper.
    The required minimum input size of the model is 63x63.
    **Arguments:**

    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    """
    model = AlexNet(**kwargs)
    if torch_weights:
        model = load_torch_weights(model, torch_weights=torch_weights)
    return model
