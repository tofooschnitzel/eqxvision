from typing import Any, Tuple

import jax.random as jr 
import equinox as eqx 
import jax 
from jaxtyping import Array

import eqxvision as eqv
from eqxvision.utils import CLASSIFICATION_URLS


def dummy_img_data(
        num_channels: int = 1, 
        img_size: int = 224,
        batches: int = 1,
        key: "jax.random.PRNGKey" = jr.PRNGKey(0)
        ) -> Array:
    return jr.normal(key=key, shape=(batches, num_channels, img_size, img_size)) 


def stateful_pretrained_model(
        num_channels: int = 3,
        num_classes: int = 10, 
        # key: "jax.random.PRNGKey" = jr.PRNGKey(0),
        ) -> Tuple[Any, Any]:
    model = eqv.models.vgg11_bn(torch_weights=CLASSIFICATION_URLS["vgg11_bn"])
    return model

    


def stateful_model(
        num_channels: int = 1,
        num_classes: int = 10, 
        pretrained: bool = False,
        key: "jax.random.PRNGKey" = jr.PRNGKey(0),
        ) -> Tuple[Any, Any]:


    model, state = eqx.nn.make_with_state(eqv.models.resnet18)(
        num_channels=num_channels, 
        num_classes=num_classes, 
        key=key
        )
    batch_model = jax.vmap(
        model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
        )
    return batch_model, state


def model(
        num_channels: int = 1,
        num_classes: int = 10, 
        pretrained: bool = False,
        key: "jax.random.PRNGKey" = jr.PRNGKey(0),
        ) -> Any:
    weights = CLASSIFICATION_URLS['alexnet'] if pretrained else None
    return eqv.models.alexnet(torch_weights=weights)


if __name__ == "__main__":
    num_channels = 3
    key = jr.PRNGKey(0)
    data = dummy_img_data(num_channels=num_channels)

    # net = model(pretrained=True)
    # preds = net(data.squeeze(), key=key)

    # net, state = stateful_model(num_channels=num_channels)
    # preds, state = net(data, state)

    net = stateful_pretrained_model(num_classes=1000)

    print(data.shape)
    # print(preds.shape)
