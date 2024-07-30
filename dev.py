import jax.random as jr 
import equinox as eqx 
import jax 

from eqxvision.models.classification.resnet import resnet18



if __name__ == "__main__":
    key = jr.PRNGKey(0)
    key, mkey, dkey = jr.split(key, 3)

    num_channels = 1
    classes = 3
    img_size = 224

    x = jr.normal(key=dkey, shape=(1, num_channels, img_size, img_size))
    print(f"input shape: {x.shape}")
    
    model, state = eqx.nn.make_with_state(resnet18)(num_channels=num_channels, num_classes=classes, key=mkey)

    batch_model = jax.vmap(
        model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )

    preds, state = batch_model(x, state)
    print(f"ouput shape: {preds.shape}")