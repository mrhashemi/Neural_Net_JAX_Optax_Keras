# 
# A simple neural network using JAX with optax optimizer and a custom defined loss function.
#
# This program is licensed under MIT
#
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable, Dict, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class TrainingHistory:
    """
    Stores the history of training, including loss and additional performance metrics.

    Attributes:
        losses (List[float]): Stores loss values over epochs.
        metrics (Dict[str, List[float]]): Tracks additional metrics (MSE, MAE, R²).
    """
    losses: List[float]
    metrics: Dict[str, List[float]]

    def add(self, loss: float, additional_metrics: Optional[Dict[str, float]] = None):
        """
        Append a loss value and optionally additional metrics.

        Args:
            loss (float): Computed loss for the current epoch.
            additional_metrics (Optional[Dict[str, float]]): Dictionary of additional metrics.
        """
        self.losses.append(float(loss))
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.metrics.setdefault(key, []).append(float(value))


class JAXMLPRegressor:
    """
    A multi-layer perceptron (MLP) regressor using JAX, Equinox, and Optax.

    Features:
    - Supports multiple activation functions.
    - Allows different optimizers (SGD, Adam).
    - Tracks training history with metrics like MSE, MAE, and R².
    - Uses cosine similarity and mean absolute error (MAE) as a combined loss function.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.01,
        activation: Union[str, Callable] = "sigmoid",
        optimizer_type: str = "sgd",
        random_seed: int = 42,
        track_history: bool = True,
    ):
        """
        Initialize the neural network.

        Args:
            layer_sizes (List[int]): Number of neurons per layer.
            learning_rate (float): Learning rate for the optimizer.
            activation (Union[str, Callable]): Activation function 
                (Options: "sigmoid", "relu", "tanh", "leaky_relu", "elu", "softplus", 
                "gelu", "swish", "selu", "softmax", "log_sigmoid", "log_softmax").
            optimizer_type (str): Optimization algorithm ("sgd" or "adam").
            random_seed (int): Random seed for reproducibility.
            track_history (bool): Whether to store training loss and metrics.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.track_history = track_history
        self.key = jax.random.PRNGKey(random_seed)

        # Set activation function
        self.activation = {
            "sigmoid": jax.nn.sigmoid,
            "relu": jax.nn.relu,
            "tanh": jax.nn.tanh,
            "leaky_relu": jax.nn.leaky_relu,
            "elu": jax.nn.elu,
            "softplus": jax.nn.softplus,
            "gelu": jax.nn.gelu,
            "swish": jax.nn.swish,
            "selu": jax.nn.selu,
            "softmax": jax.nn.softmax,
            "log_sigmoid": jax.nn.log_sigmoid,
            "log_softmax": jax.nn.log_softmax
        }.get(activation, activation) if isinstance(activation, str) else activation

        # Initialize the model
        self.model = self._initialize_model()

        # Set optimizer
        self.optimizer = {
            "sgd": optax.sgd(learning_rate),
            "adam": optax.adam(learning_rate),
        }[optimizer_type]

        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        self.history = TrainingHistory([], {}) if track_history else None

    def _initialize_model(self) -> eqx.Module:
        """Creates and returns the MLP model using Equinox."""
        class MLP(eqx.Module):
            layers: List[eqx.nn.Linear]
            activation: Callable

            def __init__(self, layer_sizes: List[int], key: jax.random.PRNGKey, activation: Callable):
                self.activation = activation
                self.layers = []
                
                for i, (input_dim, output_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                    key, subkey = jax.random.split(key)
                    self.layers.append(eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=subkey))

            def __call__(self, x):
                for layer in self.layers[:-1]:
                    x = self.activation(layer(x))
                return self.layers[-1](x)  # No activation on the last layer

        return MLP(self.layer_sizes, self.key, self.activation)

    def _compute_metrics(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> Dict[str, jnp.float16]:
        """Compute Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² metrics."""
        mse = jnp.mean((y_true - y_pred) ** 2)
        mae = jnp.mean(jnp.abs(y_true - y_pred))
        r2 = 1 - jnp.sum(jnp.square(y_true - y_pred)) / jnp.sum(jnp.square(y_true - jnp.mean(y_true)))
        return {"mse": jnp.float16(mse), "mae": jnp.float16(mae), "r2": jnp.float16(r2)}

    @eqx.filter_jit
    def _perform_training_step(
        self, model: eqx.Module, opt_state: optax.OptState, x: jnp.ndarray, y: jnp.ndarray
    ) -> Tuple[eqx.Module, optax.OptState, float, Dict[str, float]]:
        """Executes a single training step using cosine similarity + MAE loss."""
        def loss_fn(model):
            y_pred = jax.vmap(model)(x)

            # Cosine Similarity Loss
            dot_product = jnp.sum(y_pred * y, axis=1)
            norm_preds = jnp.linalg.norm(y_pred, axis=1)
            norm_y = jnp.linalg.norm(y, axis=1)
            cosine_similarity = dot_product / (norm_preds * norm_y + 1e-8)
            cosine_loss = 1 - jnp.mean(cosine_similarity)

            # Mean Absolute Error (MAE) Loss
            mae_loss = jnp.mean(jnp.abs(y_pred - y))

            # Combined Loss
            combined_loss = 0.5 * cosine_loss + 0.5 * mae_loss
            return combined_loss, y_pred

        (loss, y_pred), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, opt_state = self.optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        metrics = self._compute_metrics(y, y_pred)
        return model, opt_state, loss, metrics

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """Generates predictions for input data."""
        return jax.vmap(self.model)(x)

    def train(self, x: jnp.ndarray, y: jnp.ndarray, epochs: int = 1000, verbose: bool = True):
        """
        Trains the model.

        Args:
            x (jnp.ndarray): Input training data.
            y (jnp.ndarray): Target labels.
            epochs (int): Number of epochs to train.
            verbose (bool): Whether to print training progress.
        """
        for epoch in range(epochs):
            self.model, self.opt_state, loss, metrics = self._perform_training_step(self.model, self.opt_state, x, y)
            if self.track_history:
                self.history.add(loss, metrics)
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}, MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}")

