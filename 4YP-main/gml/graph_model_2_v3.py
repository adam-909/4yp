"""
LSTM-GCN model matching the paper's Section 5.1 exactly.

This module implements the architecture described in the paper:
- Shared LSTM across N tickers
- GCN layers with final layer outputting 1 dimension
- Single TimeDistributed dense with tanh (no residual connection)

Key differences from v2:
- No residual connection
- Final GCN layer outputs 1 dimension (not gcn_units)
- Single TimeDistributed layer (not nested)
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers


class GraphSharpeLoss(tf.keras.losses.Loss):
    """Custom loss function that maximizes Sharpe ratio."""

    def __init__(self, output_size: int = 1):
        self.output_size = output_size
        super().__init__()

    def call(self, y_true, weights):
        """
        Compute negative Sharpe ratio as loss.

        Args:
            y_true: True returns, shape (batch_size, N, time_steps, 1)
            weights: Predicted positions, shape (batch_size, N, time_steps, 1)

        Returns:
            Negative annualized Sharpe ratio
        """
        y_true = tf.reshape(y_true, [-1])
        weights = tf.reshape(weights, [-1])

        captured_returns = weights * y_true
        mean_returns = tf.reduce_mean(captured_returns)

        return -(
            mean_returns
            / tf.sqrt(
                tf.reduce_mean(tf.square(captured_returns))
                - tf.square(mean_returns)
                + 1e-9
            )
            * tf.sqrt(252.0)
        )


class GraphConvolution(layers.Layer):
    """
    Graph Convolution layer with static adjacency matrix.

    Implements: H^(k+1) = σ(A * H^(k) * W^(k) + b^(k))

    Args:
        units: Output dimension
        adjacency: Normalized adjacency matrix of shape (num_nodes, num_nodes)
    """

    def __init__(self, units, adjacency, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.adjacency = tf.constant(adjacency, dtype=tf.float32)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.weight = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="gcn_weight",
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="gcn_bias",
        )
        super(GraphConvolution, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, num_stocks, input_dim)
        A_expanded = tf.expand_dims(self.adjacency, 0)  # (1, num_stocks, num_stocks)
        Ax = tf.matmul(A_expanded, inputs)  # (batch, time_steps, num_stocks, input_dim)
        output = tf.matmul(Ax, self.weight) + self.bias
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_lstm_gcn_model(
    num_tickers: int,
    time_steps: int,
    input_size: int,
    adjacency: np.ndarray,
    hidden_layer_size: int = 10,
    gcn_units: int = 16,
    dropout_rate: float = 0.4,
    learning_rate: float = 0.001,
    max_gradient_norm: float = 0.01,
    num_gcn_layers: int = 2,
) -> keras.Model:
    """
    Build LSTM-GCN model matching the paper's Section 5.1 exactly.

    Architecture (from paper):
        1. Input X ∈ R^(δ×F×N)
        2. N shared-parameter LSTM modules → H^(0) ∈ R^(δ×H1)
        3. L GCN layers: H^(k+1) = σ(A·H^(k)·W^(k) + b^(k))
        4. Final GCN outputs Z = H^(L) ∈ R^(δ×1)
        5. TimeDistributed dense with tanh → Y ∈ R^(δ×1×N)

    Key differences from v2:
        - No residual connection
        - Final GCN layer outputs 1 dimension
        - Single TimeDistributed (not nested)

    Args:
        num_tickers: Number of stocks/nodes in the graph (N=88)
        time_steps: Number of time steps per sample (δ)
        input_size: Number of input features per time step (F)
        adjacency: Precomputed adjacency matrix of shape (num_tickers, num_tickers)
        hidden_layer_size: LSTM hidden units H1 (default: 10)
        gcn_units: GCN hidden dimension for intermediate layers (default: 16)
        dropout_rate: LSTM dropout rate (default: 0.4)
        learning_rate: Adam learning rate (default: 0.001)
        max_gradient_norm: Gradient clipping norm (default: 0.01)
        num_gcn_layers: Number of GCN layers L (default: 2)

    Returns:
        Compiled Keras model
    """
    # Input layer: (batch_size, num_tickers, time_steps, input_size)
    input_layer = keras.Input(shape=(num_tickers, time_steps, input_size))

    # Shared LSTM layer (all N modules share parameters)
    shared_lstm = layers.LSTM(
        hidden_layer_size,
        return_sequences=True,
        dropout=dropout_rate,
        activation="tanh",
        recurrent_activation="sigmoid",
        name="shared_lstm"
    )

    # Process each ticker through shared LSTM
    # Paper: "N LSTM modules, one for each underlying equity... all share common parameters"
    lstm_outputs = []
    for i in range(num_tickers):
        # Slice: (batch_size, time_steps, input_size) for ticker i
        ticker_slice = layers.Lambda(lambda x, idx=i: x[:, idx, :, :])(input_layer)
        ticker_output = shared_lstm(ticker_slice)
        lstm_outputs.append(ticker_output)

    # Stack outputs: (batch_size, time_steps, num_tickers, hidden_layer_size)
    # This is H^(0) ∈ R^(δ×H1) for each of N nodes
    stacked_lstm = layers.Lambda(lambda tensors: tf.stack(tensors, axis=2))(lstm_outputs)

    # Apply GCN layers: H^(k+1) = σ(A·H^(k)·W^(k) + b^(k))
    x = stacked_lstm

    for k in range(num_gcn_layers - 1):
        # Intermediate GCN layers use gcn_units dimensions
        x = GraphConvolution(units=gcn_units, adjacency=adjacency, name=f"gcn_{k}")(x)
        x = layers.ReLU()(x)

    # Final GCN layer outputs 1 dimension: Z = H^(L) ∈ R^(δ×1)
    # Paper: "The output H^(L) ≡ Z ∈ R^(δ×1)"
    z = GraphConvolution(units=1, adjacency=adjacency, name=f"gcn_{num_gcn_layers-1}")(x)
    # Note: No activation here - it's applied in the TimeDistributed layer

    # TimeDistributed dense with tanh: Y ∈ R^(δ×1×N)
    # Paper: "fed into a TimeDistributed dense layer with a tanh non-linearity"
    # Since z is already 1-dimensional, this is essentially tanh(w*z + b) per time step
    # We apply across the time dimension (axis 1)
    output = layers.TimeDistributed(
        layers.Dense(1, activation="tanh", kernel_constraint=keras.constraints.max_norm(3)),
        name="time_distributed_output"
    )(z)

    # Permute to match label shape: (batch_size, num_tickers, time_steps, 1)
    output = layers.Permute((2, 1, 3))(output)

    # Create and compile model
    model = keras.Model(inputs=input_layer, outputs=output)

    adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
    sharpe_loss = GraphSharpeLoss(output_size=1).call

    model.compile(
        loss=sharpe_loss,
        optimizer=adam,
    )

    return model
