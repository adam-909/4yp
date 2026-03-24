"""
Clean LSTM-GCN model for static graph training.

This module provides a simple, keras-tuner-free implementation of LSTM-GCN
for training with static (precomputed) adjacency matrices.
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

    def call(self, y_true, y_pred):
        """
        Compute negative Sharpe ratio as loss.

        Args:
            y_true: True returns, shape (batch_size, N, time_steps, 1)
            y_pred: Predicted positions, shape (batch_size, N, time_steps, 1)

        Returns:
            Negative annualized Sharpe ratio
        """
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        captured_returns = y_pred * y_true
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

    Performs: Z = A_hat * X * W + b

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
    Build LSTM-GCN model with static adjacency matrix.

    Architecture:
        1. Shared LSTM processes each ticker's time series
        2. GCN aggregates information across tickers using the graph
        3. Residual connection from LSTM to output
        4. Dense layer produces position output in [-1, 1]

    Args:
        num_tickers: Number of stocks/nodes in the graph
        time_steps: Number of time steps per sample
        input_size: Number of input features per time step
        adjacency: Precomputed adjacency matrix of shape (num_tickers, num_tickers)
        hidden_layer_size: LSTM hidden units (default: 10)
        gcn_units: GCN output dimension (default: 16)
        dropout_rate: LSTM dropout rate (default: 0.4)
        learning_rate: Adam learning rate (default: 0.001)
        max_gradient_norm: Gradient clipping norm (default: 0.01)
        num_gcn_layers: Number of GCN layers (default: 2)

    Returns:
        Compiled Keras model
    """
    # Input layer: (batch_size, num_tickers, time_steps, input_size)
    input_layer = keras.Input(shape=(num_tickers, time_steps, input_size))

    # Shared LSTM layer
    shared_lstm = layers.LSTM(
        hidden_layer_size,
        return_sequences=True,
        dropout=dropout_rate,
        activation="tanh",
        recurrent_activation="sigmoid",
        name="shared_lstm"
    )

    # Process each ticker through shared LSTM
    lstm_outputs = []
    for i in range(num_tickers):
        # Slice: (batch_size, time_steps, input_size) for ticker i
        ticker_slice = layers.Lambda(lambda x, idx=i: x[:, idx, :, :])(input_layer)
        ticker_output = shared_lstm(ticker_slice)
        lstm_outputs.append(ticker_output)

    # Stack outputs: (batch_size, time_steps, num_tickers, hidden_layer_size)
    stacked_lstm = layers.Lambda(lambda tensors: tf.stack(tensors, axis=2))(lstm_outputs)

    # Apply GCN layers
    gcn_output = GraphConvolution(units=gcn_units, adjacency=adjacency)(stacked_lstm)
    gcn_output = layers.ReLU()(gcn_output)

    if num_gcn_layers == 2:
        gcn_output = GraphConvolution(units=gcn_units, adjacency=adjacency)(gcn_output)
        gcn_output = layers.ReLU()(gcn_output)

    # Residual connection
    residual = layers.TimeDistributed(
        layers.TimeDistributed(
            keras.layers.Dense(gcn_units, activation="linear")
        )
    )(stacked_lstm)

    x = layers.Add()([gcn_output, residual])

    # Output layer
    output = layers.TimeDistributed(
        layers.TimeDistributed(
            keras.layers.Dense(
                1,
                activation=tf.nn.tanh,
                kernel_constraint=keras.constraints.max_norm(3),
            )
        )
    )(x)

    # Permute to match label shape: (batch_size, num_tickers, time_steps, 1)
    output = layers.Permute((2, 1, 3))(output)

    # Create and compile model
    model = keras.Model(inputs=input_layer, outputs=output)

    adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
    sharpe_loss = GraphSharpeLoss(output_size=1)

    model.compile(
        loss=sharpe_loss,
        optimizer=adam,
    )

    return model


def load_adjacency_matrix(graph_type: str, alpha: float = 0.5, beta: float = 0.5,
                          tau: float = 0.45, tickers: list = None) -> np.ndarray:
    """
    Load a precomputed adjacency matrix.

    Args:
        graph_type: Either "cvx" or "pearson"
        alpha: CVX optimization alpha parameter (only for cvx type)
        beta: CVX optimization beta parameter (only for cvx type)
        tau: Correlation threshold (only for pearson type)
        tickers: List of tickers to reindex the matrix (optional)

    Returns:
        Adjacency matrix as numpy array
    """
    import os

    if graph_type == "cvx":
        graph_file = os.path.join("data", "graph_structure", "cvx_opt", f"{alpha}_{beta}_cvx.csv")
    elif graph_type == "pearson":
        graph_file = os.path.join("data", "graph_structure", "pearson", f"{tau}.csv")
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}. Use 'cvx' or 'pearson'.")

    adjacency_df = pd.read_csv(graph_file, index_col=0)

    if tickers is not None:
        adjacency_df = adjacency_df.reindex(index=tickers, columns=tickers)

    return adjacency_df.values
