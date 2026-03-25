"""
Clean LSTM-GAT models for end-to-end and graph-guided training.

Per-timestep model builders (GAT runs independently per timestep):
    build_lstm_gat_model: End-to-end full attention (no graph needed)
    build_lstm_gat_sparse_model: Static graph masks attention

Trajectory model builders (LSTM outputs concatenated, GAT runs once, predict day 20 only):
    build_lstm_gat_trajectory_model: End-to-end full attention
    build_lstm_gat_trajectory_sparse_model: Static graph masks attention

Layer:
    GraphAttentionLayer: Supports both full and sparse attention modes
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from gml.graph_model_2_v2 import GraphSharpeLoss


class GraphAttentionLayer(layers.Layer):
    """
    Graph Attention layer supporting full and sparse attention.

    Full attention (adjacency=None): every node attends to every other.
    Sparse attention (adjacency provided): only attend to neighbors.

    Args:
        units: Output feature dimension per head
        attn_heads: Number of attention heads
        adjacency: Optional adjacency matrix for sparse masking (N x N)
        concat_heads: If True, concatenate heads; if False, average them
        dropout_rate: Dropout on attention weights
    """

    def __init__(self, units, attn_heads=2, adjacency=None,
                 concat_heads=True, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.dropout_rate = dropout_rate

        if adjacency is not None:
            adj_with_self = adjacency + np.eye(adjacency.shape[0])
            self.attention_mask = tf.constant(adj_with_self, dtype=tf.float32)
        else:
            self.attention_mask = None

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.W = self.add_weight(
            shape=(self.attn_heads, input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="W",
        )
        self.a_src = self.add_weight(
            shape=(self.attn_heads, self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="a_src",
        )
        self.a_dst = self.add_weight(
            shape=(self.attn_heads, self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="a_dst",
        )
        self.dropout_layer = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs, training=None):
        # inputs: (batch, time_steps, num_nodes, features)
        input_shape = tf.shape(inputs)

        if len(inputs.shape) == 4:
            batch_size = input_shape[0]
            time_steps = input_shape[1]
            num_nodes = input_shape[2]

            inputs_3d = tf.reshape(inputs, [-1, num_nodes, inputs.shape[-1]])
            output = self._attention_forward(inputs_3d, training)

            output_dim = self.units * self.attn_heads if self.concat_heads else self.units
            output = tf.reshape(output, [batch_size, time_steps, num_nodes, output_dim])
        else:
            output = self._attention_forward(inputs, training)

        return output

    def _attention_forward(self, inputs, training):
        # inputs: (batch, nodes, features)
        outputs = []
        for head in range(self.attn_heads):
            h = tf.matmul(inputs, self.W[head])  # (batch, nodes, units)

            attn_src = tf.matmul(h, self.a_src[head])  # (batch, nodes, 1)
            attn_dst = tf.matmul(h, self.a_dst[head])  # (batch, nodes, 1)

            attn_scores = attn_src + tf.transpose(attn_dst, [0, 2, 1])
            attn_scores = tf.nn.leaky_relu(attn_scores, alpha=0.2)

            # Sparse masking: mask non-neighbors before softmax
            if self.attention_mask is not None:
                mask = tf.expand_dims(self.attention_mask, 0)
                attn_scores = tf.where(
                    mask > 0, attn_scores,
                    tf.ones_like(attn_scores) * (-1e9)
                )

            attn_weights = tf.nn.softmax(attn_scores, axis=-1)
            attn_weights = self.dropout_layer(attn_weights, training=training)

            h_prime = tf.matmul(attn_weights, h)
            outputs.append(h_prime)

        if self.concat_heads:
            return tf.concat(outputs, axis=-1)
        else:
            return tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "attn_heads": self.attn_heads,
            "concat_heads": self.concat_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_lstm_gat_model(
    num_tickers: int,
    time_steps: int,
    input_size: int,
    hidden_layer_size: int = 10,
    gat_units: int = 8,
    attn_heads: int = 2,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.0005,
    max_gradient_norm: float = 0.01,
    num_gat_layers: int = 2,
) -> keras.Model:
    """
    Build end-to-end LSTM-GAT model (no precomputed graph).

    Architecture:
        1. Shared LSTM processes each ticker's time series
        2. Full GAT layers learn inter-stock attention from data
        3. Residual connection from LSTM to output
        4. Dense layer produces position output in [-1, 1]

    Args:
        num_tickers: Number of stocks/nodes (N=88)
        time_steps: Number of time steps per sample
        input_size: Number of input features per time step
        hidden_layer_size: LSTM hidden units
        gat_units: GAT output dimension per head
        attn_heads: Number of attention heads
        dropout_rate: Dropout rate
        learning_rate: Adam learning rate
        max_gradient_norm: Gradient clipping norm
        num_gat_layers: Number of GAT layers

    Returns:
        Compiled Keras model
    """
    input_layer = keras.Input(shape=(num_tickers, time_steps, input_size))

    shared_lstm = layers.LSTM(
        hidden_layer_size,
        return_sequences=True,
        dropout=dropout_rate,
        activation="tanh",
        recurrent_activation="sigmoid",
        name="shared_lstm",
    )

    lstm_outputs = []
    for i in range(num_tickers):
        ticker_slice = layers.Lambda(lambda x, idx=i: x[:, idx, :, :])(input_layer)
        lstm_outputs.append(shared_lstm(ticker_slice))

    # Stack: (batch, time_steps, num_tickers, hidden_size)
    stacked_lstm = layers.Lambda(lambda t: tf.stack(t, axis=2))(lstm_outputs)

    # GAT layers (full attention)
    x = stacked_lstm
    for k in range(num_gat_layers):
        is_last = (k == num_gat_layers - 1)
        x = GraphAttentionLayer(
            units=gat_units,
            attn_heads=attn_heads,
            adjacency=None,
            concat_heads=not is_last,
            dropout_rate=dropout_rate,
            name=f"gat_{k}",
        )(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)

    gat_output_dim = gat_units  # last layer averages heads

    # Residual connection
    residual = layers.TimeDistributed(
        layers.TimeDistributed(
            layers.Dense(gat_output_dim, activation="linear")
        )
    )(stacked_lstm)

    x = layers.Add()([x, residual])

    # Output
    output = layers.TimeDistributed(
        layers.TimeDistributed(
            layers.Dense(
                1,
                activation=tf.nn.tanh,
                kernel_constraint=keras.constraints.max_norm(3),
            )
        )
    )(x)

    output = layers.Permute((2, 1, 3))(output)

    model = keras.Model(inputs=input_layer, outputs=output)
    adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
    model.compile(loss=GraphSharpeLoss(output_size=1), optimizer=adam)

    return model


# ---------------------------------------------------------------------------
# Per-timestep sparse model
# ---------------------------------------------------------------------------

def build_lstm_gat_sparse_model(
    num_tickers: int,
    time_steps: int,
    input_size: int,
    adjacency: np.ndarray,
    hidden_layer_size: int = 10,
    gat_units: int = 8,
    attn_heads: int = 2,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.0005,
    max_gradient_norm: float = 0.01,
    num_gat_layers: int = 2,
) -> keras.Model:
    """
    Build graph-guided LSTM-GAT model with static adjacency mask.

    Same architecture as end-to-end, but attention is masked by a
    precomputed graph. Non-neighbors get -1e9 before softmax so the
    model only attends to connected stocks.

    Args:
        num_tickers: Number of stocks/nodes (N=88)
        time_steps: Number of time steps per sample
        input_size: Number of input features per time step
        adjacency: Static adjacency matrix (num_tickers, num_tickers)
        hidden_layer_size: LSTM hidden units
        gat_units: GAT output dimension per head
        attn_heads: Number of attention heads
        dropout_rate: Dropout rate
        learning_rate: Adam learning rate
        max_gradient_norm: Gradient clipping norm
        num_gat_layers: Number of GAT layers

    Returns:
        Compiled Keras model
    """
    input_layer = keras.Input(shape=(num_tickers, time_steps, input_size))

    shared_lstm = layers.LSTM(
        hidden_layer_size,
        return_sequences=True,
        dropout=dropout_rate,
        activation="tanh",
        recurrent_activation="sigmoid",
        name="shared_lstm",
    )

    lstm_outputs = []
    for i in range(num_tickers):
        ticker_slice = layers.Lambda(lambda x, idx=i: x[:, idx, :, :])(input_layer)
        lstm_outputs.append(shared_lstm(ticker_slice))

    # Stack: (batch, time_steps, num_tickers, hidden_size)
    stacked_lstm = layers.Lambda(lambda t: tf.stack(t, axis=2))(lstm_outputs)

    # GAT layers (sparse attention masked by adjacency)
    x = stacked_lstm
    for k in range(num_gat_layers):
        is_last = (k == num_gat_layers - 1)
        x = GraphAttentionLayer(
            units=gat_units,
            attn_heads=attn_heads,
            adjacency=adjacency,
            concat_heads=not is_last,
            dropout_rate=dropout_rate,
            name=f"sparse_gat_{k}",
        )(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)

    gat_output_dim = gat_units  # last layer averages heads

    # Residual connection
    residual = layers.TimeDistributed(
        layers.TimeDistributed(
            layers.Dense(gat_output_dim, activation="linear")
        )
    )(stacked_lstm)

    x = layers.Add()([x, residual])

    # Output
    output = layers.TimeDistributed(
        layers.TimeDistributed(
            layers.Dense(
                1,
                activation=tf.nn.tanh,
                kernel_constraint=keras.constraints.max_norm(3),
            )
        )
    )(x)

    output = layers.Permute((2, 1, 3))(output)

    model = keras.Model(inputs=input_layer, outputs=output)
    adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
    model.compile(loss=GraphSharpeLoss(output_size=1), optimizer=adam)

    return model


# ---------------------------------------------------------------------------
# Trajectory model builders (concat LSTM outputs, GAT once, predict day 20)
# ---------------------------------------------------------------------------

def _build_lstm_backbone(input_layer, num_tickers, hidden_layer_size, dropout_rate):
    """Shared LSTM backbone. Returns (batch, num_tickers, time_steps, hidden_size)."""
    shared_lstm = layers.LSTM(
        hidden_layer_size,
        return_sequences=True,
        dropout=dropout_rate,
        activation="tanh",
        recurrent_activation="sigmoid",
        name="shared_lstm",
    )

    lstm_outputs = []
    for i in range(num_tickers):
        ticker_slice = layers.Lambda(lambda x, idx=i: x[:, idx, :, :])(input_layer)
        lstm_outputs.append(shared_lstm(ticker_slice))

    # Stack: (batch, num_tickers, time_steps, hidden_size)
    stacked = layers.Lambda(lambda t: tf.stack(t, axis=1))(lstm_outputs)
    return stacked


def build_lstm_gat_trajectory_model(
    num_tickers: int,
    time_steps: int,
    input_size: int,
    hidden_layer_size: int = 10,
    gat_units: int = 8,
    attn_heads: int = 2,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.0005,
    max_gradient_norm: float = 0.01,
    num_gat_layers: int = 1,
) -> keras.Model:
    """
    Build trajectory-based end-to-end LSTM-GAT model.

    Instead of running GAT independently per timestep, this model:
    1. Runs shared LSTM across all timesteps per stock
    2. Flattens all LSTM hidden states into one vector per stock
    3. Runs GAT once on the 88 trajectory vectors
    4. Outputs one position per stock (day 20 only)

    Output shape: (batch, num_tickers, 1)
    """
    input_layer = keras.Input(shape=(num_tickers, time_steps, input_size))

    stacked_lstm = _build_lstm_backbone(
        input_layer, num_tickers, hidden_layer_size, dropout_rate
    )

    # Flatten all timesteps per stock: (batch, 88, time_steps * hidden_size)
    concat_dim = time_steps * hidden_layer_size
    trajectory = layers.Reshape((num_tickers, concat_dim))(stacked_lstm)

    # GAT on 3D input: (batch, 88, concat_dim)
    x = trajectory
    for k in range(num_gat_layers):
        is_last = (k == num_gat_layers - 1)
        x = GraphAttentionLayer(
            units=gat_units,
            attn_heads=attn_heads,
            adjacency=None,
            concat_heads=not is_last,
            dropout_rate=dropout_rate,
            name=f"traj_gat_{k}",
        )(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)

    # Residual: project trajectory to match GAT output dim
    residual = layers.Dense(
        gat_units, activation="linear", name="residual_proj"
    )(trajectory)
    x = layers.Add()([x, residual])

    # Output: one position per stock
    output = layers.Dense(
        1,
        activation=tf.nn.tanh,
        kernel_constraint=keras.constraints.max_norm(3),
        name="position_output",
    )(x)

    model = keras.Model(inputs=input_layer, outputs=output)
    adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
    model.compile(loss=GraphSharpeLoss(output_size=1), optimizer=adam)

    return model


def build_lstm_gat_trajectory_sparse_model(
    num_tickers: int,
    time_steps: int,
    input_size: int,
    adjacency: np.ndarray,
    hidden_layer_size: int = 10,
    gat_units: int = 8,
    attn_heads: int = 2,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.0005,
    max_gradient_norm: float = 0.01,
    num_gat_layers: int = 1,
) -> keras.Model:
    """
    Build trajectory-based graph-guided LSTM-GAT model.

    Same as build_lstm_gat_trajectory_model but attention is masked
    by a static adjacency matrix.

    Output shape: (batch, num_tickers, 1)
    """
    input_layer = keras.Input(shape=(num_tickers, time_steps, input_size))

    stacked_lstm = _build_lstm_backbone(
        input_layer, num_tickers, hidden_layer_size, dropout_rate
    )

    # Flatten all timesteps per stock: (batch, 88, time_steps * hidden_size)
    concat_dim = time_steps * hidden_layer_size
    trajectory = layers.Reshape((num_tickers, concat_dim))(stacked_lstm)

    # Sparse GAT on 3D input: (batch, 88, concat_dim)
    x = trajectory
    for k in range(num_gat_layers):
        is_last = (k == num_gat_layers - 1)
        x = GraphAttentionLayer(
            units=gat_units,
            attn_heads=attn_heads,
            adjacency=adjacency,
            concat_heads=not is_last,
            dropout_rate=dropout_rate,
            name=f"traj_sparse_gat_{k}",
        )(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)

    # Residual
    residual = layers.Dense(
        gat_units, activation="linear", name="residual_proj"
    )(trajectory)
    x = layers.Add()([x, residual])

    # Output: one position per stock
    output = layers.Dense(
        1,
        activation=tf.nn.tanh,
        kernel_constraint=keras.constraints.max_norm(3),
        name="position_output",
    )(x)

    model = keras.Model(inputs=input_layer, outputs=output)
    adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
    model.compile(loss=GraphSharpeLoss(output_size=1), optimizer=adam)

    return model
