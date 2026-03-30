"""
Improved LSTM-GAT models with GATv2 attention.

This module provides new model builders that address diagnosed performance issues
in the original LSTM-GAT implementation (graph_model_gat_v2.py) while leaving
that file untouched.

Key improvements over the original:
    - GATv2 dynamic attention (Brody et al., 2022): truly pairwise scores
    - Separate LSTM and attention dropout parameters
    - No LayerNormalization after GAT (matches GCN, preserves cross-stock variance)
    - Larger default dimensions (hidden=32, GAT units=16, heads=4)

Model builders:
    build_lstm_gat_e2e_v2: Per-timestep GATv2 with training fixes (Step 1a)
    build_lstm_gat_e2e_v2_prev_window: Previous-window attention (Step 1b)

Layer:
    GraphAttentionLayerV2: GATv2 with separate attention dropout
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from gml.graph_model_2_v2 import GraphSharpeLoss


# ---------------------------------------------------------------------------
# GATv2 Layer
# ---------------------------------------------------------------------------

class GraphAttentionLayerV2(layers.Layer):
    """
    GATv2 attention layer (Brody et al., 2022).

    Key difference from GATv1: applies LeakyReLU BEFORE the final projection,
    making attention scores truly dependent on the (source, target) pair rather
    than decomposing as f(source) + g(target).

    Supports both 3D input (batch, nodes, features) and 4D input
    (batch, time_steps, nodes, features). For 4D, attention is computed
    independently per timestep.

    Args:
        units: Output feature dimension per head.
        attn_heads: Number of attention heads.
        attn_dropout: Dropout rate on attention weights.
        concat_heads: If True, concatenate heads; if False, average them.
    """

    def __init__(self, units, attn_heads=4, attn_dropout=0.1,
                 concat_heads=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout
        self.concat_heads = concat_heads

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Per-head linear projections for source and destination
        self.W_src = self.add_weight(
            shape=(self.attn_heads, input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="W_src",
        )
        self.W_dst = self.add_weight(
            shape=(self.attn_heads, input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="W_dst",
        )
        # Attention vector (shared across all pairs, per head)
        self.a = self.add_weight(
            shape=(self.attn_heads, self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="a",
        )

        self.dropout_layer = layers.Dropout(self.attn_dropout)
        super().build(input_shape)

    def call(self, inputs, training=None):
        # inputs: (batch, time_steps, num_nodes, features) or (batch, num_nodes, features)
        if len(inputs.shape) == 4:
            input_shape = tf.shape(inputs)
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
            # GATv2: separate projections for source and destination
            h_src = tf.matmul(inputs, self.W_src[head])  # (batch, nodes, units)
            h_dst = tf.matmul(inputs, self.W_dst[head])  # (batch, nodes, units)

            # Pairwise combination: h_src[i] + h_dst[j] for all (i, j)
            # h_src: (batch, nodes, 1, units) + h_dst: (batch, 1, nodes, units)
            h_src_expanded = tf.expand_dims(h_src, 2)  # (batch, nodes, 1, units)
            h_dst_expanded = tf.expand_dims(h_dst, 1)  # (batch, 1, nodes, units)
            pairwise = h_src_expanded + h_dst_expanded  # (batch, nodes, nodes, units)

            # GATv2: nonlinearity BEFORE projection → truly pairwise scores
            pairwise = tf.nn.leaky_relu(pairwise, alpha=0.2)

            # Project to scalar attention score per pair
            attn_scores = tf.squeeze(
                tf.matmul(pairwise, self.a[head]), axis=-1
            )  # (batch, nodes, nodes)

            # Softmax over neighbors (all nodes in full attention)
            attn_weights = tf.nn.softmax(attn_scores, axis=-1)
            attn_weights = self.dropout_layer(attn_weights, training=training)

            # Message passing: aggregate neighbor features
            # Use h_src as the message (projected node features)
            h_prime = tf.matmul(attn_weights, h_src)  # (batch, nodes, units)
            outputs.append(h_prime)

        if self.concat_heads:
            return tf.concat(outputs, axis=-1)  # (batch, nodes, units * heads)
        else:
            return tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "attn_heads": self.attn_heads,
            "attn_dropout": self.attn_dropout,
            "concat_heads": self.concat_heads,
        })
        return config


# ---------------------------------------------------------------------------
# Attention extraction (for visualization)
# ---------------------------------------------------------------------------

def extract_attention_weights_v2(model, inputs, gat_layer_name="gat_v2_0"):
    """
    Extract per-sample attention weights from a trained GATv2 model.

    Args:
        model: Trained Keras model containing GraphAttentionLayerV2(s).
        inputs: Test inputs. For single-input models, shape
            (num_samples, num_tickers, time_steps, input_size).
            For dual-input models, a list/dict matching model.input.
        gat_layer_name: Name of the GATv2 layer to extract from.

    Returns:
        attention_weights: np.ndarray of shape
            (num_samples, num_heads, num_nodes, num_nodes).
            For per-timestep models, averaged across timesteps.
    """
    gat_layer = model.get_layer(gat_layer_name)
    W_src = gat_layer.W_src.numpy()  # (heads, input_dim, units)
    W_dst = gat_layer.W_dst.numpy()  # (heads, input_dim, units)
    a = gat_layer.a.numpy()          # (heads, units, 1)
    num_heads = W_src.shape[0]

    # Build sub-model to get the GAT layer's input
    gat_input_tensor = gat_layer.input
    sub_model = keras.Model(inputs=model.input, outputs=gat_input_tensor)

    # Get features feeding into the GAT layer
    gat_input = sub_model.predict(inputs, verbose=0)
    # Shape: (samples, time, nodes, features) or (samples, nodes, features)

    all_attention = []

    for sample_idx in range(gat_input.shape[0]):
        sample = gat_input[sample_idx]

        if sample.ndim == 3:
            # Per-timestep: (time_steps, nodes, features) → average attention across time
            head_attns = []
            for head in range(num_heads):
                h_src = sample @ W_src[head]  # (time, nodes, units)
                h_dst = sample @ W_dst[head]

                # Pairwise: (time, nodes, 1, units) + (time, 1, nodes, units)
                pairwise = h_src[:, :, np.newaxis, :] + h_dst[:, np.newaxis, :, :]
                pairwise = np.where(pairwise > 0, pairwise, 0.2 * pairwise)

                scores = (pairwise @ a[head]).squeeze(-1)  # (time, nodes, nodes)

                # Softmax
                exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
                attn = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-9)

                head_attns.append(attn.mean(axis=0))  # avg across time → (nodes, nodes)

            all_attention.append(np.stack(head_attns, axis=0))
        else:
            # 2D: (nodes, features) — single pass
            head_attns = []
            for head in range(num_heads):
                h_src = sample @ W_src[head]
                h_dst = sample @ W_dst[head]

                pairwise = h_src[:, np.newaxis, :] + h_dst[np.newaxis, :, :]
                pairwise = np.where(pairwise > 0, pairwise, 0.2 * pairwise)

                scores = (pairwise @ a[head]).squeeze(-1)

                exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
                attn = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-9)
                head_attns.append(attn)

            all_attention.append(np.stack(head_attns, axis=0))

    return np.stack(all_attention, axis=0)  # (samples, heads, nodes, nodes)


# ---------------------------------------------------------------------------
# Step 1a: Per-timestep GATv2 with training + architecture fixes
# ---------------------------------------------------------------------------

def build_lstm_gat_e2e_v2(
    num_tickers: int,
    time_steps: int,
    input_size: int,
    hidden_layer_size: int = 32,
    gat_units: int = 16,
    attn_heads: int = 4,
    lstm_dropout: float = 0.3,
    attn_dropout: float = 0.1,
    learning_rate: float = 0.001,
    max_gradient_norm: float = 1.0,
    num_gat_layers: int = 2,
) -> keras.Model:
    """
    Step 1a: Per-timestep LSTM-GATv2 model with training fixes.

    Improvements over build_lstm_gat_model in graph_model_gat_v2.py:
        - GATv2 dynamic attention (truly pairwise scores)
        - Separate LSTM / attention dropout (0.3 / 0.1 vs shared 0.5)
        - No LayerNormalization after GAT (matches GCN)
        - Larger dimensions: hidden=32, GAT=16, heads=4
        - Less aggressive gradient clipping: 1.0 vs 0.01

    Args:
        num_tickers: Number of stocks/nodes (N=88).
        time_steps: Lookback window length (default 20).
        input_size: Features per timestep (default 10).
        hidden_layer_size: LSTM hidden units.
        gat_units: GAT output dimension per head.
        attn_heads: Number of attention heads.
        lstm_dropout: LSTM dropout rate.
        attn_dropout: Attention weight dropout rate.
        learning_rate: Adam learning rate.
        max_gradient_norm: Gradient clipping norm.
        num_gat_layers: Number of stacked GATv2 layers.

    Returns:
        Compiled Keras model.
    """
    input_layer = keras.Input(shape=(num_tickers, time_steps, input_size))

    # Shared LSTM
    shared_lstm = layers.LSTM(
        hidden_layer_size,
        return_sequences=True,
        dropout=lstm_dropout,
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

    # GATv2 layers (no LayerNorm — just ReLU, matching GCN)
    x = stacked_lstm
    for k in range(num_gat_layers):
        is_last = (k == num_gat_layers - 1)
        x = GraphAttentionLayerV2(
            units=gat_units,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout,
            concat_heads=not is_last,
            name=f"gat_v2_{k}",
        )(x)
        x = layers.ReLU()(x)

    gat_output_dim = gat_units  # last layer averages heads

    # Residual connection
    residual = layers.TimeDistributed(
        layers.TimeDistributed(
            layers.Dense(gat_output_dim, activation="linear")
        )
    )(stacked_lstm)

    x = layers.Add()([x, residual])

    # Output: position in [-1, 1]
    output = layers.TimeDistributed(
        layers.TimeDistributed(
            layers.Dense(
                1,
                activation=tf.nn.tanh,
                kernel_constraint=keras.constraints.max_norm(3),
            )
        )
    )(x)

    # Permute to (batch, num_tickers, time_steps, 1)
    output = layers.Permute((2, 1, 3))(output)

    model = keras.Model(inputs=input_layer, outputs=output)
    adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
    model.compile(loss=GraphSharpeLoss(output_size=1), optimizer=adam)

    return model


# ---------------------------------------------------------------------------
# Step 1b: Previous-window raw features for attention computation
# ---------------------------------------------------------------------------

class PrevWindowAttentionLayer(layers.Layer):
    """
    Computes GATv2 attention weights from one set of features (previous window)
    and applies them for message passing on another set (current window LSTM
    hidden states).

    This separates graph construction from message passing:
        - Graph: computed from prev_features (raw, 200-dim per stock)
        - Messages: passed using current LSTM hidden states per timestep

    The attention weights are constant across all timesteps within a window,
    providing temporal consistency (same graph at every t).

    Args:
        units: Feature dimension per head for attention scoring.
        attn_heads: Number of attention heads.
        attn_dropout: Dropout on attention weights.
        msg_units: Output dimension per head for message passing. If None,
            defaults to units.
    """

    def __init__(self, units, attn_heads=4, attn_dropout=0.1, msg_units=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout
        self.msg_units = msg_units or units

    def build(self, input_shape):
        # input_shape is a list: [prev_features_shape, curr_hidden_shape]
        prev_dim = input_shape[0][-1]  # (batch, nodes, prev_features)
        curr_dim = input_shape[1][-1]  # (batch, time, nodes, hidden)

        # Attention scoring weights (operate on prev_features)
        self.W_src = self.add_weight(
            shape=(self.attn_heads, prev_dim, self.units),
            initializer="glorot_uniform", trainable=True, name="W_src",
        )
        self.W_dst = self.add_weight(
            shape=(self.attn_heads, prev_dim, self.units),
            initializer="glorot_uniform", trainable=True, name="W_dst",
        )
        self.a = self.add_weight(
            shape=(self.attn_heads, self.units, 1),
            initializer="glorot_uniform", trainable=True, name="a",
        )

        # Message projection weights (operate on curr_hidden per timestep)
        self.W_msg = self.add_weight(
            shape=(self.attn_heads, curr_dim, self.msg_units),
            initializer="glorot_uniform", trainable=True, name="W_msg",
        )

        self.dropout_layer = layers.Dropout(self.attn_dropout)
        super().build(input_shape)

    def call(self, inputs, training=None):
        prev_features, curr_hidden = inputs
        # prev_features: (batch, nodes, prev_dim)
        # curr_hidden: (batch, time_steps, nodes, hidden_dim)

        batch_size = tf.shape(curr_hidden)[0]
        time_steps = tf.shape(curr_hidden)[1]

        head_outputs = []
        for head in range(self.attn_heads):
            # --- Compute attention from previous window features ---
            h_src = tf.matmul(prev_features, self.W_src[head])  # (batch, N, units)
            h_dst = tf.matmul(prev_features, self.W_dst[head])  # (batch, N, units)

            pairwise = (
                tf.expand_dims(h_src, 2) + tf.expand_dims(h_dst, 1)
            )  # (batch, N, N, units)
            pairwise = tf.nn.leaky_relu(pairwise, alpha=0.2)

            scores = tf.squeeze(
                tf.matmul(pairwise, self.a[head]), axis=-1
            )  # (batch, N, N)

            attn_weights = tf.nn.softmax(scores, axis=-1)  # (batch, N, N)
            attn_weights = self.dropout_layer(attn_weights, training=training)

            # --- Apply attention to current window LSTM hidden states ---
            # Project current hidden states
            # curr_hidden: (batch, time, nodes, hidden)
            curr_3d = tf.reshape(curr_hidden, [-1, tf.shape(curr_hidden)[2], curr_hidden.shape[-1]])
            msg = tf.matmul(curr_3d, self.W_msg[head])  # (batch*time, N, msg_units)

            # Tile attention for all timesteps
            attn_tiled = tf.repeat(attn_weights, repeats=time_steps, axis=0)  # (batch*time, N, N)

            # Aggregate messages
            h_prime = tf.matmul(attn_tiled, msg)  # (batch*time, N, msg_units)
            h_prime = tf.reshape(h_prime, [batch_size, time_steps, -1, self.msg_units])

            head_outputs.append(h_prime)

        # Average heads (final layer behavior)
        output = tf.reduce_mean(tf.stack(head_outputs, axis=-1), axis=-1)
        return output  # (batch, time_steps, nodes, msg_units)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "attn_heads": self.attn_heads,
            "attn_dropout": self.attn_dropout,
            "msg_units": self.msg_units,
        })
        return config


def build_lstm_gat_e2e_v2_prev_window(
    num_tickers: int,
    time_steps: int,
    input_size: int,
    hidden_layer_size: int = 32,
    gat_units: int = 16,
    attn_heads: int = 4,
    lstm_dropout: float = 0.3,
    attn_dropout: float = 0.1,
    learning_rate: float = 0.001,
    max_gradient_norm: float = 1.0,
    prev_feature_dim: int = None,
) -> keras.Model:
    """
    Step 1b: LSTM-GATv2 with previous-window raw features for attention.

    Architecture:
        1. Previous window raw features (flattened) → GATv2 attention weights
        2. Current window → Shared LSTM → per-timestep hidden states
        3. Message passing at each timestep using the pre-computed attention
        4. Residual connection + Dense → positions in [-1, 1]

    No information leakage: attention is computed from strictly past data.
    Analogous to rolling GCN (which uses a lookback correlation window for
    graph construction), but the graph is learned end-to-end.

    Args:
        num_tickers: Number of stocks/nodes.
        time_steps: Lookback window length.
        input_size: Features per timestep.
        hidden_layer_size: LSTM hidden units.
        gat_units: Attention scoring / message output dimension per head.
        attn_heads: Number of attention heads.
        lstm_dropout: LSTM dropout rate.
        attn_dropout: Attention weight dropout rate.
        learning_rate: Adam learning rate.
        max_gradient_norm: Gradient clipping norm.
        prev_feature_dim: Dimension of flattened previous-window features per
            stock.  If None, defaults to time_steps * input_size (=200 for
            straddle features).  Set to a different value when using equity
            returns (e.g. time_steps=20 for a 20-dim return series).

    Returns:
        Compiled Keras model with two inputs:
            prev_features: (batch, num_tickers, prev_feature_dim)
            curr_features: (batch, num_tickers, time_steps, input_size)
    """
    if prev_feature_dim is None:
        prev_feature_dim = time_steps * input_size

    # --- Inputs ---
    prev_input = keras.Input(
        shape=(num_tickers, prev_feature_dim), name="prev_features"
    )
    curr_input = keras.Input(
        shape=(num_tickers, time_steps, input_size), name="curr_features"
    )

    # --- Current window through shared LSTM ---
    shared_lstm = layers.LSTM(
        hidden_layer_size,
        return_sequences=True,
        dropout=lstm_dropout,
        activation="tanh",
        recurrent_activation="sigmoid",
        name="shared_lstm",
    )

    lstm_outputs = []
    for i in range(num_tickers):
        ticker_slice = layers.Lambda(lambda x, idx=i: x[:, idx, :, :])(curr_input)
        lstm_outputs.append(shared_lstm(ticker_slice))

    # (batch, time_steps, num_tickers, hidden_size)
    stacked_lstm = layers.Lambda(lambda t: tf.stack(t, axis=2))(lstm_outputs)

    # --- Attention from previous window, message passing on current ---
    x = PrevWindowAttentionLayer(
        units=gat_units,
        attn_heads=attn_heads,
        attn_dropout=attn_dropout,
        msg_units=gat_units,
        name="prev_window_gat",
    )([prev_input, stacked_lstm])

    x = layers.ReLU()(x)

    # --- Residual connection ---
    residual = layers.TimeDistributed(
        layers.TimeDistributed(
            layers.Dense(gat_units, activation="linear")
        )
    )(stacked_lstm)

    x = layers.Add()([x, residual])

    # --- Output: position in [-1, 1] ---
    output = layers.TimeDistributed(
        layers.TimeDistributed(
            layers.Dense(
                1,
                activation=tf.nn.tanh,
                kernel_constraint=keras.constraints.max_norm(3),
            )
        )
    )(x)

    # Permute to (batch, num_tickers, time_steps, 1)
    output = layers.Permute((2, 1, 3))(output)

    model = keras.Model(inputs=[prev_input, curr_input], outputs=output)
    adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
    model.compile(loss=GraphSharpeLoss(output_size=1), optimizer=adam)

    return model


# ---------------------------------------------------------------------------
# Experiment 4a: Per-timestep GATv2, NO residual connection
# ---------------------------------------------------------------------------

def build_lstm_gat_e2e_v3(
    num_tickers: int,
    time_steps: int,
    input_size: int,
    hidden_layer_size: int = 32,
    gat_units: int = 16,
    attn_heads: int = 4,
    lstm_dropout: float = 0.3,
    attn_dropout: float = 0.1,
    learning_rate: float = 0.001,
    max_gradient_norm: float = 1.0,
    num_gat_layers: int = 1,
) -> keras.Model:
    """
    Experiment 4a: Per-timestep LSTM-GATv2 WITHOUT residual connection.

    Identical to build_lstm_gat_e2e_v2 except the residual path is removed,
    forcing all information to flow through the GAT layer. This is motivated
    by the diagnostic finding that the residual connection allows the model
    to bypass attention entirely (attention entropy = 4.44/4.48 = uniform).

    Without the residual, the model must learn meaningful attention weights
    to produce good positions, since the only path from LSTM to output goes
    through the GATv2 layer.
    """
    input_layer = keras.Input(shape=(num_tickers, time_steps, input_size))

    # Shared LSTM
    shared_lstm = layers.LSTM(
        hidden_layer_size,
        return_sequences=True,
        dropout=lstm_dropout,
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

    # GATv2 layers (no residual, no LayerNorm)
    x = stacked_lstm
    for k in range(num_gat_layers):
        is_last = (k == num_gat_layers - 1)
        x = GraphAttentionLayerV2(
            units=gat_units,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout,
            concat_heads=not is_last,
            name=f"gat_v2_{k}",
        )(x)
        x = layers.ReLU()(x)

    # NO residual — output directly from GAT

    # Output: position in [-1, 1]
    output = layers.TimeDistributed(
        layers.TimeDistributed(
            layers.Dense(
                1,
                activation=tf.nn.tanh,
                kernel_constraint=keras.constraints.max_norm(3),
            )
        )
    )(x)

    # Permute to (batch, num_tickers, time_steps, 1)
    output = layers.Permute((2, 1, 3))(output)

    model = keras.Model(inputs=input_layer, outputs=output)
    adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
    model.compile(loss=GraphSharpeLoss(output_size=1), optimizer=adam)

    return model


# ---------------------------------------------------------------------------
# Experiment 4b: Previous-window GATv2, NO residual connection
# ---------------------------------------------------------------------------

def build_lstm_gat_e2e_v3_prev_window(
    num_tickers: int,
    time_steps: int,
    input_size: int,
    hidden_layer_size: int = 32,
    gat_units: int = 16,
    attn_heads: int = 4,
    lstm_dropout: float = 0.3,
    attn_dropout: float = 0.1,
    learning_rate: float = 0.001,
    max_gradient_norm: float = 1.0,
    prev_feature_dim: int = None,
) -> keras.Model:
    """
    Experiment 4b: LSTM-GATv2 with previous-window attention, NO residual.

    Identical to build_lstm_gat_e2e_v2_prev_window except the residual path
    is removed. All information must flow through the GATv2 attention layer.

    Architecture:
        1. Previous window raw features → GATv2 attention weights
        2. Current window → Shared LSTM → per-timestep hidden states
        3. Message passing at each timestep using pre-computed attention
        4. Dense → positions in [-1, 1]  (NO residual skip)
    """
    if prev_feature_dim is None:
        prev_feature_dim = time_steps * input_size

    # --- Inputs ---
    prev_input = keras.Input(
        shape=(num_tickers, prev_feature_dim), name="prev_features"
    )
    curr_input = keras.Input(
        shape=(num_tickers, time_steps, input_size), name="curr_features"
    )

    # --- Current window through shared LSTM ---
    shared_lstm = layers.LSTM(
        hidden_layer_size,
        return_sequences=True,
        dropout=lstm_dropout,
        activation="tanh",
        recurrent_activation="sigmoid",
        name="shared_lstm",
    )

    lstm_outputs = []
    for i in range(num_tickers):
        ticker_slice = layers.Lambda(lambda x, idx=i: x[:, idx, :, :])(curr_input)
        lstm_outputs.append(shared_lstm(ticker_slice))

    # (batch, time_steps, num_tickers, hidden_size)
    stacked_lstm = layers.Lambda(lambda t: tf.stack(t, axis=2))(lstm_outputs)

    # --- Attention from previous window, message passing on current ---
    x = PrevWindowAttentionLayer(
        units=gat_units,
        attn_heads=attn_heads,
        attn_dropout=attn_dropout,
        msg_units=gat_units,
        name="prev_window_gat",
    )([prev_input, stacked_lstm])

    x = layers.ReLU()(x)

    # NO residual — output directly from GAT

    # --- Output: position in [-1, 1] ---
    output = layers.TimeDistributed(
        layers.TimeDistributed(
            layers.Dense(
                1,
                activation=tf.nn.tanh,
                kernel_constraint=keras.constraints.max_norm(3),
            )
        )
    )(x)

    # Permute to (batch, num_tickers, time_steps, 1)
    output = layers.Permute((2, 1, 3))(output)

    model = keras.Model(inputs=[prev_input, curr_input], outputs=output)
    adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
    model.compile(loss=GraphSharpeLoss(output_size=1), optimizer=adam)

    return model
