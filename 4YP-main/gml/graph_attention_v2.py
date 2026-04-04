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
                 concat_heads=True, scale_scores=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout
        self.concat_heads = concat_heads
        self.scale_scores = scale_scores

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

            # Optional sqrt scaling (normalizes score magnitude across dims)
            if self.scale_scores:
                attn_scores = attn_scores / tf.sqrt(tf.cast(self.units, tf.float32))

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
            "scale_scores": self.scale_scores,
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
                 scale_scores=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout
        self.msg_units = msg_units or units
        self.scale_scores = scale_scores

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

            if self.scale_scores:
                scores = scores / tf.sqrt(tf.cast(self.units, tf.float32))

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
            "scale_scores": self.scale_scores,
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
    scale_scores: bool = False,
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
            scale_scores=scale_scores,
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
    scale_scores: bool = False,
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
        scale_scores=scale_scores,
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


# ---------------------------------------------------------------------------
# Experiment 4c: GATv2 constrained by rolling Pearson adjacency
# ---------------------------------------------------------------------------

class DynamicMaskedGATv2Layer(layers.Layer):
    """
    GATv2 layer that receives a per-sample adjacency mask.

    Pearson correlation determines WHICH edges exist (structure).
    GATv2 learns the WEIGHT of each edge (from LSTM hidden states).

    For 4D input (batch, time, nodes, features), the same per-sample
    adjacency mask is applied at every timestep.
    """

    def __init__(self, units, attn_heads=4, attn_dropout=0.1, scale_scores=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.attn_heads = attn_heads
        self.attn_dropout = attn_dropout
        self.scale_scores = scale_scores

    def build(self, input_shape):
        features_shape = input_shape[0]
        input_dim = features_shape[-1]

        self.W_src = self.add_weight(
            shape=(self.attn_heads, input_dim, self.units),
            initializer="glorot_uniform", trainable=True, name="W_src",
        )
        self.W_dst = self.add_weight(
            shape=(self.attn_heads, input_dim, self.units),
            initializer="glorot_uniform", trainable=True, name="W_dst",
        )
        self.a = self.add_weight(
            shape=(self.attn_heads, self.units, 1),
            initializer="glorot_uniform", trainable=True, name="a",
        )
        self.dropout_layer = layers.Dropout(self.attn_dropout)
        super().build(input_shape)

    def call(self, inputs, training=None):
        features, adjacency = inputs
        # features: (batch, time, nodes, hidden) or (batch, nodes, hidden)
        # adjacency: (batch, nodes, nodes)

        if len(features.shape) == 4:
            batch_size = tf.shape(features)[0]
            time_steps = tf.shape(features)[1]
            num_nodes = tf.shape(features)[2]

            feat_3d = tf.reshape(
                features, [-1, num_nodes, features.shape[-1]]
            )
            adj_tiled = tf.repeat(adjacency, repeats=time_steps, axis=0)

            output = self._masked_attention(feat_3d, adj_tiled, training)
            output = tf.reshape(
                output, [batch_size, time_steps, num_nodes, self.units]
            )
        else:
            output = self._masked_attention(features, adjacency, training)

        return output

    def _masked_attention(self, features, adjacency, training):
        head_outputs = []

        self_loops = tf.eye(
            tf.shape(adjacency)[1], batch_shape=[tf.shape(adjacency)[0]]
        )
        mask = tf.cast((adjacency + self_loops) > 0, tf.float32)

        for head in range(self.attn_heads):
            h_src = tf.matmul(features, self.W_src[head])
            h_dst = tf.matmul(features, self.W_dst[head])

            pairwise = (
                tf.expand_dims(h_src, 2) + tf.expand_dims(h_dst, 1)
            )
            pairwise = tf.nn.leaky_relu(pairwise, alpha=0.2)

            scores = tf.squeeze(
                tf.matmul(pairwise, self.a[head]), axis=-1
            )

            # Optional sqrt scaling
            if self.scale_scores:
                scores = scores / tf.sqrt(tf.cast(self.units, tf.float32))

            # Mask non-edges to -inf before softmax
            scores = tf.where(mask > 0, scores, tf.ones_like(scores) * -1e9)

            attn_weights = tf.nn.softmax(scores, axis=-1)
            attn_weights = self.dropout_layer(attn_weights, training=training)

            h_prime = tf.matmul(attn_weights, h_src)
            head_outputs.append(h_prime)

        return tf.reduce_mean(tf.stack(head_outputs, axis=-1), axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "attn_heads": self.attn_heads,
            "attn_dropout": self.attn_dropout,
            "scale_scores": self.scale_scores,
        })
        return config


def build_lstm_gat_rolling(
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
    scale_scores: bool = False,
    use_residual: bool = False,
) -> keras.Model:
    """
    Experiment 4c: LSTM-GATv2 constrained by rolling Pearson adjacency.

    Pearson determines WHICH edges exist. GATv2 learns the WEIGHTS.
    Like rolling GCN but with learned, asymmetric, data-dependent edge weights.

    Args:
        use_residual: If True, add a residual connection from LSTM to output
            (skip connection bypassing GAT). Default False.

    Returns:
        Compiled Keras model with two inputs:
            features: (batch, num_tickers, time_steps, input_size)
            adjacency: (batch, num_tickers, num_tickers)
    """
    feature_input = keras.Input(
        shape=(num_tickers, time_steps, input_size), name="features"
    )
    adjacency_input = keras.Input(
        shape=(num_tickers, num_tickers), name="adjacency"
    )

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
        ticker_slice = layers.Lambda(lambda x, idx=i: x[:, idx, :, :])(feature_input)
        lstm_outputs.append(shared_lstm(ticker_slice))

    stacked_lstm = layers.Lambda(lambda t: tf.stack(t, axis=2))(lstm_outputs)

    x = DynamicMaskedGATv2Layer(
        units=gat_units,
        attn_heads=attn_heads,
        attn_dropout=attn_dropout,
        scale_scores=scale_scores,
        name="dynamic_masked_gat",
    )([stacked_lstm, adjacency_input])

    x = layers.ReLU()(x)

    # Optional residual connection
    if use_residual:
        residual = layers.TimeDistributed(
            layers.TimeDistributed(
                layers.Dense(gat_units, activation="linear")
            )
        )(stacked_lstm)
        x = layers.Add()([x, residual])

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

    model = keras.Model(inputs=[feature_input, adjacency_input], outputs=output)
    adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
    model.compile(loss=GraphSharpeLoss(output_size=1), optimizer=adam)

    return model


# ---------------------------------------------------------------------------
# Experiment 4cii: Concentrated attention mechanisms
# ---------------------------------------------------------------------------

def sparsemax(logits, axis=-1):
    """
    Sparsemax activation (Martins & Astudillo, 2016) in TensorFlow.

    Projects logits onto the probability simplex, producing sparse outputs
    where some entries are exactly zero.

    Handles masked inputs (entries set to -1e9 or lower) by excluding them
    from the centering and threshold computation.
    """
    # Identify masked entries (set to -1e9 by adjacency masking)
    valid_mask = tf.cast(logits > -1e8, logits.dtype)
    n_valid = tf.reduce_sum(valid_mask, axis=axis, keepdims=True)
    n_valid = tf.maximum(n_valid, 1.0)  # avoid division by zero

    # Center only over valid entries
    masked_logits = tf.where(valid_mask > 0, logits, tf.zeros_like(logits))
    valid_mean = tf.reduce_sum(masked_logits, axis=axis, keepdims=True) / n_valid
    z = logits - valid_mean
    # Zero out masked entries so they don't interfere with sorting
    z = tf.where(valid_mask > 0, z, tf.ones_like(z) * -1e9)

    z_sorted = tf.sort(z, axis=axis, direction='DESCENDING')

    dim = tf.shape(z)[-1]
    range_k = tf.cast(tf.range(1, dim + 1), z.dtype)

    cumsum = tf.cumsum(z_sorted, axis=axis)
    threshold = (cumsum - 1.0) / range_k

    support = tf.cast(z_sorted > threshold, z.dtype)
    k_star = tf.reduce_sum(support, axis=axis, keepdims=True)
    k_star = tf.maximum(k_star, 1.0)  # avoid division by zero
    tau = (tf.reduce_sum(z_sorted * support, axis=axis, keepdims=True) - 1.0) / k_star

    result = tf.maximum(z - tau, 0.0)
    # Zero out masked positions
    return result * valid_mask


def _numpy_sparsemax(logits, axis=-1):
    """NumPy sparsemax for attention extraction (inference only)."""
    # Handle masked entries (set to -1e9 by adjacency masking)
    valid_mask = (logits > -1e8).astype(logits.dtype)
    n_valid = np.maximum(valid_mask.sum(axis=axis, keepdims=True), 1.0)

    masked_logits = np.where(valid_mask > 0, logits, 0.0)
    valid_mean = masked_logits.sum(axis=axis, keepdims=True) / n_valid
    z = logits - valid_mean
    z = np.where(valid_mask > 0, z, -1e9)

    z_sorted = np.flip(np.sort(z, axis=axis), axis=axis)

    dim = z.shape[axis]
    range_k = np.arange(1, dim + 1).astype(z.dtype)

    cumsum = np.cumsum(z_sorted, axis=axis)
    threshold = (cumsum - 1.0) / range_k

    support = (z_sorted > threshold).astype(z.dtype)
    k_star = np.maximum(support.sum(axis=axis, keepdims=True), 1.0)
    tau = (np.sum(z_sorted * support, axis=axis, keepdims=True) - 1.0) / k_star

    result = np.maximum(z - tau, 0.0)
    return result * valid_mask


class ConcentratedGATv2Layer(DynamicMaskedGATv2Layer):
    """
    GATv2 layer with configurable attention concentration mechanisms.

    Extends DynamicMaskedGATv2Layer with:
        - Temperature scaling (temperature < 1 sharpens)
        - Top-k sparsity (keep only top_k logits before softmax)
        - Sparsemax (exact zeros via simplex projection)
        - Power sharpening (raise weights to power > 1 and renormalize)

    Also stores attention weights for entropy regularization.
    """

    def __init__(self, units, attn_heads=4, attn_dropout=0.1, scale_scores=False,
                 temperature=1.0, top_k=None, sharpen_power=1.0,
                 attention_type="softmax", **kwargs):
        super().__init__(
            units=units, attn_heads=attn_heads, attn_dropout=attn_dropout,
            scale_scores=scale_scores, **kwargs,
        )
        self.temperature = temperature
        self.top_k = top_k
        self.sharpen_power = sharpen_power
        self.attention_type = attention_type

    def _masked_attention(self, features, adjacency, training):
        head_outputs = []
        self._last_attn_weights_list = []

        self_loops = tf.eye(
            tf.shape(adjacency)[1], batch_shape=[tf.shape(adjacency)[0]]
        )
        mask = tf.cast((adjacency + self_loops) > 0, tf.float32)

        for head in range(self.attn_heads):
            h_src = tf.matmul(features, self.W_src[head])
            h_dst = tf.matmul(features, self.W_dst[head])

            pairwise = (
                tf.expand_dims(h_src, 2) + tf.expand_dims(h_dst, 1)
            )
            pairwise = tf.nn.leaky_relu(pairwise, alpha=0.2)

            scores = tf.squeeze(
                tf.matmul(pairwise, self.a[head]), axis=-1
            )

            if self.scale_scores:
                scores = scores / tf.sqrt(tf.cast(self.units, tf.float32))

            # Mask non-edges to -inf before attention
            scores = tf.where(mask > 0, scores, tf.ones_like(scores) * -1e9)

            # Top-k: keep only top_k logits, mask rest to -inf
            if self.top_k is not None:
                top_vals, _ = tf.math.top_k(scores, k=self.top_k)
                kth_val = top_vals[:, :, -1:]  # (batch, nodes, 1)
                scores = tf.where(scores >= kth_val, scores,
                                  tf.ones_like(scores) * -1e9)

            # Temperature scaling (applied before softmax/sparsemax)
            if self.temperature != 1.0:
                scores = scores / self.temperature

            # Attention activation
            if self.attention_type == "sparsemax":
                attn_weights = sparsemax(scores, axis=-1)
            else:
                attn_weights = tf.nn.softmax(scores, axis=-1)

            # Power sharpening (post-activation)
            if self.sharpen_power != 1.0:
                attn_weights = tf.pow(attn_weights + 1e-9, self.sharpen_power)
                attn_weights = attn_weights / (
                    tf.reduce_sum(attn_weights, axis=-1, keepdims=True) + 1e-9
                )

            self._last_attn_weights_list.append(attn_weights)
            attn_weights = self.dropout_layer(attn_weights, training=training)

            h_prime = tf.matmul(attn_weights, h_src)
            head_outputs.append(h_prime)

        # Store for entropy regularization: (batch, heads, nodes, nodes)
        self.last_attn_weights = tf.stack(self._last_attn_weights_list, axis=1)

        return tf.reduce_mean(tf.stack(head_outputs, axis=-1), axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "temperature": self.temperature,
            "top_k": self.top_k,
            "sharpen_power": self.sharpen_power,
            "attention_type": self.attention_type,
        })
        return config


def build_lstm_gat_rolling_concentrated(
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
    scale_scores: bool = False,
    use_residual: bool = False,
    # Concentration parameters
    temperature: float = 1.0,
    top_k: int = None,
    sharpen_power: float = 1.0,
    attention_type: str = "softmax",
    lambda_entropy: float = 0.0,
) -> keras.Model:
    """
    Experiment 4cii: LSTM-GATv2 with concentrated attention mechanisms.

    Same architecture as build_lstm_gat_rolling but uses ConcentratedGATv2Layer
    and optionally adds entropy regularization to the loss.

    Additional Args:
        temperature: Divide logits by T before activation. T<1 sharpens.
        top_k: Keep only top-k logits per node before activation.
        sharpen_power: Raise attention weights to this power and renormalize.
        attention_type: "softmax" or "sparsemax".
        lambda_entropy: Weight for attention entropy penalty (0 = disabled).
    """
    feature_input = keras.Input(
        shape=(num_tickers, time_steps, input_size), name="features"
    )
    adjacency_input = keras.Input(
        shape=(num_tickers, num_tickers), name="adjacency"
    )

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
        ticker_slice = layers.Lambda(lambda x, idx=i: x[:, idx, :, :])(feature_input)
        lstm_outputs.append(shared_lstm(ticker_slice))

    stacked_lstm = layers.Lambda(lambda t: tf.stack(t, axis=2))(lstm_outputs)

    gat_layer = ConcentratedGATv2Layer(
        units=gat_units,
        attn_heads=attn_heads,
        attn_dropout=attn_dropout,
        scale_scores=scale_scores,
        temperature=temperature,
        top_k=top_k,
        sharpen_power=sharpen_power,
        attention_type=attention_type,
        name="concentrated_gat",
    )

    x = gat_layer([stacked_lstm, adjacency_input])
    x = layers.ReLU()(x)

    if use_residual:
        residual = layers.TimeDistributed(
            layers.TimeDistributed(
                layers.Dense(gat_units, activation="linear")
            )
        )(stacked_lstm)
        x = layers.Add()([x, residual])

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

    model = keras.Model(inputs=[feature_input, adjacency_input], outputs=output)

    if lambda_entropy > 0:
        from gml.graph_model_2_v2 import GraphSharpeLossWithEntropy
        loss_fn = GraphSharpeLossWithEntropy(
            output_size=1, gat_layer=gat_layer, lambda_entropy=lambda_entropy,
        )
    else:
        loss_fn = GraphSharpeLoss(output_size=1)

    adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
    model.compile(loss=loss_fn, optimizer=adam)

    return model


def extract_attention_weights_concentrated(
    model, inputs, gat_layer_name="concentrated_gat", attention_type="softmax",
):
    """
    Extract attention weights from a ConcentratedGATv2Layer.

    Supports softmax and sparsemax attention types, plus top-k, temperature,
    and power sharpening — mirroring the layer's forward pass in NumPy.

    Returns:
        attention_weights: np.ndarray (num_samples, num_heads, num_nodes, num_nodes)
    """
    gat_layer = model.get_layer(gat_layer_name)
    W_src = gat_layer.W_src.numpy()
    W_dst = gat_layer.W_dst.numpy()
    a = gat_layer.a.numpy()
    num_heads = W_src.shape[0]
    temperature = gat_layer.temperature
    top_k = gat_layer.top_k
    sharpen_power = gat_layer.sharpen_power

    # Get features and adjacency feeding into the GAT layer
    gat_input_tensors = gat_layer.input  # list: [features, adjacency]
    sub_model = keras.Model(inputs=model.input, outputs=gat_input_tensors)
    gat_inputs = sub_model.predict(inputs, verbose=0)
    gat_features = gat_inputs[0]
    gat_adjacency = gat_inputs[1]

    # Use raw adjacency from inputs for masking
    if isinstance(inputs, (list, tuple)):
        raw_adjacency = inputs[1] if isinstance(inputs[1], np.ndarray) else inputs[1]
    else:
        raw_adjacency = gat_adjacency

    def _apply_attention(scores, mask_2d):
        """Apply top-k, temperature, activation, and power sharpening."""
        scores = np.where(mask_2d > 0, scores, -1e9)

        if top_k is not None:
            kth_vals = np.partition(scores, -top_k, axis=-1)[..., -top_k:]
            kth_val = kth_vals.min(axis=-1, keepdims=True)
            scores = np.where(scores >= kth_val, scores, -1e9)

        if temperature != 1.0:
            scores = scores / temperature

        if attention_type == "sparsemax":
            attn = _numpy_sparsemax(scores, axis=-1)
        else:
            exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-9)

        if sharpen_power != 1.0:
            attn = np.power(attn + 1e-9, sharpen_power)
            attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-9)

        return attn

    all_attention = []

    for sample_idx in range(gat_features.shape[0]):
        sample = gat_features[sample_idx]
        adj = raw_adjacency[sample_idx] if raw_adjacency.ndim == 3 else raw_adjacency
        mask = ((adj + np.eye(adj.shape[0])) > 0).astype(np.float32)

        if sample.ndim == 3:
            head_attns = []
            for head in range(num_heads):
                h_src = sample @ W_src[head]
                h_dst = sample @ W_dst[head]
                pairwise = h_src[:, :, np.newaxis, :] + h_dst[:, np.newaxis, :, :]
                pairwise = np.where(pairwise > 0, pairwise, 0.2 * pairwise)
                scores = (pairwise @ a[head]).squeeze(-1)
                attn = _apply_attention(scores, mask)
                head_attns.append(attn.mean(axis=0))
            all_attention.append(np.stack(head_attns, axis=0))
        else:
            head_attns = []
            for head in range(num_heads):
                h_src = sample @ W_src[head]
                h_dst = sample @ W_dst[head]
                pairwise = h_src[:, np.newaxis, :] + h_dst[np.newaxis, :, :]
                pairwise = np.where(pairwise > 0, pairwise, 0.2 * pairwise)
                scores = (pairwise @ a[head]).squeeze(-1)
                attn = _apply_attention(scores, mask)
                head_attns.append(attn)
            all_attention.append(np.stack(head_attns, axis=0))

    return np.stack(all_attention, axis=0)
