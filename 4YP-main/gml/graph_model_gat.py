import tensorflow as tf
import numpy as np
import pandas as pd
import collections
from tensorflow import keras
import keras_tuner as kt
import copy
import os

from abc import ABC, abstractmethod

from settings.hp_grid import (
    HP_HIDDEN_LAYER_SIZE_GRAPH,
    HP_DROPOUT_RATE_GRAPH,
    HP_MAX_GRADIENT_NORM_GRAPH,
    HP_LEARNING_RATE_GRAPH,
    HP_MINIBATCH_SIZE_GRAPH,
    HP_GCN_UNITS,
    HP_ALPHA,
    HP_BETA,
)
from settings.default import ALL_TICKERS
from gml.graph_model_inputs import GraphModelFeatures

from empyrical import sharpe_ratio

from gml.graph_model_2 import (
    GraphDeepMomentumNetwork,
    GraphSharpeLoss,
    GraphSharpeValidationLoss,
    GraphTunerDiversifiedSharpe,
    GraphTunerValidationLoss,
)

from settings.fixed_params import MODEL_PARAMS_GRAPH

from tensorflow.keras import layers, optimizers, constraints


class GraphAttentionLayer(layers.Layer):
    """
    Graph Attention Layer (GAT) implementation.
    Computes attention coefficients between connected nodes.

    Args:
        units: Output feature dimension
        attn_heads: Number of attention heads
        adjacency: Adjacency matrix (used as mask for sparse attention)
        concat_heads: If True, concatenate heads; if False, average them
        use_sparse: If True, only attend to neighbors; if False, full attention
    """
    def __init__(self, units, attn_heads=4, adjacency=None, concat_heads=True,
                 use_sparse=True, dropout_rate=0.0, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.use_sparse = use_sparse
        self.dropout_rate = dropout_rate

        if adjacency is not None:
            # Create attention mask from adjacency (add self-loops)
            adj_with_self = adjacency + np.eye(adjacency.shape[0])
            self.attention_mask = tf.constant(adj_with_self, dtype=tf.float32)
        else:
            self.attention_mask = None

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Weight matrices for each attention head
        self.W = self.add_weight(
            shape=(self.attn_heads, input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="W",
        )

        # Attention weights for source and target nodes
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

        self.dropout = layers.Dropout(self.dropout_rate)

        super(GraphAttentionLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        # inputs shape: (batch_size, num_nodes, input_dim) or
        #               (batch_size, time_steps, num_nodes, input_dim)

        input_shape = tf.shape(inputs)

        # Handle 4D input (batch, time, nodes, features)
        if len(inputs.shape) == 4:
            batch_size = input_shape[0]
            time_steps = input_shape[1]
            num_nodes = input_shape[2]

            # Reshape to (batch * time, nodes, features)
            inputs_reshaped = tf.reshape(inputs, [-1, num_nodes, inputs.shape[-1]])
            output = self._attention_forward(inputs_reshaped, training)

            # Reshape back to (batch, time, nodes, output_features)
            output_dim = self.units * self.attn_heads if self.concat_heads else self.units
            output = tf.reshape(output, [batch_size, time_steps, num_nodes, output_dim])
        else:
            output = self._attention_forward(inputs, training)

        return output

    def _attention_forward(self, inputs, training):
        # inputs: (batch_size, num_nodes, input_dim)
        batch_size = tf.shape(inputs)[0]
        num_nodes = tf.shape(inputs)[1]

        outputs = []
        for head in range(self.attn_heads):
            # Linear transformation: (batch, nodes, units)
            h = tf.matmul(inputs, self.W[head])

            # Compute attention scores
            # Source scores: (batch, nodes, 1)
            attn_src = tf.matmul(h, self.a_src[head])
            # Target scores: (batch, nodes, 1)
            attn_dst = tf.matmul(h, self.a_dst[head])

            # Broadcast to get pairwise scores: (batch, nodes, nodes)
            # attn[i,j] = attn_src[i] + attn_dst[j]
            attn_scores = attn_src + tf.transpose(attn_dst, [0, 2, 1])

            # Apply LeakyReLU
            attn_scores = tf.nn.leaky_relu(attn_scores, alpha=0.2)

            # Apply attention mask (sparse attention)
            if self.use_sparse and self.attention_mask is not None:
                # Mask out non-neighbors with large negative value
                mask = tf.expand_dims(self.attention_mask, 0)  # (1, nodes, nodes)
                attn_scores = tf.where(
                    mask > 0,
                    attn_scores,
                    tf.ones_like(attn_scores) * (-1e9)
                )

            # Softmax over neighbors
            attn_weights = tf.nn.softmax(attn_scores, axis=-1)

            # Apply dropout to attention weights
            attn_weights = self.dropout(attn_weights, training=training)

            # Aggregate neighbor features: (batch, nodes, units)
            h_prime = tf.matmul(attn_weights, h)
            outputs.append(h_prime)

        # Combine attention heads
        if self.concat_heads:
            output = tf.concat(outputs, axis=-1)  # (batch, nodes, units * heads)
        else:
            output = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)  # (batch, nodes, units)

        return output


class SparseGATLSTMDeepMomentumNetwork(GraphDeepMomentumNetwork):
    """
    LSTM + Sparse GAT model.
    Uses precomputed graph structure as attention mask.
    """
    def __init__(self, project_name, hp_directory, hp_minibatch_size=HP_MINIBATCH_SIZE_GRAPH, **params):
        super().__init__(project_name, hp_directory, hp_minibatch_size, **params)

    def model_builder(self, hp):
        hidden_layer_size = hp.Choice("hidden_layer_size", values=HP_HIDDEN_LAYER_SIZE_GRAPH)
        dropout_rate = hp.Choice("dropout_rate", values=HP_DROPOUT_RATE_GRAPH)
        max_gradient_norm = hp.Choice("max_gradient_norm", values=HP_MAX_GRADIENT_NORM_GRAPH)
        learning_rate = hp.Choice("learning_rate", values=HP_LEARNING_RATE_GRAPH)
        gat_units = hp.Choice("gat_units", values=HP_GCN_UNITS)
        attn_heads = hp.Choice("attn_heads", values=[2, 4, 8])
        num_gat_layers = hp.Choice("gat_layers", values=[2])

        alpha = hp.Choice("alpha", values=HP_ALPHA)
        beta = hp.Choice("beta", values=HP_BETA)

        # Load precomputed adjacency matrix
        graph_file = os.path.join("data", "graph_structure", "cvx_opt", f"{alpha}_{beta}_cvx.csv")
        adjacency_df = pd.read_csv(graph_file, index_col=0)
        adjacency_df = adjacency_df.reindex(index=ALL_TICKERS, columns=ALL_TICKERS)
        self.A = adjacency_df.values

        # Input shape: (batch_size, num_tickers, time_steps, input_size)
        input_layer = keras.Input(shape=(self.num_tickers, self.time_steps, self.input_size))

        # Shared LSTM layer
        shared_lstm = layers.LSTM(
            hidden_layer_size,
            return_sequences=True,
            dropout=dropout_rate,
            activation="tanh",
            recurrent_activation="sigmoid",
            name="shared_lstm"
        )

        lstm_outputs = []
        for i in range(self.num_tickers):
            ticker_slice = layers.Lambda(lambda x, idx=i: x[:, idx, :, :])(input_layer)
            ticker_output = shared_lstm(ticker_slice)
            lstm_outputs.append(ticker_output)

        # Stack: (batch, time_steps, num_tickers, hidden_size)
        stacked_lstm = layers.Lambda(lambda tensors: tf.stack(tensors, axis=2))(lstm_outputs)
        print("Stacked LSTM output shape:", stacked_lstm.shape)

        # Apply Sparse GAT layers with Layer Normalization
        # Layer 1: concat heads for expressiveness
        gat_output = GraphAttentionLayer(
            units=gat_units,
            attn_heads=attn_heads,
            adjacency=self.A,
            concat_heads=True,  # Concat for intermediate layer
            use_sparse=True,
            dropout_rate=dropout_rate,
            name="sparse_gat_1"
        )(stacked_lstm)
        gat_output = layers.LayerNormalization()(gat_output)  # Add layer norm
        gat_output = layers.ReLU()(gat_output)
        print("After Sparse GAT 1, shape:", gat_output.shape)

        if num_gat_layers == 2:
            # Layer 2 (final): average heads for stability
            gat_output = GraphAttentionLayer(
                units=gat_units,
                attn_heads=attn_heads,
                adjacency=self.A,
                concat_heads=False,  # Average for final layer
                use_sparse=True,
                dropout_rate=dropout_rate,
                name="sparse_gat_2"
            )(gat_output)
            gat_output = layers.LayerNormalization()(gat_output)  # Add layer norm
            gat_output = layers.ReLU()(gat_output)
            print("After Sparse GAT 2, shape:", gat_output.shape)
            gat_output_dim = gat_units  # Averaged heads = single unit dimension
        else:
            gat_output_dim = gat_units * attn_heads  # Concatenated = units * heads

        # Residual connection
        residual = layers.TimeDistributed(
            layers.TimeDistributed(
                keras.layers.Dense(gat_output_dim, activation="linear")
            )
        )(stacked_lstm)

        x = layers.Add()([gat_output, residual])

        # Output layer
        output = layers.TimeDistributed(
            layers.TimeDistributed(
                keras.layers.Dense(
                    self.output_size,
                    activation=tf.nn.tanh,
                    kernel_constraint=keras.constraints.max_norm(3),
                )
            )
        )(x)

        # Permute to (batch, num_tickers, time_steps, output_size)
        output = layers.Permute((2, 1, 3))(output)

        model = keras.Model(inputs=input_layer, outputs=output)
        print("Final model output shape:", output.shape)

        adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
        sharpe_loss = GraphSharpeLoss(self.output_size).call

        model.compile(loss=sharpe_loss, optimizer=adam)

        return model


class FullGATLSTMDeepMomentumNetwork(GraphDeepMomentumNetwork):
    """
    LSTM + Full GAT model.
    All nodes attend to all nodes (no graph structure needed).
    Learns which tickers are relevant through attention.
    """
    def __init__(self, project_name, hp_directory, hp_minibatch_size=HP_MINIBATCH_SIZE_GRAPH, **params):
        super().__init__(project_name, hp_directory, hp_minibatch_size, **params)

    def model_builder(self, hp):
        hidden_layer_size = hp.Choice("hidden_layer_size", values=HP_HIDDEN_LAYER_SIZE_GRAPH)
        dropout_rate = hp.Choice("dropout_rate", values=HP_DROPOUT_RATE_GRAPH)
        max_gradient_norm = hp.Choice("max_gradient_norm", values=HP_MAX_GRADIENT_NORM_GRAPH)
        learning_rate = hp.Choice("learning_rate", values=HP_LEARNING_RATE_GRAPH)
        gat_units = hp.Choice("gat_units", values=HP_GCN_UNITS)
        attn_heads = hp.Choice("attn_heads", values=[2, 4, 8])
        num_gat_layers = hp.Choice("gat_layers", values=[2])

        # No graph needed for full attention
        self.A = None

        # Input shape: (batch_size, num_tickers, time_steps, input_size)
        input_layer = keras.Input(shape=(self.num_tickers, self.time_steps, self.input_size))

        # Shared LSTM layer
        shared_lstm = layers.LSTM(
            hidden_layer_size,
            return_sequences=True,
            dropout=dropout_rate,
            activation="tanh",
            recurrent_activation="sigmoid",
            name="shared_lstm"
        )

        lstm_outputs = []
        for i in range(self.num_tickers):
            ticker_slice = layers.Lambda(lambda x, idx=i: x[:, idx, :, :])(input_layer)
            ticker_output = shared_lstm(ticker_slice)
            lstm_outputs.append(ticker_output)

        # Stack: (batch, time_steps, num_tickers, hidden_size)
        stacked_lstm = layers.Lambda(lambda tensors: tf.stack(tensors, axis=2))(lstm_outputs)
        print("Stacked LSTM output shape:", stacked_lstm.shape)

        # Apply Full GAT layers with Layer Normalization (no adjacency mask)
        # Layer 1: concat heads for expressiveness
        gat_output = GraphAttentionLayer(
            units=gat_units,
            attn_heads=attn_heads,
            adjacency=None,  # No mask = full attention
            concat_heads=True,  # Concat for intermediate layer
            use_sparse=False,
            dropout_rate=dropout_rate,
            name="full_gat_1"
        )(stacked_lstm)
        gat_output = layers.LayerNormalization()(gat_output)  # Add layer norm
        gat_output = layers.ReLU()(gat_output)
        print("After Full GAT 1, shape:", gat_output.shape)

        if num_gat_layers == 2:
            # Layer 2 (final): average heads for stability
            gat_output = GraphAttentionLayer(
                units=gat_units,
                attn_heads=attn_heads,
                adjacency=None,
                concat_heads=False,  # Average for final layer
                use_sparse=False,
                dropout_rate=dropout_rate,
                name="full_gat_2"
            )(gat_output)
            gat_output = layers.LayerNormalization()(gat_output)  # Add layer norm
            gat_output = layers.ReLU()(gat_output)
            print("After Full GAT 2, shape:", gat_output.shape)
            gat_output_dim = gat_units  # Averaged heads = single unit dimension
        else:
            gat_output_dim = gat_units * attn_heads  # Concatenated = units * heads

        # Residual connection
        residual = layers.TimeDistributed(
            layers.TimeDistributed(
                keras.layers.Dense(gat_output_dim, activation="linear")
            )
        )(stacked_lstm)

        x = layers.Add()([gat_output, residual])

        # Output layer
        output = layers.TimeDistributed(
            layers.TimeDistributed(
                keras.layers.Dense(
                    self.output_size,
                    activation=tf.nn.tanh,
                    kernel_constraint=keras.constraints.max_norm(3),
                )
            )
        )(x)

        # Permute to (batch, num_tickers, time_steps, output_size)
        output = layers.Permute((2, 1, 3))(output)

        model = keras.Model(inputs=input_layer, outputs=output)
        print("Final model output shape:", output.shape)

        adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
        sharpe_loss = GraphSharpeLoss(self.output_size).call

        model.compile(loss=sharpe_loss, optimizer=adam)

        return model
