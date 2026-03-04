import tensorflow as tf
import numpy as np
import pandas as pd
import collections
from tensorflow import keras
import keras_tuner as kt
import os

from abc import ABC, abstractmethod

from settings.hp_grid import (
    HP_HIDDEN_LAYER_SIZE_GRAPH,
    HP_DROPOUT_RATE_GRAPH,
    HP_MAX_GRADIENT_NORM_GRAPH,
    HP_LEARNING_RATE_GRAPH,
    HP_MINIBATCH_SIZE_GRAPH,
    HP_GRAPH_ATTENTION_DIM,
)
from settings.default import ALL_TICKERS
from gml.model_inputs import ModelFeatures
from empyrical import sharpe_ratio

from gml.deep_neural_network import (
    TunerDiversifiedSharpe,
    TunerValidationLoss,
    SharpeLoss,
    SharpeValidationLoss,
)

from settings.fixed_params import MODEL_PARAMS_GRAPH

from gml.deep_neural_network import DeepMomentumNetworkModel

from tensorflow.keras import layers, optimizers, constraints

from spektral.layers import GCNConv
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from spektral.layers import GCNConv

# Assumed to be defined elsewhere:
# ALL_TICKERS, ModelFeatures, SharpeLoss, sharpe_ratio,
# TunerDiversifiedSharpe, TunerValidationLoss, HP_MINIBATCH_SIZE_GRAPH,
# HP_HIDDEN_LAYER_SIZE_GRAPH, HP_DROPOUT_RATE_GRAPH, HP_MAX_GRADIENT_NORM_GRAPH,
# HP_LEARNING_RATE_GRAPH, and kt.
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from spektral.layers import GCNConv

# Assumed to be defined elsewhere:
# ALL_TICKERS, ModelFeatures, SharpeLoss, sharpe_ratio,
# TunerDiversifiedSharpe, TunerValidationLoss, HP_MINIBATCH_SIZE_GRAPH,
# HP_HIDDEN_LAYER_SIZE_GRAPH, HP_DROPOUT_RATE_GRAPH, HP_MAX_GRADIENT_NORM_GRAPH,
# HP_LEARNING_RATE_GRAPH, and kt.

# class GraphDeepMomentumModel(DeepMomentumNetworkModel):
#     def __init__(self, project_name, graph_directory, hp_directory, hp_minibatch_size=HP_MINIBATCH_SIZE_GRAPH, **params):
#         # Read the constant graph structure from CSV.
#         if not os.path.isfile(graph_directory):
#             raise FileNotFoundError(f"Adjacency CSV not found at: {graph_directory}")
#         graph_structure = pd.read_csv(graph_directory, index_col=0)

#         # Filter the adjacency matrix to include only tickers present in ALL_TICKERS.
#         available_tickers = set(graph_structure.index)
#         print(f"available_tickers: {available_tickers}")
#         common_tickers = list(available_tickers.intersection(set(ALL_TICKERS)))
#         print(f"common_tickers: {common_tickers}")
#         if not common_tickers:
#             raise KeyError("None of the ALL_TICKERS are present in the adjacency matrix.")
#         filtered_graph = graph_structure.loc[common_tickers, common_tickers]
#         self.A = filtered_graph  # Expected shape: (N, N), e.g. (90, 90)
#         super().__init__(project_name, hp_directory, hp_minibatch_size, **params)

#     def model_builder(self, hp):
#         # Hyperparameter choices.
#         hidden_layer_size = hp.Choice("hidden_layer_size", values=HP_HIDDEN_LAYER_SIZE_GRAPH)
#         dropout_rate = hp.Choice("dropout_rate", values=HP_DROPOUT_RATE_GRAPH)
#         max_gradient_norm = hp.Choice("max_gradient_norm", values=HP_MAX_GRADIENT_NORM_GRAPH)
#         learning_rate = hp.Choice("learning_rate", values=HP_LEARNING_RATE_GRAPH)

#         n_nodes = self.A.shape[0]  # e.g. 90 nodes.
#         # The model expects each sample to be a full graph: shape (N, d)
#         X_in = keras.Input(shape=(n_nodes, self.input_size), name="X_in")

#         # Lambda layer that tiles the constant adjacency A to match the batch size.
#         A_const = tf.convert_to_tensor(np.array(self.A, dtype=np.float32))  # shape: (N, N)
#         def tile_A(x):
#             batch_size = tf.shape(x)[0]
#             return tf.tile(tf.expand_dims(A_const, axis=0), [batch_size, 1, 1])
#         A_tiled = layers.Lambda(tile_A, name="A_tiled")(X_in)

#         # Two graph convolution layers.
#         X_1 = GCNConv(hidden_layer_size, activation="relu")([X_in, A_tiled])
#         X_1 = layers.Dropout(dropout_rate)(X_1)
#         X_2 = GCNConv(hidden_layer_size, activation="relu")([X_1, A_tiled])
#         X_2 = layers.Dropout(dropout_rate)(X_2)
#         output = layers.Dense(self.output_size, activation=tf.nn.tanh,
#                               kernel_constraint=keras.constraints.max_norm(3))(X_2)

#         model = keras.Model(inputs=X_in, outputs=output)
#         adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
#         sharpe_loss = SharpeLoss(self.output_size).call
#         model.compile(loss=sharpe_loss, optimizer=adam, sample_weight_mode="temporal")
#         return model

#     def aggregate_by_time(self, data, time):
#         """
#         Aggregates raw samples by unique time stamps.
#         data: numpy array of shape (num_samples, n_nodes, d)
#         time: 1D array of length num_samples.

#         Returns:
#           aggregated: numpy array of shape (T, n_nodes, d), where T is the number of unique time stamps.
#           unique_times: array of unique time stamps.
#         """
#         df = pd.DataFrame({"time": pd.to_datetime(time.flatten()), "data": list(data)})
#         grouped = df.groupby("time")["data"].apply(lambda x: np.mean(np.stack(x.values, axis=0), axis=0))
#         aggregated = np.stack(grouped.values, axis=0)
#         unique_times = np.array(grouped.index)
#         return aggregated, unique_times

#     def aggregate_weights_by_time(self, weights, time):
#         """
#         Aggregates sample weights by unique time stamps.
#         weights: numpy array of shape (num_samples, ...) e.g. (num_samples, 1) or (num_samples, 1, 1)
#         time: 1D array of length num_samples.

#         Returns:
#           aggregated: numpy array of shape (T, 1, 1)
#           unique_times: array of unique time stamps.
#         """
#         df = pd.DataFrame({"time": pd.to_datetime(time.flatten()), "weight": weights.flatten()})
#         grouped = df.groupby("time")["weight"].mean()
#         aggregated = grouped.values.reshape(-1, 1, 1)
#         unique_times = np.array(grouped.index)
#         return aggregated, unique_times

#     def hyperparameter_search(self, train_data, valid_data):
#         # Unpack raw training and validation data.
#         data, labels, active_flags, _, time_train = ModelFeatures._unpack(train_data)
#         val_data, val_labels, val_flags, _, time_val = ModelFeatures._unpack(valid_data)
#         n_nodes = self.A.shape[0]
#         # If data comes as (num_samples, 1, d), tile it to (num_samples, n_nodes, d)
#         if data.ndim == 3 and data.shape[1] == 1 and n_nodes != 1:
#             data = np.tile(data, (1, n_nodes, 1))
#         if val_data.ndim == 3 and val_data.shape[1] == 1 and n_nodes != 1:
#             val_data = np.tile(val_data, (1, n_nodes, 1))

#         # Aggregate features and labels by time.
#         data_agg, unique_times_data = self.aggregate_by_time(data, time_train)
#         labels_agg, unique_times_labels = self.aggregate_by_time(labels, time_train)
#         weights_agg, unique_times_weights = self.aggregate_weights_by_time(active_flags, time_train)

#         # Compute the intersection of unique time stamps among features, labels, and weights.
#         common_times = np.intersect1d(unique_times_data, np.intersect1d(unique_times_labels, unique_times_weights))
#         # Filter aggregated arrays to only these common times.
#         idx_data = [i for i, t in enumerate(unique_times_data) if t in common_times]
#         idx_labels = [i for i, t in enumerate(unique_times_labels) if t in common_times]
#         idx_weights = [i for i, t in enumerate(unique_times_weights) if t in common_times]
#         data_agg = data_agg[idx_data]
#         labels_agg = labels_agg[idx_labels]
#         weights_agg = weights_agg[idx_weights]

#         # Set training inputs and returns at the aggregated level.
#         self.inputs = data_agg  # shape: (T_common, n_nodes, d)
#         # Here, we assume that labels_agg represent returns and aggregate over nodes (by mean).
#         self.returns = np.mean(labels_agg, axis=1, keepdims=True)  # shape: (T_common, 1, 1)
#         T_common = self.inputs.shape[0]
#         self.time_indices = np.arange(T_common)
#         self.num_time = T_common

#         self.tuner.search(
#             x=data_agg,
#             y=labels_agg,
#             sample_weight=weights_agg,
#             epochs=self.num_epochs,
#             callbacks=[
#                 SharpeValidationLoss(
#                     data_agg, val_labels, self.time_indices, self.num_time,
#                     self.early_stopping_patience, self.n_multiprocessing_workers,
#                 ),
#                 tf.keras.callbacks.TerminateOnNaN(),
#             ],
#             shuffle=True,
#             use_multiprocessing=True,
#             workers=self.n_multiprocessing_workers,
#         )

#         best_hp = self.tuner.get_best_hyperparameters(num_trials=1)[0].values
#         best_model = self.tuner.get_best_models(num_models=1)[0]
#         return best_hp, best_model


#     def get_positions(self, data, model, sliding_window=True, years_geq=np.iinfo(np.int32).min, years_lt=np.iinfo(np.int32).max):
#         # Unpack raw data.
#         inputs, outputs, active_entries, identifier, time = ModelFeatures._unpack(data)
#         # Aggregate features and outputs by time.
#         data_agg, unique_times = self.aggregate_by_time(inputs, time)
#         outputs_agg, _ = self.aggregate_by_time(outputs, time)
#         # Get predictions on aggregated data.
#         positions = model.predict(
#             data_agg,
#             workers=self.n_multiprocessing_workers,
#             use_multiprocessing=True,
#         )
#         # Aggregate predictions over nodes.
#         agg_positions = np.mean(positions, axis=1, keepdims=True)
#         agg_returns = np.mean(outputs_agg, axis=1, keepdims=True)
#         T = data_agg.shape[0]
#         self.time_indices = np.arange(T)
#         self.num_time = T
#         self.returns = agg_returns  # shape: (T, 1, 1)
#         captured_returns = tf.math.unsorted_segment_mean(agg_positions * agg_returns, self.time_indices, self.num_time)
#         performance = sharpe_ratio(captured_returns.numpy().flatten())
#         results = pd.DataFrame({
#             "time": unique_times,
#             "aggregated_position": agg_positions.flatten(),
#             "aggregated_return": agg_returns.flatten(),
#             "captured_returns": captured_returns.numpy().flatten()
#         })
#         return results, performance


#####################################################


# class GraphDeepMomentumModel(DeepMomentumNetworkModel):
#     def __init__(self, project_name, graph_directory, hp_directory,
#                  hp_minibatch_size=HP_MINIBATCH_SIZE_GRAPH, **params):
#         """
#         Hybrid model that incorporates graph-based stock embeddings into an LSTM.
#         It reads a graph structure (from CSV) to determine the number of stocks.
#         The model accepts two inputs:
#           1. A time-series for one stock: shape (time_steps, input_size)
#           2. A stock index (integer) that selects the corresponding embedding.
#         """
#         # Read the constant graph structure from CSV.
#         if not os.path.isfile(graph_directory):
#             raise FileNotFoundError(f"Adjacency CSV not found at: {graph_directory}")
#         graph_structure = pd.read_csv(graph_directory, index_col=0)

#         # Filter the adjacency matrix to include only tickers present in ALL_TICKERS.
#         available_tickers = set(graph_structure.index)
#         common_tickers = list(available_tickers.intersection(set(ALL_TICKERS)))

#         if not common_tickers:
#             raise KeyError("None of the ALL_TICKERS are present in the adjacency matrix.")
#         # Use the filtered graph for potential initialization (here we simply store it).
#         self.num_stocks = len(common_tickers)
#         self.A = graph_structure.loc[common_tickers, common_tickers]

#         super().__init__(project_name, hp_directory, hp_minibatch_size, **params)

#     def model_builder(self, hp):
#         """
#         Build a hybrid model that adds a stock embedding (learned from graph information)
#         to the original LSTM model. The model accepts two inputs:
#           - A time-series input: shape (time_steps, input_size)
#           - A stock index input: a scalar integer.
#         The embedding is repeated along time and concatenated with the time series,
#         then processed by an LSTM and a TimeDistributed Dense layer.
#         """
#         hidden_layer_size = hp.Choice("hidden_layer_size", values=HP_HIDDEN_LAYER_SIZE_GRAPH)
#         dropout_rate = hp.Choice("dropout_rate", values=HP_DROPOUT_RATE_GRAPH)
#         max_gradient_norm = hp.Choice("max_gradient_norm", values=HP_MAX_GRADIENT_NORM_GRAPH)
#         learning_rate = hp.Choice("learning_rate", values=HP_LEARNING_RATE_GRAPH)
#         embedding_dim = hp.Choice("embedding_dim", values=[16, 32])

#         # Original LSTM expects input shape (time_steps, input_size)
#         time_series_input = keras.Input(shape=(self.time_steps, self.input_size), name="time_series")
#         # Stock index input: a scalar integer.
#         stock_index_input = keras.Input(shape=(), dtype=tf.int32, name="stock_index")

#         # Graph-based stock embedding.
#         stock_embedding = layers.Embedding(input_dim=self.num_stocks,
#                                            output_dim=embedding_dim,
#                                            name="stock_embedding")(stock_index_input)  # shape: (embedding_dim,)
#         # Repeat the embedding for each time step.
#         repeated_embedding = layers.RepeatVector(self.time_steps)(stock_embedding)  # (time_steps, embedding_dim)
#         # Concatenate the raw time-series with the repeated embedding along the feature dimension.
#         combined = layers.Concatenate(axis=-1)([time_series_input, repeated_embedding])
#         # Now combined has shape (time_steps, input_size + embedding_dim)

#         # Process with LSTM.
#         lstm_out = layers.LSTM(hidden_layer_size,
#                                return_sequences=True,
#                                dropout=dropout_rate,
#                                stateful=False,
#                                activation="tanh",
#                                recurrent_activation="sigmoid",
#                                recurrent_dropout=0,
#                                unroll=False,
#                                use_bias=True)(combined)
#         dropout_layer = layers.Dropout(dropout_rate)(lstm_out)
#         output = layers.TimeDistributed(layers.Dense(self.output_size,
#                                                      activation=tf.nn.tanh,
#                                                      kernel_constraint=keras.constraints.max_norm(3)))(dropout_layer)

#         model = keras.Model(inputs=[time_series_input, stock_index_input], outputs=output)
#         adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
#         sharpe_loss = SharpeLoss(self.output_size).call
#         model.compile(loss=sharpe_loss, optimizer=adam, sample_weight_mode="temporal")
#         return model

#     def aggregate_by_time(self, data, time):
#         """
#         Aggregates raw samples by unique time stamps.
#         data: numpy array of shape (num_samples, time_steps, d)
#         time: 1D array of length num_samples.

#         Returns:
#           aggregated: numpy array of shape (T, time_steps, d)
#           unique_times: array of unique time stamps.
#         """
#         df = pd.DataFrame({"time": pd.to_datetime(time.flatten()), "data": list(data)})
#         grouped = df.groupby("time")["data"].apply(lambda x: np.mean(np.stack(x.values, axis=0), axis=0))
#         aggregated = np.stack(grouped.values, axis=0)
#         unique_times = np.array(grouped.index)
#         return aggregated, unique_times

#     def aggregate_weights_by_time(self, weights, time):
#         """
#         Aggregates sample weights by unique time stamps.
#         weights: numpy array of shape (num_samples, ...) e.g., (num_samples, 1) or (num_samples, 1, 1)
#         time: 1D array of length num_samples.

#         Returns:
#           aggregated: numpy array of shape (T, 1, 1)
#           unique_times: array of unique time stamps.
#         """
#         df = pd.DataFrame({"time": pd.to_datetime(time.flatten()), "weight": weights.flatten()})
#         grouped = df.groupby("time")["weight"].mean()
#         aggregated = grouped.values.reshape(-1, 1, 1)
#         unique_times = np.array(grouped.index)
#         return aggregated, unique_times

#     def hyperparameter_search(self, train_data, valid_data):
#         """
#         Unpacks and aggregates raw training data by time.
#         Since the original LSTM model processed one stock at a time (shape: (time_steps, d)),
#         we assume here that each raw sample corresponds to one stock’s time-series.
#         We aggregate the samples by time (using mean) so that x, y, and sample_weight all share
#         the same first dimension (number of unique time stamps).
#         For the hybrid model, we also supply a dummy stock index for each aggregated sample.
#         """
#         data, labels, active_flags, _, time_train = ModelFeatures._unpack(train_data)
#         val_data, val_labels, val_flags, _, time_val = ModelFeatures._unpack(valid_data)

#         # Assume data has shape (num_samples, time_steps, d). If not, tile accordingly.
#         if data.ndim == 3 and data.shape[1] == 1:
#             # This would be the case if each sample is a single time point;
#             # but our LSTM expects self.time_steps.
#             # (Adjust as needed.)
#             data = np.tile(data, (1, self.time_steps, 1))
#         if val_data.ndim == 3 and val_data.shape[1] == 1:
#             val_data = np.tile(val_data, (1, self.time_steps, 1))

#         # Aggregate features and labels by time.
#         data_agg, unique_times_data = self.aggregate_by_time(data, time_train)
#         labels_agg, unique_times_labels = self.aggregate_by_time(labels, time_train)
#         weights_agg, unique_times_weights = self.aggregate_weights_by_time(active_flags, time_train)

#         # Compute intersection of unique time stamps.
#         common_times = np.intersect1d(unique_times_data, np.intersect1d(unique_times_labels, unique_times_weights))
#         idx_data = [i for i, t in enumerate(unique_times_data) if t in common_times]
#         idx_labels = [i for i, t in enumerate(unique_times_labels) if t in common_times]
#         idx_weights = [i for i, t in enumerate(unique_times_weights) if t in common_times]
#         data_agg = data_agg[idx_data]
#         labels_agg = labels_agg[idx_labels]
#         weights_agg = weights_agg[idx_weights]

#         # Set training inputs and returns at the aggregated level.
#         self.inputs = data_agg  # shape: (T_common, time_steps, d)
#         # Assume labels_agg are the returns; average over the time axis is not needed because they are already aggregated.
#         self.returns = np.mean(labels_agg, axis=1, keepdims=True)  # shape: (T_common, 1, 1)
#         T_common = self.inputs.shape[0]
#         self.time_indices = np.arange(T_common)
#         self.num_time = T_common

#         # For the hybrid model, supply a dummy stock index for each aggregated sample.
#         dummy_stock_indices = np.zeros((T_common,), dtype=np.int32)

#         self.tuner.search(
#             x=[data_agg, dummy_stock_indices],
#             y=labels_agg,
#             sample_weight=weights_agg,
#             epochs=self.num_epochs,
#             callbacks=[
#                 SharpeValidationLoss(
#                     data_agg, val_labels, self.time_indices, self.num_time,
#                     self.early_stopping_patience, self.n_multiprocessing_workers,
#                 ),
#                 tf.keras.callbacks.TerminateOnNaN(),
#             ],
#             shuffle=True,
#             use_multiprocessing=True,
#             workers=self.n_multiprocessing_workers,
#         )

#         best_hp = self.tuner.get_best_hyperparameters(num_trials=1)[0].values
#         best_model = self.tuner.get_best_models(num_models=1)[0]
#         return best_hp, best_model

#     def _index_times(self, val_time):
#         # Not used, as aggregation is done in hyperparameter_search.
#         indices = np.arange(len(val_time))
#         num_unique = len(np.unique(val_time))
#         return indices, num_unique

#     def get_positions(self, data, model, sliding_window=True,
#                       years_geq=np.iinfo(np.int32).min, years_lt=np.iinfo(np.int32).max):
#         """
#         Computes model predictions and aggregates them by time.
#         """
#         inputs, outputs, active_entries, identifier, time = ModelFeatures._unpack(data)
#         data_agg, unique_times = self.aggregate_by_time(inputs, time)
#         outputs_agg, _ = self.aggregate_by_time(outputs, time)
#         positions = model.predict(
#             [data_agg, np.zeros((data_agg.shape[0],), dtype=np.int32)],
#             workers=self.n_multiprocessing_workers,
#             use_multiprocessing=True,
#         )
#         # Aggregate predictions over time: here we assume each aggregated sample is already one time stamp.
#         agg_positions = np.mean(positions, axis=1, keepdims=True)
#         agg_returns = np.mean(outputs_agg, axis=1, keepdims=True)
#         T = data_agg.shape[0]
#         self.time_indices = np.arange(T)
#         self.num_time = T
#         self.returns = agg_returns  # shape: (T, 1, 1)
#         captured_returns = tf.math.unsorted_segment_mean(agg_positions * agg_returns, self.time_indices, self.num_time)
#         performance = sharpe_ratio(captured_returns.numpy().flatten())
#         results = pd.DataFrame({
#             "time": unique_times,
#             "aggregated_position": agg_positions.flatten(),
#             "aggregated_return": agg_returns.flatten(),
#             "captured_returns": captured_returns.numpy().flatten()
#         })
#         return results, performance



class Ensure4D(tf.keras.layers.Layer):
    def __init__(self, total_assets, **kwargs):
        # Mark layer as dynamic to support unknown shapes.
        kwargs['dynamic'] = True
        super(Ensure4D, self).__init__(**kwargs)
        self.total_assets = total_assets

    def call(self, x):
        # If x is 3D (batch, T, input_size), expand to (batch, T, total_assets, input_size)
        return tf.cond(
            tf.equal(tf.rank(x), 3),
            lambda: tf.tile(tf.expand_dims(x, axis=2), [1, 1, self.total_assets, 1]),
            lambda: x
        )

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 3:
            return (input_shape[0], input_shape[1], self.total_assets, input_shape[2])
        return input_shape

    def get_config(self):
        config = super(Ensure4D, self).get_config()
        config.update({"total_assets": self.total_assets})
        return config

# ----------------------------------------------------------------------
# Base class (unchanged)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Custom Graph Attention Layer (with compute_output_shape)
# ----------------------------------------------------------------------
class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, dropout_rate=0.0, adj_matrix=None, **kwargs):
        local_kwargs = kwargs.copy()
        if "adj_matrix" in local_kwargs:
            local_kwargs.pop("adj_matrix")
        super(GraphAttentionLayer, self).__init__(**local_kwargs)
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.adj_matrix = adj_matrix  # Expected shape: (total_assets, total_assets)
    def build(self, input_shape):
        # input_shape: (batch, total_assets, F)
        self.W = self.add_weight(
            shape=(input_shape[-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="W",
        )
        self.a = self.add_weight(
            shape=(2 * self.output_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
            name="a",
        )
        super(GraphAttentionLayer, self).build(input_shape)
    def call(self, inputs):
        # inputs: (batch, total_assets, F)
        h = tf.matmul(inputs, self.W)  # -> (batch, total_assets, output_dim)
        N = tf.shape(h)[1]             # total_assets
        h_i = tf.expand_dims(h, axis=2)  # (batch, total_assets, 1, output_dim)
        h_j = tf.expand_dims(h, axis=1)  # (batch, 1, total_assets, output_dim)
        a_input = tf.concat([tf.tile(h_i, [1, 1, N, 1]),
                             tf.tile(h_j, [1, N, 1, 1])],
                            axis=-1)  # (batch, total_assets, total_assets, 2*output_dim)
        e = tf.nn.leaky_relu(tf.tensordot(a_input, self.a, axes=1))  # (batch, total_assets, total_assets, 1)
        e = tf.squeeze(e, axis=-1)  # (batch, total_assets, total_assets)
        if self.adj_matrix is not None:
            A = tf.convert_to_tensor(self.adj_matrix, dtype=tf.float32)
            A = tf.expand_dims(A, axis=0)  # (1, total_assets, total_assets)
            e = e + (1.0 - A) * (-1e9)
        attention = tf.nn.softmax(e, axis=-1)  # (batch, total_assets, total_assets)
        attention = tf.nn.dropout(attention, rate=self.dropout_rate)
        output = tf.matmul(attention, h)  # (batch, total_assets, output_dim)
        return output
    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.output_dim])

# ----------------------------------------------------------------------
# New Model: GraphLstmDeepMomentumNetworkModel
# ----------------------------------------------------------------------
class GraphDeepMomentumModel(DeepMomentumNetworkModel):
    def __init__(self, project_name, graph_path, hp_directory,
                 hp_minibatch_size, **params):
        """
        Instead of passing a matrix directly, we pass a file path (graph_path)
        to the CSV file containing the adjacency matrix.
        """
        # Read the adjacency matrix from file.
        df = pd.read_csv(graph_path, index_col=0)
        # Reindex to ensure the rows and columns match ALL_TICKERS.
        df = df.reindex(index=ALL_TICKERS, columns=ALL_TICKERS, fill_value=0)
        self.adj_matrix = df.values  # shape: (90, 90)
        super().__init__(project_name, hp_directory, hp_minibatch_size, **params)
    def model_builder(self, hp):
        hidden_layer_size = hp.Choice("hidden_layer_size", values=HP_HIDDEN_LAYER_SIZE_GRAPH)
        dropout_rate = hp.Choice("dropout_rate", values=HP_DROPOUT_RATE_GRAPH)
        max_gradient_norm = hp.Choice("max_gradient_norm", values=HP_MAX_GRADIENT_NORM_GRAPH)
        learning_rate = hp.Choice("learning_rate", values=HP_LEARNING_RATE_GRAPH)
        graph_attention_dim = hp.Choice("graph_attention_dim", values=HP_GRAPH_ATTENTION_DIM)
        total_assets = len(ALL_TICKERS)  # 90
        # The model now accepts a 3D input of shape (T, input_size) where input_size is the
        # number of features per asset. We will then expand this to 4D by tiling the input for all assets.
        input_layer = keras.Input((self.time_steps, self.input_size))
        preprocessed = Ensure4D(total_assets)(input_layer)
        # preprocessed shape: (batch, T, 90, input_size)
        asset_embeddings = keras.layers.TimeDistributed(
            keras.layers.LSTM(hidden_layer_size, return_sequences=True,
                              dropout=dropout_rate, activation='tanh')
        )(preprocessed)
        # asset_embeddings shape: (batch, T, 90, hidden_layer_size)
        graph_attention_out = keras.layers.TimeDistributed(
            GraphAttentionLayer(graph_attention_dim, dropout_rate=dropout_rate, adj_matrix=self.adj_matrix)
        )(asset_embeddings)
        # graph_attention_out shape: (batch, T, 90, graph_attention_dim)
        output = keras.layers.TimeDistributed(
            keras.layers.TimeDistributed(
                keras.layers.Dense(self.output_size, activation='tanh', kernel_constraint=keras.constraints.max_norm(3))
            )
        )(graph_attention_out)
        # output shape: (batch, T, 90, output_size)
        model = keras.Model(inputs=input_layer, outputs=output)
        adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
        sharpe_loss = SharpeLoss(self.output_size).call
        model.compile(
            loss=sharpe_loss,
            optimizer=adam,
            sample_weight_mode="temporal",
        )
        return model
        
class LstmGATDeepMomentumNetworkModel(DeepMomentumNetworkModel):
    def __init__(self, project_name, adjacency_matrix_path, hp_directory, hp_minibatch_size, **params):

        df = pd.read_csv(adjacency_matrix_path, index_col=0)
        self.num_stocks = df.shape[0]
        # Reindex to ensure the rows and columns match ALL_TICKERS.
        df = df.reindex(index=ALL_TICKERS, columns=ALL_TICKERS, fill_value=0)
        
        print(df.head())
        self.adj_matrix = df.values  # shape: (90, 90)
        super().__init__(project_name, hp_directory, hp_minibatch_size, **params)
    
    def model_builder(self, hp):
        # Hyperparameter selections for the network.
        hidden_layer_size = hp.Choice("hidden_layer_size", values=HP_HIDDEN_LAYER_SIZE_GRAPH)
        dropout_rate = hp.Choice("dropout_rate", values=HP_DROPOUT_RATE_GRAPH)
        max_gradient_norm = hp.Choice("max_gradient_norm", values=HP_MAX_GRADIENT_NORM_GRAPH)
        learning_rate = hp.Choice("learning_rate", values=HP_LEARNING_RATE_GRAPH)
        graph_attention_dim = hp.Choice("graph_attention_dim", values=HP_GRAPH_ATTENTION_DIM)

        # Input shape: (batch_size, time_steps, num_stocks, input_size)
        # - time_steps: number of time steps in the sequence.
        # - num_stocks: number of stocks (nodes).
        # - input_size: number of features per stock.
        input_layer = keras.Input((self.time_steps, self.input_size))  # (None, 63, 8)
        print(f"self.time_steps: {self.time_steps}")
        print(f"self.num_stocks: {self.num_stocks}")
        print(f"self.input_size: {self.input_size}")
        
        # -------------------------------------------------------------
        # 1) Process each stock's time series with an LSTM.
        # We want the LSTM to be applied to each stock individually.
        # To do this, we permute the dimensions so that the stock axis comes first.
        # From: (batch, time_steps, num_stocks, input_size)
        # To:   (batch, num_stocks, time_steps, input_size)
        permuted = keras.layers.Permute((2, 1, 3))(input_layer)
        
        # Apply TimeDistributed LSTM across stocks.
        # This applies the LSTM to each stock's time series of shape (time_steps, input_size).
        # The LSTM returns a sequence of hidden states: (batch, num_stocks, time_steps, hidden_layer_size)
        stock_embedding = keras.layers.TimeDistributed(
            keras.layers.LSTM(
                hidden_layer_size,
                return_sequences=True,
                dropout=dropout_rate,
                stateful=False,
                activation="tanh",
                recurrent_activation="sigmoid",
                recurrent_dropout=0,
                unroll=False,
                use_bias=True,
            )
        )(permuted)
        
        # Apply dropout to the LSTM outputs.
        stock_embedding = keras.layers.Dropout(dropout_rate)(stock_embedding)
        
        # -------------------------------------------------------------
        # 2) Apply Graph Attention across stocks at each time step.
        # We want to combine the per-stock features using the fixed adjacency matrix A.
        # First, permute to bring the time dimension to the front:
        # From: (batch, num_stocks, time_steps, hidden_layer_size)
        # To:   (batch, time_steps, num_stocks, hidden_layer_size)
        permuted_for_graph = keras.layers.Permute((2, 1, 3))(stock_embedding)
        
        # Apply a TimeDistributed Graph Attention layer.
        # The GraphAttentionLayer should be defined to accept inputs of shape (batch, num_stocks, features)
        # and use the predefined adjacency matrix A.
        graph_attention_out = keras.layers.TimeDistributed(
            GraphAttentionLayer(graph_attention_dim, dropout_rate=dropout_rate)
        )(permuted_for_graph)
        
        graph_attention_out = keras.layers.Dropout(dropout_rate)(graph_attention_out)
        
        # -------------------------------------------------------------
        # 3) Final TimeDistributed Dense layer:
        # Apply a Dense mapping to each stock's feature at each time step.
        # This maps from the graph attention output dimension to the final output dimension.
        # Output shape will be (batch, time_steps, num_stocks, output_size)
        output = keras.layers.TimeDistributed(
            keras.layers.TimeDistributed(
                keras.layers.Dense(
                    self.output_size,
                    activation=tf.nn.tanh,
                    kernel_constraint=keras.constraints.max_norm(3),
                )
            )
        )(graph_attention_out)
        
        # If needed, you can permute or aggregate across stocks.
        # For instance, if you want one output per time step (e.g. aggregated across stocks),
        # you could apply a pooling operation here.
        # For now, we assume the output per stock is desired.
        
        # Build the final Keras Model.
        model = keras.Model(inputs=input_layer, outputs=output)
        
        # Configure the optimizer with gradient clipping.
        adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
        
        # Define the custom Sharpe loss function.
        sharpe_loss = SharpeLoss(self.output_size).call
        
        # Compile the model with the Sharpe loss and Adam optimizer.
        # sample_weight_mode="temporal" indicates that sample weights may vary over time steps.
        model.compile(
            loss=sharpe_loss,
            optimizer=adam,
            sample_weight_mode="temporal",
        )
        
        return model
    
##############################################################################
from tensorflow.keras import layers

class GCLSTMCell(layers.Layer):
    def __init__(self, units, adjacency_matrix, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.adjacency_matrix = adjacency_matrix
        # LSTM parameters
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        
        # Example of a simple graph convolution weight for demonstration
        self.graph_weight = self.add_weight(
            shape=(adjacency_matrix.shape[1], adjacency_matrix.shape[1]),
            initializer='glorot_uniform',
            trainable=True,
            name='graph_weight'
        )

    def call(self, inputs, states, **kwargs):
        """
        inputs: shape (batch_size, features)
        states: previous hidden/cell states for LSTM
        """
        # 1) Graph convolution step (example)
        #    If 'inputs' is shape (batch_size, N, features),
        #    you might multiply adjacency with inputs or hidden states.
        #    But in a single LSTMCell call, we typically have shape (batch_size, features).
        #    For a full graph, you'd do adjacency * hidden states across nodes, etc.
        
        # For demonstration, let's pretend 'inputs' is already shaped for a graph:
        # shape (batch_size, num_nodes, features). 
        # We'll do a simple adjacency multiplication:
        #     X' = A * X * W
        # But this is just an example; real usage depends on how your data is batched.
        
        # NOTE: If your real data is shape (batch_size, num_nodes, features),
        #       you'd do something like:
        # X_gc = tf.einsum('ij,bjf->bif', self.adjacency_matrix, inputs)
        # X_gc = tf.einsum('ij,bjf->bif', self.graph_weight, X_gc)
        
        # For a minimal example (assuming 2D inputs), we’ll skip that:
        x_gc = inputs  # no actual GC for the simplified example

        # 2) Pass the "graph-convolved" inputs into the LSTM cell
        output, new_states = self.lstm_cell(x_gc, states=states)
        return output, new_states

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.lstm_cell.get_initial_state(
            inputs=inputs, batch_size=batch_size, dtype=dtype
        )

# Wrap the custom cell in a Keras RNN
class GCLSTM(layers.RNN):
    def __init__(self, units, adjacency_matrix, return_sequences=False, **kwargs):
        cell = GCLSTMCell(units, adjacency_matrix)
        super().__init__(cell, return_sequences=return_sequences, **kwargs)

    def call(self, inputs, **kwargs):
        return super().call(inputs, **kwargs)
   
    
    
    
    
    
    
    
    
    
    
    
class GCLstmDeepMomentumNetworkModel(DeepMomentumNetworkModel):
    def __init__(
        self,
        project_name,
        hp_directory,
        hp_minibatch_size=HP_MINIBATCH_SIZE_GRAPH,
        adjacency_matrix_path=None,  # New parameter for the adjacency matrix file path
        **params
    ):
        if adjacency_matrix_path is None:
            raise ValueError("An adjacency matrix path must be provided for GC_LSTM.")
        self.adjacency_matrix_path = adjacency_matrix_path
        super().__init__(project_name, hp_directory, hp_minibatch_size, **params)

    def model_builder(self, hp):
        # Hyperparameter selections for the network.
        hidden_layer_size = hp.Choice("hidden_layer_size", values=HP_HIDDEN_LAYER_SIZE_GRAPH)
        dropout_rate = hp.Choice("dropout_rate", values=HP_DROPOUT_RATE_GRAPH)
        max_gradient_norm = hp.Choice("max_gradient_norm", values=HP_MAX_GRADIENT_NORM_GRAPH)
        learning_rate = hp.Choice("learning_rate", values=HP_LEARNING_RATE_GRAPH)

        # 1) Load the adjacency matrix from a CSV file.
        #    - We assume the CSV has row labels (e.g., stock tickers) in the first column
        #      and column labels in the first row. For example:
        #          , AAPL, ABT, ACN, ADBE, ...
        #        AAPL, 0,   1,   0,   0,    ...
        #        ABT,  1,   0,   1,   0,    ...
        #        ...
        #    - `index_col=0` will use the first column as row labels,
        #      so the matrix values will be in the DataFrame body.
        df_adj = pd.read_csv(self.adjacency_matrix_path, index_col=0)
        A = df_adj.values  # Convert to a NumPy array
        # Optional: ensure float type, if needed:
        # A = df_adj.to_numpy(dtype="float32")

        # Input shape: (batch_size, time_steps, input_size)
        input_layer = keras.Input((self.time_steps, self.input_size))
        print(f"self.time_steps: {self.time_steps}")
        print(f"self.input_size: {self.input_size}")

        # 2) Use GC_LSTM instead of a standard LSTM. We pass in A (the adjacency matrix).
        gc_lstm = GCLSTM(
            hidden_layer_size,
            adjacency_matrix=A,
            return_sequences=True,
            dropout=dropout_rate,
            stateful=False,
            activation="tanh",
            recurrent_activation="sigmoid",
            recurrent_dropout=0,
            unroll=False,
            use_bias=True,
        )(input_layer)
        
        # 3) Apply dropout to the GC_LSTM outputs to prevent overfitting.
        dropout = keras.layers.Dropout(dropout_rate)(gc_lstm)

        # 4) TimeDistributed Dense layer to map each time step to the desired output.
        output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                self.output_size,
                activation=tf.nn.tanh,
                kernel_constraint=keras.constraints.max_norm(3),
            )
        )(dropout)

        # 5) Build the final Keras Model.
        model = keras.Model(inputs=input_layer, outputs=output)

        # 6) Configure the optimizer with gradient clipping.
        adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)

        # 7) Define the custom Sharpe loss function.
        sharpe_loss = SharpeLoss(self.output_size).call

        # 8) Compile the model with the Sharpe loss and Adam optimizer.
        model.compile(
            loss=sharpe_loss,
            optimizer=adam,
            sample_weight_mode="temporal",
        )
        
        return model