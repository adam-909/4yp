print("=" * 70)
print("LOADING graph_model_2.py - VERSION WITH run_trial DEBUG PRINTS")
print("=" * 70)

import tensorflow as tf
import numpy as np
import pandas as pd
import collections
from tensorflow import keras
import keras_tuner as kt
import copy
import os
import matplotlib.pyplot as plt

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
    HP_CORRELATION_LOOKBACK,
    HP_CORRELATION_THRESHOLD,
)
from settings.default import ALL_TICKERS
# from gml.model_inputs import ModelFeatures
from gml.graph_model_inputs import GraphModelFeatures

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

# from spektral.layers import GCNConv  # Not used - using custom GraphConvolution


class GraphSharpeLoss(tf.keras.losses.Loss):
    def __init__(self, output_size: int = 1):
        self.output_size = output_size
        super().__init__()

    def call(self, y_true, weights):
        """
        y_true  shape: (batch_size, N, 1)
        y_pred  shape: (batch_size, N, 1)
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


class GraphSharpeValidationLoss(keras.callbacks.Callback):
    def __init__(
        self, 
        inputs,
        returns, 
        time_indices,
        num_time, 
        early_stopping_patience, 
        n_multiprocessing_workers,
        weights_save_location="tmp/checkpoint",
        min_delta=1e-4,
    ):
        super(keras.callbacks.Callback, self).__init__()
        self.inputs = inputs
        self.returns = returns
        self.time_indices = time_indices
        self.n_multiprocessing_workers = n_multiprocessing_workers
        self.early_stopping_patience = early_stopping_patience
        self.num_time = num_time
        self.min_delta = min_delta

        self.best_sharpe = -np.inf  # since calculating positive Sharpe...
        self.weights_save_location = weights_save_location

    def set_weights_save_loc(self, weights_save_location):
        self.weights_save_location = weights_save_location

    def on_train_begin(self, logs=None):
        self.patience_counter = 0
        self.stopped_epoch = 0
        self.best_sharpe = -np.inf

    def on_epoch_end(self, epoch, logs=None):

        positions = self.model.predict(self.inputs)

        # ???
        positions_flat = tf.reshape(positions, [-1])
        returns_flat = tf.reshape(self.returns, [-1])
        time_indices_flat = tf.reshape(self.time_indices, [-1])
        
        # captured_returns = positions_flat * returns_flat
        # daily_returns = tf.math.unsorted_segment_mean(
        #     captured_returns, time_indices_flat, self.num_time,
        # )[1:]
        
        
        captured_returns = tf.math.unsorted_segment_mean(
            positions * self.returns, self.time_indices, self.num_time
        )[1:]
    
        # sharpe = (
        #     tf.reduce_mean(daily_returns)
        #     / tf.sqrt(
        #         tf.math.reduce_variance(daily_returns)
        #         + tf.constant(1e-9, dtype=tf.float64)
        #     )
        #     * tf.sqrt(tf.constant(252.0, dtype=tf.float64))
        # ).numpy()
        
        sharpe = (
            tf.reduce_mean(captured_returns)
            / tf.sqrt(
                tf.math.reduce_variance(captured_returns)
                + tf.constant(1e-9, dtype=tf.float64)
            )
            * tf.sqrt(tf.constant(252.0, dtype=tf.float64))
        ).numpy()
        
        if sharpe > self.best_sharpe + self.min_delta:
            self.best_sharpe = sharpe
            self.patience_counter = 0  # reset the count
            # self.best_weights = self.model.get_weights()
            self.model.save_weights(self.weights_save_location)
        else:
            # if self.verbose: #TODO
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.load_weights(self.weights_save_location)
        logs["sharpe"] = sharpe  # for keras tuner
        print(f"\nval_sharpe {logs['sharpe']}")
        
        
class GraphTunerValidationLoss(kt.RandomSearch): # TODO changed
    def __init__(
        self,
        hypermodel,
        objective,
        max_trials,
        hp_minibatch_size,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        **kwargs,
    ):
        self.hp_minibatch_size = hp_minibatch_size
        self._trained_models = {}  # Store trained models in memory
        super().__init__(
            hypermodel,
            objective,
            max_trials,
            seed,
            hyperparameters,
            tune_new_entries,
            allow_new_entries,
            **kwargs,
        )

    def run_trial(self, trial, *args, **kwargs):
        print(f"\n{'='*60}")
        print(f"DEBUG: GraphTunerValidationLoss.run_trial() CALLED")
        print(f"DEBUG: trial.trial_id = {trial.trial_id}")
        print(f"{'='*60}\n")

        kwargs["batch_size"] = trial.hyperparameters.Choice(
            "batch_size", values=self.hp_minibatch_size
        )
        # Add restore_best_weights to keep best weights in memory
        original_callbacks = kwargs.get("callbacks", [])
        for cb in original_callbacks:
            if isinstance(cb, tf.keras.callbacks.EarlyStopping):
                cb.restore_best_weights = True

        # Build and train model, then store in memory
        print("DEBUG: Building model...")
        model = self.hypermodel.build(trial.hyperparameters)
        print("DEBUG: Starting model.fit()...")
        history = model.fit(*args, **kwargs)
        print("DEBUG: model.fit() completed!")
        self._trained_models[trial.trial_id] = model
        return history

    def get_best_models(self, num_models=1):
        """Return models from memory instead of loading from disk."""
        best_trials = self.oracle.get_best_trials(num_models)
        models = []
        for trial in best_trials:
            if trial.trial_id in self._trained_models:
                models.append(self._trained_models[trial.trial_id])
            else:
                # Try loading from disk, rebuild if that fails
                try:
                    models.append(super().load_model(trial))
                except (FileNotFoundError, OSError):
                    # Checkpoint missing - rebuild model with best HP (untrained)
                    print(f"Warning: Could not load weights for {trial.trial_id}, rebuilding model")
                    model = self.hypermodel.build(trial.hyperparameters)
                    models.append(model)
        return models


class GraphTunerDiversifiedSharpe(kt.RandomSearch):
    def __init__(
        self,
        hypermodel,
        objective,
        max_trials,
        hp_minibatch_size,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        **kwargs,
    ):
        self.hp_minibatch_size = hp_minibatch_size
        self._trained_models = {}  # Store trained models in memory

        super().__init__(
            hypermodel,
            objective,
            max_trials,
            seed,
            hyperparameters,
            tune_new_entries,
            allow_new_entries,
            **kwargs,
        )

    def run_trial(self, trial, *args, **kwargs):
        print(f"\n{'='*60}")
        print(f"DEBUG: GraphTunerDiversifiedSharpe.run_trial() CALLED")
        print(f"DEBUG: trial.trial_id = {trial.trial_id}")
        print(f"DEBUG: executions_per_trial = {self.executions_per_trial}")
        print(f"{'='*60}\n")

        kwargs["batch_size"] = trial.hyperparameters.Choice(
            "batch_size", values=self.hp_minibatch_size
        )

        original_callbacks = kwargs.pop("callbacks", [])

        # Run the training process multiple times.
        metrics = collections.defaultdict(list)
        best_model = None
        for execution in range(self.executions_per_trial):
            print(f"DEBUG: Starting execution {execution + 1}/{self.executions_per_trial}")
            copied_fit_kwargs = copy.copy(kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)

            # Set checkpoint path AFTER deep copy to ensure it's preserved
            for callback in callbacks:
                if isinstance(callback, GraphSharpeValidationLoss):
                    callback.set_weights_save_loc(
                        self._get_checkpoint_fname(trial.trial_id, self._reported_step)
                    )

            self._configure_tensorboard_dir(callbacks, trial, execution)
            callbacks.append(kt.engine.tuner_utils.TunerCallback(self, trial))
            copied_fit_kwargs["callbacks"] = callbacks

            # Build model explicitly so we can store it
            model = self.hypermodel.build(trial.hyperparameters)
            history = model.fit(*args, **copied_fit_kwargs)

            for metric, epoch_values in history.history.items():
                if self.oracle.objective.direction == "min":
                    best_value = np.min(epoch_values)
                else:
                    best_value = np.max(epoch_values)
                metrics[metric].append(best_value)

            # Keep the last trained model
            best_model = model

        # Store the trained model in memory
        self._trained_models[trial.trial_id] = best_model

        # Average the results across executions and send to the Oracle.
        averaged_metrics = {}
        for metric, execution_values in metrics.items():
            averaged_metrics[metric] = np.mean(execution_values)
        self.oracle.update_trial(
            trial.trial_id, metrics=averaged_metrics, step=self._reported_step
        )

    def get_best_models(self, num_models=1):
        """Return models from memory instead of loading from disk."""
        best_trials = self.oracle.get_best_trials(num_models)
        models = []
        for trial in best_trials:
            if trial.trial_id in self._trained_models:
                models.append(self._trained_models[trial.trial_id])
            else:
                # Try loading from disk, rebuild if that fails
                try:
                    models.append(super().load_model(trial))
                except (FileNotFoundError, OSError):
                    # Checkpoint missing - rebuild model with best HP (untrained)
                    print(f"Warning: Could not load weights for {trial.trial_id}, rebuilding model")
                    model = self.hypermodel.build(trial.hyperparameters)
                    models.append(model)
        return models


class GraphDeepMomentumNetwork(ABC):
    # def __init__(self, project_name, hp_directory, hp_minibatch_size, graph_directory, **params):
    def __init__(self, project_name, hp_directory, hp_minibatch_size, **params):

        self.time_steps = int(params["total_time_steps"])
        self.input_size = int(params["input_size"])
        self.output_size = int(params["output_size"])
        self.num_tickers = int(params["num_tickers"])
        self.n_multiprocessing_workers = int(params["multiprocessing_workers"])
        self.num_epochs = int(params["num_epochs"])
        self.early_stopping_patience = int(params["early_stopping_patience"])
        # self.sliding_window = params["sliding_window"]
        self.random_search_iterations = params["random_search_iterations"]
        self.evaluate_diversified_val_sharpe = params["evaluate_diversified_val_sharpe"]
        self.force_output_sharpe_length = params["force_output_sharpe_length"]
        self.A = None

        # df = pd.read_csv(graph_directory, index_col=0)
        # df = df.reindex(index=ALL_TICKERS, columns=ALL_TICKERS)
        # self.A = df.values
        
        print("\n")
        print("Deep Momentum Network params:")
        for k in params:
            print(f"{k} = {params[k]}")
        print("\n")
        
        # To build model
        def model_builder(hp):
            return self.model_builder(hp)

        if self.evaluate_diversified_val_sharpe:
            print(f"DEBUG: Creating GraphTunerDiversifiedSharpe with overwrite=True")
            print(f"DEBUG: directory={hp_directory}, project_name={project_name}")
            self.tuner = GraphTunerDiversifiedSharpe(
                model_builder,
                # objective="val_loss",
                objective=kt.Objective("sharpe", "max"),
                hp_minibatch_size=hp_minibatch_size,
                max_trials=self.random_search_iterations,
                directory=hp_directory,
                project_name=project_name,
                overwrite=True,  # Clear corrupted oracle state
            )
            print(f"DEBUG: Tuner created. executions_per_trial={self.tuner.executions_per_trial}")
        else:
            print(f"DEBUG: Creating GraphTunerValidationLoss with overwrite=True")
            print(f"DEBUG: directory={hp_directory}, project_name={project_name}")
            self.tuner = GraphTunerValidationLoss(
                model_builder,
                objective="val_loss",
                hp_minibatch_size=hp_minibatch_size,
                max_trials=self.random_search_iterations,
                directory=hp_directory,
                project_name=project_name,
                overwrite=True,  # Clear corrupted oracle state
            )
            print(f"DEBUG: Tuner created.")

    @abstractmethod
    def model_builder(self, hp):
        return

    @staticmethod
    def _index_times(val_time):
        val_time_unique = np.sort(np.unique(val_time))
        if val_time_unique[0]:  # check if ""
            val_time_unique = np.insert(val_time_unique, 0, "")
        mapping = dict(zip(val_time_unique, range(len(val_time_unique))))

        @np.vectorize
        def get_indices(t):
            return mapping[t]

        return get_indices(val_time), len(mapping)

    def hyperparameter_search(self, train_data, valid_data):
        # Unpack training and validation data
        data, labels, active_flags, identifier, _ = GraphModelFeatures._unpack(train_data)
        val_data, val_labels, val_flags, val_identifier, val_time = GraphModelFeatures._unpack(valid_data)
        
        print("\n")
        print("train")
        print("data.shape:", data.shape)
        print("labels.shape:", labels.shape)
        print("active_flags.shape:", active_flags.shape)
        print(identifier)
        print("identifier.shape:", identifier.shape)
        
        print("\n")
        print("val")
        print("val_data.shape:", val_data.shape)
        print("\n")
        print("val_time type:", type(val_time))
        print("val_identifier shape:", val_identifier.shape)
        print("val_time shape:", val_time.shape)
        
        # Optionally, adjust active_flags dimensions if necessary
        # if active_flags.ndim == 3:
        #     active_flags = np.tile(active_flags, (1, self.num_tickers, 1, 1))
        
        if self.evaluate_diversified_val_sharpe:
            val_time_indices, num_val_time = self._index_times(val_time)
            callbacks = [
                GraphSharpeValidationLoss(
                    val_data,
                    val_labels,
                    val_time_indices,
                    num_val_time,
                    self.early_stopping_patience,
                    self.n_multiprocessing_workers,
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ]
            # print("\n")
            # print("---------------------")
            # print("hyperparam debugging")
            # print(identifier)
            # print("\n")
            # print("x:", data)
            # print("\n")
            # print("y:", labels)
            print(f"\nDEBUG: Calling tuner.search() with epochs={self.num_epochs}")
            print(f"DEBUG: data.shape={data.shape}, labels.shape={labels.shape}")
            self.tuner.search(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                callbacks=callbacks,
                shuffle=True,
            )
            print("DEBUG: tuner.search() returned")
        else:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    min_delta=1e-4,
                ),
            ]
            self.tuner.search(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                validation_data=(val_data, val_labels, val_flags),
                callbacks=callbacks,
                shuffle=True,
            )

        print("completed HP search")
        best_hp = self.tuner.get_best_hyperparameters(num_trials=1)[0].values
        print("best_hp:", best_hp)

        # Get the already-trained best model (not a fresh one with random weights)
        best_model = self.tuner.get_best_models(num_models=1)[0]
        print("best_model loaded from HP search")

        # Just evaluate, no fine-tuning
        train_loss = best_model.evaluate(data, labels, sample_weight=active_flags, verbose=0)
        val_loss = best_model.evaluate(val_data, val_labels, sample_weight=val_flags, verbose=0)
        print(f"Loaded model - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        return best_hp, best_model, {"train_loss": train_loss, "val_loss": val_loss}

    def load_model(
        self,
        hyperparameters,
    ) -> tf.keras.Model:
        hyp = kt.engine.hyperparameters.HyperParameters()
        hyp.values = hyperparameters
        return self.tuner.hypermodel.build(hyp)

    def fit(
        self,
        train_data: np.array,
        valid_data: np.array,
        hyperparameters,
        temp_folder: str,
    ):
        data, labels, active_flags, _, _ = GraphModelFeatures._unpack(train_data)
        val_data, val_labels, val_flags, _, val_time = GraphModelFeatures._unpack(valid_data)

        model = self.load_model(hyperparameters)

        if self.evaluate_diversified_val_sharpe:
            val_time_indices, num_val_time = self._index_times(val_time)
            callbacks = [
                GraphSharpeValidationLoss(
                    val_data,
                    val_labels,
                    val_time_indices,
                    num_val_time,
                    self.early_stopping_patience,
                    self.n_multiprocessing_workers,
                    weights_save_location=temp_folder,
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ]
            # self.model.run_eagerly = True
            model.fit(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                batch_size=hyperparameters["batch_size"],
                callbacks=callbacks,
                shuffle=True,
            )
            model.load_weights(temp_folder)
        else:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    min_delta=1e-4,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ]
            # self.model.run_eagerly = True
            model.fit(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                batch_size=hyperparameters["batch_size"],
                validation_data=(
                    val_data,
                    val_labels,
                    val_flags,
                ),
                callbacks=callbacks,
                shuffle=True,
            )
        return model

    def evaluate(self, data, model):
        """Applies evaluation metric to the training data.

        Args:
          data: Dataframe for evaluation
          eval_metric: Evaluation metic to return, based on model definition.

        Returns:
          Computed evaluation loss.
        """

        inputs, outputs, active_entries, _, _ = GraphModelFeatures._unpack(data)

        if self.evaluate_diversified_val_sharpe:
            _, performance = self.get_positions(data, model, False)
            return performance

        else:
            metric_values = model.evaluate(
                x=inputs,
                y=outputs,
                sample_weight=active_entries,
            )

            metrics = pd.Series(metric_values, model.metrics_names)
            return metrics["loss"]

    def get_positions(
        self,
        data,
        model,
        sliding_window=True,
        years_geq=np.iinfo(np.int32).min,
        years_lt=np.iinfo(np.int32).max,
    ):
        # Unpack data; expect shapes:
        # inputs: (batch, n_stocks, time_steps, n_features)
        # outputs, identifier, time: (batch, n_stocks, time_steps, output_size)
        inputs, outputs, _, identifier, time = GraphModelFeatures._unpack(data)
        
        if sliding_window:
            # For sliding windows, extract the final time step for each stock.
            # New shapes:
            # time: (batch, n_stocks, -1, 0) => (batch, n_stocks) then flatten to (batch*n_stocks,)
            time = pd.to_datetime(time[:, :, -1, 0].reshape(-1))
            years = time.map(lambda t: t.year)
            identifier = identifier[:, :, -1, 0].reshape(-1)
            returns = outputs[:, :, -1, 0].reshape(-1)
        else:
            time = pd.to_datetime(time.flatten())
            years = time.map(lambda t: t.year)
            identifier = identifier.flatten()
            returns = outputs.flatten()
        
        mask = (years >= years_geq) & (years < years_lt)
        
        positions = model.predict(inputs)
        if sliding_window:
            # Expect predictions to have shape: (batch, n_stocks, time_steps, 1)
            # Extract the final time step for each stock and flatten.
            positions = positions[:, :, -1, 0].reshape(-1)
        else:
            positions = positions.flatten()
        
        captured_returns = returns * positions
        results = pd.DataFrame(
            {
                "identifier": identifier[mask],
                "time": time[mask],
                "returns": returns[mask],
                "position": positions[mask],
                "captured_returns": captured_returns[mask],
            }
        )
        
        performance = sharpe_ratio(results.groupby("time")["captured_returns"].sum())
        
        return results, performance
        
        
class GraphConvolution(layers.Layer):
    """
    A simple Graph Convolution layer:
      Z = A_hat * X * W + b
    where:
      - A_hat: the normalized adjacency matrix of shape (num_stocks, num_stocks)
      - X: node features of shape (batch_size, num_stocks, input_dim)
      - W: trainable weight matrix (input_dim x units)
      - b: bias (units,)
    """
    def __init__(self, units, adjacency, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        # Store the (typically pre-normalized) adjacency matrix.
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
        # inputs shape: (batch_size, num_stocks, input_dim)
        # Expand adjacency so that it broadcasts across the batch dimension.
        A_expanded = tf.expand_dims(self.adjacency, 0)  # shape: (1, num_stocks, num_stocks)
        Ax = tf.matmul(A_expanded, inputs)  # shape: (batch_size, num_stocks, input_dim)
        output = tf.matmul(Ax, self.weight) + self.bias  # shape: (batch_size, num_stocks, units)
        return output
    
     
class GraphLSTMDeepMomentumNetwork(GraphDeepMomentumNetwork):
    def __init__(self, project_name, hp_directory, hp_minibatch_size=HP_MINIBATCH_SIZE_GRAPH, **params):
        super().__init__(project_name, hp_directory, hp_minibatch_size, **params)
        
    def model_builder(self, hp):
        # Hyperparameter selections for the network.
        # hidden_layer_size = hp.Choice("hidden_layer_size", values=[10])
        # dropout_rate      = hp.Choice("dropout_rate",      values=[0.2])
        # max_gradient_norm = hp.Choice("max_gradient_norm", values=[5.0])
        # learning_rate     = hp.Choice("learning_rate",     values=[1e-3])
        # gcn_units         = hp.Choice("gcn_units",         values=[32])
        
        
        hidden_layer_size = hp.Choice("hidden_layer_size", values=HP_HIDDEN_LAYER_SIZE_GRAPH)
        dropout_rate      = hp.Choice("dropout_rate",      values=HP_DROPOUT_RATE_GRAPH)
        max_gradient_norm = hp.Choice("max_gradient_norm", values=HP_MAX_GRADIENT_NORM_GRAPH)
        learning_rate     = hp.Choice("learning_rate",     values=HP_LEARNING_RATE_GRAPH)
        gcn_units         = hp.Choice("gcn_units",         values=HP_GCN_UNITS)
        num_gcn_layers = hp.Choice("gcn_layers", values=[2])
        
        alpha = hp.Choice("alpha", values=HP_ALPHA)
        beta  = hp.Choice("beta",  values=HP_BETA)

        tau = 0.45
        # Define the file path for the precomputed ensemble graph.
        
        
        graph_file = os.path.join("data", "graph_structure", "cvx_opt", f"{alpha}_{beta}_cvx.csv")
        # graph_file = os.path.join("data", "graph_structure", "pearson", f"{tau}.csv")

        # Load the precomputed ensemble and normalized adjacency matrix.
        adjacency_df = pd.read_csv(graph_file, index_col=0)
        adjacency_df = adjacency_df.reindex(index=ALL_TICKERS, columns=ALL_TICKERS)
        self.A = adjacency_df.values
        

               
        # Input shape: (batch_size, num_tickers, time_steps, input_size)
        input_layer = keras.Input(shape=(self.num_tickers, self.time_steps, self.input_size))
        
        # Define a shared LSTM layer that will be reused for all tickers.
        shared_lstm = layers.LSTM(
            hidden_layer_size,
            return_sequences=True,  # preserve the time dimension
            dropout=dropout_rate,
            activation="tanh",
            recurrent_activation="sigmoid",
            name="shared_lstm"
        )
        
        lstm_outputs = []
        # print("\n")
        for i in range(self.num_tickers):
            # Slice the input to get (batch_size, time_steps, input_size) for ticker i
            ticker_slice = layers.Lambda(lambda x: x[:, i, :, :])(input_layer)
            
            # Process the slice through the shared LSTM layer.
            ticker_output = shared_lstm(ticker_slice)
            lstm_outputs.append(ticker_output)
            
            # print(f"LSTM input shape for ticker {i}:", ticker_slice.shape)
            # print(f"LSTM output shape for ticker {i}:", ticker_output.shape)

        # Stack the LSTM outputs along a new ticker dimension.
        # This gives a tensor of shape: (batch_size, time_steps, num_tickers, hidden_layer_size)
        stacked_lstm = layers.Lambda(lambda tensors: tf.stack(tensors, axis=2))(lstm_outputs)
        print("Stacked LSTM output shape:", stacked_lstm.shape)
        
        # Apply the GCN layer.
        # The GCN layer aggregates information across the ticker (node) dimension.
        gcn_output = GraphConvolution(units=gcn_units, adjacency=self.A)(stacked_lstm)
        gcn_output = layers.ReLU()(gcn_output)
        print("After GCN, shape:", gcn_output.shape)
        
        if num_gcn_layers == 2:
            gcn_output = GraphConvolution(units=gcn_units, adjacency=self.A)(gcn_output)
            gcn_output = layers.ReLU()(gcn_output)
        
        # **Add a residual connection:**  
        # For instance, we can upsample (or project) the original LSTM outputs so that
        # they have the same feature dimension as the GCN output, and then add them.
        residual = layers.TimeDistributed(
                        layers.TimeDistributed(
                            keras.layers.Dense(gcn_units, activation="linear")
                        )
                   )(stacked_lstm)
        print("Residual shape:", residual.shape)
        
        x = layers.Add()([gcn_output, residual])
        print("After adding residual, shape:", x.shape)
        
        # Instead of flattening the ticker dimension, apply a nested TimeDistributed Dense layer.
        output = layers.TimeDistributed(
                    layers.TimeDistributed(
                        keras.layers.Dense(
                            self.output_size,
                            activation=tf.nn.tanh,
                            kernel_constraint=keras.constraints.max_norm(3),
                        )
                    )
                 )(x)
        
        # Now, output shape is: (batch_size, time_steps, num_tickers, output_size).
        # If your labels have shape (batch_size, num_tickers, time_steps, output_size), permute axes:
        output = layers.Permute((2, 1, 3))(output)
        
        # Create the model.
        model = keras.Model(inputs=input_layer, outputs=output)
        print("Final model input shape:", input_layer.shape)
        print("Final model output shape:", output.shape)
        
        # Configure the optimizer with gradient clipping.
        adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
        sharpe_loss = GraphSharpeLoss(self.output_size).call
        
        model.compile(
            loss=sharpe_loss,
            optimizer=adam,
        )
        
        # model.summary()
        return model
    
# class GraphConvolution(layers.Layer):
#     """
#     A simple Graph Convolution layer:
#       Z = A_hat * X * W + b
#     where:
#       - A_hat: the normalized adjacency matrix of shape (num_stocks, num_stocks)
#       - X: node features of shape (batch_size, num_stocks, input_dim)
#       - W: trainable weight matrix (input_dim x units)
#       - b: bias (units,)
#     """
#     def __init__(self, units, adjacency, **kwargs):
#         super(GraphConvolution, self).__init__(**kwargs)
#         self.units = units
#         # Store the (typically pre-normalized) adjacency matrix.
#         self.adjacency = tf.constant(adjacency, dtype=tf.float32)

#     def build(self, input_shape):
#         input_dim = input_shape[-1]
#         self.weight = self.add_weight(
#             shape=(input_dim, self.units),
#             initializer="glorot_uniform",
#             trainable=True,
#             name="gcn_weight",
#         )
#         self.bias = self.add_weight(
#             shape=(self.units,),
#             initializer="zeros",
#             trainable=True,
#             name="gcn_bias",
#         )
#         super(GraphConvolution, self).build(input_shape)

#     def call(self, inputs):
#         # inputs shape: (batch_size, num_stocks, input_dim)
#         # Expand adjacency so that it broadcasts across the batch dimension.
#         A_expanded = tf.expand_dims(self.adjacency, 0)  # shape: (1, num_stocks, num_stocks)
#         Ax = tf.matmul(A_expanded, inputs)  # shape: (batch_size, num_stocks, input_dim)
#         output = tf.matmul(Ax, self.weight) + self.bias  # shape: (batch_size, num_stocks, units)
#         return output
    
     
# class GraphLSTMDeepMomentumNetwork(GraphDeepMomentumNetwork):
#     def __init__(self, project_name, hp_directory, hp_minibatch_size=HP_MINIBATCH_SIZE_GRAPH, **params):
#         super().__init__(project_name, hp_directory, hp_minibatch_size, **params)
        
#     def model_builder(self, hp):
#         # Hyperparameter selections for the network.
#         hidden_layer_size = hp.Choice("hidden_layer_size", values=[10])
#         dropout_rate      = hp.Choice("dropout_rate",      values=[0.2])
#         max_gradient_norm = hp.Choice("max_gradient_norm", values=[5.0])
#         learning_rate     = hp.Choice("learning_rate",     values=[1e-3])
#         gcn_units         = hp.Choice("gcn_units",         values=[32])
        
#         # Input shape: (batch_size, num_tickers, time_steps, input_size)
#         input_layer = keras.Input(shape=( self.time_steps, self.num_tickers, self.input_size))
        
#         # Define a shared LSTM layer that will be reused for all tickers.
#         shared_lstm = layers.LSTM(
#             hidden_layer_size,
#             return_sequences=True,  # preserve the time dimension
#             dropout=dropout_rate,
#             activation="tanh",
#             recurrent_activation="sigmoid",
#             name="shared_lstm"
#         )
        
#         lstm_outputs = []
#         # print("\n")
#         for i in range(self.num_tickers):
#             # Slice the input to get (batch_size, time_steps, input_size) for ticker i
#             ticker_slice = layers.Lambda(lambda x: x[:, :, i, :])(input_layer)
            
#             # Process the slice through the shared LSTM layer.
#             ticker_output = shared_lstm(ticker_slice)
#             lstm_outputs.append(ticker_output)
            
#             # print(f"LSTM input shape for ticker {i}:", ticker_slice.shape)
#             # print(f"LSTM output shape for ticker {i}:", ticker_output.shape)

#         # Stack the LSTM outputs along a new ticker dimension.
#         # This gives a tensor of shape: (batch_size, time_steps, num_tickers, hidden_layer_size)
#         stacked_lstm = layers.Lambda(lambda tensors: tf.stack(tensors, axis=2))(lstm_outputs)
#         print("Stacked LSTM output shape:", stacked_lstm.shape)
        
#         # Apply the GCN layer.
#         # The GCN layer aggregates information across the ticker (node) dimension.
#         gcn_output = GraphConvolution(units=gcn_units, adjacency=self.A)(stacked_lstm)
#         gcn_output = layers.ReLU()(gcn_output)
#         print("After GCN, shape:", gcn_output.shape)
        
#         # **Add a residual connection:**  
#         # For instance, we can upsample (or project) the original LSTM outputs so that
#         # they have the same feature dimension as the GCN output, and then add them.
#         residual = layers.TimeDistributed(
#                         layers.TimeDistributed(
#                             keras.layers.Dense(gcn_units, activation="linear")
#                         )
#                    )(stacked_lstm)
#         print("Residual shape:", residual.shape)
        
#         x = layers.Add()([gcn_output, residual])
#         print("After adding residual, shape:", x.shape)
        
#         # Instead of flattening the ticker dimension, apply a nested TimeDistributed Dense layer.
#         output = layers.TimeDistributed(
#                     layers.TimeDistributed(
#                         keras.layers.Dense(
#                             self.output_size,
#                             activation=tf.nn.tanh,
#                             kernel_constraint=keras.constraints.max_norm(3),
#                         )
#                     )
#                  )(x)
        
#         # Now, output shape is: (batch_size, time_steps, num_tickers, output_size).
#         # If your labels have shape (batch_size, num_tickers, time_steps, output_size), permute axes:
#         # output = layers.Permute((2, 1, 3))(output)
        
#         # Create the model.
#         model = keras.Model(inputs=input_layer, outputs=output)
#         print("Final model input shape:", input_layer.shape)
#         print("Final model output shape:", output.shape)
        
#         # Configure the optimizer with gradient clipping.
#         adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
#         sharpe_loss = GraphSharpeLoss(self.output_size).call
        
#         model.compile(
#             loss=sharpe_loss,
#             optimizer=adam,
#         )
        
#         # model.summary()
#         return model


class DynamicGraphConvolution(layers.Layer):
    """
    Graph Convolution layer that accepts per-sample adjacency matrices.

    Unlike GraphConvolution which stores a static adjacency matrix,
    this layer receives the adjacency matrix as an input tensor,
    allowing different graphs per sample in the batch.

    Inputs (as a list):
        - node_features: (batch_size, time_steps, num_stocks, input_dim)
        - adjacency: (batch_size, num_stocks, num_stocks)

    Output: (batch_size, time_steps, num_stocks, units)
    """
    def __init__(self, units, **kwargs):
        super(DynamicGraphConvolution, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # input_shape is a list: [node_features_shape, adjacency_shape]
        features_shape = input_shape[0]
        input_dim = features_shape[-1]
        self.weight = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="dynamic_gcn_weight",
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="dynamic_gcn_bias",
        )
        super(DynamicGraphConvolution, self).build(input_shape)

    def call(self, inputs):
        # inputs is a list: [node_features, adjacency]
        node_features, adjacency = inputs
        # node_features: (batch, time_steps, num_stocks, input_dim)
        # adjacency: (batch, num_stocks, num_stocks)

        # Expand adjacency for time dimension: (batch, 1, num_stocks, num_stocks)
        A_expanded = tf.expand_dims(adjacency, 1)

        # Perform graph convolution per time step
        # Ax: (batch, time_steps, num_stocks, input_dim)
        Ax = tf.matmul(A_expanded, node_features)

        # Linear transform
        output = tf.matmul(Ax, self.weight) + self.bias
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class RollingGraphLSTMDeepMomentumNetwork(GraphDeepMomentumNetwork):
    """
    LSTM-GCN model that uses per-sample rolling Pearson correlation graphs.

    Instead of a single static adjacency matrix, this model accepts
    a batch of adjacency matrices (one per sample) computed from
    rolling correlation windows.
    """
    def __init__(self, project_name, hp_directory, hp_minibatch_size=HP_MINIBATCH_SIZE_GRAPH, **params):
        super().__init__(project_name, hp_directory, hp_minibatch_size, **params)

    def model_builder(self, hp):
        # Hyperparameter selections for the network
        hidden_layer_size = hp.Choice("hidden_layer_size", values=HP_HIDDEN_LAYER_SIZE_GRAPH)
        dropout_rate      = hp.Choice("dropout_rate",      values=HP_DROPOUT_RATE_GRAPH)
        max_gradient_norm = hp.Choice("max_gradient_norm", values=HP_MAX_GRADIENT_NORM_GRAPH)
        learning_rate     = hp.Choice("learning_rate",     values=HP_LEARNING_RATE_GRAPH)
        gcn_units         = hp.Choice("gcn_units",         values=HP_GCN_UNITS)
        num_gcn_layers    = hp.Choice("gcn_layers",        values=[2])

        # Rolling Pearson hyperparameters (stored for reference, actual computation done in data prep)
        correlation_lookback  = hp.Choice("correlation_lookback",  values=HP_CORRELATION_LOOKBACK)
        correlation_threshold = hp.Choice("correlation_threshold", values=HP_CORRELATION_THRESHOLD)

        # Input layers: features AND adjacency matrix
        feature_input = keras.Input(
            shape=(self.num_tickers, self.time_steps, self.input_size),
            name="features"
        )
        adjacency_input = keras.Input(
            shape=(self.num_tickers, self.num_tickers),
            name="adjacency"
        )

        # Define a shared LSTM layer that will be reused for all tickers
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
            # Slice the input to get (batch_size, time_steps, input_size) for ticker i
            ticker_slice = layers.Lambda(lambda x, idx=i: x[:, idx, :, :])(feature_input)

            # Process the slice through the shared LSTM layer
            ticker_output = shared_lstm(ticker_slice)
            lstm_outputs.append(ticker_output)

        # Stack the LSTM outputs along a new ticker dimension
        # Shape: (batch_size, time_steps, num_tickers, hidden_layer_size)
        stacked_lstm = layers.Lambda(lambda tensors: tf.stack(tensors, axis=2))(lstm_outputs)
        print("Stacked LSTM output shape:", stacked_lstm.shape)

        # Apply the Dynamic GCN layer with per-sample adjacency
        gcn_output = DynamicGraphConvolution(units=gcn_units)([stacked_lstm, adjacency_input])
        gcn_output = layers.ReLU()(gcn_output)
        print("After GCN, shape:", gcn_output.shape)

        if num_gcn_layers == 2:
            gcn_output = DynamicGraphConvolution(units=gcn_units)([gcn_output, adjacency_input])
            gcn_output = layers.ReLU()(gcn_output)

        # Add a residual connection
        residual = layers.TimeDistributed(
                        layers.TimeDistributed(
                            keras.layers.Dense(gcn_units, activation="linear")
                        )
                   )(stacked_lstm)
        print("Residual shape:", residual.shape)

        x = layers.Add()([gcn_output, residual])
        print("After adding residual, shape:", x.shape)

        # Apply nested TimeDistributed Dense layer for output
        output = layers.TimeDistributed(
                    layers.TimeDistributed(
                        keras.layers.Dense(
                            self.output_size,
                            activation=tf.nn.tanh,
                            kernel_constraint=keras.constraints.max_norm(3),
                        )
                    )
                 )(x)

        # Permute to match label shape: (batch_size, num_tickers, time_steps, output_size)
        output = layers.Permute((2, 1, 3))(output)

        # Create the model with two inputs
        model = keras.Model(
            inputs=[feature_input, adjacency_input],
            outputs=output
        )
        print("Final model feature input shape:", feature_input.shape)
        print("Final model adjacency input shape:", adjacency_input.shape)
        print("Final model output shape:", output.shape)

        # Configure the optimizer with gradient clipping
        adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
        sharpe_loss = GraphSharpeLoss(self.output_size).call

        model.compile(
            loss=sharpe_loss,
            optimizer=adam,
        )

        return model

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# class GraphLSTMDeepMomentumNetwork(GraphDeepMomentumNetwork):
#     def __init__(self, project_name, hp_directory, hp_minibatch_size=32, **params):
#         super().__init__(project_name, hp_directory, hp_minibatch_size, **params)
        
#     def model_builder(self, hp):
#         # Hyperparameter selections.
#         hidden_layer_size = hp.Choice("hidden_layer_size", values=[10])
#         dropout_rate      = hp.Choice("dropout_rate",      values=[0.2])
#         max_gradient_norm = hp.Choice("max_gradient_norm", values=[5.0])
#         learning_rate     = hp.Choice("learning_rate",     values=[1e-3])
#         gcn_units         = hp.Choice("gcn_units",         values=[32])
        
#         # Input: (batch_size, num_tickers, time_steps, input_size)
#         input_layer = keras.Input(shape=(self.num_tickers, self.time_steps, self.input_size))
        
#         # Shared LSTM layer (reused for each ticker).
#         shared_lstm = layers.LSTM(
#             hidden_layer_size,
#             return_sequences=True,  # preserve the time dimension
#             dropout=dropout_rate,
#             activation="tanh",
#             recurrent_activation="sigmoid",
#             name="shared_lstm"
#         )
        
#         lstm_outputs = []
#         for i in range(self.num_tickers):
#             # Slice the input: shape becomes (batch_size, time_steps, input_size)
#             ticker_slice = layers.Lambda(lambda x: x[:, i, :, :])(input_layer)
#             ticker_output = shared_lstm(ticker_slice)
#             lstm_outputs.append(ticker_output)
        
#         # Stack LSTM outputs along a new ticker axis.
#         # New shape: (batch_size, time_steps, num_tickers, hidden_layer_size)
#         stacked_lstm = layers.Lambda(lambda tensors: tf.stack(tensors, axis=2))(lstm_outputs)
#         print("Stacked LSTM output shape:", stacked_lstm.shape)
        
#         # Apply the GCN layer to aggregate info across tickers.
#         gcn_output = GraphConvolution(units=gcn_units, adjacency=self.A)(stacked_lstm)
#         gcn_output = layers.ReLU()(gcn_output)
#         print("After GCN, shape:", gcn_output.shape)
        
#         # Residual connection: project original LSTM outputs to gcn_units.
#         residual = layers.TimeDistributed(
#                         layers.TimeDistributed(
#                             keras.layers.Dense(gcn_units, activation="linear")
#                         )
#                    )(stacked_lstm)
#         print("Residual shape:", residual.shape)
        
#         # Add residual connection.
#         x = layers.Add()([gcn_output, residual])
#         print("After adding residual, shape:", x.shape)
        
#         # Apply a nested TimeDistributed Dense layer.
#         output = layers.TimeDistributed(
#                     layers.TimeDistributed(
#                         keras.layers.Dense(
#                             self.output_size,
#                             activation=tf.nn.tanh,
#                             kernel_constraint=keras.constraints.max_norm(3),
#                         )
#                     )
#                  )(x)
#         # At this point: shape is (batch_size, time_steps, num_tickers, output_size)
        
#         # Permute axes so that tickers come first:
#         # New shape: (batch_size, num_tickers, time_steps, output_size)
#         output = layers.Permute((2, 1, 3))(output)
        
#         # --- Modification for 2D Output ---
#         # For one prediction per stock, extract the last time step.
#         output = layers.Lambda(lambda x: x[:, :, -1, :])(output)
#         # Now shape: (batch_size, num_tickers, output_size)
        
#         # Flatten the ticker and output_size dimensions.
#         output = layers.Flatten()(output)
#         # Final shape: (batch_size, num_tickers * output_size) e.g. (batch_size, 88) if output_size==1
        
#         # Build the model.
#         model = keras.Model(inputs=input_layer, outputs=output)
#         print("Final model input shape:", input_layer.shape)
#         print("Final model output shape:", output.shape)
        
#         # Configure the optimizer with gradient clipping.
#         adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)
#         sharpe_loss = GraphSharpeLoss(self.output_size).call
        
#         model.compile(
#             loss=sharpe_loss,
#             optimizer=adam,
#             sample_weight_mode="temporal",
#         )
        
#         model.summary()
#         return model