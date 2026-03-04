import tensorflow as tf
import numpy as np
import pandas as pd
import collections
from tensorflow import keras
import keras_tuner as kt
# from keras_tuner import TunerCallback


from keras_tuner import RandomSearch
import copy
from abc import ABC, abstractmethod

from settings.hp_grid import (
    HP_HIDDEN_LAYER_SIZE,
    HP_DROPOUT_RATE,
    HP_MAX_GRADIENT_NORM,
    HP_LEARNING_RATE,
    HP_MINIBATCH_SIZE,
)
from gml.model_inputs import ModelFeatures
from empyrical import sharpe_ratio


from settings.fixed_params import MODEL_PARAMS


class SharpeLoss(tf.keras.losses.Loss):
    def __init__(self, output_size: int = 1):
        self.output_size = output_size  # in case we have multiple targets => output dim[-1] = output_size * n_quantiles
        super().__init__()

    def call(self, y_true, weights):
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


class SharpeValidationLoss(keras.callbacks.Callback):
    # TODO check if weights already exist and pass in best sharpe
    def __init__(
        self,
        inputs,
        returns,
        time_indices,
        num_time,  # including a count for nulls which will be indexed as 0
        early_stopping_patience,
        n_multiprocessing_workers,
        weights_save_location="tmp/checkpoint",
        # verbose=0,
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
        # self.best_weights = None
        self.weights_save_location = weights_save_location
        # self.verbose = verbose

    def set_weights_save_loc(self, weights_save_location):
        self.weights_save_location = weights_save_location

    def on_train_begin(self, logs=None):
        self.patience_counter = 0
        self.stopped_epoch = 0
        self.best_sharpe = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        positions = self.model.predict(self.inputs)
        print(self.inputs.shape)
        print(positions.shape)
        print(self.returns.shape)
        captured_returns = tf.math.unsorted_segment_mean(
            positions * self.returns, self.time_indices, self.num_time
        )[1:]
        print(captured_returns.shape)
        # ignoring null times

        # TODO sharpe
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

# class CustomTuner(kt.tuners.RandomSearch):
#     def _get_trial_dir(self, trial):
#         return f"trial_{trial.trial_id[:5]}" 

class TunerValidationLoss(kt.RandomSearch): # TODO changed
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
        # self._reported_step = 0 # TECKY
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
        kwargs["batch_size"] = trial.hyperparameters.Choice(
            "batch_size", values=self.hp_minibatch_size
        )
        super(TunerValidationLoss, self).run_trial(trial, *args, **kwargs)


class TunerDiversifiedSharpe(kt.RandomSearch):
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
        # self._reported_step = 0 # TECKY

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
        kwargs["batch_size"] = trial.hyperparameters.Choice(
            "batch_size", values=self.hp_minibatch_size
        )

        original_callbacks = kwargs.pop("callbacks", [])

        # Shorten trial ID when creating checkpoint path
        short_trial_id = trial.trial_id  # Shorten trial ID TODO

        for callback in original_callbacks:
            if isinstance(callback, SharpeValidationLoss):
                callback.set_weights_save_loc(
                    self._get_checkpoint_fname(short_trial_id, self._reported_step)
                )

        # Run the training process multiple times.
        metrics = collections.defaultdict(list)
        for execution in range(self.executions_per_trial):
            copied_fit_kwargs = copy.copy(kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial, execution)
            callbacks.append(kt.engine.tuner_utils.TunerCallback(self, trial))
            # Only checkpoint the best epoch across all executions.
            # callbacks.append(model_checkpoint)
            copied_fit_kwargs["callbacks"] = callbacks

            history = self._build_and_fit_model(trial, args, copied_fit_kwargs)
            for metric, epoch_values in history.history.items():
                if self.oracle.objective.direction == "min":
                    best_value = np.min(epoch_values)
                else:
                    best_value = np.max(epoch_values)
                metrics[metric].append(best_value)

        # Average the results across executions and send to the Oracle.
        averaged_metrics = {}
        for metric, execution_values in metrics.items():
            averaged_metrics[metric] = np.mean(execution_values)
        self.oracle.update_trial(
            trial.trial_id, metrics=averaged_metrics, step=self._reported_step
        )


class DeepMomentumNetworkModel(ABC):
    def __init__(self, project_name, hp_directory, hp_minibatch_size, **params):
        params = params.copy()

        self.time_steps = int(params["total_time_steps"])
        self.input_size = int(params["input_size"])
        self.output_size = int(params["output_size"])
        self.n_multiprocessing_workers = int(params["multiprocessing_workers"])
        self.num_epochs = int(params["num_epochs"])
        self.early_stopping_patience = int(params["early_stopping_patience"])
        # self.sliding_window = params["sliding_window"]
        self.random_search_iterations = params["random_search_iterations"]
        self.evaluate_diversified_val_sharpe = params["evaluate_diversified_val_sharpe"]
        self.force_output_sharpe_length = params["force_output_sharpe_length"]

        print("Deep Momentum Network params:")
        for k in params:
            print(f"{k} = {params[k]}")

        # To build model
        def model_builder(hp):
            return self.model_builder(hp)

        if self.evaluate_diversified_val_sharpe:
            self.tuner = TunerDiversifiedSharpe(
                model_builder,
                # objective="val_loss",
                objective=kt.Objective("sharpe", "max"),
                hp_minibatch_size=hp_minibatch_size,
                max_trials=self.random_search_iterations,
                directory=hp_directory,
                project_name=project_name,
            )
        else:
            self.tuner = TunerValidationLoss(
                model_builder,
                objective="val_loss",
                hp_minibatch_size=hp_minibatch_size,
                max_trials=self.random_search_iterations,
                directory=hp_directory,
                project_name=project_name,
            )

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
        data, labels, active_flags, _, _ = ModelFeatures._unpack(train_data)
        val_data, val_labels, val_flags, _, val_time = ModelFeatures._unpack(valid_data)

        if self.evaluate_diversified_val_sharpe:
            val_time_indices, num_val_time = self._index_times(val_time)
            callbacks = [
                SharpeValidationLoss(
                    val_data,
                    val_labels,
                    val_time_indices,
                    num_val_time,
                    self.early_stopping_patience,
                    self.n_multiprocessing_workers,
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ]
            self.tuner.search(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                callbacks=callbacks,
                shuffle=True,
            )
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

        best_hp = self.tuner.get_best_hyperparameters(num_trials=1)[0].values
        # Build fresh model with best hyperparameters (get_best_models requires saved checkpoints)
        best_model = self.load_model(best_hp)

        # Additional training to capture the training history (with both loss and val_loss)
        new_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                min_delta=1e-4,
            )
        ]
        training_history = best_model.fit(
            x=data,
            y=labels,
            sample_weight=active_flags,
            epochs=self.num_epochs,
            validation_data=(val_data, val_labels, val_flags),
            callbacks=new_callbacks,
            shuffle=True,
        )
        
        # Save the full training history including both loss and val_loss to a JSON file
        import json
        with open("training_history.json", "w") as f:
            json.dump(training_history.history, f)

        return best_hp, best_model, training_history

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
        data, labels, active_flags, _, _ = ModelFeatures._unpack(train_data)
        val_data, val_labels, val_flags, _, val_time = ModelFeatures._unpack(valid_data)

        model = self.load_model(hyperparameters)

        if self.evaluate_diversified_val_sharpe:
            val_time_indices, num_val_time = self._index_times(val_time)
            callbacks = [
                SharpeValidationLoss(
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

        inputs, outputs, active_entries, _, _ = ModelFeatures._unpack(data)

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
        inputs, outputs, _, identifier, time = ModelFeatures._unpack(data)
        if sliding_window:
            time = pd.to_datetime(
                time[:, -1, 0].flatten()
            )  # TODO to_datetime maybe not needed
            years = time.map(lambda t: t.year)
            identifier = identifier[:, -1, 0].flatten()
            returns = outputs[:, -1, 0].flatten()
        else:
            time = pd.to_datetime(time.flatten())
            years = time.map(lambda t: t.year)
            identifier = identifier.flatten()
            returns = outputs.flatten()
        mask = (years >= years_geq) & (years < years_lt)

        positions = model.predict(inputs)
        if sliding_window:
            positions = positions[:, -1, 0].flatten()
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

        # don't need to divide sum by n because not storing here
        # mean does not work as well (related to days where no information)
        performance = sharpe_ratio(results.groupby("time")["captured_returns"].sum())

        return results, performance


class LstmDeepMomentumNetworkModel(DeepMomentumNetworkModel):
    def __init__(
        self, project_name, hp_directory, hp_minibatch_size=HP_MINIBATCH_SIZE, **params
    ):
        super().__init__(project_name, hp_directory, hp_minibatch_size, **params)

    def model_builder(self, hp):
        # Hyperparameter selections for the network.
        hidden_layer_size = hp.Choice("hidden_layer_size", values=HP_HIDDEN_LAYER_SIZE)
        dropout_rate = hp.Choice("dropout_rate", values=HP_DROPOUT_RATE)
        max_gradient_norm = hp.Choice("max_gradient_norm", values=HP_MAX_GRADIENT_NORM)
        learning_rate = hp.Choice("learning_rate", values=HP_LEARNING_RATE)

        # Input shape: (batch_size, time_steps, input_size)
        # time_steps: number of time steps in the sequence
        # input_size: number of features per time step (e.g., stock prices/returns for each stock)
        input_layer = keras.Input((self.time_steps, self.input_size))
        print("\n")
        print(f"self.time_steps: {self.time_steps}")
        print(f"self.input_size: {self.input_size}")
        print("\n")

        print("LSTM Input shape:", input_layer.shape)
        # LSTM layer processes the temporal sequence.
        # Output shape: (batch_size, time_steps, hidden_layer_size)
        # Each time step is transformed into a hidden representation of size hidden_layer_size.
        lstm = tf.keras.layers.LSTM(
            hidden_layer_size,
            return_sequences=True,
            dropout=dropout_rate,
            stateful=False,
            activation="tanh",
            recurrent_activation="sigmoid",
            recurrent_dropout=0,
            unroll=False,
            use_bias=True,
        )(input_layer)

        print("LSTM output shape:", lstm.shape)
        print("\n")

        # Apply dropout to the LSTM outputs to prevent overfitting.
        dropout = keras.layers.Dropout(dropout_rate)(lstm)

        # TimeDistributed Dense layer:
        # Applies a Dense layer to each time step independently.
        # The Dense layer maps from hidden_layer_size to output_size.
        # Output shape: (batch_size, time_steps, output_size)
        output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                self.output_size,
                activation=tf.nn.tanh,
                kernel_constraint=keras.constraints.max_norm(3),
            )
        )(dropout)

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
        
        model.summary()
        
        return model