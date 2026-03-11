import numpy as np
import sklearn.preprocessing
import pandas as pd
import datetime as dt
import enum

from sklearn.preprocessing import MinMaxScaler

from gml.model_inputs import DataTypes, InputTypes, ModelFeatures, get_single_col_by_input_type, extract_cols_from_data_type

    
# class GraphModelFeatures(ModelFeatures):
#     """
#     A subclass of ModelFeatures that creates batches suitable for a graph‐based network.
#     Instead of producing 3‐D arrays of shape
#         (num_samples, total_time_steps, input_size),
#     this class pivots the data so that the inputs have shape
#         (num_windows, num_tickers, total_time_steps, input_size),
#     where num_tickers is the number of unique entities (nodes). The outputs (and related fields)
#     remain 4‐D so that their first dimension (num_windows) is consistent across inputs (x),
#     outputs (y), and sample weights.
#     """
    
#     def _batch_data(self, data, sliding_window):
#         data = data.copy()
#         data["date"] = data.index.strftime("%Y-%m-%d")
    
#         id_col = get_single_col_by_input_type(InputTypes.ID, self._column_definition)
#         time_col = get_single_col_by_input_type(InputTypes.TIME, self._column_definition)
#         target_col = get_single_col_by_input_type(InputTypes.TARGET, self._column_definition)
    
#         input_cols = [tup[0] for tup in self._column_definition 
#                       if tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}]
    
#         data_map = {}
#         # Use sorted tickers so that each stock’s windows are kept together.
#         tickers = sorted(data[id_col].unique())
    
#         if sliding_window:
#             def _batch_single_entity(input_data):
#                 time_steps = len(input_data)
#                 lags = self.total_time_steps
#                 x = input_data.values
#                 if time_steps >= lags:
#                     return np.stack([x[i: time_steps - (lags - 1) + i, :] 
#                                      for i in range(lags)], axis=1)
#                 else:
#                     return None
    
#             for ticker in tickers:
#                 sliced = data[data[id_col] == ticker]
#                 col_mappings = {
#                     "identifier": [id_col],
#                     "date": [time_col],
#                     "outputs": [target_col],
#                     "inputs": input_cols,
#                 }
#                 for k in col_mappings:
#                     cols = col_mappings[k]
#                     arr = _batch_single_entity(sliced[cols].copy())
#                     data_map.setdefault(k, []).append(arr)
    
#             # Stack the per-ticker lists along axis 1.
#             for k in data_map:
#                 data_map[k] = np.stack(data_map[k], axis=1)
    
#             # Create active_entries if not provided.
#             data_map["active_entries"] = np.ones_like(data_map["outputs"])
    
#         else:
#             for ticker in tickers:
#                 sliced = data[data[id_col] == ticker]
#                 col_mappings = {
#                     "identifier": [id_col],
#                     "date": [time_col],
#                     "inputs": input_cols,
#                     "outputs": [target_col],
#                 }
#                 time_steps = len(sliced)
#                 lags = self.total_time_steps
#                 additional_time_steps_required = lags - (time_steps % lags)
    
#                 def _batch_single_entity(input_data):
#                     x = input_data.values
#                     if additional_time_steps_required > 0:
#                         x = np.concatenate([x, np.zeros((additional_time_steps_required, x.shape[1]))])
#                     return x.reshape(-1, lags, x.shape[1])
    
#                 # Process outputs first.
#                 k = "outputs"
#                 cols = col_mappings[k]
#                 arr = _batch_single_entity(sliced[cols].copy())
#                 batch_size = arr.shape[0]
#                 sequence_lengths = [(lags if i != batch_size - 1 else lags - additional_time_steps_required)
#                                       for i in range(batch_size)]
#                 active_entries = np.ones((arr.shape[0], arr.shape[1], arr.shape[2]))
#                 for i in range(batch_size):
#                     active_entries[i, sequence_lengths[i]:, :] = 0
#                 sequence_lengths = np.array(sequence_lengths, dtype=np.int32)
    
#                 data_map.setdefault("active_entries", []).append(active_entries[sequence_lengths > 0, :, :])
#                 data_map.setdefault(k, []).append(arr[sequence_lengths > 0, :, :])
    
#                 for k in set(col_mappings) - {"outputs"}:
#                     cols = col_mappings[k]
#                     arr = _batch_single_entity(sliced[cols].copy())
#                     data_map.setdefault(k, []).append(arr[sequence_lengths > 0, :, :])
    
#             for k in data_map:
#                 data_map[k] = np.stack(data_map[k], axis=1)
    
#         # ----- CULL WINDOWS WITH INSUFFICIENT DATA -----
#         # Assume identifier is 4-D: (num_windows, num_tickers, total_time_steps, id_dim)
#         # If id_dim == 1, squeeze the last dimension for masking.
#         if data_map["identifier"].ndim == 4 and data_map["identifier"].shape[-1] == 1:
#             identifier_4d = np.squeeze(data_map["identifier"], axis=-1)
#         else:
#             identifier_4d = data_map["identifier"]
    
#         # A window is valid if every identifier entry is nonzero (or not "0").
#         valid_mask = np.all((identifier_4d != 0) & (identifier_4d != "0") & (identifier_4d != 0.0), axis=(1, 2))
    
#         # Apply the mask along the window dimension (axis 0) to all fields.
#         data_map["inputs"] = data_map["inputs"][valid_mask]
#         data_map["outputs"] = data_map["outputs"][valid_mask]
#         data_map["active_entries"] = data_map["active_entries"][valid_mask]
#         data_map["identifier"] = data_map["identifier"][valid_mask]
#         data_map["date"] = data_map["date"][valid_mask]
    
#         print("After culling:")
#         print("inputs.shape:", data_map["inputs"].shape)      # (new_num_windows, num_tickers, total_time_steps, input_size)
#         print("outputs.shape:", data_map["outputs"].shape)    # (new_num_windows, num_tickers, total_time_steps, output_size)
#         print("active_entries.shape:", data_map["active_entries"].shape)
#         print("identifier.shape:", data_map["identifier"].shape)
#         print("date.shape:", data_map["date"].shape)
    
#         return data_map
    
    
#     def _batch_data_smaller_output(self, data, sliding_window, output_length):
#         """Batches data for training for graph-based networks.
    
#         Converts raw dataframe from a 2-D tabular format to a batched 4-D array
#         to feed into a graph-based Keras model.
        
#         Returns arrays with shape:
#           - inputs:  (num_windows, num_tickers, total_time_steps, input_size)
#           - outputs: (num_windows, num_tickers, total_time_steps, 1)
#         (active_entries, identifier, and date remain 4-D.)
    
#         Args:
#           data: DataFrame to batch.
#           sliding_window: Boolean indicating sliding window batching.
#           output_length: Used to trim the final portion of identifier and date arrays.
#         """
#         data = data.copy()
#         data["date"] = data.index.strftime("%Y-%m-%d")
    
#         id_col = get_single_col_by_input_type(InputTypes.ID, self._column_definition)
#         time_col = get_single_col_by_input_type(InputTypes.TIME, self._column_definition)
#         target_col = get_single_col_by_input_type(InputTypes.TARGET, self._column_definition)
    
#         input_cols = [tup[0] for tup in self._column_definition
#                       if tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}]
    
#         data_map = {}
#         col_mappings = {
#             "identifier": [id_col],
#             "date": [time_col],
#             "inputs": input_cols,
#             "outputs": [target_col],
#         }
    
#         # Use sorted tickers so that each stock’s data is kept together.
#         tickers = sorted(data[id_col].unique())
    
#         if sliding_window:
#             for ticker in tickers:
#                 sliced = data[data[id_col] == ticker]
#                 time_steps = len(sliced)
#                 batch_size = time_steps - self.total_time_steps + 1
#                 seq_len = self.total_time_steps
#                 for k in col_mappings:
#                     cols = col_mappings[k]
#                     arr = sliced[cols].copy().values
#                     ticker_windows = np.concatenate(
#                         [arr[start: start + seq_len] for start in range(0, batch_size)]
#                     ).reshape(-1, seq_len, arr.shape[1])
#                     data_map.setdefault(k, []).append(ticker_windows)
#         else:
#             for ticker in tickers:
#                 sliced = data[data[id_col] == ticker]
#                 time_steps = len(sliced)
#                 batch_size = (time_steps - self.total_time_steps + output_length) // output_length
#                 active_time_steps = batch_size * output_length + (self.total_time_steps - output_length)
#                 disregard_time_steps = time_steps % active_time_steps
#                 seq_len = self.total_time_steps
#                 for k in col_mappings:
#                     cols = col_mappings[k]
#                     arr = sliced[cols].copy().values[disregard_time_steps:]
#                     ticker_windows = np.concatenate(
#                         [arr[start: start + seq_len] for start in range(0, output_length * batch_size, output_length)]
#                     ).reshape(-1, seq_len, arr.shape[1])
#                     data_map.setdefault(k, []).append(ticker_windows)
    
#         # Ensure consistent number of windows per ticker.
#         for k in data_map:
#             min_windows = min(arr.shape[0] for arr in data_map[k])
#             trimmed = [arr[:min_windows] for arr in data_map[k]]
#             data_map[k] = np.stack(trimmed, axis=1)
    
#         # If active_entries not provided, create it.
#         if "active_entries" not in data_map:
#             num_windows, num_tickers, seq_len, _ = data_map["outputs"].shape
#             data_map["active_entries"] = np.ones((num_windows, num_tickers, seq_len))
    
#         # ----- CULL WINDOWS WITH INSUFFICIENT DATA -----
#         if data_map["identifier"].ndim == 4 and data_map["identifier"].shape[-1] == 1:
#             identifier_4d = np.squeeze(data_map["identifier"], axis=-1)
#         else:
#             identifier_4d = data_map["identifier"]
    
#         valid_mask = np.all((identifier_4d != 0) & (identifier_4d != "0") & (identifier_4d != 0.0), axis=(1, 2))
    
#         data_map["inputs"] = data_map["inputs"][valid_mask]
#         data_map["outputs"] = data_map["outputs"][valid_mask]
#         data_map["active_entries"] = data_map["active_entries"][valid_mask]
#         data_map["identifier"] = data_map["identifier"][valid_mask]
#         data_map["date"] = data_map["date"][valid_mask]
    
#         print("After culling:")
#         print("inputs.shape:", data_map["inputs"].shape)       # (new_num_windows, num_tickers, seq_len, input_size)
#         print("outputs.shape:", data_map["outputs"].shape)     # (new_num_windows, num_tickers, seq_len, output_size)
#         print("active_entries.shape:", data_map["active_entries"].shape)
#         print("identifier.shape:", data_map["identifier"].shape)
#         print("date.shape:", data_map["date"].shape)
    
#         return data_map



class GraphModelFeatures(ModelFeatures):
    def _batch_data(self, data, sliding_window):
        """
        Builds 4D arrays for graph-based models:
          (num_windows, num_tickers, total_time_steps, num_features).
        Dimension 1 enumerates tickers, dimension 2 enumerates time steps,
        and dimension 3 enumerates features.
        """
        data = data.copy()

        # Handle date column: either from DatetimeIndex or existing column
        if isinstance(data.index, pd.DatetimeIndex):
            data["date"] = data.index.strftime("%Y-%m-%d")
        elif "date" in data.columns:
            # Ensure date is string format for consistency
            if pd.api.types.is_datetime64_any_dtype(data["date"]):
                data["date"] = data["date"].dt.strftime("%Y-%m-%d")

        id_col = get_single_col_by_input_type(InputTypes.ID, self._column_definition)
        time_col = get_single_col_by_input_type(InputTypes.TIME, self._column_definition)
        target_col = get_single_col_by_input_type(InputTypes.TARGET, self._column_definition)

        # Identify which columns are inputs vs. the single target
        input_cols = [
            tup[0]
            for tup in self._column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
        ]

        # Gather data for final stacking
        inputs_list = []
        outputs_list = []
        active_entries_list = []
        identifier_list = []
        date_list = []

        # Sort tickers so dimension 1 (tickers) is consistent
        tickers = sorted(data[id_col].unique())

        for ticker in tickers:
            # 1) Filter rows for this ticker and sort by time
            df_ticker = data[data[id_col] == ticker].copy()
            df_ticker.sort_values(by=time_col, inplace=True)

            # 2) Convert to 3D arrays: (num_windows_for_ticker, time_steps, features)
            arr_inputs, arr_outputs, arr_active, arr_identifier, arr_dates = \
                self._build_windows_for_single_ticker(
                    df_ticker,
                    id_col,
                    time_col,
                    target_col,
                    input_cols,
                    sliding_window
                )

            # Append to lists
            inputs_list.append(arr_inputs)
            outputs_list.append(arr_outputs)
            active_entries_list.append(arr_active)
            identifier_list.append(arr_identifier)
            date_list.append(arr_dates)

        # 3) Trim all tickers to the same number of windows
        min_windows = min(arr.shape[0] for arr in inputs_list)
        inputs_list      = [arr[:min_windows] for arr in inputs_list]
        outputs_list     = [arr[:min_windows] for arr in outputs_list]
        active_entries_list = [arr[:min_windows] for arr in active_entries_list]
        identifier_list  = [arr[:min_windows] for arr in identifier_list]
        date_list        = [arr[:min_windows] for arr in date_list]

        # 4) Stack along axis=1 => (num_windows, num_tickers, time_steps, features)
        data_map = {}
        data_map["inputs"]         = np.stack(inputs_list, axis=1)
        data_map["outputs"]        = np.stack(outputs_list, axis=1)
        data_map["active_entries"] = np.stack(active_entries_list, axis=1)
        data_map["identifier"]     = np.stack(identifier_list, axis=1)
        data_map["date"]           = np.stack(date_list, axis=1)

        # 5) Cull windows with invalid identifiers if needed
        data_map = self._cull_invalid_windows(data_map)

        # Print final shapes
        print("After final stacking and culling:")
        print("inputs.shape:", data_map["inputs"].shape)
        print("outputs.shape:", data_map["outputs"].shape)
        print("active_entries.shape:", data_map["active_entries"].shape)
        print("identifier.shape:", data_map["identifier"].shape)
        print("date.shape:", data_map["date"].shape)

        return data_map

    def _build_windows_for_single_ticker(
        self,
        df_ticker,
        id_col,
        time_col,
        target_col,
        input_cols,
        sliding_window
    ):
        """
        Returns 5 arrays of shape (num_windows_for_ticker, time_steps, feature_dim):
          inputs, outputs, active_entries, identifier, date
        where time_steps == self.total_time_steps if enough data is present.
        """
        arr_inputs = []
        arr_outputs = []
        arr_active = []
        arr_identifier = []
        arr_dates = []

        x = df_ticker[input_cols].values  # shape: (time_steps, num_input_features)
        y = df_ticker[[target_col]].values  # shape: (time_steps, 1)
        ids = df_ticker[[id_col]].values   # shape: (time_steps, 1)
        dts = df_ticker[[time_col]].values # shape: (time_steps, 1)

        time_steps = len(df_ticker)
        lags = self.total_time_steps

        if sliding_window:
            # For sliding windows, each start in [0.. time_steps-lags]
            # yields a window of size (lags).
            for start in range(time_steps - lags + 1):
                end = start + lags
                arr_inputs.append(x[start:end])
                arr_outputs.append(y[start:end])
                arr_identifier.append(ids[start:end])
                arr_dates.append(dts[start:end])
                # Active entries is fully 1 if we have complete data
                arr_active.append(np.ones((lags, 1)))
        else:
            # Non-sliding windows:
            # For example, chunk the series in blocks of size 'lags'
            # (pad if needed).
            remainder = time_steps % lags
            if remainder != 0:
                # Optionally pad or discard remainder
                needed = lags - remainder
                x = np.concatenate([x, np.zeros((needed, x.shape[1]))])
                y = np.concatenate([y, np.zeros((needed, y.shape[1]))])
                ids = np.concatenate([ids, np.zeros((needed, ids.shape[1]))])
                dts = np.concatenate([dts, np.full((needed, dts.shape[1]), "0")])
                time_steps = x.shape[0]  # updated length

            # Now we can reshape directly
            num_windows = time_steps // lags
            x_reshaped = x.reshape(num_windows, lags, x.shape[1])
            y_reshaped = y.reshape(num_windows, lags, y.shape[1])
            id_reshaped = ids.reshape(num_windows, lags, ids.shape[1])
            dt_reshaped = dts.reshape(num_windows, lags, dts.shape[1])

            # Active entries
            arr_active_reshaped = np.ones((num_windows, lags, 1))

            arr_inputs = [window for window in x_reshaped]
            arr_outputs = [window for window in y_reshaped]
            arr_identifier = [window for window in id_reshaped]
            arr_dates = [window for window in dt_reshaped]
            arr_active = [window for window in arr_active_reshaped]

        # Convert lists of windows => single 3D arrays
        arr_inputs     = np.stack(arr_inputs, axis=0) if arr_inputs else np.empty((0, lags, x.shape[1]))
        arr_outputs    = np.stack(arr_outputs, axis=0) if arr_outputs else np.empty((0, lags, y.shape[1]))
        arr_active     = np.stack(arr_active, axis=0)  if arr_active  else np.empty((0, lags, 1))
        arr_identifier = np.stack(arr_identifier, axis=0) if arr_identifier else np.empty((0, lags, ids.shape[1]))
        arr_dates      = np.stack(arr_dates, axis=0)      if arr_dates      else np.empty((0, lags, dts.shape[1]))

        return arr_inputs, arr_outputs, arr_active, arr_identifier, arr_dates

    def _cull_invalid_windows(self, data_map):
        """
        Removes windows whose 'identifier' field contains 0 or "0" anywhere
        along (window, time_step) dimensions.
        """
        # If 'identifier' has shape (..., 1), squeeze out that last dim
        if data_map["identifier"].ndim == 4 and data_map["identifier"].shape[-1] == 1:
            identifier_4d = np.squeeze(data_map["identifier"], axis=-1)
        else:
            identifier_4d = data_map["identifier"]

        # valid_mask shape => (num_windows, num_tickers, time_steps)
        valid_mask = np.all(
            (identifier_4d != 0) & (identifier_4d != "0") & (identifier_4d != 0.0),
            axis=(1, 2)
        )
        # valid_mask has length num_windows, one boolean per window

        for k in ["inputs", "outputs", "active_entries", "identifier", "date"]:
            data_map[k] = data_map[k][valid_mask]

        return data_map
    
# import numpy as np
# class GraphModelFeatures(ModelFeatures):
#     def _batch_data(self, data, sliding_window):
#         """
#         Builds 4D arrays for graph-based models:
#           (num_windows, num_tickers, total_time_steps, num_features).
#         Dimension 1 enumerates tickers, dimension 2 enumerates time steps,
#         and dimension 3 enumerates features.
#         """
#         data = data.copy()
#         data["date"] = data.index.strftime("%Y-%m-%d")

#         id_col = get_single_col_by_input_type(InputTypes.ID, self._column_definition)
#         time_col = get_single_col_by_input_type(InputTypes.TIME, self._column_definition)
#         target_col = get_single_col_by_input_type(InputTypes.TARGET, self._column_definition)

#         # Identify which columns are inputs vs. the single target
#         input_cols = [
#             tup[0]
#             for tup in self._column_definition
#             if tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
#         ]

#         # Gather data for final stacking
#         inputs_list = []
#         outputs_list = []
#         active_entries_list = []
#         identifier_list = []
#         date_list = []

#         # Sort tickers so dimension 1 (tickers) is consistent
#         tickers = sorted(data[id_col].unique())

#         for ticker in tickers:
#             # 1) Filter rows for this ticker and sort by time
#             df_ticker = data[data[id_col] == ticker].copy()
#             df_ticker.sort_values(by=time_col, inplace=True)

#             # 2) Convert to 3D arrays: (num_windows_for_ticker, time_steps, features)
#             arr_inputs, arr_outputs, arr_active, arr_identifier, arr_dates = \
#                 self._build_windows_for_single_ticker(
#                     df_ticker,
#                     id_col,
#                     time_col,
#                     target_col,
#                     input_cols,
#                     sliding_window
#                 )

#             # Append to lists
#             inputs_list.append(arr_inputs)
#             outputs_list.append(arr_outputs)
#             active_entries_list.append(arr_active)
#             identifier_list.append(arr_identifier)
#             date_list.append(arr_dates)

#         # 3) Trim all tickers to the same number of windows
#         min_windows = min(arr.shape[0] for arr in inputs_list)
#         inputs_list       = [arr[:min_windows] for arr in inputs_list]
#         outputs_list      = [arr[:min_windows] for arr in outputs_list]
#         active_entries_list = [arr[:min_windows] for arr in active_entries_list]
#         identifier_list   = [arr[:min_windows] for arr in identifier_list]
#         date_list         = [arr[:min_windows] for arr in date_list]

#         # 4) Stack along axis=1 => (num_windows, num_tickers, time_steps, features)
#         data_map = {}
#         data_map["inputs"]         = np.stack(inputs_list, axis=1)
#         data_map["outputs"]        = np.stack(outputs_list, axis=1)
#         data_map["active_entries"] = np.stack(active_entries_list, axis=1)
#         data_map["identifier"]     = np.stack(identifier_list, axis=1)
#         data_map["date"]           = np.stack(date_list, axis=1)

#         # --- Modification: Swap the second and third dimensions ---
#         # for inputs, active_entries, identifier, and date.
#         data_map["inputs"] = np.transpose(data_map["inputs"], (0, 2, 1, 3))
#         data_map["outputs"] = np.transpose(data_map["active_entries"], (0, 2, 1, 3))
#         data_map["active_entries"] = np.transpose(data_map["active_entries"], (0, 2, 1, 3))
#         data_map["identifier"] = np.transpose(data_map["identifier"], (0, 2, 1, 3))
#         data_map["date"] = np.transpose(data_map["date"], (0, 2, 1, 3))

#         # 5) Cull windows with invalid identifiers if needed
#         data_map = self._cull_invalid_windows(data_map)

#         # --- Revert outputs shape change ---
#         # Do not reshape outputs. They remain with shape:
#         # (num_windows, num_tickers, time_steps, output_size)

#         # Remove the extra singleton dimension for active_entries, identifier, and date if present.
#         if data_map["active_entries"].ndim == 4 and data_map["active_entries"].shape[-1] == 1:
#             data_map["active_entries"] = np.squeeze(data_map["active_entries"], axis=-1)
#         if data_map["identifier"].ndim == 4 and data_map["identifier"].shape[-1] == 1:
#             data_map["identifier"] = np.squeeze(data_map["identifier"], axis=-1)
#         if data_map["date"].ndim == 4 and data_map["date"].shape[-1] == 1:
#             data_map["date"] = np.squeeze(data_map["date"], axis=-1)

#         # Print final shapes
#         print("After final stacking and culling:")
#         print("inputs.shape:", data_map["inputs"].shape)
#         print("outputs.shape:", data_map["outputs"].shape)
#         print("active_entries.shape:", data_map["active_entries"].shape)
#         print("identifier.shape:", data_map["identifier"].shape)
#         print("date.shape:", data_map["date"].shape)

#         return data_map

#     def _build_windows_for_single_ticker(
#         self,
#         df_ticker,
#         id_col,
#         time_col,
#         target_col,
#         input_cols,
#         sliding_window
#     ):
#         """
#         Returns 5 arrays of shape (num_windows_for_ticker, time_steps, feature_dim):
#           inputs, outputs, active_entries, identifier, date
#         where time_steps == self.total_time_steps if enough data is present.
#         """
#         arr_inputs = []
#         arr_outputs = []
#         arr_active = []
#         arr_identifier = []
#         arr_dates = []

#         x = df_ticker[input_cols].values  # shape: (time_steps, num_input_features)
#         y = df_ticker[[target_col]].values  # shape: (time_steps, 1)
#         ids = df_ticker[[id_col]].values    # shape: (time_steps, 1)
#         dts = df_ticker[[time_col]].values    # shape: (time_steps, 1)

#         time_steps = len(df_ticker)
#         lags = self.total_time_steps

#         if sliding_window:
#             # For sliding windows, each start in [0.. time_steps-lags]
#             # yields a window of size (lags).
#             for start in range(time_steps - lags + 1):
#                 end = start + lags
#                 arr_inputs.append(x[start:end])
#                 arr_outputs.append(y[start:end])
#                 arr_identifier.append(ids[start:end])
#                 arr_dates.append(dts[start:end])
#                 # Active entries is fully 1 if we have complete data.
#                 arr_active.append(np.ones((lags, 1)))
#         else:
#             # Non-sliding windows: chunk the series in blocks of size 'lags' (pad if needed).
#             remainder = time_steps % lags
#             if remainder != 0:
#                 # Optionally pad or discard remainder.
#                 needed = lags - remainder
#                 x = np.concatenate([x, np.zeros((needed, x.shape[1]))])
#                 y = np.concatenate([y, np.zeros((needed, y.shape[1]))])
#                 ids = np.concatenate([ids, np.zeros((needed, ids.shape[1]))])
#                 dts = np.concatenate([dts, np.full((needed, dts.shape[1]), "0")])
#                 time_steps = x.shape[0]  # updated length

#             # Now we can reshape directly.
#             num_windows = time_steps // lags
#             x_reshaped = x.reshape(num_windows, lags, x.shape[1])
#             y_reshaped = y.reshape(num_windows, lags, y.shape[1])
#             id_reshaped = ids.reshape(num_windows, lags, ids.shape[1])
#             dt_reshaped = dts.reshape(num_windows, lags, dts.shape[1])

#             # Active entries.
#             arr_active_reshaped = np.ones((num_windows, lags, 1))

#             arr_inputs = [window for window in x_reshaped]
#             arr_outputs = [window for window in y_reshaped]
#             arr_identifier = [window for window in id_reshaped]
#             arr_dates = [window for window in dt_reshaped]
#             arr_active = [window for window in arr_active_reshaped]

#         # Convert lists of windows to single 3D arrays.
#         arr_inputs     = np.stack(arr_inputs, axis=0) if arr_inputs else np.empty((0, lags, x.shape[1]))
#         arr_outputs    = np.stack(arr_outputs, axis=0) if arr_outputs else np.empty((0, lags, y.shape[1]))
#         arr_active     = np.stack(arr_active, axis=0)  if arr_active  else np.empty((0, lags, 1))
#         arr_identifier = np.stack(arr_identifier, axis=0) if arr_identifier else np.empty((0, lags, ids.shape[1]))
#         arr_dates      = np.stack(arr_dates, axis=0)      if arr_dates      else np.empty((0, lags, dts.shape[1]))

#         return arr_inputs, arr_outputs, arr_active, arr_identifier, arr_dates

#     def _cull_invalid_windows(self, data_map):
#         """
#         Removes windows whose 'identifier' field contains 0 or "0" anywhere
#         along (window, time_step) dimensions.
#         """
#         # If 'identifier' has shape (..., 1), squeeze out that last dimension.
#         if data_map["identifier"].ndim == 4 and data_map["identifier"].shape[-1] == 1:
#             identifier_4d = np.squeeze(data_map["identifier"], axis=-1)
#         else:
#             identifier_4d = data_map["identifier"]

#         # valid_mask shape => (num_windows, num_tickers, time_steps)
#         valid_mask = np.all(
#             (identifier_4d != 0) & (identifier_4d != "0") & (identifier_4d != 0.0),
#             axis=(1, 2)
#         )
#         # valid_mask has one boolean per window.
#         for k in ["inputs", "outputs", "active_entries", "identifier", "date"]:
#             data_map[k] = data_map[k][valid_mask]

#         return data_map


class RollingGraphModelFeatures(GraphModelFeatures):
    """
    Extends GraphModelFeatures to compute rolling Pearson correlation
    adjacency matrices for each sample window.

    In addition to the standard batched data, this class also generates
    per-sample adjacency matrices based on rolling correlation windows.
    """

    def __init__(
        self,
        df,
        total_time_steps,
        correlation_lookback=20,
        correlation_threshold=0.5,
        returns_column="daily_returns",
        **kwargs
    ):
        """
        Args:
            df: Input DataFrame with features
            total_time_steps: Number of time steps per sample (default 20)
            correlation_lookback: Number of time steps to use for correlation (20, 40, 60)
            correlation_threshold: Threshold for Pearson correlation (tau)
            returns_column: Column name containing returns for correlation computation
            **kwargs: Additional arguments passed to parent class
        """
        self.correlation_lookback = correlation_lookback
        self.correlation_threshold = correlation_threshold
        self.returns_column = returns_column
        # Store the full dataframe for adjacency computation
        self._full_df = df.copy()
        super().__init__(df, total_time_steps, **kwargs)

    def _batch_data(self, data, sliding_window):
        """
        Override parent's _batch_data to also compute rolling adjacency matrices.
        """
        # Call parent's _batch_data
        data_map = super()._batch_data(data, sliding_window)

        # Compute rolling adjacency matrices using the stored full dataframe
        adjacencies = self._compute_rolling_adjacencies(data, data_map, sliding_window)
        data_map["adjacency"] = adjacencies

        print("adjacency.shape:", data_map["adjacency"].shape)

        return data_map

    def _compute_correlation_matrix(self, returns_df):
        """
        Compute Pearson correlation matrix from returns DataFrame.

        Args:
            returns_df: DataFrame of shape (lookback, num_tickers)

        Returns:
            Correlation matrix of shape (num_tickers, num_tickers)
        """
        corr = returns_df.corr().values
        corr = np.nan_to_num(corr, nan=0.0)
        return corr

    def _build_adjacency_from_correlation(self, corr_matrix):
        """
        Build adjacency matrix from correlation using threshold.

        Args:
            corr_matrix: Pearson correlation matrix (N x N)

        Returns:
            Adjacency matrix (N x N)
        """
        N = corr_matrix.shape[0]
        A = np.zeros((N, N))

        # Apply threshold on absolute correlation
        mask = np.abs(corr_matrix) >= self.correlation_threshold
        A[mask] = self.correlation_threshold

        # Zero out diagonal (no self-loops)
        np.fill_diagonal(A, 0)

        return A

    def _normalize_adjacency(self, A, add_self_loops=True):
        """
        Normalize adjacency matrix using symmetric normalization.

        Â = D̃^{-1/2} Ã D̃^{-1/2}

        Args:
            A: Adjacency matrix (N x N)
            add_self_loops: Whether to add self-loops

        Returns:
            Normalized adjacency matrix
        """
        if add_self_loops:
            A_tilde = A + np.eye(A.shape[0])
        else:
            A_tilde = A.copy()

        # Compute degree
        d = A_tilde.sum(axis=1)
        d_inv_sqrt = np.power(d, -0.5, where=d > 0)
        d_inv_sqrt[d == 0] = 0

        D_inv_sqrt = np.diag(d_inv_sqrt)

        # Symmetric normalization
        A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

        return A_norm

    def _compute_rolling_adjacencies(self, data, data_map, sliding_window):
        """
        Compute per-sample Pearson correlation adjacency matrices.

        For each sample window, compute correlation using the lookback period
        of returns data, apply threshold to create adjacency, and normalize.

        Args:
            data: Original DataFrame with all data
            data_map: Batched data containing dates for each window
            sliding_window: Whether using sliding windows

        Returns:
            adjacencies: np.ndarray of shape (num_windows, num_tickers, num_tickers)
        """
        id_col = get_single_col_by_input_type(InputTypes.ID, self._column_definition)
        time_col = get_single_col_by_input_type(InputTypes.TIME, self._column_definition)

        # Get sorted tickers
        tickers = sorted(data[id_col].unique())
        num_tickers = len(tickers)
        num_windows = data_map["inputs"].shape[0]

        # Use FULL dataframe for lookback (not just current split)
        # This ensures we have enough historical data for correlation lookback
        data_copy = self._full_df.copy()
        if not isinstance(data_copy.index, pd.DatetimeIndex):
            data_copy = data_copy.reset_index()

        # Get returns data for each ticker
        returns_pivot = data_copy.pivot_table(
            index=time_col,
            columns=id_col,
            values=self.returns_column,
            aggfunc='first'
        )
        returns_pivot = returns_pivot[tickers]  # Ensure ticker order matches
        returns_pivot = returns_pivot.sort_index()

        # Get the dates for each window from the batched data
        # date shape: (num_windows, num_tickers, time_steps, 1) or similar
        dates = data_map["date"]
        if dates.ndim == 4 and dates.shape[-1] == 1:
            dates = np.squeeze(dates, axis=-1)

        adjacencies = []

        for window_idx in range(num_windows):
            # Use the day BEFORE the window starts to avoid look-ahead bias
            # This ensures adjacency only contains information available before the window
            window_dates = dates[window_idx, 0, :]  # (time_steps,)
            window_start_date = window_dates[0]  # First date in window

            # Find the position of window_start_date in the returns index
            try:
                window_start_pos = returns_pivot.index.get_loc(window_start_date)
            except KeyError:
                window_start_pos = returns_pivot.index.searchsorted(window_start_date)
                window_start_pos = min(window_start_pos, len(returns_pivot) - 1)

            # End position is ONE DAY BEFORE the window starts (no look-ahead)
            end_pos = window_start_pos - 1

            # Compute start position based on lookback
            start_pos = max(0, end_pos - self.correlation_lookback + 1)

            # Get returns for this lookback period (entirely before the window)
            if end_pos >= start_pos and end_pos >= 0:
                lookback_returns = returns_pivot.iloc[start_pos:end_pos + 1]
            else:
                lookback_returns = pd.DataFrame()  # Not enough historical data

            # Handle case where we don't have enough data
            if len(lookback_returns) < 2:
                # Not enough data for correlation, use identity matrix
                A_norm = np.eye(num_tickers)
            else:
                # Compute correlation matrix
                corr_matrix = self._compute_correlation_matrix(lookback_returns)

                # Build adjacency from correlation
                A = self._build_adjacency_from_correlation(corr_matrix)

                # Normalize
                A_norm = self._normalize_adjacency(A, add_self_loops=True)

            adjacencies.append(A_norm)

        return np.stack(adjacencies, axis=0)

    def make_rolling_graph_dataset(
        self,
        sliding_window=True,
    ):
        """
        Return the datasets with rolling adjacency matrices.

        The adjacencies are already computed during __init__ when _batch_data is called.

        Args:
            sliding_window: If True, return test_sliding; otherwise return test_fixed

        Returns:
            Dictionary containing train, valid, test data with adjacency matrices
        """
        test_data = self.test_sliding if sliding_window else self.test_fixed

        return {
            "train": self.train,
            "valid": self.valid,
            "test": test_data,
        }