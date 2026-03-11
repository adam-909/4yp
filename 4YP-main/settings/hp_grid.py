#import keras_tuner as kt

# LSTM Only

# For proper training

# HP_HIDDEN_LAYER_SIZE = [5, 10, 20, 40, 80, 160]
# HP_DROPOUT_RATE = [0.1, 0.2, 0.3, 0.4, 0.5]
# HP_MINIBATCH_SIZE = [32, 64, 128]
# HP_LEARNING_RATE = [1e-4, 1e-3, 1e-2, 1e-1]
# HP_MAX_GRADIENT_NORM = [0.01, 1.0, 100.0]


# For testing the code

HP_HIDDEN_LAYER_SIZE = [5]
HP_DROPOUT_RATE = [0.3]
HP_MINIBATCH_SIZE = [32]
HP_LEARNING_RATE = [1e-2]
HP_MAX_GRADIENT_NORM = [0.01]

# LSTM + GCN

HP_HIDDEN_LAYER_SIZE_GRAPH = [10]#5, 10, 20, 40, 80, 160]
HP_DROPOUT_RATE_GRAPH = [0.4]#0.1, 0.2, 0.3, 0.4, 0.5]
HP_MINIBATCH_SIZE_GRAPH = [32]#, 64, 128]
HP_LEARNING_RATE_GRAPH = [0.001]#1e-4, 1e-3, 1e-2, 1e-1]
HP_MAX_GRADIENT_NORM_GRAPH = [0.01]#, 1.0, 100.0]
HP_GCN_UNITS = [16]#, 32, 64]

HP_ALPHA = [100]#0.1, 1.0, 10.0, 100.0]
HP_BETA = [0.01]#0.01, 0.1, 1.0, 10.0]

# GAT-specific hyperparameters
HP_ATTN_HEADS = [4]  # Number of attention heads [2, 4, 8]

# Rolling Pearson graph hyperparameters
HP_CORRELATION_LOOKBACK = [20, 40, 60]  # Time steps for correlation window
HP_CORRELATION_THRESHOLD = [0.3, 0.4, 0.5, 0.6]  # Tau threshold for adjacency


# HP_HIDDEN_LAYER_SIZE = [20]
# HP_DROPOUT_RATE = [0.1]
# HP_MINIBATCH_SIZE = [128]
# HP_LEARNING_RATE = [1e-2]
# HP_MAX_GRADIENT_NORM = [100]

# # HP_MINIBATCH_SIZE= [32, 64, 128]



# hidden_layer_size = hp.Choice("hidden_layer_size", values=[10])
        # dropout_rate      = hp.Choice("dropout_rate",      values=[0.2])
        # max_gradient_norm = hp.Choice("max_gradient_norm", values=[5.0])
        # learning_rate     = hp.Choice("learning_rate",     values=[1e-3])
        # gcn_units         = hp.Choice("gcn_units",         values=[32])
        
# HP_HIDDEN_LAYER_SIZE_GRAPH = [10]
# HP_DROPOUT_RATE_GRAPH = [0.2]
# HP_MINIBATCH_SIZE_GRAPH = [32]
# HP_LEARNING_RATE_GRAPH = [1e-3]
# HP_MAX_GRADIENT_NORM_GRAPH = [5.0]
# HP_GCN_UNITS = [32]

# HP_ALPHA = [100]
# HP_BETA = [0.01]

















