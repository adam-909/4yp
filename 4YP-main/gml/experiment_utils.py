"""
Standardized experiment saving and loading for thesis results.

All experiment outputs are saved to a unified FINAL_RESULTS/ directory
(typically on Google Drive when running on Colab). This replaces ad-hoc
saving logic scattered across individual notebooks.

Directory structure:
    {base_dir}/{experiment_name}/seed_{seed}/          (single config)
    {base_dir}/{experiment_name}/{config_name}/seed_{seed}/  (multiple configs)

Usage (in notebook):
    from gml.experiment_utils import save_experiment_results

    save_experiment_results(
        experiment_name="4a_GATv2_per_ts_nores",
        seed=40,
        predictions=predictions,
        results_df=results_df,
        daily_returns=daily_returns,
        metrics_raw=metrics_raw,
        metrics_norm=metrics_norm,
        yearly_sharpes=yearly_sharpes,
        training_history=history.history,
        hyperparams={...},
        test_dates=test_dates_arr,
        attention_weights=all_attn,
    )
"""

import json
import os

import numpy as np
import pandas as pd


def save_experiment_results(
    experiment_name,
    seed,
    predictions,
    results_df,
    daily_returns,
    metrics_raw,
    metrics_norm,
    yearly_sharpes,
    training_history,
    hyperparams,
    test_dates=None,
    attention_weights=None,
    adjacency=None,
    graph_stats=None,
    model=None,
    base_dir="/content/drive/MyDrive/FINAL_RESULTS",
    config_name=None,
):
    """
    Save all experiment outputs to a standardized directory structure.

    Args:
        experiment_name: e.g. "4a_GATv2_per_ts_nores"
        seed: Random seed used for this run (e.g. 40)
        predictions: Raw model output, shape (num_windows, num_tickers, time_steps, 1)
        results_df: DataFrame with columns [time, identifier, position, returns,
            captured_returns]
        daily_returns: pd.Series of aggregated daily portfolio returns
        metrics_raw: Dict of raw performance metrics
        metrics_norm: Dict of vol-normalized performance metrics
        yearly_sharpes: Dict mapping year -> Sharpe ratio
        training_history: Dict from history.history (train/val loss per epoch)
        hyperparams: Dict of all configuration values
        test_dates: Array of test window dates for alignment with VIX/regimes.
            Shape (num_windows,) of datetime-like values.
        attention_weights: GAT only. Shape (num_windows, heads, nodes, nodes).
        adjacency: GCN only. Static (nodes, nodes) or rolling
            (num_windows, nodes, nodes).
        graph_stats: Dict of arrays with per-window graph statistics
            (edge_count, mean_degree, etc.).
        model: Keras model to save weights from. Optional.
        base_dir: Root directory for all results.
        config_name: Sub-directory for hyperparameter variants
            (e.g. "lb20_th05"). If None, no sub-directory is created.
    """
    if config_name is not None:
        results_dir = os.path.join(
            base_dir, experiment_name, config_name, f"seed_{seed}"
        )
    else:
        results_dir = os.path.join(base_dir, experiment_name, f"seed_{seed}")

    os.makedirs(results_dir, exist_ok=True)

    # --- Standard CSV files ---
    results_df.to_csv(
        os.path.join(results_dir, "captured_returns_sw.csv"), index=False
    )

    pd.DataFrame([metrics_raw]).to_csv(
        os.path.join(results_dir, "metrics_raw.csv"), index=False
    )

    pd.DataFrame([metrics_norm]).to_csv(
        os.path.join(results_dir, "metrics_vol_normalized.csv"), index=False
    )

    pd.DataFrame(
        list(yearly_sharpes.items()), columns=["Year", "Sharpe"]
    ).to_csv(os.path.join(results_dir, "yearly_sharpes.csv"), index=False)

    daily_returns.to_csv(os.path.join(results_dir, "daily_returns.csv"))

    # --- JSON files ---
    # Convert any non-serializable values in training_history
    serializable_history = {
        k: [float(v) for v in vals] for k, vals in training_history.items()
    }
    with open(os.path.join(results_dir, "training_history.json"), "w") as f:
        json.dump(serializable_history, f, indent=2)

    # Convert numpy types in hyperparams to native Python for JSON
    clean_hyperparams = {}
    for k, v in hyperparams.items():
        if isinstance(v, (np.integer,)):
            clean_hyperparams[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean_hyperparams[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean_hyperparams[k] = v.tolist()
        else:
            clean_hyperparams[k] = v

    with open(os.path.join(results_dir, "hyperparams.json"), "w") as f:
        json.dump(clean_hyperparams, f, indent=2)

    # --- NumPy arrays ---
    np.save(os.path.join(results_dir, "predictions.npy"), predictions)

    if test_dates is not None:
        np.save(os.path.join(results_dir, "test_dates.npy"), test_dates)

    if attention_weights is not None:
        np.save(
            os.path.join(results_dir, "attention_weights.npy"), attention_weights
        )

    if adjacency is not None:
        np.save(os.path.join(results_dir, "adjacency.npy"), adjacency)

    # --- Graph statistics ---
    if graph_stats is not None:
        pd.DataFrame(graph_stats).to_csv(
            os.path.join(results_dir, "graph_stats.csv"), index=False
        )

    # --- Model weights ---
    if model is not None:
        weights_dir = os.path.join(results_dir, "model_weights")
        os.makedirs(weights_dir, exist_ok=True)
        model.save_weights(os.path.join(weights_dir, "weights.weights.h5"))

    print(f"Results saved to: {results_dir}")
    _print_saved_files(results_dir)


def _print_saved_files(results_dir):
    """Print summary of what was saved."""
    files = []
    for root, dirs, filenames in os.walk(results_dir):
        for f in filenames:
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            rel = os.path.relpath(path, results_dir)
            if size > 1024 * 1024:
                files.append(f"  {rel} ({size / 1024 / 1024:.1f} MB)")
            elif size > 1024:
                files.append(f"  {rel} ({size / 1024:.1f} KB)")
            else:
                files.append(f"  {rel} ({size} B)")
    print(f"Saved {len(files)} files:")
    for f in files:
        print(f)


def load_experiment_results(
    experiment_name, seed, config_name=None, base_dir="FINAL_RESULTS"
):
    """
    Load a saved experiment.

    Returns:
        Dict with keys matching what was saved: metrics_raw, metrics_norm,
        yearly_sharpes, daily_returns, predictions, training_history,
        hyperparams, attention_weights, adjacency, graph_stats, test_dates.
        Missing files return None for their key.
    """
    if config_name is not None:
        results_dir = os.path.join(
            base_dir, experiment_name, config_name, f"seed_{seed}"
        )
    else:
        results_dir = os.path.join(base_dir, experiment_name, f"seed_{seed}")

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"No results found at {results_dir}")

    result = {"experiment_name": experiment_name, "seed": seed}

    # CSV files
    csv_files = {
        "captured_returns": "captured_returns_sw.csv",
        "metrics_raw": "metrics_raw.csv",
        "metrics_norm": "metrics_vol_normalized.csv",
        "yearly_sharpes": "yearly_sharpes.csv",
        "daily_returns": "daily_returns.csv",
        "graph_stats": "graph_stats.csv",
    }
    for key, filename in csv_files.items():
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            if key == "daily_returns":
                result[key] = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
            elif key in ("metrics_raw", "metrics_norm"):
                result[key] = pd.read_csv(path).iloc[0].to_dict()
            elif key == "yearly_sharpes":
                df = pd.read_csv(path)
                result[key] = dict(zip(df["Year"], df["Sharpe"]))
            else:
                result[key] = pd.read_csv(path)
        else:
            result[key] = None

    # JSON files
    for key, filename in [
        ("training_history", "training_history.json"),
        ("hyperparams", "hyperparams.json"),
    ]:
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            with open(path) as f:
                result[key] = json.load(f)
        else:
            result[key] = None

    # NumPy files
    for key, filename in [
        ("predictions", "predictions.npy"),
        ("attention_weights", "attention_weights.npy"),
        ("adjacency", "adjacency.npy"),
        ("test_dates", "test_dates.npy"),
    ]:
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            result[key] = np.load(path, allow_pickle=True)
        else:
            result[key] = None

    return result


def load_all_experiments(base_dir="FINAL_RESULTS"):
    """
    Load all experiments and seeds from the results directory.

    Returns:
        Nested dict: {experiment_name: {config_or_seed: {seed: result_dict}}}
        For experiments without config_name: {experiment_name: {"seed_40": result_dict}}
        For experiments with config_name: {experiment_name: {"config": {"seed_40": result_dict}}}
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Results directory not found: {base_dir}")

    all_results = {}

    for exp_name in sorted(os.listdir(base_dir)):
        exp_dir = os.path.join(base_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        all_results[exp_name] = {}

        for sub in sorted(os.listdir(exp_dir)):
            sub_dir = os.path.join(exp_dir, sub)
            if not os.path.isdir(sub_dir):
                continue

            if sub.startswith("seed_"):
                # Direct seed directory (no config_name)
                seed = int(sub.replace("seed_", ""))
                try:
                    all_results[exp_name][sub] = load_experiment_results(
                        exp_name, seed, base_dir=base_dir
                    )
                except FileNotFoundError:
                    pass
            else:
                # Config sub-directory — look for seeds inside
                config_name = sub
                all_results[exp_name][config_name] = {}
                for seed_dir in sorted(os.listdir(sub_dir)):
                    if seed_dir.startswith("seed_"):
                        seed = int(seed_dir.replace("seed_", ""))
                        try:
                            all_results[exp_name][config_name][
                                seed_dir
                            ] = load_experiment_results(
                                exp_name,
                                seed,
                                config_name=config_name,
                                base_dir=base_dir,
                            )
                        except FileNotFoundError:
                            pass

    return all_results


def compute_graph_stats(graphs, threshold=0.02):
    """
    Compute per-window graph statistics from attention or adjacency matrices.

    Args:
        graphs: Array of shape (num_windows, nodes, nodes).
        threshold: Edge threshold for counting edges.

    Returns:
        Dict of arrays, each of length num_windows:
            num_edges, mean_degree, max_degree, min_degree,
            mean_edge_weight, std_edge_weight, max_attn, mean_entropy.
    """
    stats = {
        "num_edges": [],
        "mean_degree": [],
        "max_degree": [],
        "min_degree": [],
        "mean_edge_weight": [],
        "std_edge_weight": [],
        "max_attn": [],
        "mean_entropy": [],
    }
    n = graphs.shape[-1]

    for g in graphs:
        sym = (g + g.T) / 2
        np.fill_diagonal(sym, 0)

        mask = sym > threshold
        edge_count = mask.sum() / 2
        degree = mask.sum(axis=1)
        stats["num_edges"].append(edge_count)
        stats["mean_degree"].append(degree.mean())
        stats["max_degree"].append(degree.max())
        stats["min_degree"].append(degree.min())

        off_diag = sym[~np.eye(n, dtype=bool)]
        stats["mean_edge_weight"].append(off_diag.mean())
        stats["std_edge_weight"].append(off_diag.std())
        stats["max_attn"].append(off_diag.max())

        ent = -np.sum(g * np.log(g + 1e-9), axis=-1)
        stats["mean_entropy"].append(ent.mean())

    return {k: np.array(v) for k, v in stats.items()}


def perturb_adjacencies(adjacencies, mode, fraction, base_seed=42):
    """Perturb each per-window adjacency by adding or removing edges.

    Preserves real edge weights under subtract mode. For add mode, assigns
    fake edges a weight drawn from the distribution of existing real edge
    weights at that window. Re-applies symmetric normalisation
    D^{-1/2}(A+I)D^{-1/2} after perturbation.

    Args:
        adjacencies: (num_windows, N, N) array of normalised adjacency matrices
            (as produced by RollingGraphModelFeatures, already symmetrically
            normalised with self-loops).
        mode: 'add' or 'subtract'.
        fraction: float in [0, 1+]. Fraction of current real edges to perturb.
            - 'subtract': remove round(fraction * current_edges) real edges.
              At fraction=1.0, all real edges are removed (graph becomes
              self-loops only after normalisation).
            - 'add': add round(fraction * current_edges) fake edges.
              Capped at the number of available non-edge pairs.
        base_seed: base RNG seed; window_idx is added per window for
            reproducibility across GCN and GAT runs.

    Returns:
        Array of same shape as input, re-normalised. Self-loops re-added
        during normalisation.

    Raises:
        ValueError: if mode is not 'add' or 'subtract', or fraction < 0.
    """
    if mode not in ("add", "subtract"):
        raise ValueError(f"mode must be 'add' or 'subtract', got {mode!r}")
    if fraction < 0:
        raise ValueError(f"fraction must be >= 0, got {fraction}")

    num_windows, n, _ = adjacencies.shape
    result = np.zeros_like(adjacencies)

    for t in range(num_windows):
        rng = np.random.default_rng(base_seed + t)
        A = adjacencies[t].copy()
        # Work with the off-diagonal, symmetric part
        A_no_diag = A.copy()
        np.fill_diagonal(A_no_diag, 0)

        # Find current real edges (upper triangle only, undirected)
        iu, ju = np.triu_indices(n, k=1)
        edge_mask = A_no_diag[iu, ju] > 0
        real_edge_indices = np.where(edge_mask)[0]
        non_edge_indices = np.where(~edge_mask)[0]

        real_weights = A_no_diag[iu[real_edge_indices], ju[real_edge_indices]]
        n_real_edges = len(real_edge_indices)

        # Start from current adjacency (off-diagonal, symmetric)
        new_adj = A_no_diag.copy()

        if mode == "subtract":
            n_to_remove = int(round(fraction * n_real_edges))
            n_to_remove = min(n_to_remove, n_real_edges)
            if n_to_remove > 0:
                remove_choice = rng.choice(n_real_edges, size=n_to_remove, replace=False)
                remove_idx = real_edge_indices[remove_choice]
                i_rem = iu[remove_idx]
                j_rem = ju[remove_idx]
                new_adj[i_rem, j_rem] = 0
                new_adj[j_rem, i_rem] = 0

        elif mode == "add":
            n_to_add = int(round(fraction * n_real_edges))
            n_to_add = min(n_to_add, len(non_edge_indices))
            if n_to_add > 0 and len(real_weights) > 0:
                add_choice = rng.choice(len(non_edge_indices), size=n_to_add, replace=False)
                add_idx = non_edge_indices[add_choice]
                i_add = iu[add_idx]
                j_add = ju[add_idx]
                # Sample weights from the distribution of real edge weights
                sampled_weights = rng.choice(real_weights, size=n_to_add, replace=True)
                new_adj[i_add, j_add] = sampled_weights
                new_adj[j_add, i_add] = sampled_weights

        # Re-apply symmetric normalisation: D^{-1/2} (A + I) D^{-1/2}
        np.fill_diagonal(new_adj, 0)  # ensure no self-loops before adding identity
        A_with_self = new_adj + np.eye(n)
        d = A_with_self.sum(axis=1)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        result[t] = D_inv_sqrt @ A_with_self @ D_inv_sqrt

    return result
