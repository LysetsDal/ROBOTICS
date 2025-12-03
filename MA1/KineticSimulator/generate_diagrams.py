import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SIMULATOR = ROOT / "simulator_swarm_robotics.py"


def run_collect_command(scenario, output_path, runtime, sample_interval, runs, seed):
    command = [
        sys.executable,
        str(SIMULATOR),
        "collect",
        scenario,
        "--runtime",
        f"{runtime}",
        "--sample-interval",
        f"{sample_interval}",
        "--runs",
        str(runs),
        "--seed",
        str(seed),
        "--output",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def load_metrics(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def align_runs(runs, metric_key):
    if not runs:
        raise ValueError("No runs available for plotting")
    min_len = min(len(run[metric_key]) for run in runs)
    if min_len == 0:
        raise ValueError(f"Metric '{metric_key}' has no samples")
    times = np.array(runs[0]["time"][:min_len], dtype=float)
    values = np.vstack(
        [np.array(run[metric_key][:min_len], dtype=float) for run in runs]
    )
    return times, values


def plot_time_series(runs, metric_key, title, ylabel, file_path):
    times, values = align_runs(runs, metric_key)
    mean = values.mean(axis=0)
    std = values.std(axis=0)

    plt.figure(figsize=(8, 4))
    for row in values:
        plt.plot(times, row, alpha=0.35, linewidth=1, color="#888888")
    plt.plot(times, mean, color="tab:blue", linewidth=2, label="mean")
    plt.fill_between(times, mean - std, mean + std, color="tab:blue", alpha=0.2)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path, dpi=200)
    plt.close()


def plot_histogram(runs, metric_key, title, xlabel, file_path, bins=10):
    final_values = [run[metric_key][-1] for run in runs if run[metric_key]]
    if not final_values:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(final_values, bins=bins, color="tab:purple", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(file_path, dpi=200)
    plt.close()


def plot_cluster_heatmap(runs, file_path):
    times, values = align_runs(runs, "num_flocks")
    values = values.astype(int)
    unique_clusters = sorted(set(values.flatten()))
    if not unique_clusters:
        return
    freq = np.zeros((len(unique_clusters), values.shape[1]))
    for idx, cluster in enumerate(unique_clusters):
        freq[idx] = (values == cluster).mean(axis=0)

    plt.figure(figsize=(8, 4))
    extent = [times[0], times[-1], unique_clusters[0] - 0.5, unique_clusters[-1] + 0.5]
    im = plt.imshow(
        freq,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(im, label="Frequency")
    plt.yticks(unique_clusters, [str(c) for c in unique_clusters])
    plt.xlabel("Time (s)")
    plt.ylabel("Cluster count")
    plt.title("Flocking: Cluster Count Heatmap")
    plt.tight_layout()
    plt.savefig(file_path, dpi=200)
    plt.close()


def generate_dispersion_plots(data, output_dir):
    runs = data["runs"]
    plot_time_series(
        runs,
        "avg_nn",
        "Dispersion: Average Nearest Neighbor",
        "Distance (px)",
        output_dir / "dispersion_avg_nn.png",
    )
    plot_time_series(
        runs,
        "hull_area",
        "Dispersion: Convex Hull Area",
        "Area (px^2)",
        output_dir / "dispersion_hull_area.png",
    )
    plot_histogram(
        runs,
        "avg_nn",
        "Dispersion: Final Average Nearest Neighbor",
        "Distance (px)",
        output_dir / "dispersion_avg_nn_hist.png",
    )
    plot_histogram(
        runs,
        "hull_area",
        "Dispersion: Final Hull Area",
        "Area (px^2)",
        output_dir / "dispersion_hull_hist.png",
    )


def generate_flocking_plots(data, output_dir):
    runs = data["runs"]
    plot_time_series(
        runs,
        "alignment",
        "Flocking: Heading Alignment",
        "Alignment (0-1)",
        output_dir / "flocking_alignment.png",
    )
    plot_time_series(
        runs,
        "avg_nn",
        "Flocking: Cohesion (Average Nearest Neighbor)",
        "Distance (px)",
        output_dir / "flocking_avg_nn.png",
    )
    plot_time_series(
        runs,
        "hull_area",
        "Flocking: Convex Hull Area",
        "Area (px^2)",
        output_dir / "flocking_hull_area.png",
    )
    plot_time_series(
        runs,
        "num_flocks",
        "Flocking: Cluster Count",
        "Number of clusters",
        output_dir / "flocking_cluster_count.png",
    )
    plot_histogram(
        runs,
        "alignment",
        "Flocking: Final Alignment",
        "Alignment (0-1)",
        output_dir / "flocking_alignment_hist.png",
    )
    plot_histogram(
        runs,
        "num_flocks",
        "Flocking: Final Cluster Count",
        "Number of clusters",
        output_dir / "flocking_cluster_hist.png",
        bins=max(5, len(runs)),
    )
    plot_cluster_heatmap(
        runs,
        output_dir / "flocking_cluster_heatmap.png",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dispersion and flocking diagrams"
    )
    parser.add_argument("--runtime", type=float, default=120.0)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "analysis_data",
        help="Directory for cached metric JSON",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=ROOT / "analysis_plots",
        help="Directory for generated diagrams",
    )
    parser.add_argument(
        "--skip-sim",
        action="store_true",
        help="Reuse existing JSON data instead of rerunning simulations",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    plots_dir = args.plots_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dispersion_json = data_dir / "dispersion_metrics.json"
    flocking_json = data_dir / "flocking_metrics.json"

    if not args.skip_sim:
        run_collect_command(
            "dispersion",
            dispersion_json,
            runtime=args.runtime,
            sample_interval=args.sample_interval,
            runs=args.runs,
            seed=args.seed,
        )
        run_collect_command(
            "flocking",
            flocking_json,
            runtime=args.runtime,
            sample_interval=args.sample_interval,
            runs=args.runs,
            seed=args.seed,
        )

    if not dispersion_json.exists() or not flocking_json.exists():
        raise FileNotFoundError(
            "Metric JSON files are missing; run without --skip-sim first."
        )

    dispersion_data = load_metrics(dispersion_json)
    flocking_data = load_metrics(flocking_json)

    generate_dispersion_plots(dispersion_data, plots_dir)
    generate_flocking_plots(flocking_data, plots_dir)

    print(f"Saved diagrams to {plots_dir}")
if __name__ == "__main__":
    main()
