"""Run the publication correlation renderer on explicitly synthetic inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recipes.stnsnr.stats.walk.walk_lfp_fogq_3m_correlation import (
    FOGQ_COLUMNS,
    compute_walk_fogq_correlation,
)
from recipes.stnsnr.viz.walk.walk_lfp_fogq_3m_correlation_heatmap import (
    PANEL_ORDER,
    prepare_heatmap_matrices,
)
from recipes.stnsnr.viz.walk.walk_lfp_gait_pca_3m_correlation_heatmap import (
    plot_layered_correlation_heatmap,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    rng = np.random.default_rng(2026)
    ids = [f"synthetic-{number:02d}" for number in range(1, 13)]
    latent = np.linspace(-1.0, 1.0, len(ids))
    outcomes = pd.DataFrame({
        "ID": ids,
        FOGQ_COLUMNS[0]: latent + rng.normal(0, 0.35, len(ids)),
        FOGQ_COLUMNS[1]: -latent + rng.normal(0, 0.45, len(ids)),
    })
    outcomes.to_csv(output / "synthetic_outcomes.csv", index=False)
    feature_names = ["synthetic_theta", "synthetic_beta", "synthetic_gamma"]
    r_plots, p_plots = [], []
    for pose in PANEL_ORDER:
        features = pd.DataFrame({
            "ID": ids,
            feature_names[0]: latent + rng.normal(0, 0.5, len(ids)),
            feature_names[1]: -latent + rng.normal(0, 0.6, len(ids)),
            feature_names[2]: rng.normal(0, 1, len(ids)),
        })
        features.to_csv(output / f"synthetic_{pose}_features.csv", index=False)
        result = compute_walk_fogq_correlation(features, outcomes)
        for name in ["r", "p_raw", "p_adj", "n"]:
            getattr(result, name).to_csv(output / f"{pose}_{name}.csv")
        r_plot, p_plot = prepare_heatmap_matrices(
            result.r, result.p_raw, feature_order=feature_names,
            x_labels=["Synthetic theta", "Synthetic beta", "Synthetic gamma"],
            pose=pose,
        )
        r_plots.append(r_plot)
        p_plots.append(p_plot)
    fig = plot_layered_correlation_heatmap(
        r_plots, p_plots, label_top="Synthetic data",
    )
    fig.savefig(output / "synthetic_correlation.pdf", bbox_inches="tight")
    fig.savefig(output / "synthetic_correlation.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    rows = []
    intercepts = rng.normal(0, 0.3, len(ids))
    for timepoint in [25.0, 75.0]:
        for index, ident in enumerate(ids):
            for region in ["SNr", "STN"]:
                for epoch in range(3):
                    rows.append({
                        "ID": ident, "Subject": ident, "Region": region,
                        "Side": "R", "Side_cycle": "R", "Hemi": "MAH",
                        "Lat": "Ipsi", "Task": "Single", "Epoch": str(epoch),
                        "Param": "tfr", "Subparam": "period", "Metric": "synthetic_power",
                        "MetricClass": "local", "Band": "Theta", "t_pct": timepoint,
                        "Value": intercepts[index] + (0.35 if region == "STN" else -0.15)
                        + (0.15 if timepoint == 25 else -0.15) + rng.normal(0, 0.25),
                    })
    pd.DataFrame(rows).to_csv(output / "timepoint_input.csv", index=False)
    print(f"Synthetic correlation matrices: 2 x 3 for {len(PANEL_ORDER)} poses")
    print(f"Synthetic pointwise input: {len(rows)} rows; 12 participants; 2 regions; 2 phase points")
    print(f"Outputs: {output.resolve()}")
    print(f"Elapsed seconds: {perf_counter() - start:.3f}")


if __name__ == "__main__":
    main()
