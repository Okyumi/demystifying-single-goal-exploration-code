#!/usr/bin/env python3
"""
Compare results across conditions within each study.

Generates side-by-side comparison plots for all metrics.

Usage:
  python experiments/isolation_study/compare_results.py --study a --seed 42
  python experiments/isolation_study/compare_results.py --study b --seed 42
  python experiments/isolation_study/compare_results.py --study both --seed 42
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from visualization import (
    plot_metric_comparison,
    plot_bar_comparison,
    plot_adaptation_comparison,
    plot_m8_alignment_comparison,
    plot_approach_direction_stats,
    plot_approach_over_time,
)


def load_metrics(path):
    with open(os.path.join(path, "metrics.json")) as f:
        return json.load(f)


def load_env_config(path):
    with open(os.path.join(path, "env_config.json")) as f:
        return json.load(f)


# -----------------------------------------------------------------------
# Study A comparisons
# -----------------------------------------------------------------------

def compare_study_a(results_dir, output_dir):
    print("Generating Study A comparison plots...")
    os.makedirs(output_dir, exist_ok=True)

    base_dir = os.path.join(results_dir, "A1-baseline")
    allreplay_dir = os.path.join(results_dir, "A2-stochastic-allreplay")
    successonly_dir = os.path.join(results_dir, "A2-stochastic-successonly")

    base_m = load_metrics(base_dir)
    conditions = [("A1-baseline (Deterministic)", base_m)]

    if os.path.exists(allreplay_dir):
        allreplay_m = load_metrics(allreplay_dir)
        conditions.append(("A2-allreplay (Stochastic)", allreplay_m))
    if os.path.exists(successonly_dir):
        successonly_m = load_metrics(successonly_dir)
        conditions.append(("A2-successonly (Stochastic)", successonly_m))

    # M1: Success rate comparison
    for i in range(1, len(conditions)):
        label_t, m_t = conditions[i]
        tag = "allreplay" if "allreplay" in label_t else "successonly"
        plot_metric_comparison(
            np.array(base_m["m1_rolling_success_rate"]),
            np.array(m_t["m1_rolling_success_rate"]),
            "Episode", "Success Rate (rolling 50)",
            f"M1: Success Rate — Baseline vs {tag.title()}",
            os.path.join(output_dir, f"m1_success_rate_vs_{tag}.png"),
            baseline_label="A1-baseline",
            treatment_label=f"A2-{tag}",
        )

    # M3: Exploitation ratio comparison
    for i in range(1, len(conditions)):
        label_t, m_t = conditions[i]
        tag = "allreplay" if "allreplay" in label_t else "successonly"
        b_exploit = {int(k): v for k, v in base_m["m3_exploitation_ratio"].items()}
        t_exploit = {int(k): v for k, v in m_t["m3_exploitation_ratio"].items()}
        plot_bar_comparison(
            b_exploit, t_exploit,
            "Exploitation Ratio", f"M3: Exploitation Ratio — Baseline vs {tag.title()}",
            os.path.join(output_dir, f"m3_exploitation_ratio_vs_{tag}.png"),
            baseline_label="A1-baseline",
            treatment_label=f"A2-{tag}",
        )

    # M4+M5: Adaptation metrics
    for i in range(1, len(conditions)):
        label_t, m_t = conditions[i]
        tag = "allreplay" if "allreplay" in label_t else "successonly"
        plot_adaptation_comparison(
            base_m["m4_policy_adaptation_index"],
            m_t["m4_policy_adaptation_index"],
            base_m["m5_representation_adaptation_rate"],
            m_t["m5_representation_adaptation_rate"],
            os.path.join(output_dir, f"m4_m5_adaptation_vs_{tag}.png"),
            baseline_label="A1-baseline",
            treatment_label=f"A2-{tag}",
        )

    # M6: Suboptimality gap (stochastic conditions only)
    for i in range(1, len(conditions)):
        label_t, m_t = conditions[i]
        tag = "allreplay" if "allreplay" in label_t else "successonly"
        if "m6_suboptimality_gap" in m_t:
            # Baseline has no M6 (deterministic), so plot treatment alone
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(m_t["m6_suboptimality_gap"], linewidth=1.5,
                    color="#E64A19", alpha=0.8)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Suboptimality Gap")
            ax.set_title(f"M6: Suboptimality Gap — {tag.title()}", fontweight="bold")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            savepath = os.path.join(output_dir, f"m6_suboptimality_gap_{tag}.png")
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # M7: Approach direction entropy (stochastic conditions only)
    for i in range(1, len(conditions)):
        label_t, m_t = conditions[i]
        tag = "allreplay" if "allreplay" in label_t else "successonly"
        if "m7_approach_direction_entropy" in m_t:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(m_t["m7_approach_direction_entropy"], linewidth=1.5,
                    color="#E64A19", alpha=0.8)
            ax.axhline(np.log(4), color="gray", linestyle="--", alpha=0.5,
                        label="Max entropy (uniform)")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Shannon Entropy")
            ax.set_title(f"M7: Approach Direction Entropy — {tag.title()}",
                         fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            savepath = os.path.join(output_dir, f"m7_approach_entropy_{tag}.png")
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # M8: Representation-Route Alignment
    for i in range(1, len(conditions)):
        label_t, m_t = conditions[i]
        tag = "allreplay" if "allreplay" in label_t else "successonly"
        plot_m8_alignment_comparison(
            base_m["m8_representation_route_alignment"],
            m_t["m8_representation_route_alignment"],
            os.path.join(output_dir, f"m8_alignment_vs_{tag}.png"),
            baseline_label="A1-baseline",
            treatment_label=f"A2-{tag}",
        )

    # AllReplay vs SuccessOnly direct comparison
    if os.path.exists(allreplay_dir) and os.path.exists(successonly_dir):
        allreplay_m = load_metrics(allreplay_dir)
        successonly_m = load_metrics(successonly_dir)
        plot_metric_comparison(
            np.array(allreplay_m["m1_rolling_success_rate"]),
            np.array(successonly_m["m1_rolling_success_rate"]),
            "Episode", "Success Rate (rolling 50)",
            "M1: AllReplay vs SuccessOnly",
            os.path.join(output_dir, "m1_allreplay_vs_successonly.png"),
            baseline_label="A2-allreplay",
            treatment_label="A2-successonly",
        )

    print(f"  Study A comparisons saved to {output_dir}")


# -----------------------------------------------------------------------
# Study B comparisons
# -----------------------------------------------------------------------

def compare_study_b(results_dir, output_dir):
    print("Generating Study B comparison plots...")
    os.makedirs(output_dir, exist_ok=True)

    base_dir = os.path.join(results_dir, "B1-baseline")
    change_dir = os.path.join(results_dir, "B2-changing")

    base_m = load_metrics(base_dir)
    change_m = load_metrics(change_dir)

    phase_transitions = change_m.get("phase_transition_episodes", [])
    phase_names = ["fourrooms", "corridor_shift", "l_wall"]

    # M1: Success rate
    plot_metric_comparison(
        np.array(base_m["m1_rolling_success_rate"]),
        np.array(change_m["m1_rolling_success_rate"]),
        "Episode", "Success Rate (rolling 50)",
        "M1: Success Rate — Static vs Changing Dynamics",
        os.path.join(output_dir, "m1_success_rate.png"),
        baseline_label="B1-static",
        treatment_label="B2-changing",
        phase_transitions=phase_transitions,
        phase_names=phase_names,
    )

    # M3: Exploitation ratio
    b_exploit = {int(k): v for k, v in base_m["m3_exploitation_ratio"].items()}
    t_exploit = {int(k): v for k, v in change_m["m3_exploitation_ratio"].items()}
    plot_bar_comparison(
        b_exploit, t_exploit,
        "Exploitation Ratio", "M3: Exploitation Ratio — Static vs Changing",
        os.path.join(output_dir, "m3_exploitation_ratio.png"),
        baseline_label="B1-static",
        treatment_label="B2-changing",
        phase_names=phase_names,
    )

    # M4+M5: Adaptation metrics
    plot_adaptation_comparison(
        base_m["m4_policy_adaptation_index"],
        change_m["m4_policy_adaptation_index"],
        base_m["m5_representation_adaptation_rate"],
        change_m["m5_representation_adaptation_rate"],
        os.path.join(output_dir, "m4_m5_adaptation.png"),
        baseline_label="B1-static",
        treatment_label="B2-changing",
        phase_transitions=phase_transitions,
    )

    # M8: Representation-Route Alignment
    plot_m8_alignment_comparison(
        base_m["m8_representation_route_alignment"],
        change_m["m8_representation_route_alignment"],
        os.path.join(output_dir, "m8_alignment.png"),
        baseline_label="B1-static",
        treatment_label="B2-changing",
        phase_transitions=phase_transitions,
    )

    # M9: Phase-transition recovery
    recovery = change_m.get("m9_phase_transition_recovery", [])
    if recovery:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        trans_eps = [r["transition_episode"] for r in recovery]
        rec_times = [r["recovery_episodes"] if r["recovery_episodes"] is not None
                     else 1000 for r in recovery]
        colors = ["#4CAF50" if r["recovery_episodes"] is not None
                  else "#F44336" for r in recovery]
        bars = ax.bar(range(len(trans_eps)), rec_times, color=colors)
        ax.set_xticks(range(len(trans_eps)))
        ax.set_xticklabels([f"Ep {e}" for e in trans_eps])
        ax.set_ylabel("Episodes to Recover (50% of pre-level)")
        ax.set_title("M9: Phase-Transition Recovery Time (B2-changing)",
                     fontweight="bold")
        for i, r in enumerate(recovery):
            if r["recovery_episodes"] is None:
                ax.text(i, rec_times[i], "Never", ha="center", va="bottom",
                        fontweight="bold", color="#F44336")
        fig.tight_layout()
        savepath = os.path.join(output_dir, "m9_recovery.png")
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # M2: Path diversity comparison
    b_div = base_m.get("m2_path_diversity", {})
    t_div = change_m.get("m2_path_diversity", {})
    if b_div or t_div:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Jaccard
        b_jaccard = {int(k): v["mean_jaccard"] for k, v in b_div.items()}
        t_jaccard = {int(k): v["mean_jaccard"] for k, v in t_div.items()}
        all_phases = sorted(set(b_jaccard) | set(t_jaccard))
        x = np.arange(len(all_phases))
        w = 0.35
        axes[0].bar(x - w/2, [b_jaccard.get(p, 0) for p in all_phases], w,
                     label="B1-static", color="#1976D2")
        axes[0].bar(x + w/2, [t_jaccard.get(p, 0) for p in all_phases], w,
                     label="B2-changing", color="#E64A19")
        axes[0].set_xticks(x)
        labels = [phase_names[p] if p < len(phase_names) else f"P{p}" for p in all_phases]
        axes[0].set_xticklabels(labels, rotation=30, ha="right")
        axes[0].set_title("Mean Jaccard Distance")
        axes[0].legend(fontsize=8)

        # Clusters
        b_clust = {int(k): v["num_clusters"] for k, v in b_div.items()}
        t_clust = {int(k): v["num_clusters"] for k, v in t_div.items()}
        axes[1].bar(x - w/2, [b_clust.get(p, 0) for p in all_phases], w,
                     label="B1-static", color="#1976D2")
        axes[1].bar(x + w/2, [t_clust.get(p, 0) for p in all_phases], w,
                     label="B2-changing", color="#E64A19")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=30, ha="right")
        axes[1].set_title("# Route Clusters")
        axes[1].legend(fontsize=8)

        # Entropy
        b_ent = {int(k): v["entropy"] for k, v in b_div.items()}
        t_ent = {int(k): v["entropy"] for k, v in t_div.items()}
        axes[2].bar(x - w/2, [b_ent.get(p, 0) for p in all_phases], w,
                     label="B1-static", color="#1976D2")
        axes[2].bar(x + w/2, [t_ent.get(p, 0) for p in all_phases], w,
                     label="B2-changing", color="#E64A19")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, rotation=30, ha="right")
        axes[2].set_title("Route Distribution Entropy")
        axes[2].legend(fontsize=8)

        fig.suptitle("M2: Path Diversity — Static vs Changing",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        savepath = os.path.join(output_dir, "m2_path_diversity.png")
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Study B comparisons saved to {output_dir}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare isolation study results")
    parser.add_argument("--study", type=str, default="both",
                        choices=["a", "b", "both"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_base", type=str, default=None)
    args = parser.parse_args()

    base = args.results_base or os.path.join(str(SCRIPT_DIR), "results")

    if args.study in ("a", "both"):
        a_dir = os.path.join(base, f"study_a_seed_{args.seed}")
        a_out = os.path.join(a_dir, "comparisons")
        if os.path.exists(a_dir):
            compare_study_a(a_dir, a_out)
        else:
            print(f"Study A results not found at {a_dir}")

    if args.study in ("b", "both"):
        b_dir = os.path.join(base, f"study_b_seed_{args.seed}")
        b_out = os.path.join(b_dir, "comparisons")
        if os.path.exists(b_dir):
            compare_study_b(b_dir, b_out)
        else:
            print(f"Study B results not found at {b_dir}")


if __name__ == "__main__":
    main()
