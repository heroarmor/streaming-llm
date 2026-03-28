#!/usr/bin/env python3
"""
Analyze avg_kept_mass statistics from Quest attention task results.

This script reads prediction JSONL files and extracts statistics about:
- avg_kept_mass: Attention probability mass of kept tokens
- avg_real_budget: Actual number of tokens kept per head
- avg_real_sparsity: Fraction of tokens pruned

Usage:
    python analyze_kept_mass.py [pred_dir]
"""

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def analyze_kept_mass(pred_dir: Path) -> dict:
    """Analyze kept mass statistics from prediction files.

    Args:
        pred_dir: Directory containing prediction JSONL files

    Returns:
        Dictionary with statistics per task
    """
    # Get all jsonl files (excluding summary files)
    jsonl_files = [f for f in pred_dir.glob("*.jsonl") if f.name not in ["submission.jsonl"]]

    results = {}

    for jsonl_file in sorted(jsonl_files):
        task_name = jsonl_file.stem
        kept_masses = []
        budgets = []
        sparsities = []

        with open(jsonl_file) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "avg_kept_mass" in data:
                        kept_masses.append(data["avg_kept_mass"])
                    if "avg_real_budget" in data:
                        budgets.append(data["avg_real_budget"])
                    if "avg_real_sparsity" in data:
                        sparsities.append(data["avg_real_sparsity"])
                except json.JSONDecodeError:
                    continue

        if kept_masses:
            results[task_name] = {
                "num_samples": len(kept_masses),
                "avg_kept_mass": float(np.mean(kept_masses)),
                "std_kept_mass": float(np.std(kept_masses)),
                "min_kept_mass": float(np.min(kept_masses)),
                "max_kept_mass": float(np.max(kept_masses)),
                "avg_budget": float(np.mean(budgets)) if budgets else 0.0,
                "avg_sparsity": float(np.mean(sparsities)) if sparsities else 0.0,
            }

    return results


def print_analysis(
    results: dict, target_mass: float = 0.70, budget_ratio: float = 0.25, output_file: Path | None = None
) -> str:
    """Print formatted analysis of kept mass statistics.

    Args:
        results: Dictionary with statistics per task
        target_mass: Target coverage (default: 0.70)
        budget_ratio: Budget ratio (default: 0.25)
        output_file: Optional path to save text output

    Returns:
        String containing the formatted analysis
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Quest Attention Statistics Analysis")
    lines.append("=" * 80)
    lines.append(f"\nTarget: {target_mass * 100:.0f}% coverage ({target_mass:.2f} kept mass)")
    lines.append(f"Budget ratio: {budget_ratio:.2f} ({budget_ratio * 100:.0f}% of tokens)\n")

    lines.append(
        f"{'Task':<20} {'Samples':<10} {'Avg Mass':<12} {'Std':<10} "
        f"{'Min':<10} {'Max':<10} {'Budget':<10} {'Sparsity':<10}"
    )
    lines.append("-" * 100)

    for task_name in sorted(results.keys()):
        r = results[task_name]
        lines.append(
            f"{task_name:<20} {r['num_samples']:<10} {r['avg_kept_mass']:<12.4f} "
            f"{r['std_kept_mass']:<10.4f} {r['min_kept_mass']:<10.4f} {r['max_kept_mass']:<10.4f} "
            f"{r['avg_budget']:<10.1f} {r['avg_sparsity']:<10.4f}"
        )

    # Overall statistics
    all_masses = [r["avg_kept_mass"] for r in results.values()]
    all_budgets = [r["avg_budget"] for r in results.values()]
    all_sparsities = [r["avg_sparsity"] for r in results.values()]

    lines.append("-" * 100)
    lines.append(
        f"{'OVERALL':<20} {sum(r['num_samples'] for r in results.values()):<10} "
        f"{np.mean(all_masses):<12.4f} {np.std(all_masses):<10.4f} "
        f"{np.min(all_masses):<10.4f} {np.max(all_masses):<10.4f} "
        f"{np.mean(all_budgets):<10.1f} {np.mean(all_sparsities):<10.4f}"
    )

    # Analysis
    lines.append("\n" + "=" * 80)
    lines.append("Analysis:")
    lines.append("=" * 80)

    avg_mass = np.mean(all_masses)

    lines.append(f"\n1. Average kept mass across all tasks: {avg_mass:.4f}")
    lines.append(f"   Target coverage: {target_mass:.4f}")
    lines.append(f"   Difference: {avg_mass - target_mass:+.4f} ({(avg_mass - target_mass) / target_mass * 100:+.2f}%)")

    lines.append(f"\n2. Average budget: {np.mean(all_budgets):.1f} tokens")
    lines.append(f"   Average sparsity: {np.mean(all_sparsities):.4f} ({np.mean(all_sparsities) * 100:.2f}%)")

    tasks_below_target = sum(1 for m in all_masses if m < target_mass)
    lines.append(f"\n3. Tasks below target ({target_mass}): {tasks_below_target}/{len(all_masses)}")

    if tasks_below_target > 0:
        lines.append("\n   Tasks below target:")
        for task_name in sorted(results.keys()):
            if results[task_name]["avg_kept_mass"] < target_mass:
                mass = results[task_name]["avg_kept_mass"]
                lines.append(f"   - {task_name}: {mass:.4f} (deficit: {target_mass - mass:.4f})")

    lines.append("\n4. Variance across tasks:")
    lines.append(f"   Std deviation: {np.std(all_masses):.4f}")
    lines.append(f"   Range: [{np.min(all_masses):.4f}, {np.max(all_masses):.4f}]")

    output = "\n".join(lines)

    # Print to console
    print(output)

    # Save to file if specified
    if output_file:
        with open(output_file, "w") as f:
            f.write(output)
        print(f"\n\nText analysis saved to: {output_file}")

    return output


def plot_results(results: dict, pred_dir: Path, target_mass: float = 0.70) -> None:
    """Create visualizations of the kept mass analysis.

    Args:
        results: Dictionary with statistics per task
        pred_dir: Directory to save plots
        target_mass: Target coverage (default: 0.70)
    """
    # Set up the plot style
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Quest Attention Kept Mass Analysis", fontsize=16, fontweight="bold")

    # Sort tasks by kept mass
    sorted_tasks = sorted(results.items(), key=lambda x: x[1]["avg_kept_mass"], reverse=True)
    task_names = [t[0] for t in sorted_tasks]

    # Shorten task names for better display
    display_names = [name.replace("niah_", "").replace("_", " ") for name in task_names]

    # Extract data
    avg_masses = [results[t]["avg_kept_mass"] for t in task_names]
    std_masses = [results[t]["std_kept_mass"] for t in task_names]
    min_masses = [results[t]["min_kept_mass"] for t in task_names]
    max_masses = [results[t]["max_kept_mass"] for t in task_names]
    budgets = [results[t]["avg_budget"] for t in task_names]
    # sparsities = [results[t]["avg_sparsity"] for t in task_names]

    # 1. Bar plot of average kept mass by task
    ax1 = axes[0, 0]
    bars = ax1.barh(display_names, avg_masses, color="steelblue", alpha=0.8)
    ax1.axvline(target_mass, color="red", linestyle="--", linewidth=2, label=f"Target ({target_mass:.2f})")
    ax1.axvline(
        np.mean(avg_masses), color="green", linestyle="--", linewidth=2, label=f"Mean ({np.mean(avg_masses):.4f})"
    )
    ax1.set_xlabel("Average Kept Mass", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Task", fontsize=11, fontweight="bold")
    ax1.set_title("Average Kept Mass by Task", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.set_xlim([0.7, 1.0])
    ax1.grid(axis="x", alpha=0.3)

    # Add value labels on bars
    for _i, (bar, val) in enumerate(zip(bars, avg_masses, strict=True)):
        ax1.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=9)

    # 2. Error bars showing variance
    ax2 = axes[0, 1]
    x_pos = np.arange(len(task_names))
    ax2.errorbar(
        x_pos,
        avg_masses,
        yerr=std_masses,
        fmt="o",
        markersize=8,
        capsize=5,
        capthick=2,
        color="steelblue",
        ecolor="gray",
        alpha=0.8,
    )
    ax2.axhline(target_mass, color="red", linestyle="--", linewidth=2, label=f"Target ({target_mass:.2f})")
    ax2.axhline(
        np.mean(avg_masses), color="green", linestyle="--", linewidth=2, label=f"Mean ({np.mean(avg_masses):.4f})"
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(display_names, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Kept Mass", fontsize=11, fontweight="bold")
    ax2.set_title("Kept Mass with Standard Deviation", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.set_ylim([0.7, 1.0])
    ax2.grid(alpha=0.3)

    # 3. Min-Max range visualization
    ax3 = axes[1, 0]
    for i, (_task, avg, min_m, max_m) in enumerate(zip(display_names, avg_masses, min_masses, max_masses, strict=True)):
        ax3.plot([min_m, max_m], [i, i], "o-", linewidth=2, markersize=6, color="steelblue", alpha=0.6)
        ax3.plot(avg, i, "o", markersize=10, color="darkblue", alpha=0.9)
    ax3.axvline(target_mass, color="red", linestyle="--", linewidth=2, label=f"Target ({target_mass:.2f})")
    ax3.set_yticks(range(len(display_names)))
    ax3.set_yticklabels(display_names, fontsize=9)
    ax3.set_xlabel("Kept Mass", fontsize=11, fontweight="bold")
    ax3.set_title("Min-Avg-Max Range by Task", fontsize=12, fontweight="bold")
    ax3.legend(loc="lower right")
    ax3.set_xlim([0.7, 1.0])
    ax3.grid(axis="x", alpha=0.3)

    # 4. Scatter: Kept Mass vs Budget
    ax4 = axes[1, 1]
    colors = ["green" if m >= target_mass else "red" for m in avg_masses]
    ax4.scatter(budgets, avg_masses, c=colors, s=150, alpha=0.7, edgecolors="black", linewidth=1.5)

    # Add task labels
    for _i, (name, x, y) in enumerate(zip(display_names, budgets, avg_masses, strict=True)):
        ax4.annotate(name, (x, y), fontsize=8, ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")

    ax4.axhline(target_mass, color="red", linestyle="--", linewidth=2, label=f"Target Mass ({target_mass:.2f})")
    ax4.axhline(
        np.mean(avg_masses), color="blue", linestyle=":", linewidth=2, label=f"Mean Mass ({np.mean(avg_masses):.4f})"
    )
    ax4.axvline(
        np.mean(budgets), color="purple", linestyle=":", linewidth=2, label=f"Mean Budget ({np.mean(budgets):.1f})"
    )
    ax4.set_xlabel("Average Budget (tokens)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Average Kept Mass", fontsize=11, fontweight="bold")
    ax4.set_title("Kept Mass vs Budget Usage", fontsize=12, fontweight="bold")
    ax4.legend(loc="lower right", fontsize=9)
    ax4.grid(alpha=0.3)

    # Create custom legend for scatter colors
    green_patch = mpatches.Patch(color="green", label="Above target", alpha=0.7)
    red_patch = mpatches.Patch(color="red", label="Below target", alpha=0.7)
    ax4.legend(handles=[green_patch, red_patch], loc="upper left", fontsize=9)

    plt.tight_layout()

    # Save figure
    plot_file = pred_dir / "kept_mass_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {plot_file}")

    plt.close()


def save_results(results: dict, pred_dir: Path, target_mass: float = 0.70, budget_ratio: float = 0.25) -> None:
    """Save analysis results to JSON and text files.

    Args:
        results: Dictionary with statistics per task
        pred_dir: Directory containing prediction files
        target_mass: Target coverage (default: 0.70)
        budget_ratio: Budget ratio (default: 0.25)
    """
    # Calculate overall statistics
    all_masses = [r["avg_kept_mass"] for r in results.values()]
    all_budgets = [r["avg_budget"] for r in results.values()]
    all_sparsities = [r["avg_sparsity"] for r in results.values()]

    tasks_below_target = sum(1 for m in all_masses if m < target_mass)

    # Create comprehensive results dictionary
    output_data = {
        "config": {"target_mass": target_mass, "budget_ratio": budget_ratio, "pred_dir": str(pred_dir)},
        "overall": {
            "avg_kept_mass": float(np.mean(all_masses)),
            "std_kept_mass": float(np.std(all_masses)),
            "min_kept_mass": float(np.min(all_masses)),
            "max_kept_mass": float(np.max(all_masses)),
            "avg_budget": float(np.mean(all_budgets)),
            "avg_sparsity": float(np.mean(all_sparsities)),
            "total_samples": sum(r["num_samples"] for r in results.values()),
            "num_tasks": len(results),
            "tasks_below_target": tasks_below_target,
        },
        "per_task": results,
    }

    # Save JSON
    json_file = pred_dir / "kept_mass_analysis.json"
    with open(json_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"JSON results saved to: {json_file}")

    # Save text analysis
    text_file = pred_dir / "kept_mass_analysis.txt"
    print_analysis(results, target_mass, budget_ratio, text_file)

    # Generate plots
    plot_results(results, pred_dir, target_mass)


def main() -> None:
    """Main entry point."""
    if len(sys.argv) > 1:
        pred_dir = Path(sys.argv[1])
    else:
        # Default directory
        pred_dir = Path(
            "benchmark/ruler/ruler_eval_result/meta-llama/Llama-3.1-8B-Instruct/synthetic/16384/Quest-25/pred/"
        )

    if not pred_dir.exists():
        print(f"Error: Directory {pred_dir} does not exist")
        sys.exit(1)

    results = analyze_kept_mass(pred_dir)

    if not results:
        print("No results found in the specified directory")
        sys.exit(1)

    save_results(results, pred_dir)


if __name__ == "__main__":
    main()
