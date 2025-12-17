"""
Create all visualizations for Reddit post.
Uses both match-level model (win probability) and game-level analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Set style
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


def plot_match_timeline(match_analysis: dict, save: bool = True):
    """
    Win probability timeline for a single match.
    Uses the 90-minute match-level model.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), height_ratios=[2, 1], sharex=True
    )

    probs = match_analysis["win_probability"]
    importance = match_analysis["importance"]
    n_points = len(probs)
    x = range(n_points)

    # Win probability
    ax1.plot(x, probs, "b-", linewidth=1.5)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax1.fill_between(
        x,
        0.5,
        probs,
        where=[p > 0.5 for p in probs],
        alpha=0.3,
        color="blue",
        label="Player 1 favored",
    )
    ax1.fill_between(
        x,
        0.5,
        probs,
        where=[p <= 0.5 for p in probs],
        alpha=0.3,
        color="red",
        label="Player 2 favored",
    )
    ax1.set_ylabel("P(Player 1 wins)", fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.set_title(
        f"{match_analysis['player1']} vs {match_analysis['player2']}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="upper right")

    dp = match_analysis["decision_point"]
    ax1.axvline(x=dp, color="green", linestyle=":", alpha=0.7)
    ax1.annotate(
        "Decision point",
        xy=(dp, 0.5),
        xytext=(dp + 10, 0.7),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="green"),
    )

    # Point importance
    ax2.bar(x, importance, width=1, alpha=0.7, color="purple")
    ax2.set_ylabel("Point Importance", fontsize=12)
    ax2.set_xlabel("Point Number", fontsize=12)

    for idx in match_analysis["top_important_points"][:5]:
        if idx < len(importance):
            ax2.bar(idx, importance[idx], width=1, color="red", alpha=0.9)

    plt.tight_layout()

    if save:
        safe_name = match_analysis["match_id"][:40].replace("/", "_")
        filename = f"match_{safe_name}.png"
        plt.savefig(FIGURES_DIR / filename, dpi=150)
        print(f"Saved {FIGURES_DIR / filename}")

    plt.close()
    return fig


def plot_clutch_vs_choke():
    """
    Heatmap showing where players beat or miss expectations.
    This is the GOD TIER visualization.
    """
    with open("data/processed/actual_vs_theoretical.json") as f:
        comparison = json.load(f)

    # Build heatmap of differences
    heatmap = np.zeros((4, 4))

    for key, data in comparison.items():
        pts = eval(key)
        if pts[0] <= 3 and pts[1] <= 3:
            heatmap[pts[0], pts[1]] = data["diff"] * 100  # Convert to percentage points

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use diverging colormap (red = choke, green = clutch)
    im = ax.imshow(heatmap, cmap="RdYlGn", aspect="equal", vmin=-10, vmax=10)

    labels = ["0", "15", "30", "40"]
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel("Receiver's Points", fontsize=14)
    ax.set_ylabel("Server's Points", fontsize=14)
    ax.set_title(
        "Clutch vs Choke: Reality vs Mathematical Expectation",
        fontsize=16,
        fontweight="bold",
    )

    # Add values
    for i in range(4):
        for j in range(4):
            val = heatmap[i, j]
            color = "white" if abs(val) > 5 else "black"
            sign = "+" if val > 0 else ""
            ax.text(
                j,
                i,
                f"{sign}{val:.1f}%",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=color,
            )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Difference from Expected (%)", fontsize=12)

    # Add annotation
    ax.text(
        0.5,
        -0.15,
        "Green = CLUTCH (beat expectations)  |  Red = CHOKE (worse than expected)",
        ha="center",
        transform=ax.transAxes,
        fontsize=11,
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "clutch_vs_choke.png", dpi=150)
    print(f"Saved {FIGURES_DIR / 'clutch_vs_choke.png'}")
    plt.close()


def plot_leverage_chart():
    """Bar chart showing point leverage (swing) at each score."""
    with open("data/processed/point_leverage.json") as f:
        leverage = json.load(f)

    # Sort by swing
    items = [(k, v["swing"]) for k, v in leverage.items()]
    items.sort(key=lambda x: x[1], reverse=True)

    scores = []
    swings = []
    for score_str, swing in items:
        pts = eval(score_str)
        label = f"{['0','15','30','40'][pts[0]]}-{['0','15','30','40'][pts[1]]}"
        scores.append(label)
        swings.append(swing * 100)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [
        "#d62728" if s > 30 else "#ff7f0e" if s > 20 else "#2ca02c" for s in swings
    ]
    bars = ax.bar(scores, swings, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Point Score", fontsize=12)
    ax.set_ylabel("Swing (Percentage Points)", fontsize=12)
    ax.set_title(
        "Point Leverage: How Much Each Point Swings Hold Probability",
        fontsize=14,
        fontweight="bold",
    )

    # Add value labels
    for bar, val in zip(bars, swings):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylim(0, max(swings) + 10)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "point_leverage.png", dpi=150)
    print(f"Saved {FIGURES_DIR / 'point_leverage.png'}")
    plt.close()


def plot_momentum_analysis():
    """Visualization of momentum findings."""
    with open("data/processed/momentum_analysis.json") as f:
        momentum = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ["After Winning\nPrevious Point", "After Losing\nPrevious Point"]
    values = [momentum["p_win_after_win"] * 100, momentum["p_win_after_loss"] * 100]
    colors = ["#2ca02c", "#d62728"]

    bars = ax.bar(
        categories, values, color=colors, edgecolor="black", linewidth=1, width=0.6
    )

    # Add baseline
    baseline = (momentum["p_win_after_win"] + momentum["p_win_after_loss"]) / 2 * 100
    ax.axhline(
        y=baseline,
        color="gray",
        linestyle="--",
        linewidth=2,
        label=f"Average: {baseline:.1f}%",
    )

    ax.set_ylabel("P(Server Wins Point) %", fontsize=12)
    ax.set_title("Is Momentum Real?", fontsize=16, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    # Add effect size annotation
    effect = momentum["momentum_effect"] * 100
    verdict = "MYTH" if abs(effect) < 2 else "REAL"
    ax.text(
        0.5,
        0.85,
        f"Momentum Effect: {effect:+.1f}%\nVerdict: {verdict}",
        ha="center",
        transform=ax.transAxes,
        fontsize=14,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.set_ylim(0, max(values) + 10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "momentum_analysis.png", dpi=150)
    print(f"Saved {FIGURES_DIR / 'momentum_analysis.png'}")
    plt.close()


def plot_first_point_impact():
    """Visualization of first point importance."""
    with open("data/processed/first_point_analysis.json") as f:
        first_pt = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ["Won First Point", "Lost First Point"]
    values = [
        first_pt["p_hold_if_won_first"] * 100,
        first_pt["p_hold_if_lost_first"] * 100,
    ]
    colors = ["#2ca02c", "#d62728"]

    bars = ax.bar(
        categories, values, color=colors, edgecolor="black", linewidth=1, width=0.6
    )

    ax.set_ylabel("P(Server Holds Game) %", fontsize=12)
    ax.set_title(
        "How Much Does the First Point Matter?", fontsize=16, fontweight="bold"
    )

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    impact = first_pt["first_point_impact"] * 100
    ax.text(
        0.5,
        0.85,
        f"First Point Impact: {impact:+.1f}%",
        ha="center",
        transform=ax.transAxes,
        fontsize=14,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "first_point_impact.png", dpi=150)
    print(f"Saved {FIGURES_DIR / 'first_point_impact.png'}")
    plt.close()


def plot_hold_probability_heatmap():
    """Basic hold probability heatmap."""
    with open("data/processed/hold_probability_by_score.json") as f:
        score_probs = json.load(f)

    heatmap = np.zeros((4, 4))
    for key, value in score_probs.items():
        pts = eval(key)
        if pts[0] <= 3 and pts[1] <= 3:
            heatmap[pts[0], pts[1]] = value

    fig, ax = plt.subplots(figsize=(9, 8))

    im = ax.imshow(heatmap, cmap="RdYlGn", aspect="equal", vmin=0.2, vmax=1.0)

    labels = ["0", "15", "30", "40"]
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel("Receiver's Points", fontsize=14)
    ax.set_ylabel("Server's Points", fontsize=14)
    ax.set_title("P(Server Holds Game) by Point Score", fontsize=16, fontweight="bold")

    for i in range(4):
        for j in range(4):
            val = heatmap[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.0%}",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=color,
            )

    plt.colorbar(im, ax=ax, label="Hold Probability", shrink=0.8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hold_probability_heatmap.png", dpi=150)
    print(f"Saved {FIGURES_DIR / 'hold_probability_heatmap.png'}")
    plt.close()


def generate_all_visualizations():
    """Generate all visualizations for the Reddit post."""

    print("=" * 60)
    print("Generating all visualizations...")
    print("=" * 60)

    # Game-level visualizations
    print("\n[1/5] Hold probability heatmap...")
    plot_hold_probability_heatmap()

    print("\n[2/5] Clutch vs Choke heatmap...")
    plot_clutch_vs_choke()

    print("\n[3/5] Point leverage chart...")
    plot_leverage_chart()

    print("\n[4/5] Momentum analysis...")
    plot_momentum_analysis()

    print("\n[5/5] First point impact...")
    plot_first_point_impact()

    # Match-level visualization (uses 90-min model)
    print("\n[Bonus] Generating match timeline for a famous match...")

    try:
        from src.importance import load_model, analyze_match

        with open("data/processed/matches.json") as f:
            matches = json.load(f)

        model = load_model()

        # Find Wimbledon 2019 final
        wimbledon_2019 = [
            m
            for m in matches
            if "2019" in m["match_id"] and "Wimbledon-F" in m["match_id"]
        ]

        if wimbledon_2019:
            match = wimbledon_2019[0]
            print(f"    Analyzing: {match['player1']} vs {match['player2']}")
            analysis = analyze_match(model, match)
            plot_match_timeline(analysis)
        else:
            print("    Wimbledon 2019 final not found, skipping...")

    except Exception as e:
        print(f"    Could not generate match timeline: {e}")

    print("\n" + "=" * 60)
    print(f"All visualizations saved to {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_visualizations()
