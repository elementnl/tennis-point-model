"""
Analyze a specific match by searching for it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from src.importance import load_model, analyze_match
from scripts.visualize import plot_match_timeline


def find_match(matches: list[dict], search_term: str) -> list[dict]:
    """Find matches containing search term in match_id or player names."""
    results = []
    search_lower = search_term.lower()

    for m in matches:
        if (
            search_lower in m["match_id"].lower()
            or search_lower in m["player1"].lower()
            or search_lower in m["player2"].lower()
        ):
            results.append(m)

    return results


def main():
    # Load data
    with open("data/processed/matches.json") as f:
        matches = json.load(f)

    # Search for Wimbledon 2019 final
    search = "wimbledon-f"
    results = find_match(matches, search)

    print(f"Found {len(results)} matches matching '{search}':\n")
    for i, m in enumerate(results[:20]):  # Show first 20
        print(f"{i:3d}. {m['match_id'][:50]}...")
        print(f"     {m['player1']} vs {m['player2']}")
        print()

    # Look specifically for 2019
    results_2019 = [m for m in results if "2019" in m["match_id"]]

    if results_2019:
        print(f"\n{'='*60}")
        print("2019 Wimbledon Finals:")
        print("=" * 60)
        for m in results_2019:
            print(f"{m['player1']} vs {m['player2']}")
            print(f"Match ID: {m['match_id']}")

            # Analyze it
            print("\nAnalyzing...")
            model = load_model()
            analysis = analyze_match(model, m)

            print(f"Total points: {analysis['n_points']}")
            print(
                f"Decision point: {analysis['decision_point']} / {analysis['n_points']}"
            )
            print(f"Top 5 important points: {analysis['top_important_points'][:5]}")
            print(f"Final win prob (Player 1): {analysis['final_prob']:.3f}")
            print(
                f"Winner: Player {analysis['winner']} ({m['player1'] if analysis['winner'] == 1 else m['player2']})"
            )

            # Generate visualization
            print("\nGenerating visualization...")
            plot_match_timeline(analysis)
            print("Done! Check figures/ directory.")
    else:
        print("No 2019 Wimbledon final found in dataset.")
        print("\nAvailable Wimbledon finals:")
        for m in results:
            print(f"  {m['match_id'][:30]}... - {m['player1']} vs {m['player2']}")


if __name__ == "__main__":
    main()
