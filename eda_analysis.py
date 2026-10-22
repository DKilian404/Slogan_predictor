"""Exploratory analysis of a slogan word co-occurrence network."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean, median, stdev
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

START = "<START>"
END = "<END>"
TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def find_text_column(data: pd.DataFrame) -> str:
    """Return the slogan/text column, preferring conventional column names."""
    preferred = ("slogan", "text")
    columns_by_lower_name = {str(column).lower(): column for column in data.columns}
    for name in preferred:
        if name in columns_by_lower_name:
            return columns_by_lower_name[name]

    string_columns = data.select_dtypes(include="object").columns
    if len(string_columns) == 0:
        raise ValueError("The CSV does not contain a text column.")
    return string_columns[-1]


def tokenize(text: str) -> list[str]:
    """Lowercase text and keep basic word characters as tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def load_slogans(csv_path: Path) -> list[str]:
    """Load, clean, and deduplicate non-empty slogans from a CSV file."""
    data = pd.read_csv(csv_path)
    column = find_text_column(data)
    slogans = (
        data[column]
        .dropna()
        .astype(str)
        .str.strip()
    )
    slogans = slogans[slogans.ne("")].drop_duplicates()
    tokenized = [tokenize(slogan) for slogan in slogans]
    return [" ".join(tokens) for tokens in tokenized if tokens]


def build_network(slogans: Iterable[str]) -> nx.DiGraph:
    """Build a weighted directed transition network from normalized slogans."""
    graph = nx.DiGraph()
    for slogan in slogans:
        tokens = [START, *slogan.split(), END]
        for source, target in zip(tokens, tokens[1:]):
            if graph.has_edge(source, target):
                graph[source][target]["weight"] += 1
            else:
                graph.add_edge(source, target, weight=1)
    return graph


def summary(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
    return {
        "mean": mean(values),
        "median": median(values),
        "max": max(values),
        "min": min(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def print_summary(graph: nx.DiGraph, slogan_count: int) -> None:
    """Print the requested network statistics."""
    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    in_centrality = nx.in_degree_centrality(graph)
    out_centrality = nx.out_degree_centrality(graph)
    undirected = graph.to_undirected()
    components = list(nx.weakly_connected_components(graph))

    print(f"Unique slogans: {slogan_count}")
    print(f"Nodes (N): {graph.number_of_nodes()}")
    print(f"Directed edges (E): {graph.number_of_edges()}")
    print(f"Network density: {nx.density(graph):.6f}")
    print(f"In-degree summary: {summary(in_degrees.values())}")
    print(f"Out-degree summary: {summary(out_degrees.values())}")
    print("\nTop 10 nodes by in-degree centrality:")
    print(sorted(in_centrality.items(), key=lambda item: item[1], reverse=True)[:10])
    print("\nTop 10 nodes by out-degree centrality:")
    print(sorted(out_centrality.items(), key=lambda item: item[1], reverse=True)[:10])
    print("\nTop 10 outgoing words from <START>:")
    print(sorted(graph[START].items(), key=lambda item: item[1]["weight"], reverse=True)[:10])
    print("\nTop 10 incoming words to <END>:")
    incoming = ((node, graph[node][END]) for node in graph.predecessors(END))
    print(sorted(incoming, key=lambda item: item[1]["weight"], reverse=True)[:10])
    print(f"\nAverage clustering coefficient (undirected): "
          f"{nx.average_clustering(undirected, weight='weight'):.6f}")
    largest_component = max((len(component) for component in components), default=0)
    print(f"Weakly connected components: {len(components)}")
    print(f"Largest weakly connected component: {largest_component} nodes")


def save_degree_plot(graph: nx.DiGraph, output_path: Path) -> None:
    """Save a log-log scatter plot of in- and out-degree distributions."""
    in_counts = pd.Series(dict(graph.in_degree())).value_counts().sort_index()
    out_counts = pd.Series(dict(graph.out_degree())).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(in_counts.index, in_counts.values, "o-", label="In-degree")
    ax.loglog(out_counts.index, out_counts.values, "s-", label="Out-degree")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of nodes")
    ax.set_title("In-degree and out-degree distributions")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_transition_plot(graph: nx.DiGraph, output_path: Path) -> None:
    """Save a horizontal bar chart of the 15 most frequent transitions."""
    transitions = sorted(
        graph.edges(data="weight"),
        key=lambda edge: edge[2],
        reverse=True,
    )[:15]
    labels = [f"{source} -> {target}" for source, target, _ in transitions][::-1]
    weights = [weight for _, _, weight in transitions][::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(labels, weights)
    ax.set_xlabel("Transition frequency")
    ax.set_title("Top 15 word transitions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, nargs="?", default=Path("slogans.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("eda_outputs"),
        help="Directory for generated PNG plots (default: eda_outputs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slogans = load_slogans(args.csv_path)
    graph = build_network(slogans)
    print_summary(graph, len(slogans))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_degree_plot(graph, args.output_dir / "degree_distribution.png")
    save_transition_plot(graph, args.output_dir / "top_transitions.png")
    print(f"\nPlots saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
