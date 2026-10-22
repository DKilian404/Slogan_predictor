import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / "processed" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def save_statistics(graph, records, company_counts, output):
    """Save the basic statistics as a Markdown table."""
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()

    no_loops = graph.copy()
    no_loops.remove_edges_from(nx.selfloop_edges(no_loops))
    undirected = no_loops.to_undirected()
    
    weak = list(nx.weakly_connected_components(graph))
    strong = list(nx.strongly_connected_components(graph))
    stats = {
        "source_records": len(records),
        "usable_slogans": sum(r["usable"] for r in records),
        "distinct_company_labels": sum(bool(c) for c in company_counts),
        "nodes": nodes,
        "directed_edges": edges,
        "total_edge_weight": int(graph.size(weight="weight")),
        "self_loops": nx.number_of_selfloops(graph),
        "density_excluding_loops": nx.density(no_loops),
        "average_in_degree": edges / nodes if nodes else 0,
        "average_out_degree": edges / nodes if nodes else 0,
        "average_in_strength": graph.size(weight="weight") / nodes if nodes else 0,
        "average_out_strength": graph.size(weight="weight") / nodes if nodes else 0,
        "average_undirected_clustering": nx.average_clustering(undirected) if nodes else 0,
        "undirected_transitivity": nx.transitivity(undirected),
        "weak_components": len(weak),
        "largest_weak_component": max(map(len, weak), default=0),
        "strong_components": len(strong),
        "largest_strong_component": max(map(len, strong), default=0),
        "isolated_words": nx.number_of_isolates(graph),
    }
    table = "# Basic network statistics\n\n| Statistic | Value |\n| --- | ---: |\n"
    for name, value in stats.items():
        displayed = f"{value:.6f}" if isinstance(value, float) else f"{value:,}"
        table += f"| {name.replace('_', ' ')} | {displayed} |\n"
    table += "\nClustering and transitivity use the unweighted undirected graph without self-loops.\n"
    (output / "statistics.md").write_text(table, encoding="utf-8")


def make_plots(graph, records, company_counts, output):
    """Save a data overview and a small network illustration as PNG files."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), layout="constrained")
    fig.suptitle("Advertising slogans: data and word network", fontsize=20)

    lengths = [sum(map(len, r["sentences"])) for r in records if r["usable"]]
    axes[0, 0].hist(lengths, bins=[i - 0.5 for i in range(1, max(lengths, default=1) + 2)], color="#287d8e")
    axes[0, 0].set(title="Slogan lengths", xlabel="Words per original slogan", ylabel="Slogans")

    common = sorted(graph.nodes(data="frequency"), key=lambda pair: pair[1], reverse=True)[:12]
    axes[0, 1].barh([w for w, _ in common][::-1], [c for _, c in common][::-1], color="#287d8e")
    axes[0, 1].set(title="Most frequent words", xlabel="Occurrences")

    companies = Counter(count for name, count in company_counts.items() if name)
    axes[0, 2].bar(list(companies), list(companies.values()), color="#287d8e")
    axes[0, 2].set(title="Slogans per company label", xlabel="Source records per label", ylabel="Company labels", yscale="log")

    for degree, label in [(graph.in_degree(), "In-degree"), (graph.out_degree(), "Out-degree")]:
        counts = sorted(Counter(value for _, value in degree).items())
        axes[1, 0].scatter([k for k, _ in counts], [v for _, v in counts], s=18, alpha=0.7, label=label)
    axes[1, 0].set_xscale("symlog", linthresh=1)
    axes[1, 0].set(title="Degree distributions", xlabel="Distinct edges per word", ylabel="Words", yscale="log")
    axes[1, 0].legend()

    weights = Counter(data["weight"] for _, _, data in graph.edges(data=True))
    axes[1, 1].scatter(list(weights), list(weights.values()), s=20, color="#287d8e")
    axes[1, 1].set(title="Edge-weight distribution", xlabel="Transition occurrences", ylabel="Edges", xscale="log", yscale="log")

    boundary_words = sorted(graph, key=lambda w: graph.nodes[w]["start_count"] + graph.nodes[w]["end_count"], reverse=True)[:10]
    positions = list(range(len(boundary_words)))
    axes[1, 2].barh([p - 0.2 for p in positions], [graph.nodes[w]["start_count"] for w in boundary_words], height=0.4, label="Start")
    axes[1, 2].barh([p + 0.2 for p in positions], [graph.nodes[w]["end_count"] for w in boundary_words], height=0.4, label="End")
    axes[1, 2].set_yticks(positions, boundary_words)
    axes[1, 2].invert_yaxis()
    axes[1, 2].set(title="Common boundary words", xlabel="Sentence occurrences")
    axes[1, 2].legend()
    fig.savefig(output / "overview.png", dpi=160)
    plt.close(fig)

    top_edges = sorted(graph.edges(data=True), key=lambda edge: edge[2]["weight"], reverse=True)[:18]
    sample = graph.edge_subgraph([(a, b) for a, b, _ in top_edges])
    fig, ax = plt.subplots(figsize=(13, 9), layout="constrained")
    positions = nx.spring_layout(sample, seed=42, weight=None, k=1.4, iterations=150)
    sizes = [600 + 0.6 * graph.nodes[w]["frequency"] for w in sample]
    nx.draw_networkx(sample, pos=positions, ax=ax, node_size=sizes,
                     node_color="#c4e5e9", edge_color="#607d8b", arrowsize=20,
                     font_size=11, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_edge_labels(sample, pos=positions, ax=ax,
                                edge_labels=nx.get_edge_attributes(sample, "weight"), font_size=9,
                                connectionstyle="arc3,rad=0.08")
    ax.set_title("18 strongest word transitions (selected edges only)", fontsize=17)
    ax.margins(0.18)
    ax.axis("off")
    fig.savefig(output / "network_example.png", dpi=160)
    plt.close(fig)
