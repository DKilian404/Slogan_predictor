import os
import csv
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / "processed" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def weighted_pagerank(graph, alpha=0.85, tolerance=1e-10, max_iterations=1000):
    """Calculate PageRank using transition counts and uniform teleportation."""
    nodes = list(graph)
    if not nodes:
        return {}
    n = len(nodes)
    rank = dict.fromkeys(nodes, 1 / n)
    strength = dict(graph.out_degree(weight="weight"))
    for _ in range(max_iterations):
        dangling = sum(rank[w] for w in nodes if strength[w] == 0)
        updated = dict.fromkeys(nodes, (1 - alpha + alpha * dangling) / n)
        for source, target, data in graph.edges(data=True):
            updated[target] += alpha * rank[source] * data["weight"] / strength[source]
        if sum(abs(updated[w] - rank[w]) for w in nodes) < tolerance:
            return updated
        rank = updated
    raise RuntimeError("PageRank did not converge.")


def save_centrality_analysis(graph, output):
    """Save centrality rankings and strength-distribution plots for the training graph."""
    output.mkdir(parents=True, exist_ok=True)
    paths = graph.copy()
    paths.remove_edges_from(nx.selfloop_edges(paths))
    measures = {
        "in_degree": dict(graph.in_degree()), "out_degree": dict(graph.out_degree()),
        "in_strength": dict(graph.in_degree(weight="weight")),
        "out_strength": dict(graph.out_degree(weight="weight")),
        "in_degree_centrality": nx.in_degree_centrality(paths),
        "out_degree_centrality": nx.out_degree_centrality(paths),
        "pagerank": weighted_pagerank(graph),
        "in_closeness": nx.closeness_centrality(paths),
        "out_closeness": nx.closeness_centrality(paths.reverse(copy=False)),
        "betweenness": nx.betweenness_centrality(paths, k=min(256, len(paths)), seed=42),
    }
    with (output / "centrality.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", *measures])
        writer.writeheader()
        for word in sorted(graph):
            writer.writerow({"word": word, **{name: values[word] for name, values in measures.items()}})
    lines = ["# Training-network centrality", "",
             "Degree counts distinct directed edges; strength sums transition occurrences. "
             "Raw degree and strength include self-loops. Normalized degree, closeness and betweenness exclude loops.", "",
             "PageRank uses transition weights, damping 0.85 and uniform teleportation/dangling redistribution. "
             "Closeness uses directed, unweighted paths with the disconnected-graph correction: inward measures "
             "how easily other words reach a word, outward measures how easily it reaches others. "
             "Betweenness uses directed, unweighted shortest paths, normalized, with up to 256 sampled sources (seed 42). "
             "It is an estimate. Transition counts are not treated as distances.", ""]
    for name in ("in_degree", "out_degree", "in_strength", "out_strength", "pagerank",
                 "in_closeness", "out_closeness", "betweenness"):
        lines += [f"## {name.replace('_', ' ').title()}", "", "| Word | Score |", "| --- | ---: |"]
        for word, value in sorted(measures[name].items(), key=lambda item: (-item[1], item[0]))[:10]:
            lines.append(f"| {word} | {value:.6g} |")
        lines.append("")
    lines += ["These scores describe structural importance, not grammatical quality or suitability as a slogan word.", "",
              "[All node scores](centrality.csv) · [Strength distributions](strength_distributions.png)"]
    (output / "centrality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    distribution = []
    for direction in ("in", "out"):
        values = measures[direction + "_strength"]
        counts = Counter(values.values())
        remaining = len(values)
        xs, ys, ccdf = [], [], []
        for strength, count in sorted(counts.items()):
            xs.append(strength); ys.append(count); ccdf.append(remaining / len(values))
            distribution.append({"direction": direction, "strength": strength, "nodes": count,
                                 "fraction_at_least": remaining / len(values)})
            remaining -= count
        axes[0].scatter(xs, ys, s=18, alpha=0.6, label=direction)
        axes[1].step(xs, ccdf, where="post", label=direction)
        axes[2].scatter(list(measures[direction + "_degree"].values()), list(values.values()),
                        s=10, alpha=0.3, label=direction)
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1)
        ax.legend()
    axes[0].set(title="Strength distribution", xlabel="Strength", ylabel="Words", yscale="log")
    axes[1].set(title="Cumulative tail", xlabel="Strength", ylabel="Fraction with at least this strength", yscale="log")
    axes[2].set(title="Degree versus strength", xlabel="Degree", ylabel="Strength")
    axes[2].set_yscale("symlog", linthresh=1)
    fig.savefig(output / "strength_distributions.png", dpi=160)
    plt.close(fig)
    with (output / "strength_distribution.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["direction", "strength", "nodes", "fraction_at_least"])
        writer.writeheader(); writer.writerows(distribution)


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
