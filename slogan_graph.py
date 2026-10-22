from collections import Counter
from pathlib import Path

import networkx as nx

from preprocess import load_slogans


def build_graph(records: list[dict]) -> nx.DiGraph:
    """Build weighted word edges from the sentence lists in processed records."""
    graph = nx.DiGraph()

    for record in records:
        for words in record["sentences"]:
            if not words:
                continue

            for word in words:
                if word not in graph:
                    graph.add_node(word, frequency=0, start_count=0, end_count=0)
                graph.nodes[word]["frequency"] += 1

            graph.nodes[words[0]]["start_count"] += 1
            graph.nodes[words[-1]]["end_count"] += 1

            # Adjacent pairs only: [a, b, c] gives a -> b and b -> c.
            for source, target in zip(words, words[1:]):
                if graph.has_edge(source, target):
                    graph[source][target]["weight"] += 1
                else:
                    graph.add_edge(source, target, weight=1)

    return graph


def graph_from_csv_files(
    csv_paths: str | Path | list[str | Path],
) -> tuple[nx.DiGraph, list[dict], dict[str, int]]:
    """Normalize/tokenize one or more CSVs and return graph, records, counts."""
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]

    records = []
    company_counts = Counter()

    for csv_path in csv_paths:
        file_records, file_counts = load_slogans(csv_path)
        for record in file_records:
            record["source_file"] = str(Path(csv_path).resolve())
        records.extend(file_records)
        company_counts.update(file_counts)

    return build_graph(records), records, dict(company_counts)
