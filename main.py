from pathlib import Path

import networkx as nx

from preprocess import save_company_counts
from slogan_graph import graph_from_csv_files
from analysis import save_statistics, make_plots


def main() -> None:
    folder = Path(__file__).parent
    files = sorted((folder / "data").glob("*.csv"))
    output = folder / "processed"
    if not files:
        raise FileNotFoundError("Put your CSV files in the data folder first.")

    # This is the reusable call: pass one CSV path or a list of paths.
    graph, records, company_counts = graph_from_csv_files(files)

    save_company_counts(company_counts, output)
    # GraphML preserves direction, edge weights, and all node attributes.
    nx.write_graphml(graph, output / "slogan_graph.graphml")
    save_statistics(graph, records, company_counts, output)
    make_plots(graph, records, company_counts, output)

    print(f"CSV files: {len(files)}")
    print(f"Records: {len(records)}; usable: {sum(r['usable'] for r in records)}")
    print(f"Word nodes: {graph.number_of_nodes()}")
    print(f"Unique directed edges: {graph.number_of_edges()}")
    print(f"Total transition count: {int(graph.size(weight='weight'))}")
    print(f"Saved results to: {output.resolve()}")


if __name__ == "__main__":
    main()
