from pathlib import Path
from collections import Counter

import networkx as nx

from preprocess import save_company_counts
from generation import SloganDataset, FirstOrderGenerator, record_outcomes, save_length_statistics
from analysis import save_statistics, make_plots


def main() -> None:
    folder = Path(__file__).parent
    files = sorted((folder / "data").glob("*.csv"))
    output = folder / "processed"
    if not files:
        raise FileNotFoundError("Put your CSV files in the data folder first.")

    dataset = SloganDataset(files)
    training_records, test_records = dataset.split(train_fraction=0.8, seed=42)
    model = FirstOrderGenerator(training_records, seed=42)
    graph = model.graph

    save_company_counts(dataset.company_counts, output)
    dataset.save_split(output / "split.csv")
    save_length_statistics(dataset, output / "length_statistics.txt")
    # GraphML preserves direction, edge weights, and all node attributes.
    nx.write_graphml(graph, output / "training_graph.graphml")
    # Keep training analysis separate from the earlier full-dataset analysis.
    training_output = output / "training"
    training_output.mkdir(exist_ok=True)
    training_companies = Counter(record["company"] for record in training_records)
    save_statistics(graph, training_records, training_companies, training_output)
    make_plots(graph, training_records, training_companies, training_output)
    record_outcomes(model, dataset, output / "generation_outcomes.txt")

    print(f"CSV files: {len(files)}")
    print(f"Training records: {len(training_records)}; test records: {len(test_records)}")
    print(f"Word nodes: {graph.number_of_nodes()}")
    print(f"Unique directed edges: {graph.number_of_edges()}")
    print(f"Total transition count: {int(graph.size(weight='weight'))}")
    print(f"Saved results to: {output.resolve()}")


if __name__ == "__main__":
    main()
