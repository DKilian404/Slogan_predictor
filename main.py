from pathlib import Path
from collections import Counter

import networkx as nx

from preprocess import save_company_counts
from generation import (SloganDataset, FirstOrderGenerator, SecondOrderGenerator,
                        SecondOrderFallbackGenerator, ContextualEndingGenerator, PredictedLinkGenerator,
                        save_length_statistics)
from evaluation import (GenerationExperiment, save_order_comparison, save_two_word_failures,
                        save_mixed_second_order_review, save_ending_coverage, save_all_model_review)
from analysis import save_statistics, make_plots, save_centrality_analysis
from link_prediction import LinkPredictionExperiment


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
    save_centrality_analysis(graph, training_output)
    LinkPredictionExperiment(training_records).run(output / "link_prediction")
    experiment = GenerationExperiment(model, company_labels=dataset.company_counts)
    first_results = experiment.run(output / "evaluation")
    second_model = SecondOrderGenerator(training_records, seed=42)
    second_experiment = GenerationExperiment(second_model, company_labels=dataset.company_counts)
    second_results = second_experiment.run(output / "evaluation_second_order")
    save_order_comparison(output / "order_comparison.md", first_results, second_results)
    save_two_word_failures(output / "evaluation_second_order" / "two_word_failures.md",
                          second_model, second_results[0])
    fallback_model = SecondOrderFallbackGenerator(training_records, seed=42)
    fallback_experiment = GenerationExperiment(fallback_model, company_labels=dataset.company_counts)
    fallback_results = fallback_experiment.run(output / "evaluation_second_order_fallback")
    save_order_comparison(output / "fallback_comparison.md", second_results, fallback_results, fallback=True)
    save_mixed_second_order_review(output / "evaluation_second_order_mixed",
                                   second_results[0], fallback_results[0], dataset.company_counts)
    contextual_model = ContextualEndingGenerator(training_records, seed=42)
    contextual_experiment = GenerationExperiment(contextual_model, company_labels=dataset.company_counts)
    contextual_output = output / "evaluation_contextual_endings"
    contextual_results = contextual_experiment.run(contextual_output)
    save_order_comparison(output / "contextual_ending_comparison.md", fallback_results,
                          contextual_results, contextual=True)
    save_ending_coverage(contextual_output / "ending_coverage.md", contextual_model,
                         fallback_results[0], contextual_results[0])
    predicted_model = PredictedLinkGenerator(training_records, seed=42)
    predicted_experiment = GenerationExperiment(predicted_model, company_labels=dataset.company_counts)
    predicted_results = predicted_experiment.run(output / "evaluation_predicted_links")
    save_order_comparison(output / "predicted_link_comparison.md", fallback_results,
                          predicted_results, predicted=True)
    save_all_model_review(output)

    print(f"CSV files: {len(files)}")
    print(f"Training records: {len(training_records)}; test records: {len(test_records)}")
    print(f"Word nodes: {graph.number_of_nodes()}")
    print(f"Unique directed edges: {graph.number_of_edges()}")
    print(f"Total transition count: {int(graph.size(weight='weight'))}")
    print(f"Saved results to: {output.resolve()}")


if __name__ == "__main__":
    main()
