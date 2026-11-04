import math
import random
from collections import defaultdict
from pathlib import Path

import networkx as nx

from analysis import plt
from evaluation import write_csv
from slogan_graph import build_graph


def split_training_records(records, seed=43):
    """Reserve 20% of training records for validation while keeping duplicate slogans together."""
    groups = defaultdict(list)
    for record in records:
        if record["usable"]:
            groups[tuple(tuple(words) for words in record["sentences"])].append(record)
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    target = sum(map(len, groups)) * 0.2
    fit, validation = [], []
    for group in groups:
        if abs(len(validation) + len(group) - target) < abs(len(validation) - target):
            validation.extend(group)
        else:
            fit.extend(group)
    if not fit or not validation:
        raise ValueError("Need enough distinct slogans for a validation split.")
    return fit, validation


def ranking_metrics(labels, scores, k=100):
    """Calculate ranking metrics and curves with threshold-based handling of tied scores."""
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        raise ValueError("Ranking evaluation needs both positive and negative examples.")
    groups = defaultdict(list)
    for label, score in zip(labels, scores):
        groups[score].append(label)
    tp = fp = 0
    auc = ap = 0.0
    roc, pr = [(0.0, 0.0)], [(0.0, 1.0)]
    for score in sorted(groups, reverse=True):
        group = groups[score]
        old_tp, old_fp = tp, fp
        tp += sum(group); fp += len(group) - sum(group)
        auc += (fp - old_fp) / negatives * (tp + old_tp) / (2 * positives)
        ap += (tp - old_tp) / positives * tp / (tp + fp)
        roc.append((fp / negatives, tp / positives))
        pr.append((tp / positives, tp / (tp + fp)))
    # Shuffle before sorting so top-k ties do not favour word spelling or labels.
    indices = list(range(len(labels)))
    random.Random(44).shuffle(indices)
    indices.sort(key=lambda i: scores[i], reverse=True)
    k = min(k, len(labels))
    hits = sum(labels[i] for i in indices[:k])
    return {"roc_auc": auc, "average_precision": ap, "k": k,
            "hits_at_k": hits, "precision_at_k": hits / k,
            "recall_at_k": hits / positives}, roc, pr, indices


class LinkPredictionExperiment:
    """Compare simple directed link scores using a validation split inside training."""

    def __init__(self, training_records, seed=43, negatives_per_positive=10):
        """Build fit and validation graphs without accessing the final test records."""
        if negatives_per_positive < 1:
            raise ValueError("Use at least one negative per positive.")
        self.seed = seed
        self.negatives_per_positive = negatives_per_positive
        self.fit, self.validation = split_training_records(training_records, seed)
        self.graph = build_graph(self.fit)
        self.validation_graph = build_graph(self.validation)
        self.graph.remove_edges_from(list(nx.selfloop_edges(self.graph)))
        self.successors = {w: set(self.graph.successors(w)) for w in self.graph}
        self.predecessors = {w: set(self.graph.predecessors(w)) for w in self.graph}

    def score(self, source, target):
        """Score a directed missing link using degree and two-step-path baselines."""
        bridges = self.successors[source] & self.predecessors[target]
        degrees = [len(self.successors[w]) + len(self.predecessors[w]) for w in sorted(bridges)]
        return {
            "preferential_attachment": len(self.successors[source]) * len(self.predecessors[target]),
            "common_neighbors": len(bridges),
            "adamic_adar": sum(1 / math.log(degree) for degree in degrees),
            "resource_allocation": sum(1 / degree for degree in degrees),
        }

    def make_candidates(self):
        """Combine unseen validation edges with uniformly sampled unobserved directed pairs."""
        nodes = sorted(self.graph)
        observed = set(self.graph.edges())
        validation_edges = {(a, b) for a, b in self.validation_graph.edges() if a != b}
        known = {(a, b) for a, b in validation_edges if a in self.graph and b in self.graph}
        positives = known - observed
        if not positives:
            raise ValueError("No new validation edges between known training words.")
        possible = len(nodes) * (len(nodes) - 1) - len(observed | known)
        target = min(len(positives) * self.negatives_per_positive, possible)
        if target < 1:
            raise ValueError("No unobserved pairs available for negative sampling.")
        rng = random.Random(self.seed)
        negatives = set()
        while len(negatives) < target:
            source, destination = rng.sample(nodes, 2)
            edge = (source, destination)
            if edge not in observed and edge not in known:
                negatives.add(edge)
        self.coverage = {
            "fit_records": len(self.fit), "validation_records": len(self.validation),
            "fit_nodes": len(nodes), "fit_edges": len(observed),
            "validation_edges_excluding_loops": len(validation_edges),
            "validation_edges_with_unseen_words": len(validation_edges - known),
            "validation_edges_already_in_fit": len(known & observed),
            "positive_new_edges": len(positives), "sampled_negatives": len(negatives),
            "all_possible_negative_pairs": possible,
        }
        candidates = [(a, b, 1) for a, b in sorted(positives)]
        candidates += [(a, b, 0) for a, b in sorted(negatives)]
        rng.shuffle(candidates)
        return candidates

    def run(self, output):
        """Save validation scores, ranking metrics, top predictions and evaluation curves."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        candidates = self.make_candidates()
        rng = random.Random(self.seed + 1)
        rows = [{"source": a, "target": b, "validation_edge": label,
                 **self.score(a, b), "random": rng.random()} for a, b, label in candidates]
        write_csv(output / "candidate_scores.csv", rows)
        split = [{"source_file": r["source_file"], "record_id": r["record_id"], "split": name}
                 for name, records in [("fit", self.fit), ("validation", self.validation)] for r in records]
        write_csv(output / "validation_split.csv", split)
        methods = ["preferential_attachment", "common_neighbors", "adamic_adar", "resource_allocation", "random"]
        labels = [r["validation_edge"] for r in rows]
        metrics, predictions = [], []
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        for method in methods:
            result, roc, pr, order = ranking_metrics(labels, [r[method] for r in rows])
            metrics.append({"method": method, **result})
            axes[0].plot(*zip(*roc), label=method.replace("_", " "))
            axes[1].step(*zip(*pr), where="pre", label=method.replace("_", " "))
            for rank, index in enumerate(order[:50], 1):
                r = rows[index]
                predictions.append({"method": method, "rank": rank, "source": r["source"],
                                    "target": r["target"], "score": r[method],
                                    "observed_in_validation": r["validation_edge"]})
        prevalence = sum(labels) / len(labels)
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[1].axhline(prevalence, color="black", linestyle="--", alpha=0.4)
        axes[0].set(title="Validation ROC", xlabel="False positive rate", ylabel="True positive rate")
        axes[1].set(title="Validation precision-recall", xlabel="Recall", ylabel="Precision")
        for ax in axes:
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
        fig.savefig(output / "validation_curves.png", dpi=160)
        plt.close(fig)
        write_csv(output / "metrics.csv", metrics)
        write_csv(output / "top_predictions.csv", predictions)
        lines = ["# Classical link prediction: validation", "",
                 f"Grouped 80/20 split of the existing training records, seed {self.seed}. "
                 "The final held-out test set is not accessed. Duplicate tokenized slogans stay together.", "",
                 "Predict new directed edges between fit-graph words. Positive examples occur in validation but not fit. "
                 "Negatives are sampled ordered pairs absent from both fit and validation; absence is not proof of linguistic impossibility. "
                 "Self-links are excluded. Validation edges with unseen words are excluded and reported below.", "",
                 f"Negative sampling: up to {self.negatives_per_positive} per positive. Candidate positive fraction: {prevalence:.2%}. "
                 "Precision, average precision and top-100 results apply to this sampled candidate set, not all possible word pairs. "
                 "Scores use no validation labels. Tied top-k scores use a reproducible shuffle; AUC/AP group tied scores.", "",
                 "## Coverage", "", "| Quantity | Count |", "| --- | ---: |"]
        lines += [f"| {name.replace('_', ' ')} | {value:,} |" for name, value in self.coverage.items()]
        lines += ["", "## Directed adaptations", "",
                  "For u -> v, common neighbours means intermediate words w with u -> w -> v. "
                  "Preferential attachment is out_degree(u) * in_degree(v). Adamic-Adar sums 1/log(total_degree(w)); "
                  "resource allocation sums 1/total_degree(w). Total degree is in-degree plus out-degree. "
                  "These are explicitly directed adaptations of the course's simple-graph scores, using unweighted topology. "
                  "A two-step word path does not necessarily imply a grammatical direct transition.", "",
                  "## Results", "", "| Method | ROC AUC | Average precision | Hits / top 100 | Precision@100 | Recall@100 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for r in metrics:
            lines.append(f"| {r['method']} | {r['roc_auc']:.3f} | {r['average_precision']:.3f} | "
                         f"{r['hits_at_k']} / {r['k']} | {r['precision_at_k']:.2%} | {r['recall_at_k']:.2%} |")
        lines += ["", "This is one validation split, not a final test result. Predicted links have not been added to any generation graph. "
                  "Recovering observed transitions does not establish improved slogan coherence.", "",
                  "[Curves](validation_curves.png) · [Top predictions](top_predictions.csv) · [Candidate scores](candidate_scores.csv)"]
        (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return metrics


if __name__ == "__main__":
    from generation import SloganDataset
    folder = Path(__file__).parent
    dataset = SloganDataset(sorted((folder / "data").glob("*.csv")))
    training, _ = dataset.split()
    LinkPredictionExperiment(training).run(folder / "processed" / "link_prediction")
