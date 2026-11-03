import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from slogan_graph import build_graph, records_from_csv_files


class SloganDataset:
    """Store the original slogan records and their training/test split."""

    def __init__(self, csv_paths):
        """Read and tokenize the supplied CSV files."""
        self.records, self.company_counts = records_from_csv_files(csv_paths)
        self.train = []
        self.test = []
        self.split_seed = None
        self.train_fraction = None

    def split(self, train_fraction=0.8, seed=42):
        """Split whole slogans while keeping identical tokenized slogans together."""
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1.")
        groups = defaultdict(list)
        for record in self.records:
            if record["usable"]:
                key = tuple(tuple(words) for words in record["sentences"])
                groups[key].append(record)
        if len(groups) < 2:
            raise ValueError("At least two distinct usable slogans are needed to split.")

        groups = list(groups.values())
        random.Random(seed).shuffle(groups)
        target = sum(map(len, groups)) * (1 - train_fraction)
        self.train, self.test = [], []
        # Add a whole group only when it brings the test size closer to 20%.
        for group in groups:
            if abs(len(self.test) + len(group) - target) < abs(len(self.test) - target):
                self.test.extend(group)
            else:
                self.train.extend(group)
        if not self.train or not self.test:
            raise ValueError("These group sizes cannot produce two nonempty splits.")
        self.split_seed = seed
        self.train_fraction = train_fraction
        return self.train, self.test

    def save_split(self, path):
        """Save the assignment of every original row for reproducibility."""
        train_ids = {id(record) for record in self.train}
        test_ids = {id(record) for record in self.test}
        if not train_ids or not test_ids:
            raise ValueError("Call split before saving the assignments.")
        with Path(path).open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["source_file", "record_id", "split"])
            for record in self.records:
                split = "train" if id(record) in train_ids else "test" if id(record) in test_ids else "excluded_empty"
                writer.writerow([record["source_file"], record["record_id"], split])


class FirstOrderGenerator:
    """Generate word sequences from observed training transitions."""

    def __init__(self, training_records, seed=42):
        """Build a training graph and prepare its weighted choices."""
        self.graph = build_graph(training_records)
        self.random = random.Random(seed)
        self.training_sentences = {
            tuple(words) for record in training_records for words in record["sentences"]
        }
        self.starts = sorted(w for w in self.graph if self.graph.nodes[w]["start_count"])
        if not self.starts:
            raise ValueError("Training records must contain at least one usable sentence.")
        self.start_weights = [self.graph.nodes[w]["start_count"] for w in self.starts]
        self.multiword_starts = [w for w in self.starts if self.graph.out_degree(w) > 0]
        self.multiword_start_weights = [self.graph.nodes[w]["start_count"] for w in self.multiword_starts]
        self.training_trigrams = {tuple(words[i:i + 3]) for words in self.training_sentences
                                  for i in range(len(words) - 2)}

    def choose_start(self, target_length):
        """Choose a starting word with an outgoing edge for multiword requests."""
        starts = self.multiword_starts if target_length > 1 else self.starts
        weights = self.multiword_start_weights if target_length > 1 else self.start_weights
        return self.random.choices(starts, weights=weights, k=1)[0] if starts else None

    def generate_fixed_attempt(self, target_length):
        """Follow transition weights until exactly N words or an early dead end."""
        if target_length < 1:
            raise ValueError("target_length must be positive.")
        start = self.choose_start(target_length)
        if start is None:
            return [], "no_start"
        words = [start]
        while len(words) < target_length:
            neighbors = sorted(self.graph.successors(words[-1]))
            if not neighbors:
                return words, "dead_end"
            weights = [self.graph[words[-1]][w]["weight"] for w in neighbors]
            words.append(self.random.choices(neighbors, weights=weights, k=1)[0])
        return words, "completed"

    def generate_fixed_length(self, target_length):
        """Return a fixed-length walk or None if it gets stuck early."""
        words, status = self.generate_fixed_attempt(target_length)
        return words if status == "completed" else None

    def generate_controlled_length(self, target_length):
        """Return a controlled-length walk or None if no valid completion is found."""
        words, status = self.generate_controlled_attempt(target_length)
        return words if status == "completed" else None

    def generate_controlled_attempt(self, target_length):
        """Return the words and outcome of one controlled-length attempt."""
        if target_length < 1:
            raise ValueError("target_length must be positive.")
        if target_length == 1:
            weights = [self.graph.nodes[w]["start_count"] * self.graph.nodes[w]["end_count"]
                       for w in self.starts]
            if not sum(weights):
                return [], "no_ending"
            return self.random.choices(self.starts, weights=weights, k=1), "completed"

        start = self.choose_start(target_length)
        if start is None:
            return [], "no_start"
        words = [start]
        while len(words) < target_length:
            neighbors = sorted(self.graph.successors(words[-1]))
            if not neighbors:
                return words, "dead_end"
            if len(words) == target_length - 1:
                weights = [self.graph.nodes[w]["end_count"] for w in neighbors]
            else:
                weights = [self.graph[words[-1]][w]["weight"] for w in neighbors]
            if not sum(weights):
                return words, "no_ending"
            words.append(self.random.choices(neighbors, weights=weights, k=1)[0])
        return words, "completed"

    def generate_at_length(self, target_length, max_attempts=100, method="controlled_length"):
        """Retry the selected length rule until it succeeds or reaches the attempt limit."""
        if target_length < 1 or max_attempts < 1:
            raise ValueError("Length and attempt limit must be positive.")
        methods = {"fixed_length": self.generate_fixed_length,
                   "controlled_length": self.generate_controlled_length}
        if method not in methods:
            raise ValueError("method must be fixed_length or controlled_length.")
        for attempt in range(1, max_attempts + 1):
            words = methods[method](target_length)
            if words is not None:
                return words, attempt
        return None, max_attempts


def save_length_statistics(dataset, path):
    """Save word-count summaries and distributions for slogans and sentences."""
    lines = []
    for label, records in [("Full dataset", dataset.records), ("Training", dataset.train)]:
        usable = [record for record in records if record["usable"]]
        slogans = [sum(map(len, record["sentences"])) for record in usable]
        sentences = [len(words) for record in usable for words in record["sentences"]]
        lines.append(label)
        for unit, lengths in [("Whole slogans", slogans), ("Sentence sequences", sentences)]:
            lines.append(f"{unit}: n={len(lengths)}, mean={mean(lengths):.2f}, "
                         f"median={median(lengths):g}, range={min(lengths)}-{max(lengths)} words")
        slogan_counts, sentence_counts = Counter(slogans), Counter(sentences)
        lines.append("Words | Whole slogans | Sentence sequences")
        for length in range(1, max(slogans) + 1):
            lines.append(f"{length:5} | {slogan_counts[length]} ({slogan_counts[length]/len(slogans):.2%}) | "
                         f"{sentence_counts[length]} ({sentence_counts[length]/len(sentences):.2%})")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
