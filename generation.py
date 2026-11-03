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
        self.choices = {}
        for word in self.graph:
            neighbors = sorted(self.graph.successors(word))
            weights = [self.graph[word][next_word]["weight"] for next_word in neighbors]
            # None represents stopping; it is not a word node in the graph.
            self.choices[word] = (
                neighbors + [None], weights + [self.graph.nodes[word]["end_count"]]
            )

    def generate(self, max_words=30):
        """Return the sampled words and whether the sequence ended naturally."""
        if max_words < 1:
            raise ValueError("max_words must be positive.")
        words = [self.random.choices(self.starts, weights=self.start_weights, k=1)[0]]
        while True:
            choices, weights = self.choices[words[-1]]
            next_word = self.random.choices(choices, weights=weights, k=1)[0]
            if next_word is None:
                return words, True
            if len(words) == max_words:
                return words, False
            words.append(next_word)

    def generate_fixed_length(self, target_length):
        """Sample a fixed-length walk using ending counts for its final word."""
        if target_length < 1:
            raise ValueError("target_length must be positive.")
        if target_length == 1:
            weights = [self.graph.nodes[w]["start_count"] * self.graph.nodes[w]["end_count"]
                       for w in self.starts]
            if not sum(weights):
                return None
            return self.random.choices(self.starts, weights=weights, k=1)

        words = [self.random.choices(self.starts, weights=self.start_weights, k=1)[0]]
        while len(words) < target_length:
            neighbors = sorted(self.graph.successors(words[-1]))
            if len(words) == target_length - 1:
                weights = [self.graph.nodes[w]["end_count"] for w in neighbors]
            else:
                weights = [self.graph[words[-1]][w]["weight"] for w in neighbors]
            if not sum(weights):
                return None
            words.append(self.random.choices(neighbors, weights=weights, k=1)[0])
        return words

    def generate_at_length(self, target_length, max_attempts=100):
        """Retry controlled-length walks until one succeeds or the limit is reached."""
        if target_length < 1 or max_attempts < 1:
            raise ValueError("Length and attempt limit must be positive.")
        for attempt in range(1, max_attempts + 1):
            words = self.generate_fixed_length(target_length)
            if words is not None:
                return words, attempt
        return None, max_attempts


def record_outcomes(model, dataset, path, seed=42, attempts=1000, max_words=30):
    """Save generation counts and examples for random walks and controlled lengths."""
    if attempts < 1:
        raise ValueError("attempts must be positive.")
    model.random.seed(seed)
    natural = [model.generate(max_words) for _ in range(attempts)]
    completed = [words for words, ended in natural if ended]
    lines = [
        f"Split seed: {dataset.split_seed}; generation seed: {seed}",
        f"Train: {len(dataset.train)}; test: {len(dataset.test)}",
        f"Training nodes: {len(model.graph)}; edges: {model.graph.number_of_edges()}",
        "", f"Random walks: {len(completed)}/{attempts} completed; word limit: {max_words}",
        f"Mean completed random-walk length: {mean(map(len, completed)) if completed else 0:.2f} words",
        "Words | Random walks | Controlled lengths | Controlled failures | Controlled training matches",
    ]
    controlled = {}
    for length in (3, 5, 8, 10):
        model.random.seed(seed + length)
        outcomes = [model.generate_fixed_length(length) for _ in range(attempts)]
        accepted = [words for words in outcomes if words is not None]
        controlled[length] = outcomes
        natural_count = sum(len(words) == length for words in completed)
        matches = sum(tuple(words) in model.training_sentences for words in accepted)
        lines.append(f"{length:5} | {natural_count}/{attempts} ({natural_count / attempts:.1%}) | "
                     f"{len(accepted)}/{attempts} ({len(accepted) / attempts:.1%}) | "
                     f"{attempts - len(accepted)} | {matches}")
    lines += ["", "Controlled seeds: base seed + length; one walk per attempt.",
              "Final-word weight: end_count among outgoing neighbors.",
              "Examples: first five successful controlled walks per length."]
    for length, outcomes in controlled.items():
        lines += ["", f"{length} words:"]
        accepted = [words for words in outcomes if words is not None]
        lines.extend(" ".join(words) for words in accepted[:5])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"natural": natural, "controlled": controlled}


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
