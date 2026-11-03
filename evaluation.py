"""Run and record reproducible comparisons of the two generation rules."""

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def write_csv(path, rows):
    """Write dictionaries with the same fields to a CSV file."""
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class GenerationExperiment:
    """Compare generation methods over fixed lengths and random seeds."""

    def __init__(self, model, lengths=(3, 4, 5, 8, 10), seeds=(42, 43, 44, 45, 46),
                 attempts=1000, company_labels=None):
        """Store the model and the experiment settings."""
        if not lengths or not seeds or attempts < 1 or min(lengths) < 1:
            raise ValueError("Provide lengths, seeds, and a positive attempt count.")
        if len(set(lengths)) != len(lengths) or len(set(seeds)) != len(seeds):
            raise ValueError("Lengths and seeds must not contain duplicates.")
        self.model = model
        self.lengths = lengths
        self.seeds = seeds
        self.attempts = attempts
        self.company_labels = sorted({label.strip() for label in (company_labels or []) if label.strip()})

    def run(self, output):
        """Save every attempt, per-seed metrics, and mean results across seeds."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        rows, metrics = [], []
        for method in ("fixed_length", "controlled_length"):
            for length in self.lengths:
                for seed in self.seeds:
                    # Give each condition its own deterministic random stream.
                    stream_seed = f"{method}:{length}:{seed}"
                    self.model.random.seed(stream_seed)
                    batch = []
                    for attempt in range(1, self.attempts + 1):
                        if method == "fixed_length":
                            words, status = self.model.generate_fixed_attempt(length)
                        else:
                            words, status = self.model.generate_controlled_attempt(length)
                        completed = status == "completed"
                        success = completed and len(words) == length
                        pairs = list(zip(words, words[1:]))
                        triples = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
                        ending = self.model.graph.nodes[words[-1]] if words else None
                        row = {
                            "attempt_id": f"{method}-{length}-{seed}-{attempt}",
                            "method": method, "seed": seed, "stream_seed": stream_seed,
                            "target_length": length, "attempt": attempt,
                            "generated_text": " ".join(words), "word_count": len(words),
                            "completed": completed, "target_achieved": success,
                            "status": status,
                            "repeats_word": len(set(words)) < len(words),
                            "training_match": tuple(words) in self.model.training_sentences,
                            "repeats_bigram": len(set(pairs)) < len(pairs),
                            "observed_ending": bool(ending and ending["end_count"] > 0),
                            "ending_proportion": ending["end_count"] / ending["frequency"] if ending else None,
                            "trigram_support": sum(t in self.model.training_trigrams for t in triples) / len(triples) if triples else None,
                        }
                        batch.append(row)
                    rows.extend(batch)
                    metrics.append(self.score(batch))
        write_csv(output / "attempts.csv", rows)
        write_csv(output / "metrics_by_seed.csv", metrics)
        summary = self.summarize(metrics)
        write_csv(output / "summary.csv", summary)
        self.save_summary(output / "summary.md", summary)
        self.save_seed_tables(output / "metrics_by_seed.md", metrics)
        self.save_examples(output / "examples.md", rows)
        self.save_rating_sample(output, rows)
        return rows, metrics, summary

    def score(self, batch):
        """Calculate automatic measures for one method, length, and seed."""
        accepted = [row for row in batch if row["target_achieved"]]
        count = len(accepted)
        # Match diversity sample size across conditions when enough outputs exist.
        sample = accepted[:20]
        tokens = [word for row in sample for word in row["generated_text"].split()]
        return {
            "method": batch[0]["method"], "target_length": batch[0]["target_length"],
            "seed": batch[0]["seed"], "attempts": len(batch), "successes": count,
            "success_rate": count / len(batch),
            "dead_ends": sum(r["status"] == "dead_end" for r in batch),
            "no_ending": sum(r["status"] == "no_ending" for r in batch),
            "no_start": sum(r["status"] == "no_start" for r in batch),
            "bigram_repetition_rate": mean(r["repeats_bigram"] for r in accepted) if count else None,
            "observed_ending_rate": mean(r["observed_ending"] for r in accepted) if count else None,
            "ending_proportion": mean(r["ending_proportion"] for r in accepted) if count else None,
            "trigram_support": mean(r["trigram_support"] for r in accepted if r["trigram_support"] is not None) if any(r["trigram_support"] is not None for r in accepted) else None,
            "repetition_rate": sum(r["repeats_word"] for r in accepted) / count if count else None,
            "training_match_rate": sum(r["training_match"] for r in accepted) / count if count else None,
            "novelty_rate": sum(not r["training_match"] for r in accepted) / count if count else None,
            "diversity_sample_size": len(sample),
            "type_token_ratio_20": len(set(tokens)) / len(tokens) if len(sample) == 20 else None,
        }

    def summarize(self, metrics):
        """Calculate means and sample standard deviations across generation seeds."""
        groups = defaultdict(list)
        for row in metrics:
            groups[row["method"], row["target_length"]].append(row)
        summary = []
        for (method, length), group in groups.items():
            result = {"method": method, "target_length": length, "runs": len(group)}
            for name in ("success_rate", "repetition_rate", "training_match_rate",
                         "novelty_rate", "type_token_ratio_20", "bigram_repetition_rate",
                         "observed_ending_rate", "ending_proportion", "trigram_support"):
                values = [r[name] for r in group if r[name] is not None]
                result[name + "_runs"] = len(values)
                result[name + "_mean"] = mean(values) if values else None
                result[name + "_sd"] = stdev(values) if len(values) > 1 else None
            summary.append(result)
        return summary

    def save_summary(self, path, summary):
        """Save formatted Markdown tables comparing both generation methods."""
        lines = ["# Generation results", "",
                 f"Seeds: {', '.join(map(str, self.seeds))}. "
                 f"Attempts per method/length/seed: {self.attempts:,}.", "",
                 "Fixed-length random walk: transition weights at every step, stopping at word N. "
                 "Controlled-length random walk: transition weights for middle words, ending counts for the final word.", "",
                 "Multiword starts require start_count > 0 and out_degree > 0; the graph itself is unchanged.", "",
                 "Values are means ± sample SD across generation seeds, in percent. "
                 "SD measures variation, not a confidence interval.", ""]
        groups = {(row["method"], row["target_length"]): row for row in summary}
        measures = [
            ("Length completion", "success_rate", "Denominator: all attempts."),
            ("Word repetition", "repetition_rate", "Accepted outputs containing at least one repeated word."),
            ("Exact training matches", "training_match_rate", "Accepted outputs identical to a training sentence."),
            ("Novelty", "novelty_rate", "Accepted outputs absent from training sentences."),
            ("Vocabulary diversity", "type_token_ratio_20", "Distinct words / tokens in the first 20 accepted outputs per seed; compare within the same length."),
            ("Repeated two-word phrases", "bigram_repetition_rate", "Accepted outputs repeating an adjacent word pair."),
            ("Observed ending words", "observed_ending_rate", "Accepted outputs ending on a word with positive end_count; guaranteed for controlled generation."),
            ("Ending suitability", "ending_proportion", "Mean end_count / frequency of the final word across accepted outputs; a boundary diagnostic, not a grammar score."),
            ("Three-word phrase support", "trigram_support", "Mean fraction of adjacent triples observed in training, per accepted output; measures familiar phrasing, not coherence."),
        ]
        for title, metric, note in measures:
            lines += [f"## {title}", "", note, "",
                      "| Words | Fixed-length random walks | Controlled-length random walks | Valid seeds (FL / CL) |",
                      "| ---: | ---: | ---: | ---: |"]
            for length in self.lengths:
                cells, counts = [], []
                for method in ("fixed_length", "controlled_length"):
                    row = groups[method, length]
                    value, sd = row[metric + "_mean"], row[metric + "_sd"]
                    cell = "—" if value is None else f"{100 * value:.2f}%"
                    if value is not None and sd is not None:
                        cell += f" ± {100 * sd:.2f}"
                    cells.append(cell)
                    counts.append(str(row[metric + "_runs"]))
                lines.append(f"| {length} | {' | '.join(cells)} | {' / '.join(counts)} |")
            lines.append("")
        lines += ["Accepted = completed at the requested length. Missing values appear as —. "
                  "These automatic measures do not assess coherence.", "",
                  "[Per-seed results](metrics_by_seed.md) · [Summary CSV](summary.csv) · [All attempts](attempts.csv)"]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def save_seed_tables(self, path, metrics):
        """Save per-seed counts and automatic measures as Markdown tables."""
        lines = ["# Results by seed", "", "[Overall summary](summary.md)", ""]
        for method, label in [("fixed_length", "Fixed-length random walks"), ("controlled_length", "Controlled-length random walks")]:
            lines += [f"## {label}", "", "### Completion and failures", "",
                      "| Words | Seed | Attempts | Successes | Success rate | Dead end | No ending | No start |",
                      "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            selected = [row for row in metrics if row["method"] == method]
            for r in selected:
                lines.append(f"| {r['target_length']} | {r['seed']} | {r['attempts']} | {r['successes']} | "
                             f"{r['success_rate']:.2%} | {r['dead_ends']} | {r['no_ending']} | {r['no_start']} |")
            lines += ["", "### Measures on successful outputs", "",
                      "| Words | Seed | Repetition | Training matches | Novelty | Diversity | Diversity sample |",
                      "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for r in selected:
                values = ["—" if r[name] is None else f"{r[name]:.2%}" for name in
                          ("repetition_rate", "training_match_rate", "novelty_rate", "type_token_ratio_20")]
                lines.append(f"| {r['target_length']} | {r['seed']} | {' | '.join(values)} | {r['diversity_sample_size']} |")
            lines += ["", "### Phrase and ending diagnostics", "",
                      "| Words | Seed | Repeated pairs | Observed ending | Ending proportion | Three-word support |",
                      "| ---: | ---: | ---: | ---: | ---: | ---: |"]
            for r in selected:
                values = ["—" if r[name] is None else f"{r[name]:.2%}" for name in
                          ("bigram_repetition_rate", "observed_ending_rate", "ending_proportion", "trigram_support")]
                lines.append(f"| {r['target_length']} | {r['seed']} | {' | '.join(values)} |")
            lines.append("")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def save_examples(self, path, rows, count=10):
        """Save actual successful sequences and failed prefixes in readable tables."""
        lines = ["# Generated sequences", "",
                 "First 10 successful attempts per method and length, without quality filtering. "
                 "Failures show the words reached before the walk got stuck.", "",
                 "[Summary](summary.md) · [All attempts](attempts.csv)", ""]
        for length in self.lengths:
            lines += [f"## {length} words", ""]
            for method, label in [("fixed_length", "Fixed-length random walk"),
                                  ("controlled_length", "Controlled-length random walk")]:
                group = [r for r in rows if r["method"] == method and r["target_length"] == length]
                successes = [r for r in group if r["target_achieved"]][:count]
                lines += [f"### {label}", "",
                          "| Sequence | Seed | Attempt | Exact training match |",
                          "| --- | ---: | ---: | --- |"]
                for row in successes:
                    sequence = row["generated_text"].replace("|", "\\|")
                    match = "Yes" if row["training_match"] else "No"
                    lines.append(f"| {sequence} | {row['seed']} | {row['attempt']} | {match} |")
                failures = [r for r in group if not r["target_achieved"]][:3]
                if failures:
                    lines += ["", "**Failed attempts (partial sequences)**", "",
                              "| Partial sequence | Words reached | Reason |",
                              "| --- | ---: | --- |"]
                    for row in failures:
                        reason = {"dead_end": "No outgoing edges", "no_ending": "No eligible ending word", "no_start": "No eligible starting word"}[row["status"]]
                        sequence = row["generated_text"].replace("|", "\\|")
                        lines.append(f"| {sequence} | {row['word_count']} | {reason} |")
                lines.append("")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def save_rating_sample(self, output, rows, per_condition=15):
        """Save shared reviewer sheets with 80% plain and 20% company-prefixed items."""
        groups = defaultdict(list)
        for row in rows:
            if row["target_achieved"]:
                groups[row["method"], row["target_length"]].append(row)
        expected = [(method, length) for method in ("fixed_length", "controlled_length")
                    for length in self.lengths]
        if per_condition < 5 or per_condition % 5:
            raise ValueError("per_condition must be a positive multiple of five.")
        if not self.company_labels:
            raise ValueError("Provide company_labels to create company-prefixed review items.")
        if any(len(groups[key]) < per_condition for key in expected):
            raise ValueError("Not enough successful attempts for a balanced review sample.")
        rng = random.Random(2026)
        brand_rng = random.Random(2027)
        sample = []
        for key in expected:
            selected = rng.sample(groups[key], per_condition)
            branded_indices = set(brand_rng.sample(range(per_condition), per_condition // 5))
            for index, row in enumerate(selected):
                company = brand_rng.choice(self.company_labels) if index in branded_indices else ""
                sample.append({**row, "assigned_company": company})
        rng.shuffle(sample)
        fingerprint = hashlib.sha256(repr(sample).encode()).hexdigest()[:12]
        review = output / "human_review" / fingerprint
        review.mkdir(parents=True, exist_ok=True)
        sheet, key = [], []
        for number, row in enumerate(sample, 1):
            sample_id = f"S{number:03}"
            company = row["assigned_company"]
            text = f"{company}: {row['generated_text']}" if company else row["generated_text"]
            sheet.append({"sample_id": sample_id, "text": text,
                          "readability_1_to_5": "", "meaningfulness_1_to_5": "", "notes": ""})
            key.append({"sample_id": sample_id, "attempt_id": row["attempt_id"],
                        "method": row["method"], "target_length": row["target_length"],
                        "has_company": bool(company), "company": company,
                        "generated_text": row["generated_text"]})
        for reviewer in ("Hugo", "Kilian", "Vlad"):
            path = review / f"{reviewer}.csv"
            if not path.exists():
                write_csv(path, sheet)
        write_csv(review / "method_key.csv", key)
        (review / "rubric.txt").write_text(
            "Rate independently before opening method_key.csv.\n"
            "Rate the displayed item, including company context when shown.\n"
            "Company labels are randomly assigned examples, not real company slogans.\n"
            "Readability: 1=very broken wording; 3=understandable but awkward; 5=natural wording.\n"
            "Meaningfulness: 1=no clear meaning; 3=partly interpretable; 5=clear, sensible idea.\n"
            "Use 2 and 4 for intermediate judgments. Short advertising fragments are allowed.\n"
            "Leave genuinely unrateable items blank and explain in notes.\n",
            encoding="utf-8")
