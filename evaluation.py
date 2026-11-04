"""Run and record reproducible comparisons of the two generation rules."""

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
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


def save_order_comparison(path, first, second, fallback=False, contextual=False, predicted=False):
    """Save automatic results for both orders using matching experiment settings."""
    labels = ("Strict second order", "With fallback") if fallback else ("First order", "Second order")
    if contextual:
        labels = ("Global endings", "Contextual endings")
    description = ("All versions use the same start-word and second-word rules. Strict second order uses "
                   "observed pair continuations; the fallback version uses first-order counts only when the pair has none. "
                   "Controlled endings still require positive word end counts, with no extra fallback for missing endings.")
    if contextual:
        description = ("Both models use second order with first-order fallback. Only the controlled final-word weights change: "
                       "the new model prefers sentence-final triple counts and uses global ending counts when the pair has no ending evidence. "
                       "Fixed-length generation is unchanged. Matching seeds give identical prefixes and completion outcomes; "
                       "only final-word choices can differ. Human ratings are needed to test whether these endings read better.")
    if predicted:
        labels = ("Second order with fallback", "With predicted-link fallback")
        description = ("Both models use the same starts, observed transitions and global controlled-ending counts. "
                       "The new version tries predecessor-context similarity only after both observed transition rules fail. "
                       "It borrows continuations from similar words; this is a new heuristic, not the classical bridge scores. "
                       "Prediction scores are not observed frequencies. Individual attempts are not paired after the random streams diverge. "
                       "Completion gains do not establish improved coherence.")
    lines = [f"# {labels[0]} versus {labels[1]}", "",
             "Same training split, lengths, five seeds and 1,000 attempts per method/length/seed. "
             "Each model has 50,000 attempts.", "", description, ""]
    first_metrics = {(r["method"], r["target_length"], r["seed"]): r for r in first[1]}
    second_metrics = {(r["method"], r["target_length"], r["seed"]): r for r in second[1]}
    if first_metrics.keys() != second_metrics.keys() or any(
            r["attempts"] != second_metrics[key]["attempts"] for key, r in first_metrics.items()):
        raise ValueError("Both orders must use matching lengths, seeds and attempt counts.")
    lines += ["## Completion counts", "",
              f"| Method | Words | Attempts per model | {labels[0]} successes | {labels[1]} successes | Success change | {labels[0]} failures | {labels[1]} failures |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in first[2]:
        method, length = row["method"], row["target_length"]
        groups = [[r for r in result[1] if r["method"] == method and r["target_length"] == length]
                  for result in (first, second)]
        totals = [sum(r["successes"] for r in group) for group in groups]
        attempts = sum(r["attempts"] for r in groups[0])
        a, b = totals
        lines.append(f"| {method} | {length} | {attempts:,} | {a:,} | {b:,} | {b-a:+,} | {attempts-a:,} | {attempts-b:,} |")
    measures = [("Completion", "success_rate"), ("Word repetition", "repetition_rate"),
                ("Training matches", "training_match_rate"), ("Novelty", "novelty_rate"),
                ("Vocabulary diversity", "type_token_ratio_20"),
                ("Repeated pairs", "bigram_repetition_rate"),
                ("Observed ending", "observed_ending_rate"),
                ("Ending proportion", "ending_proportion"), ("Three-word support", "trigram_support")]
    other = {(r["method"], r["target_length"]): r for r in second[2]}
    lines += ["", "Rates below are means ± sample SD across seeds. "
              "Completion uses all attempts; other measures use successful outputs. "
              "Diversity uses the first 20 successes per seed and should be compared within a length.", ""]
    for title, metric in measures:
        lines += [f"## {title}", "", f"| Method | Words | {labels[0]} | {labels[1]} | Change (pp) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for row in first[2]:
            counterpart = other[row["method"], row["target_length"]]
            cells = []
            for result in (row, counterpart):
                value, sd = result[metric + "_mean"], result[metric + "_sd"]
                cell = "—" if value is None else f"{value:.2%}"
                if value is not None and sd is not None:
                    cell += f" ± {100*sd:.2f}"
                cells.append(cell)
            a, b = row[metric + "_mean"], counterpart[metric + "_mean"]
            change = "—" if a is None or b is None else f"{100*(b-a):+.2f}"
            lines.append(f"| {row['method']} | {row['target_length']} | {' | '.join(cells)} | {change} |")
        lines.append("")
    lines += ["Three-word support is guaranteed only for strict second order by construction. "
              "It is not independent evidence of coherence. Human ratings are needed to assess readability and meaning. "
              "No held-out test data is used in these diagnostics.", "",
              "[First-order results](evaluation/summary.md) · [Strict second-order results](evaluation_second_order/summary.md) · "
              "[Fallback results](evaluation_second_order_fallback/summary.md) · "
              "[Contextual endings](evaluation_contextual_endings/summary.md) · "
              "[Predicted-link fallback](evaluation_predicted_links/summary.md)"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_ending_coverage(path, model, baseline_rows, rows):
    """Report contextual ending coverage and sentence-final triple matches."""
    lines = ["# Contextual ending coverage", "",
             "Controlled-length attempts only. Contextual counts include sentence-final triples, never internal triples. "
             "Global fallback keeps the same candidate continuations and uses their word end_count. "
             "No test data is used.", "",
             "| Words | Attempts | Contextual endings used | Global ending rule tried | Stopped before final choice | Successes | Final triple observed at a sentence end: global | Final triple observed at a sentence end: contextual |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for length in sorted({r["target_length"] for r in rows}):
        group = [r for r in rows if r["method"] == "controlled_length" and r["target_length"] == length]
        if not group:
            continue
        rules = Counter(r["ending_rule"] for r in group)
        matches = []
        for source in (baseline_rows, rows):
            accepted = [r for r in source if r["method"] == "controlled_length"
                        and r["target_length"] == length and r["target_achieved"]]
            count = 0
            for row in accepted:
                words = row["generated_text"].split()
                if len(words) >= 3 and model.ending_counts.get(tuple(words[-3:-1]), {}).get(words[-1], 0):
                    count += 1
            matches.append(f"{count}/{len(accepted)} ({count/len(accepted):.2%})" if accepted else "—")
        success = sum(r["target_achieved"] for r in group)
        lines.append(f"| {length} | {len(group):,} | {rules['contextual']:,} | {rules['global']:,} | "
                     f"{rules['not_reached']:,} | {success:,} | {' | '.join(matches)} |")
    counts = [sum(c.values()) for c in model.ending_counts.values()]
    lines += ["", f"Training pairs with sentence-ending evidence: {len(counts):,}. "
              f"Pairs supported by just one sentence-ending occurrence: {sum(c == 1 for c in counts):,}.", "",
              "This ending-match measure directly reflects the new selection rule, not independent evidence of coherence. "
              "Completion should stay unchanged because only the final-word weights change and global fallback remains available. "
              "Global-rule attempts can still fail if no candidate has positive end_count."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_two_word_failures(path, model, rows):
    """Summarize strict second-order failures after exactly two words."""
    lines = ["# Strict second-order failures after two words", "",
             "These counts describe training-data transitions, not the held-out test set. "
             "A dead-end pair was observed together but always at a sentence end. "
             "A no_ending failure has continuations, but none qualify as the final word.", "",
             "| Method | Target | Attempts | All failures | Two-word dead ends | Two-word no-ending failures | All two-word failures | Share of all failures |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"], row["target_length"]].append(row)
    for (method, length), group in groups.items():
        failures = [r for r in group if not r["target_achieved"]]
        two = [r for r in failures if r["word_count"] == 2]
        reasons = Counter(r["status"] for r in two)
        share = f"{len(two)/len(failures):.2%}" if failures else "—"
        lines.append(f"| {method} | {length} | {len(group):,} | {len(failures):,} | "
                     f"{reasons['dead_end']:,} | {reasons['no_ending']:,} | {len(two):,} | {share} |")
    dead = [r for r in rows if r["word_count"] == 2 and r["status"] == "dead_end"]
    pairs = Counter(tuple(r["generated_text"].split()) for r in dead)
    for pair in pairs:
        if not model.graph.has_edge(*pair) or model.pair_counts.get(pair):
            raise ValueError("An observed two-word dead end has inconsistent training counts.")
    recoverable = sum(model.graph.out_degree(pair[-1]) > 0 for pair in
                      (tuple(r["generated_text"].split()) for r in dead))
    lines += ["", f"Two-word dead ends: {len(dead):,}, covering {len(pairs):,} distinct pairs. "
              f"Of these attempts, {recoverable:,} end on a word with first-order continuations; "
              "fallback can offer a next step there, but does not guarantee completion or an eligible controlled ending.", "",
              "## Most frequent dead-end pairs", "",
              "| Pair | Failed attempts | Training occurrences (all at ends) | Last word has outgoing edges |",
              "| --- | ---: | ---: | --- |"]
    for pair, count in pairs.most_common(15):
        lines.append(f"| {' '.join(pair)} | {count} | {model.graph[pair[0]][pair[1]]['weight']} | "
                     f"{'Yes' if model.graph.out_degree(pair[-1]) else 'No'} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


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
                        if self.model.order == 2:
                            row["model_order"] = 2
                        if self.model.fallback:
                            row["fallback_requests"] = self.model.fallback_requests
                            row["fallback_steps"] = self.model.fallback_steps
                        if self.model.contextual_endings:
                            row["ending_rule"] = self.model.ending_rule
                        if self.model.predicted_fallback:
                            row["prediction_requests"] = self.model.prediction_requests
                            row["predicted_steps"] = len(self.model.predicted_links)
                            row["predicted_links"] = json.dumps(self.model.predicted_links, ensure_ascii=False)
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
        if self.model.fallback:
            self.save_fallback_usage(output / "fallback_usage.md", rows)
        if self.model.predicted_fallback:
            self.save_prediction_usage(output, rows)
        return rows, metrics, summary

    def save_prediction_usage(self, output, rows):
        """Save predicted-edge provenance and recovery counts for each condition."""
        lines = ["# Predicted-link recovery", "",
                 f"At an observed dead end, choose up to {self.model.donor_limit} donor words with outgoing edges "
                 "ranked by Jaccard similarity of their predecessor sets. "
                 "For each candidate, sum similarity times its observed transition probability from each donor. "
                 f"Keep the top {self.model.candidate_limit} candidates (alphabetical tie-break). Interior choices sample "
                 "proportionally to these scores; the controlled final choice uses global end_count within that pool. "
                 "No shared predecessors means no recovery. Missing eligible endings do not trigger prediction. "
                 "All counts and similarities use training data only; observed graph edges and weights are unchanged.", "",
                 "These scores are an unvalidated similarity heuristic, not calibrated probabilities or observed counts. "
                 "The donor and candidate limits are fixed defaults, not tuned parameters. "
                 "Predicted edges are not inserted into the graph. After each step, try observed pair context again.", "",
                 "| Method | Words | Attempts | Requested prediction | Used prediction | Predicted steps | Successes | Successes with prediction |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for method in ("fixed_length", "controlled_length"):
            for length in self.lengths:
                group = [r for r in rows if r["method"] == method and r["target_length"] == length]
                requested = sum(r["prediction_requests"] > 0 for r in group)
                used = sum(r["predicted_steps"] > 0 for r in group)
                steps = sum(r["predicted_steps"] for r in group)
                successes = sum(r["target_achieved"] for r in group)
                recovered = sum(r["target_achieved"] and r["predicted_steps"] > 0 for r in group)
                lines.append(f"| {method} | {length} | {len(group):,} | {requested:,} | {used:,} | {steps:,} | {successes:,} | {recovered:,} |")
        lines += ["", "Successes with prediction are not paired rescues of specific baseline attempts. "
                  "Inspect attempts.csv for every chosen predicted edge, its position and score. "
                  "Human or validated LLM ratings are needed to assess whether these outputs read better."]
        (output / "prediction_usage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        candidates = [{"source": source, "target": target, "prediction_score": score,
                       "donors": json.dumps(self.model.donors[source], ensure_ascii=False)}
                      for source in sorted(self.model.predictions)
                      for target, score in self.model.predictions[source].items()]
        write_csv(output / "predicted_candidates.csv", candidates)

    def save_fallback_usage(self, path, rows):
        """Report fallback requests and actual first-order steps for every condition."""
        lines = ["# Fallback usage", "",
                 "A request occurs when a pair has no continuation. A step counts only when a word is "
                 "actually added using fallback. Initialization is not counted. Missing controlled endings "
                 "do not trigger additional fallback. Pair context is tried again after every word.", "",
                 "| Method | Words | Attempts | Requested fallback | Used fallback | Fallback steps | Successes | Successes using fallback |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for method in ("fixed_length", "controlled_length"):
            for length in self.lengths:
                group = [r for r in rows if r["method"] == method and r["target_length"] == length]
                requested = sum(r["fallback_requests"] > 0 for r in group)
                used = sum(r["fallback_steps"] > 0 for r in group)
                steps = sum(r["fallback_steps"] for r in group)
                successes = sum(r["target_achieved"] for r in group)
                helped = sum(r["target_achieved"] and r["fallback_steps"] > 0 for r in group)
                lines.append(f"| {method} | {length} | {len(group):,} | {requested:,} ({requested/len(group):.2%}) | "
                             f"{used:,} ({used/len(group):.2%}) | {steps:,} | {successes:,} | {helped:,} |")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

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
        suffix = " with fallback" if self.model.fallback else ""
        if self.model.predicted_fallback:
            suffix += " and predicted-link recovery"
        if self.model.contextual_endings:
            suffix += " and contextual endings"
        lines = [f"# Order {self.model.order}{suffix} generation results", "",
                 f"Seeds: {', '.join(map(str, self.seeds))}. "
                 f"Attempts per method/length/seed: {self.attempts:,}.", "",
                 "Fixed-length random walk: transition weights at every step, stopping at word N. "
                 "Controlled-length random walk: transition weights for middle words, ending counts for the final word.", "",
                 "Multiword starts require start_count > 0 and out_degree > 0; the graph itself is unchanged.", "",
                 "Values are means ± sample SD across generation seeds, in percent. "
                 "SD measures variation, not a confidence interval.", ""]
        if self.model.predicted_fallback:
            lines += ["Try second-order transitions, then first-order transitions, then predicted links only at an observed dead end. "
                      "Prediction borrows continuations from words sharing preceding contexts. "
                      "Controlled final choices retain global word-ending counts among the selected candidates. "
                      "The observed graph is unchanged.", "", "[Prediction usage and method](prediction_usage.md)", ""]
        elif self.model.contextual_endings:
            lines += ["Interior steps use second order with first-order fallback when a pair has no continuation. "
                      "At the controlled final step, sample using counts of sentence-final triples for the current pair. "
                      "If none exist, use the existing global word-ending counts among available continuations. "
                      "Fixed-length generation is unchanged.", "",
                      "[Ending coverage](ending_coverage.md) · [Fallback usage](fallback_usage.md)", ""]
        elif self.model.fallback:
            lines += ["The first two words use the same initialization as first order. Later steps prefer pair counts. "
                      "When a pair has no continuation, use first-order counts for that step, then try pair context again. "
                      "Controlled endings still use raw end_count; missing eligible endings do not trigger another fallback.", "",
                      "Three-word support is no longer guaranteed. [Fallback usage](fallback_usage.md)", ""]
        elif self.model.order == 2:
            lines += ["The first two words use the same initialization as first order. "
                      "Later choices require an observed continuation of the previous two words, with no fallback. "
                      "The final controlled choice uses word end_count among those continuations.", "",
                      "Three-word support is guaranteed by construction and is not independent evidence of better language.", ""]
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
        suffix = " with fallback" if self.model.fallback else ""
        if self.model.predicted_fallback:
            suffix += " and predicted-link recovery"
        if self.model.contextual_endings:
            suffix += " and contextual endings"
        lines = [f"# Order {self.model.order}{suffix} generated sequences", "",
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
                        reason = {"dead_end": "No continuation for the current context", "no_ending": "No eligible ending word", "no_start": "No eligible starting word"}[row["status"]]
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
        return save_review_sample(output, sample)


def save_mixed_second_order_review(output, strict_rows, fallback_rows, company_labels):
    """Mix 150 strict and fallback outputs with 20% random company labels."""
    companies = sorted({label.strip() for label in company_labels if label.strip()})
    if not companies:
        raise ValueError("Company labels are required for the mixed review.")
    rng, brand_rng = random.Random(2028), random.Random(2029)
    sample = []
    for variant_index, (variant, rows) in enumerate(
            [("strict_second_order", strict_rows), ("second_order_fallback", fallback_rows)]):
        for length_index, length in enumerate((3, 4, 5, 8, 10)):
            for method_index, method in enumerate(("fixed_length", "controlled_length")):
                # Alternate 7/8 items and 1/2 branded items across the conditions.
                extra = (variant_index + length_index + method_index) % 2
                count, branded = 7 + extra, 1 + extra
                candidates = [r for r in rows if r["target_achieved"]
                              and r["target_length"] == length and r["method"] == method]
                if len(candidates) < count:
                    raise ValueError("Not enough successful outputs for the mixed review.")
                selected = rng.sample(candidates, count)
                brand_indices = set(brand_rng.sample(range(count), branded))
                for index, row in enumerate(selected):
                    company = brand_rng.choice(companies) if index in brand_indices else ""
                    sample.append({**row, "model_variant": variant, "assigned_company": company})
    rng.shuffle(sample)
    return save_review_sample(output, sample)


def save_all_model_review(processed):
    """Create a blinded 250-item review with five outputs per model, rule and length."""
    processed = Path(processed)
    models = {"first_order": "evaluation", "strict_second_order": "evaluation_second_order",
              "second_order_fallback": "evaluation_second_order_fallback",
              "contextual_endings": "evaluation_contextual_endings",
              "predicted_link_recovery": "evaluation_predicted_links"}
    rng = random.Random(2030)
    sample = []
    for variant, folder in models.items():
        path = processed / folder / "attempts.csv"
        if not path.exists():
            raise FileNotFoundError("Run main.py first to generate attempts for all five models.")
        groups = defaultdict(list)
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                if row["target_achieved"] == "True":
                    groups[row["method"], int(row["target_length"])].append(row)
        for method in ("fixed_length", "controlled_length"):
            for length in (3, 4, 5, 8, 10):
                candidates = groups[method, length]
                if len(candidates) < 5:
                    raise ValueError(f"Not enough successful outputs for {variant}, {method}, {length}.")
                for row in rng.sample(candidates, 5):
                    sample.append({"model_variant": variant, "attempt_id": row["attempt_id"],
                                   "method": method, "target_length": length,
                                   "generated_text": row["generated_text"], "assigned_company": ""})
    rng.shuffle(sample)
    review = save_review_sample(processed / "evaluation_all_models", sample)
    (review / "rubric.txt").write_text(
        "250 items: 50 per model, 25 per length rule per model, 10 per target length per model.\n"
        "No company labels have been added; words generated from the dataset are kept as produced.\n"
        "Everyone rates the same shuffled items independently; do not open method_key.csv first.\n"
        "Readability: 1=very broken wording; 3=understandable but awkward; 5=natural wording.\n"
        "Meaningfulness: 1=no clear meaning; 3=partly interpretable; 5=clear, sensible idea.\n"
        "Use 2 and 4 for intermediate scores. Short advertising fragments are allowed.\n"
        "Items are sampled successful attempts without quality filtering; duplicate text is possible.\n"
        "Keep this batch separate from earlier ratings and preserve its folder ID when returning files.\n",
        encoding="utf-8")
    return review


def save_review_sample(output, sample):
    """Save blinded reviewer sheets without overwriting existing ratings."""
    output = Path(output)
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
    if sample and "model_variant" in sample[0]:
        for entry, row in zip(key, sample):
            entry["model_variant"] = row["model_variant"]
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
    return review
