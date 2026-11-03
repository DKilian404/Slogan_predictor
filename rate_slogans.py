"""Run this script to rate slogans interactively and save scores to a reviewer CSV."""

import argparse
import csv
import os
import tempfile
from pathlib import Path


def choose_file():
    """Let the reviewer choose one of the generated rating sheets."""
    folder = Path(__file__).parent / "processed" / "evaluation" / "human_review"
    files = sorted(path for name in ("Hugo", "Kilian", "Vlad")
                   for path in folder.glob(f"*/{name}.csv"))
    if not files:
        raise FileNotFoundError("No reviewer CSVs found; run main.py first.")
    for number, path in enumerate(files, 1):
        print(f"{number}. {path.parent.name}/{path.name}")
    while True:
        answer = input("Choose your reviewer file (or q to quit): ").strip()
        if answer.lower() == "q":
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(files):
            return files[int(answer) - 1]
        print("Enter one of the file numbers.")


def save_scores(path, fields, rows):
    """Replace the CSV only after the updated contents have been fully written."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def ask_score(label):
    """Ask for a score from one to five or a command to skip or quit."""
    while True:
        answer = input(f"{label} (1–5, s=skip, q=quit): ").strip().lower()
        if answer in {"1", "2", "3", "4", "5", "s", "q"}:
            return answer
        print("Please enter 1, 2, 3, 4, 5, s, or q.")


def rate_file(path):
    """Show unanswered items and save each new score to the selected CSV."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        required = {"sample_id", "text", "readability_1_to_5", "meaningfulness_1_to_5"}
        if not required <= set(fields) or len(fields) != len(set(fields)):
            raise ValueError("Choose a reviewer CSV with text and both rating columns.")
        rows = list(reader)
    score_fields = [("readability_1_to_5", "Readability"),
                    ("meaningfulness_1_to_5", "Meaningfulness")]
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("The CSV has a row with the wrong number of fields.")
        for field, _ in score_fields:
            if row[field] not in {"", "1", "2", "3", "4", "5"}:
                raise ValueError(f"Invalid existing score for {row['sample_id']}: {row[field]}")

    print(f"\nFile: {path.name}")
    print("Readability: 1=very broken, 3=understandable but awkward, 5=natural.")
    print("Meaningfulness: 1=no clear meaning, 3=partly interpretable, 5=clear idea.")
    print("Use 2 and 4 for intermediate scores; short slogan fragments are allowed.")
    print("Scores save immediately; use s to leave an item for later or q to quit.\n")
    try:
        for index, row in enumerate(rows, 1):
            if all(row[field] for field, _ in score_fields):
                continue
            print(f"[{index}/{len(rows)}] {row['sample_id']}: {row['text']}")
            for field, label in score_fields:
                if row[field]:
                    print(f"{label}: {row[field]} (saved)")
                    continue
                answer = ask_score(label)
                if answer == "q":
                    print("Progress saved. Run again to continue.")
                    return
                if answer == "s":
                    break
                row[field] = answer
                save_scores(path, fields, rows)
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped; previously saved scores are kept.")
        return
    remaining = sum(not all(row[field] for field, _ in score_fields) for row in rows)
    print(f"Saved. {len(rows) - remaining}/{len(rows)} items complete; {remaining} remaining.")


def main():
    """Open an optional CSV path or prompt the user to choose a reviewer file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", type=Path)
    args = parser.parse_args()
    try:
        path = args.csv_path if args.csv_path else choose_file()
        if path is not None:
            rate_file(path)
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
