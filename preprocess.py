import csv
import json
import re
import unicodedata
from pathlib import Path


# Keep Unicode letters/numbers and apostrophes inside words (e.g. "it's").
WORD_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
SENTENCE_BOUNDARY = re.compile(r"[.!?…]+")


def clean_slogan(text: str) -> str:
    """Normalize case, apostrophes and whitespace, retaining sentence breaks."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().replace("’", "'").replace("‘", "'")
    return " ".join(text.split())


def tokenize_slogan(text: str) -> list[list[str]]:
    """Return a separate word list for each nonempty sentence."""
    sentences = []
    for sentence in SENTENCE_BOUNDARY.split(clean_slogan(text)):
        words = WORD_PATTERN.findall(sentence)
        if words:
            sentences.append(words)
    return sentences


def load_slogans(csv_path: str | Path) -> tuple[list[dict], dict[str, int]]:
    """Return (records, company_counts), preserving one record per CSV row."""
    records = []
    company_counts = {}

    with Path(csv_path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        missing = {"company", "slogan"} - set(columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        if len(columns) != len(set(columns)):
            raise ValueError("CSV contains duplicate column names.")

        for record_id, row in enumerate(reader, start=1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV record {record_id} has an incorrect number of fields.")

            company = row["company"].strip()
            slogan = row["slogan"]
            sentences = tokenize_slogan(slogan)

            # Count each original observation once, even if it has many sentences.
            company_counts[company] = company_counts.get(company, 0) + 1

            records.append({
                "record_id": record_id,
                "original": row,
                "company": company,
                "cleaned_slogan": clean_slogan(slogan),
                "sentences": sentences,
                "usable": bool(sentences),
            })

    return records, company_counts


def get_token_lists(records: list[dict]) -> list[list[list[str]]]:
    """Return slogan -> sentences -> words, keeping original record boundaries."""
    return [record["sentences"] for record in records]


def save_company_counts(company_counts: dict[str, int], output_dir: str | Path) -> None:
    """Save the number of slogan records per company as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = dict(sorted(company_counts.items(), key=lambda item: (-item[1], item[0])))
    with (output_dir / "company_counts.json").open("w", encoding="utf-8") as destination:
        json.dump(counts, destination, ensure_ascii=False, indent=2)
        destination.write("\n")

