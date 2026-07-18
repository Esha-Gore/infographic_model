import argparse
import pandas as pd
import json


def parse_args():
    p = argparse.ArgumentParser(
        description="Flatten all_detections.json into a per-element CSV (captions_flat.csv)."
    )
    p.add_argument("--input", required=True, help="all_detections.json")
    p.add_argument("--output", required=True, help="captions_flat.csv output path")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    df = pd.json_normalize(data, record_path=["elements"], meta=["uuid", "country", "image"])
    df.rename(columns={"type": "element_type"}, inplace=True)

    # Filter (kept off so every element is retained in order — the downstream
    # groupby('uuid').cumcount() in build_joined_filtered.py relies on this).
    # df = df[df["content"].notna()]
    # df = df[df["content"].str.strip() != ""]
    # df = df[df["content"].str.split().str.len() >= 4]

    df.to_csv(args.output, index=False)
    print(f"Done! {len(df)} rows saved to {args.output}")


if __name__ == "__main__":
    main()
