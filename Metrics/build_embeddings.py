import os
import argparse
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Build per-uuid topic-count embedding vectors from a topics CSV. "
                    "One row per uuid: country, category, then a count column per topic. "
                    "No category filter and no topic-count filter (matches Metrics/new.py)."
    )
    p.add_argument("--input", required=True,
                   help="topics CSV (needs uuid, country, category, final_topic)")
    p.add_argument("--output", default=None,
                   help="output CSV path (default: embeddings.csv beside --input)")
    return p.parse_args()


def build_and_save_embeddings(df, out_path):
    """
    Build a uuid x topic count matrix for ALL uuids and save it, with no category
    filter and no topic-count filter. The saved CSV has columns:
    uuid, country, category, <topic_1>, <topic_2> ...
    """
    counts = pd.crosstab(df["uuid"], df["final_topic"])

    # uuid -> country and uuid -> category (first occurrence per uuid)
    meta = (
        df.drop_duplicates("uuid")
          .set_index("uuid")[["country", "category"]]
          .reindex(counts.index)
    )

    embeddings = meta.join(counts)
    embeddings.to_csv(out_path)  # index=uuid

    print(f"embeddings: {len(embeddings)} uuid(s) -> {out_path}")
    return embeddings


def main():
    args = parse_args()
    base = os.path.dirname(args.input)
    out = args.output or os.path.join(base, "embeddings.csv")

    df = pd.read_csv(args.input)
    build_and_save_embeddings(df, out)


if __name__ == "__main__":
    main()
