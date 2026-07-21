import os
import argparse
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Build per-uuid topic-count embedding vectors from a topics CSV. "
                    "This is count_embeddings.py minus the within/between-country distance "
                    "analysis (which needs >=2 websites) - just the embeddings."
    )
    p.add_argument("--input", required=True,
                   help="topics CSV (needs uuid, country, category, final_topic)")
    p.add_argument("--og-out", default=None,
                   help="output for all-uuid embeddings (default: embeddings_og.csv beside --input)")
    p.add_argument("--filtered-out", default=None,
                   help="output for >min-topics uuids (default: embeddings_filtered.csv beside --input)")
    p.add_argument("--min-topics", type=int, default=4,
                   help="filtered file keeps uuids with MORE than this many total topics (default 4)")
    return p.parse_args()


def build_and_save_embeddings(df, og_path, filtered_path, min_topics):
    """
    Build a uuid x topic count matrix and save two versions:
      - og_path: every uuid
      - filtered_path: only uuids with more than `min_topics` total topics
    Each saved CSV has columns: uuid, country, category, <topic_1>, <topic_2> ...
    """
    counts = pd.crosstab(df["uuid"], df["final_topic"])

    # uuid -> country and uuid -> category (first occurrence per uuid)
    meta = (
        df.drop_duplicates("uuid")
          .set_index("uuid")[["country", "category"]]
          .reindex(counts.index)
    )

    og = meta.join(counts)
    og.to_csv(og_path)  # index=uuid

    enough = counts.values.sum(axis=1) > min_topics
    filtered = og[enough]
    filtered.to_csv(filtered_path)  # index=uuid

    print(f"embeddings: {len(og)} uuid(s) -> {og_path}")
    print(f"filtered (>{min_topics} topics): {len(filtered)} uuid(s) -> {filtered_path}")
    return og, filtered


def main():
    args = parse_args()
    base = os.path.dirname(args.input)
    og_out = args.og_out or os.path.join(base, "embeddings_og.csv")
    filtered_out = args.filtered_out or os.path.join(base, "embeddings_filtered.csv")

    df = pd.read_csv(args.input)
    build_and_save_embeddings(df, og_out, filtered_out, args.min_topics)


if __name__ == "__main__":
    main()
