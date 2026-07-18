import os
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


CATEGORIES = [
    'Automotive', 'Business and Finance', 'Communication', 'Education',
    'Entertainment', 'Medical Health', 'Politics', 'Sensitive Topics',
    'Shopping', 'Sports', 'Technology & Computing', 'Travel', 'Video Gaming'
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Build per-uuid topic-count vectors (reduced_topics.csv -> embeddings CSVs) "
                    "and report within/between-country distance gaps."
    )
    p.add_argument("--input", required=True,
                   help="reduced_topics.csv (needs uuid, country, category, final_topic)")
    p.add_argument("--og-out", default=None,
                   help="output for all-uuid embeddings (default: embeddings_og.csv beside --input)")
    p.add_argument("--filtered-out", default=None,
                   help="output for >4-topic uuids (default: embeddings_filtered.csv beside --input)")
    return p.parse_args()


args = parse_args()
CSV_PATH = args.input
_base = os.path.dirname(CSV_PATH)
OG_OUT = args.og_out or os.path.join(_base, "embeddings_og.csv")
FILTERED_OUT = args.filtered_out or os.path.join(_base, "embeddings_filtered.csv")

df = pd.read_csv(CSV_PATH)

# Keep only rows in the allowed categories
df = df[df["category"].isin(CATEGORIES)]


def build_and_save_embeddings(df, og_path, filtered_path):
    """
    Builds uuid x topic count matrices and saves two versions:
      - og_path: all uuids, regardless of total topic count
      - filtered_path: only uuids with more than 4 total topics 
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

    enough = counts.values.sum(axis=1) > 4
    filtered = og[enough]
    filtered.to_csv(filtered_path)  # index=uuid

    return og, filtered


def country_gap(sub_df, metric="cosine"):
    """
    With-in vs between-country distances for a df
    metric: any pdist metric (curr: cosine or "euclidean).
    Returns (within_mean, between_mean, n_websites)
    
    """

    # Make a df with the uuid and all of the topics as columns consisting of their counts. 
    counts = pd.crosstab(sub_df["uuid"], sub_df["final_topic"])

    # Country to uuid mapping, in order of df above ^^ 
    labels = (
        sub_df.drop_duplicates("uuid")
              .set_index("uuid")["country"]
              .reindex(counts.index)
    )

    # Keep only uuids with more than 4 topics (row-sum = total topics per uuid)
    enough = counts.values.sum(axis=1) > 4
    counts = counts[enough]
    labels = labels[enough]

    if len(counts) < 2:
        return None  # need at least 2 websites to have any pair

    # Run cosine distance on eevry pair and reshapes into an array
    D = squareform(pdist(counts.values, metric=metric))

    # Figure out pairs of the uuids within the same country
    countries = labels.values
    same_country = countries[:, None] == countries[None, :]   # N x N: True if same country

    i, j = np.triu_indices(len(countries), k=1)
    pair_distances = D[i, j]
    pair_same = same_country[i, j]

    # Separate the distances between pairs of within the same country vs different country. 
    within = pair_distances[pair_same]     # same-country pairs
    between = pair_distances[~pair_same]   # different-country pairs

    if within.size == 0 or between.size == 0:
        return None  # need both same- and different-country pairs to compare

    return within.mean(), between.mean(), len(counts)


def run_report(name, metric):
    print(f"{name}")
    overall = country_gap(df, metric=metric)
    w, b, n = overall
    print(f"{'Ooverall':25s} within={w:.4f} between={b:.4f} gap={b - w:.4f} (n={n:,})")
    for category in CATEGORIES:
        result = country_gap(df[df["category"] == category], metric=metric)
        if result is None:
            continue
        w, b, n = result
        print(f"{category:25s} within={w:.4f} between={b:.4f} gap={b - w:.4f} (n={n:,})")
    print()


build_and_save_embeddings(
    df,
    og_path=OG_OUT,
    filtered_path=FILTERED_OUT,
)

run_report("Cosine (counts)", metric="cosine")
run_report("Euclidean (counts)", metric="euclidean")