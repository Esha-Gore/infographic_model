import pandas as pd
import argparse

parser = argparse.ArgumentParser(
    description="Fill the 'category' column in a target CSV from a lookup CSV, matching on 'uuid'."
)
parser.add_argument("target_csv", help="CSV whose 'category' column should be filled")
parser.add_argument("lookup_csv", help="CSV providing uuid -> category")
parser.add_argument("output_csv", help="Path to write the result")
args = parser.parse_args()

target = pd.read_csv(args.target_csv)
lookup = pd.read_csv(args.lookup_csv)

for name, dfx in [("target", target), ("lookup", lookup)]:
    if "uuid" not in dfx.columns:
        raise ValueError(f"{name} CSV missing 'uuid' column")
if "category" not in lookup.columns:
    raise ValueError("lookup CSV missing 'category' column")
if "category" not in target.columns:
    target["category"] = pd.NA  # create it if it doesn't exist

# Build uuid -> category map (drop dup uuids in lookup, keep first)
category_map = (
    lookup.drop_duplicates(subset="uuid")
          .set_index("uuid")["category"]
)

# Overwrite category for every matched uuid
mapped = target["uuid"].map(category_map)
matched_mask = mapped.notna()
target.loc[matched_mask, "category"] = mapped[matched_mask]

# Flag rows whose uuid was not found in the lookup
target["category_unmatched"] = ~matched_mask

target.to_csv(args.output_csv, index=False)

n_total = len(target)
n_matched = int(matched_mask.sum())
print(f"Rows: {n_total}")
print(f"Matched (category filled): {n_matched}")
print(f"Unmatched (flagged category_unmatched=True): {n_total - n_matched}")
print(f"Written to {args.output_csv}")