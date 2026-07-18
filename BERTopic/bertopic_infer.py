import argparse
import pandas as pd
from bertopic import BERTopic


def parse_args():
    p = argparse.ArgumentParser(
        description="Assign topics to captions using a saved (pre-trained) BERTopic model."
    )
    p.add_argument("--input", required=True, help="joined_filtered.csv")
    p.add_argument("--model", required=True, help="saved BERTopic model (bertopic_model_reduced_600)")
    p.add_argument("--output", required=True, help="output CSV (all input columns + final_topic)")
    return p.parse_args()


args = parse_args()

df = pd.read_csv(args.input)
docs = df["combined_text_filtered"].tolist()
print(f"Loaded {len(docs)} captions")

topic_model = BERTopic.load(args.model)
print("Model loaded")

topics, probs = topic_model.transform(docs)

# Write as final_topic,  not topic
df["final_topic"] = topics
df.to_csv(args.output, index=False)
print(f"Saved {len(df)} rows with topics to {args.output}")
