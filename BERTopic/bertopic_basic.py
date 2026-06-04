import pandas as pd
from bertopic import BERTopic
import json
import os

with open("bert_config.json") as f:
    config = json.load(f)

name = config["run_name"]
data_csv = config["input"]
topics =  config.get("nr_topics", None)
# Create output directory
out_dir = name
os.makedirs(out_dir, exist_ok=True)

#df = pd.read_csv("/gscratch/scrubbed/eshagore/omniparser_dataset/dataset_results/joined_en.csv")
df = pd.read_csv(data_csv)
docs = df["combined_text_filtered"].tolist()
print(f"Running BERTopic on {len(docs)} captions...")

topic_model = BERTopic(language="english", verbose=True, nr_topics=topics)
topics, probs = topic_model.fit_transform(docs)

print(topic_model.get_topic_info())

# Save CSV
df["topic"] = topics
df.to_csv(f"{out_dir}/{name}_captions_with_topics.csv", index=False)
print("CSV Saved :)")

# Save model
topic_model.save(f"{out_dir}/bertopic_model_{name}", serialization="pickle")
print("Model saved :)")

# Save visualizations
topic_model.visualize_topics().write_html(f"{out_dir}/{name}_viz_topics.html")
topic_model.visualize_barchart(top_n_topics=10).write_html(f"{out_dir}/{name}_viz_barchart.html")
topic_model.visualize_heatmap().write_html(f"{out_dir}/{name}_viz_heatmap.html")
print("Visualizations saved :)")


