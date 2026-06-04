import pandas as pd
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import numpy as np
import json

with open("bert_config.json") as f:
    config = json.load(f)

run_name = config["run_name"]
data_csv = config["input"]
out_dir = run_name

df = pd.read_csv(data_csv)
docs = df["combined_text_en"].tolist()
tokenized_docs = [doc.lower().split() for doc in docs]
dictionary = Dictionary(tokenized_docs)
print(f"Loaded {len(docs)} documents")

def evaluate_model(model_path, model_name):
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*50}")

    model = BERTopic.load(model_path)
    info = model.get_topic_info()

    n_topics = len(info[info.Topic != -1])
    total = info['Count'].sum()
    print(f"\n--- Basic Info ---")
    print(f"Number of topics: {n_topics}")
    print(f"Total docs:       {total}")

    noise_count = info[info.Topic == -1]['Count'].values[0]
    noise_ratio = noise_count / total
    print(f"\n--- Noise ---")
    print(f"Noise docs:  {noise_count}")
    print(f"Noise ratio: {noise_ratio:.2%}")


    real_topics = info[info.Topic != -1]['Count']
    print(f"\n--- Topic Size Distribution ---")
    print(f"Largest topic:  {real_topics.max()} docs")
    print(f"Smallest topic: {real_topics.min()} docs")
    print(f"Average topic:  {real_topics.mean():.1f} docs")
    print(f"Median topic:   {real_topics.median():.1f} docs")

    print(f"\n--- Coherence ---")

    try:
        topics_words = [
            [word for word, _ in model.get_topic(t)
            if word in dictionary.token2id and word != '']
            for t in info.Topic
            if t != -1
        ]

        topics_words = [t for t in topics_words if len(t) > 0]

        coherence_model = CoherenceModel(
            topics=topics_words,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v'
        )

        coherence_score = coherence_model.get_coherence()
        print(f"Coherence (c_v): {coherence_score:.4f}")
    except Exception as e:
        print(f"Coherence failed: {e}")
        coherence_score = None
    
    return {
        'run_name': model_name,
        'n_topics': n_topics,
        'noise_ratio': f"{noise_ratio:.2%}",
        'largest_topic': real_topics.max(),
        'smallest_topic': real_topics.min(),
        'avg_topic_size': round(real_topics.mean(), 1),
        'median_topic_size': round(real_topics.median(), 1),
        'coherence': round(coherence_score, 4) if coherence_score else None
    }

result = evaluate_model(
            f"{out_dir}/bertopic_model_{run_name}",
                run_name
                )

print(f"\n{'='*50}")
print("RESULTS")
print(f"{'='*50}")
summary = pd.DataFrame([result])
print(summary.to_string(index=False))

out_path = f"{out_dir}/{run_name}_metrics.csv"
summary.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
