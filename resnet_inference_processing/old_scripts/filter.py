import json
import pandas as pd

CONFIDENCE_THRESHOLD = 0.8

with open('/gscratch/scrubbed/eshagore/omniparser_dataset/dataset_results/joined_en.json') as f:
    joined = json.load(f)

for item in joined:
    content = item.get('content_en') or item.get('content', '')
    if item.get('confidence', 0) >= CONFIDENCE_THRESHOLD:
        item['combined_text_filtered'] = f"{item['pred_class']}: {content}"
    else:
        item['combined_text_filtered'] = content

with open('/gscratch/scrubbed/eshagore/omniparser_dataset/dataset_results/joined_filtered.json', 'w') as f:
    json.dump(joined, f, indent=2)

df = pd.DataFrame(joined)
df.to_csv('/gscratch/scrubbed/eshagore/omniparser_dataset/dataset_results/joined_filtered.csv', index=False)

print(f"Done. {len(joined)} entries saved.")
print(f"Above threshold: {sum(1 for i in joined if i.get('confidence', 0) >= CONFIDENCE_THRESHOLD)}")
print(f"Below threshold: {sum(1 for i in joined if i.get('confidence', 0) < CONFIDENCE_THRESHOLD)}")
