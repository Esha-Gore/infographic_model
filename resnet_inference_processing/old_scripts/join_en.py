import json
import pandas as pd
import re

with open('/gscratch/scrubbed/eshagore/omniparser_dataset/dataset_results/joined.json') as f:
    joined = json.load(f)

translated = pd.read_csv('/gscratch/scrubbed/eshagore/omniparser_dataset/captions_translated.csv')

def clean_translation(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'^.*?:\s*', '', text, count=1, flags=re.IGNORECASE)
    return text.strip()

translated['content_en'] = translated['content_en'].apply(clean_translation)
icon_mask = translated['element_type'] == 'icon'
translated.loc[icon_mask, 'element_index'] = translated[icon_mask].groupby('uuid').cumcount()
translated_icons = translated[icon_mask]
lookup = {(row['uuid'], int(row['element_index'])): row['content_en']
          for _, row in translated_icons.iterrows()}

for item in joined:
    key = (item['uuid'], item['element_index'])
    item['content_en'] = lookup.get(key, '')
    item['combined_text_en'] = f"{item['pred_class']}: {item['content_en']}" if item['content_en'] else item['combined_text']

with open('/gscratch/scrubbed/eshagore/omniparser_dataset/dataset_results/joined_en.json', 'w') as f:
    json.dump(joined, f, indent=2)

df_out = pd.DataFrame(joined)
df_out.to_csv('/gscratch/scrubbed/eshagore/omniparser_dataset/dataset_results/joined_en.csv', index=False)

print(f"Done, saved {len(joined)} entries")
for item in joined[:3]:
    print(item['combined_text_en'])
