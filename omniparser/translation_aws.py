import argparse
import boto3
import pandas as pd
from botocore.exceptions import ClientError
from langdetect import detect

DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_REGION   = "us-east-1"


def parse_args():
    p = argparse.ArgumentParser(
        description="Translate non-English captions to English via Bedrock (Claude), "
                    "writing a content_en column."
    )
    p.add_argument("--input", required=True, help="captions_flat.csv")
    p.add_argument("--output", required=True, help="captions_translated.csv output path")
    p.add_argument("--max-spend", type=float, default=5.00,
                   help="stop once estimated spend reaches this many USD (default 5.00)")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--region", default=DEFAULT_REGION)
    return p.parse_args()


args = parse_args()
INPUT_FILE  = args.input
OUTPUT_FILE = args.output
MODEL_ID    = args.model_id
REGION      = args.region
MAX_SPEND   = args.max_spend

df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} rows")

client = boto3.client("bedrock-runtime", region_name=REGION)

def translate_caption(caption, element_type):
    conversation = [
        {
            "role": "user",
            "content": [{"text": f"""Translate the following image caption to English. Output ONLY the translated text, with no preamble, quotes, explanations, or notes.
        Rules:
        - If the caption is already in English, output it unchanged.
        - If the caption is unintelligible or gibberish, output the origin caption exactly.
        - Do not add commentary, language labels, or context.
        - Preserve emojis and proper nouns as-is.
        Caption: {caption}"""}]
            }
    ]
    try:
        response = client.converse(
            modelId=MODEL_ID,
            messages=conversation,
            inferenceConfig={"maxTokens": 256, "temperature": 0.5}
        )
        return response["output"]["message"]["content"][0]["text"].strip()
    except (ClientError, Exception) as e:
        print(f"ERROR: {e}")
        return None

def is_english(text):
    try:
        return detect(str(text)) == "en"
    except:
        return False

mask = ~df["content"].apply(is_english)
non_english_df = df[mask].copy()
print(f"Non-English rows: {len(non_english_df)} / {len(df)}")

results = {}
running_cost = 0.0

for i, (idx, row) in enumerate(non_english_df.iterrows()):
    result = translate_caption(row["content"], row["element_type"])
    results[idx] = result if result is not None else row["content"]

    running_cost += (len(results[idx]) / 4 / 1_000_000) * (0.25 + 1.25)

    if (i + 1) % 100 == 0:
        print(f"{i + 1} / {len(non_english_df)} done | Est. spend: ${running_cost:.4f}")

    if running_cost >= MAX_SPEND:
        print(f"Cost limit hit (${running_cost:.2f}) — stopping early")
        break

non_english_df["content_en"] = pd.Series(results)
df["content_en"] = df["content"]
df.update(non_english_df[["content_en"]])

df.to_csv(OUTPUT_FILE, index=False)
print(f" Saved to {OUTPUT_FILE} | Total est. spend: ${running_cost:.4f}")