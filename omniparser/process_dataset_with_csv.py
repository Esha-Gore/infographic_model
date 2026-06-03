import sys
sys.path.append('.')
from util.utils import get_yolo_model, get_caption_model_processor, get_som_labeled_img
from PIL import Image
import os
import json
import base64
import io
import pandas as pd

# ===== CONFIGURATION =====
IMAGE_FOLDER = '/gscratch/scrubbed/atgao/kaleidoscope_data'
CSV_FILE = '/gscratch//stf/eshagore/updated_images/websites_onlyvalid_top1000.csv'
OUTPUT_FOLDER = '/gscratch/scrubbed/eshagore/dataset_results_new_weights'

IMAGE_COLUMN = 'uuid'
COUNTRY_COLUMN = 'country'

FILTER_COLUMN = 'is_porn'
FILTER_VALUE = True
# =========================

print("Loading models...")
yolo_model = get_yolo_model(
    model_path='/gscratch/stf/eshagore/weights/icon_detect_florence/model.pt'
)
caption_model_processor = get_caption_model_processor(
    model_name='florence2',
    model_name_or_path='/gscratch/stf/eshagore/weights/icon_caption_florence',
    device='cuda'
)
print("Models loaded successfully\n")

# Load and filter CSV
print(f"Loading CSV from {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df)} total entries from CSV")

if FILTER_COLUMN and FILTER_VALUE is not None:
    original_count = len(df)
    df = df[df[FILTER_COLUMN] != FILTER_VALUE]
    filtered_count = original_count - len(df)
    print(f"Filtered out {filtered_count} entries where {FILTER_COLUMN}={FILTER_VALUE}")
    print(f"Processing {len(df)} remaining entries\n")
else:
    print(f"No filtering applied - processing all {len(df)} entries\n")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

total_images = 0
total_elements = 0
skipped_images = 0
error_images = 0
all_results = []

master_results = open(f'{OUTPUT_FOLDER}/all_detections.txt', 'w')
master_results.write('OmniParser Detection Results - All Countries\n\n')

error_log = open(f'{OUTPUT_FOLDER}/errors.log', 'w')
error_log.write('Error Log\n\n')

countries = df[COUNTRY_COLUMN].unique()
print(f"Found {len(countries)} countries: {', '.join(sorted(countries))}\n")

def safe_get_element_data(elem):
    """Safely extract data from element whether it's a dict or list"""
    if isinstance(elem, dict):
        return {
            'type': elem.get('type', 'icon'),
            'bbox': elem.get('bbox', []),
            'content': elem.get('content', ''),
            'interactivity': elem.get('interactivity', True)
        }
    else:
        # If it's a list/tuple, assume it's bbox coordinates
        return {
            'type': 'icon',
            'bbox': elem if isinstance(elem, (list, tuple)) else [],
            'content': '',
            'interactivity': True
        }

for country in sorted(countries):
    country_df = df[df[COUNTRY_COLUMN] == country]
    country_output = os.path.join(OUTPUT_FOLDER, country)
    os.makedirs(country_output, exist_ok=True)
    
    print(f'\nProcessing: {country}')
    print(f'Images in this country: {len(country_df)}')
    
    country_results = open(f'{country_output}/detections.txt', 'w')
    country_results.write(f'Results for {country}\n\n')
    
    master_results.write(f'\n{country}\n')
    
    country_processed = 0
    
    for idx, row in country_df.iterrows():
        img_uuid = row[IMAGE_COLUMN]
        img_file = f"{img_uuid}_1920x1080.png"
        img_path = os.path.join(IMAGE_FOLDER, img_file)
        
        if not os.path.exists(img_path):
            print(f'  SKIP: {img_file} not found')
            error_log.write(f'{img_file}: File not found\n')
            skipped_images += 1
            continue
        
        try:
            file_size = os.path.getsize(img_path)
            if file_size == 0:
                print(f'  SKIP: {img_file} is empty')
                error_log.write(f'{img_file}: Empty file (0 bytes)\n')
                skipped_images += 1
                continue
            
            try:
                image = Image.open(img_path)
                image.verify()
                image = Image.open(img_path)
            except Exception as img_error:
                print(f'  SKIP: {img_file} corrupted image')
                error_log.write(f'{img_file}: Corrupted image - {img_error}\n')
                skipped_images += 1
                continue
            
            # Run OmniParser
            encoded_image, label_coordinates, filtered_boxes_elem = get_som_labeled_img(
                image,
                yolo_model,
                ocr_bbox=[],
                ocr_text=[],
                caption_model_processor=caption_model_processor
            )
            
            # Decode and save annotated image
            image_data = base64.b64decode(encoded_image)
            annotated_image = Image.open(io.BytesIO(image_data))
            output_path = os.path.join(country_output, f'annotated_{img_file}')
            annotated_image.save(output_path)
            
            # Process elements safely
            processed_elements = [safe_get_element_data(elem) for elem in filtered_boxes_elem]
            
            num_elements = len(processed_elements)
            num_text = sum(1 for elem in processed_elements if elem['type'] == 'text')
            num_icons = sum(1 for elem in processed_elements if elem['type'] == 'icon')
            total_images += 1
            country_processed += 1
            total_elements += num_elements
            
            print(f'  OK: {img_file} - {num_elements} elements')
            
            result = {
                'country': country,
                'image': img_file,
                'uuid': img_uuid,
                'num_elements': num_elements,
                'num_text': num_text,
                'num_icons': num_icons,
                'elements': processed_elements
            }
            all_results.append(result)
            
            country_results.write(f'\nImage: {img_file}\n')
            country_results.write(f'UUID: {img_uuid}\n')
            country_results.write(f'Elements detected: {num_elements} ({num_text} text, {num_icons} icons)\n\n')
            for i, elem in enumerate(processed_elements, 1):
                country_results.write(f'  {i}. Type: {elem["type"]}\n')
                country_results.write(f'     Caption: {elem["content"]}\n')
                country_results.write(f'     BBox: {elem["bbox"]}\n')
                country_results.write('\n')
            
            master_results.write(f'  {img_file}: {num_elements} elements\n')
            
        except Exception as e:
            print(f'  ERROR: {img_file} - {str(e)[:100]}')
            error_log.write(f'{img_file}: {str(e)}\n')
            error_images += 1
            country_results.write(f'\nERROR - {img_file}: {str(e)}\n\n')
    
    country_results.close()
    print(f'{country} complete: {country_processed} images processed')

summary = f'\n{"="*60}\nSUMMARY\n{"="*60}\n'
summary += f'Total images processed: {total_images}\n'
summary += f'Total elements detected: {total_elements}\n'
summary += f'Images skipped: {skipped_images}\n'
summary += f'Images with errors: {error_images}\n'
summary += f'{"="*60}\n'

master_results.write(summary)
master_results.close()
error_log.close()

json_path = os.path.join(OUTPUT_FOLDER, 'all_detections.json')
with open(json_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(summary)
print(f'Results saved to: {OUTPUT_FOLDER}')
