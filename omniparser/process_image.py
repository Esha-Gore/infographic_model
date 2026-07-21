import sys
sys.path.append('.')
from util.utils import get_yolo_model, get_caption_model_processor, get_som_labeled_img
from PIL import Image
import os
import json
import base64
import io
import argparse


def parse_args():
    p = argparse.ArgumentParser(
        description="Run OmniParser (YOLO detector + Florence-2 captioner) on a SINGLE image, "
                    "writing all_detections.json + an annotated image."
    )
    p.add_argument("--image", required=True, help="path to one screenshot")
    p.add_argument("--uuid", required=True, help="id for this image (join key / crop prefix)")
    p.add_argument("--country", required=True, help="country tag for this image")
    p.add_argument("--category", required=True, help="category tag for this image")

    p.add_argument("--output-dir", required=True, help="where to write all_detections.json + annotated img")
    p.add_argument("--yolo-weights", required=True, help="path to icon_detect model.pt")
    p.add_argument("--florence-weights", required=True, help="path to icon_caption_florence dir")
    p.add_argument("--device", default="cuda", help="torch device for the captioner (default cuda)")
    return p.parse_args()


args = parse_args()
OUTPUT_FOLDER = args.output_dir

print("Loading models...")
yolo_model = get_yolo_model(model_path=args.yolo_weights)
caption_model_processor = get_caption_model_processor(
    model_name='florence2',
    model_name_or_path=args.florence_weights,
    device=args.device
)
print("Models loaded successfully\n")


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


os.makedirs(OUTPUT_FOLDER, exist_ok=True)

img_uuid = args.uuid
country = args.country
img_file = os.path.basename(args.image)
img_path = args.image

country_output = os.path.join(OUTPUT_FOLDER, country)
os.makedirs(country_output, exist_ok=True)

error_log = open(f'{OUTPUT_FOLDER}/errors.log', 'w')
error_log.write('Error Log\n\n')

all_results = []

print(f'Processing: {img_file} (uuid={img_uuid}, country={country})')

if not os.path.exists(img_path):
    sys.exit(f"image not found: {img_path}")

try:
    if os.path.getsize(img_path) == 0:
        sys.exit(f"image is empty (0 bytes): {img_path}")

    try:
        image = Image.open(img_path)
        image.verify()
        image = Image.open(img_path)
    except Exception as img_error:
        sys.exit(f"corrupted image {img_path}: {img_error}")

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
    annotated_image.save(os.path.join(country_output, f'annotated_{img_file}'))

    processed_elements = [safe_get_element_data(elem) for elem in filtered_boxes_elem]
    num_elements = len(processed_elements)
    num_text = sum(1 for elem in processed_elements if elem['type'] == 'text')
    num_icons = sum(1 for elem in processed_elements if elem['type'] == 'icon')

    print(f'  OK: {img_file} - {num_elements} elements ({num_text} text, {num_icons} icons)')

    all_results.append({
        'country': country,
        'category': args.category,
        'image': img_file,
        'uuid': img_uuid,
        'num_elements': num_elements,
        'num_text': num_text,
        'num_icons': num_icons,
        'elements': processed_elements
    })

except Exception as e:
    print(f'  ERROR: {img_file} - {str(e)[:100]}')
    error_log.write(f'{img_file}: {str(e)}\n')

error_log.close()

json_path = os.path.join(OUTPUT_FOLDER, 'all_detections.json')
with open(json_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\nResults saved to: {json_path}')
