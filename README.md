# Icon Detection and Clustering Model
---- Add in Blurb ----
## FLOWCHART OF DATA + FILES

```mermaid
flowchart TD
    C[/"countries.csv"/] --> P
    S[/"screenshots"/] --> P
    P["process_data.py<br/>(OmniParser)"] --> BI[/"bounded images<br/>(country folders)"/]
    P --> AD[/"all_detections.json"/]

    AD --> CR["icon_crops.py"]
    S --> CR
    CR --> CROPS[/"icon crops"/]
    CROPS --> RN["infer.py<br/>(ResNet50)"]
    RN --> PRED[/"predictions.json"/]

    AD --> CF["caption_flatten.py"]
    CF --> FLAT[/"captions_flat.csv"/]
    FLAT --> AWS["aws_translate.py"]
    AWS --> TRANS[/"captions_translated.csv"/]

    PRED --> BJ["build_joined_filtered.py"]
    TRANS --> BJ
    AD --> BJ
    BJ --> JF[/"joined_filtered.json + .csv"/]
    JF --> BT["BERTopic"]
    BT --> TOPICS[/"topic clusters"/]

    classDef script fill:#cce5cc,stroke:#5a8a5a,color:#1b3b1b;
    classDef data fill:#cce0f0,stroke:#4a7aa5,color:#143047;
    class P,CR,RN,CF,AWS,BJ,BT script;
    class C,S,BI,AD,CROPS,PRED,FLAT,TRANS,JF,TOPICS data;
```

### Omniparser
We ran Microsoft OmniParser v2 (YOLO detector + finetuned Florence-2 captioner) on each screenshot. Omniparser is a vision language model that produces per-image bounding boxes, semantic captions, and annotated images. See omniparser/README.md for setup, weights download, and the flash_attn / OCR-import patches required.

### RESNET50

We cropped each detected icon region from the source screenshots (icon_crops.py) and classifies them with a finetuned ResNet50 (infer.py). This produces per-icon class predictions.

### BERTOPIC

We used the classifications form Resnet50 and the captiosn from Omniparser as test input into BERTopic to produce Topic clusters for icons.
