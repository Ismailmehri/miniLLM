# Car Preprocessing Project

This project provides a pipeline to preprocess car images for downstream tasks. It uses **SAM (Segment Anything Model)** for segmentation and **YOLO (Ultralytics)** for bounding box detection.

## Features
- **Car Detection**: Uses YOLO to detect cars in images.
- **Segmentation**: Uses SAM to generate high-quality masks for the detected cars.
- **Background Removal**: Applies a white background to the segmented cars.
- **License Plate Replacement**: Replaces license plates with "PLANY.TN" (placeholder logic).
- **Annotation Management**: Saves bounding boxes, masks, and metadata in JSON format.

## Structure
```
projects/car_preprocessing/
 ├── data/
 │    ├── raw/              # Input images
 │    ├── processed/        # Output images (white bg + plate replacement)
 │    └── annotations/      # JSON annotations & masks
 ├── src/
 │    ├── dataset_builder/  # Core logic
 │    ├── training/         # Training scripts (placeholder)
 │    └── utils/            # Helper functions
 ├── notebooks/             # Jupyter notebooks
 ├── models/                # Model checkpoints (SAM, YOLO)
 └── configs/               # Configuration files
```

## Setup
1. Install dependencies:
   ```bash
   pip install -r ../../requirements.txt
   ```
2. Download SAM checkpoint (e.g., `sam_vit_h_4b8939.pth`) and place it in `models/` or update the path in `mask_generator.py`.
3. Ensure you have `yolov8n.pt` or let Ultralytics download it automatically.

## Usage

### Python Script
```python
from dataset_builder.dataset_builder import DatasetBuilder

# Initialize builder
builder = DatasetBuilder(
    raw_data_dir="data/raw",
    output_dir="data"
)

# Run pipeline
builder.run()
```

### Notebook
Open `notebooks/prepare_dataset.ipynb` and follow the steps.
