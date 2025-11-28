import os
import cv2
import glob
from tqdm import tqdm
from .mask_generator import MaskGenerator
from .bbox_detector import BBoxDetector
from .label_manager import LabelManager
from .utils_image import apply_white_background, replace_license_plate

class DatasetBuilder:
    def __init__(self, raw_data_dir, output_dir):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.processed_dir = os.path.join(output_dir, "processed")
        self.annotations_dir = os.path.join(output_dir, "annotations")
        
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        
        # Initialize components
        self.mask_generator = MaskGenerator()
        self.bbox_detector = BBoxDetector()
        self.label_manager = LabelManager(self.annotations_dir)

    def run(self):
        image_paths = glob.glob(os.path.join(self.raw_data_dir, "*.*"))
        # Filter for images
        image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(image_paths)} images to process.")
        
        for image_path in tqdm(image_paths):
            self.process_image(image_path)

    def process_image(self, image_path):
        filename = os.path.basename(image_path)
        print(f"Processing {filename}...")
        
        # 1. Read Image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading {image_path}")
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Detect BBox (Car)
        car_bbox = self.bbox_detector.detect_car(image_rgb)
        if car_bbox is None:
            print(f"No car detected in {filename}")
            return
            
        # 3. Generate Mask
        mask = self.mask_generator.generate_mask(image_rgb, bbox=car_bbox)
        
        # 4. Apply White Background
        processed_img = apply_white_background(image_rgb, mask)
        
        # 5. Detect & Replace License Plate (Optional/Placeholder)
        # Assuming we might want to detect plate here or use the car bbox logic if we had a plate detector
        # For now, let's skip plate replacement or use a dummy logic if we had a plate bbox.
        # Since BBoxDetector.detect_license_plate returns None, this effectively does nothing currently.
        plate_bbox = self.bbox_detector.detect_license_plate(image_rgb)
        if plate_bbox is not None:
            processed_img = replace_license_plate(processed_img, plate_bbox)
            
        # 6. Save Processed Image
        save_path = os.path.join(self.processed_dir, filename)
        # Convert back to BGR for OpenCV saving
        cv2.imwrite(save_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
        
        # 7. Save Annotations
        # We can save the mask path relative to annotations or absolute
        mask_filename = f"{os.path.splitext(filename)[0]}_mask.png"
        mask_path = os.path.join(self.annotations_dir, mask_filename)
        cv2.imwrite(mask_path, mask)
        
        self.label_manager.save_annotation(
            image_name=filename,
            bbox=car_bbox,
            mask_path=mask_filename,
            labels={"processed": True}
        )
