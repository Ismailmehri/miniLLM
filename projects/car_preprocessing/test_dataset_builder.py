import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))

from dataset_builder.dataset_builder import DatasetBuilder

def main():
    raw_dir = os.path.abspath("data/raw")
    output_dir = os.path.abspath("data")
    
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
        print(f"Created {raw_dir}. Please put some images there.")
        return

    # Check for images
    images = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print(f"No images found in {raw_dir}. Please add some images.")
        return

    print("Initializing DatasetBuilder...")
    # Note: Ensure models/sam_vit_h_4b8939.pth exists or MaskGenerator might fail/warn
    builder = DatasetBuilder(raw_dir, output_dir)
    
    print("Running DatasetBuilder...")
    builder.run()
    print("Done!")

if __name__ == "__main__":
    main()
