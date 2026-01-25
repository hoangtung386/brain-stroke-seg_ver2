import numpy as np
import pydicom
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os

def explore_cpaisd_dataset(dataset_path):
    """Khám phá dataset CPAISD"""
    print(f"Exploring dataset at: {dataset_path}")
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"ERROR: Path {dataset_path} does not exist!")
        return
        
    # Đọc metadata tổng quan
    try:
        with open(dataset_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        print("="*60)
        print("DATASET METADATA")
        print("="*60)
        print(json.dumps(metadata, indent=2))
    except Exception as e:
        print(f"Could not read metadata.json: {e}")
    
    # Lấy một sample để phân tích
    try:
        train_path = dataset_path / "train"
        if not train_path.exists():
             print(f"Train path {train_path} does not exist. Listing root:")
             print([x.name for x in dataset_path.iterdir()])
             return

        sample_study = list(train_path.iterdir())[0]
        sample_slice = list(sample_study.iterdir())[0]
        
        print(f"\n{'='*60}")
        print("ANALYZING SAMPLE SLICE")
        print(f"{'='*60}")
        print(f"Study: {sample_study.name}")
        print(f"Slice: {sample_slice.name}")
        
        # 1. Kiểm tra DICOM raw
        dcm_path = sample_slice / "raw.dcm"
        if dcm_path.exists():
            dcm = pydicom.dcmread(dcm_path)
            
            print(f"\n--- DICOM Information ---")
            print(f"Modality: {dcm.Modality}")
            print(f"Manufacturer: {dcm.get('Manufacturer', 'Unknown')}")
            print(f"Slice Thickness: {dcm.get('SliceThickness', 'Unknown')} mm")
            print(f"Pixel Spacing: {dcm.get('PixelSpacing', 'Unknown')}")
            print(f"Rows x Columns: {dcm.Rows} x {dcm.Columns}")
            print(f"Bits Stored: {dcm.BitsStored}")
            
            # HU window information (QUAN TRỌNG!)
            print(f"\n--- HU Window Info ---")
            print(f"Rescale Intercept: {dcm.get('RescaleIntercept', 'N/A')}")
            print(f"Rescale Slope: {dcm.get('RescaleSlope', 'N/A')}")
            print(f"Window Center: {dcm.get('WindowCenter', 'N/A')}")
            print(f"Window Width: {dcm.get('WindowWidth', 'N/A')}")
            
            # Pixel array
            pixel_array = dcm.pixel_array
            hu_array = pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
            
            print(f"\n--- Pixel Statistics ---")
            print(f"Raw Pixel Range: [{pixel_array.min()}, {pixel_array.max()}]")
            print(f"HU Range: [{hu_array.min():.1f}, {hu_array.max():.1f}]")
            print(f"HU Mean: {hu_array.mean():.1f}")
            print(f"HU Std: {hu_array.std():.1f}")
        else:
            print(f"No raw.dcm found at {dcm_path}")

        # 2. Kiểm tra NPZ image
        if (sample_slice / "image.npz").exists():
            image_npz = np.load(sample_slice / "image.npz")
            image_array = image_npz['image']
            
            print(f"\n--- NPZ Image ---")
            print(f"Shape: {image_array.shape}")
            print(f"Dtype: {image_array.dtype}")
            print(f"Range: [{image_array.min()}, {image_array.max()}]")
            print(f"Mean: {image_array.mean():.3f}")
            print(f"Std: {image_array.std():.3f}")
        else:
             print("No image.npz found")
        
        # 3. Kiểm tra Mask
        if (sample_slice / "mask.npz").exists():
            mask_npz = np.load(sample_slice / "mask.npz")
            mask_array = mask_npz['mask']
            
            print(f"\n--- Mask Information ---")
            print(f"Shape: {mask_array.shape}")
            print(f"Dtype: {mask_array.dtype}")
            print(f"Unique values: {np.unique(mask_array)}")
            
            unique, counts = np.unique(mask_array, return_counts=True)
            print(f"\n--- Class Distribution ---")
            for val, count in zip(unique, counts):
                percentage = (count / mask_array.size) * 100
                class_name = {0: 'Background', 1: 'Core', 2: 'Penumbra'}.get(val, 'Unknown')
                print(f"Class {val} ({class_name}): {count} pixels ({percentage:.2f}%)")
        else:
            print("No mask.npz found")

    except Exception as e:
        print(f"Error analyzing sample: {e}")
        import traceback
        traceback.print_exc()

def analyze_all_masks(dataset_path):
    """Kiểm tra toàn bộ masks trong dataset"""
    print(f"\n{'='*60}")
    print("MASK ANALYSIS ACROSS ENTIRE DATASET")
    print(f"{'='*60}")
    
    dataset_path = Path(dataset_path)
    all_classes = set()
    class_counts = {0: 0, 1: 0, 2: 0}  # Background, Core, Penumbra
    
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if not split_path.exists(): continue

        print(f"Scanning {split}...")
        for study in split_path.iterdir():
            if not study.is_dir(): continue
            for slice_dir in study.iterdir():
                mask_path = slice_dir / "mask.npz"
                if mask_path.exists():
                    try:
                        mask = np.load(mask_path)['mask']
                        unique_vals = np.unique(mask)
                        all_classes.update(unique_vals.tolist())
                        
                        # Count pixels
                        for val in unique_vals:
                            count = np.sum(mask == val)
                            class_counts[val] = class_counts.get(val, 0) + count
                    except:
                        pass
    
    print(f"All unique values found: {sorted(all_classes)}")
    print(f"\nClass distribution:")
    total = sum(class_counts.values())
    if total > 0:
        for cls, count in class_counts.items():
            name = {0: 'Background', 1: 'Core', 2: 'Penumbra'}.get(cls, 'Unknown')
            print(f"  Class {cls} ({name}): {count:,} pixels ({100*count/total:.2f}%)")
    else:
        print("No masks found or empty.")

if __name__ == "__main__":
    # Try to find the dataset path
    possible_paths = [
        "C:/Users/Admin/Projects/brain-stroke-segmentation_ver2/dataset_APIS/dataset",
        "dataset_APIS/dataset",
        "dataset",
        "C:/Users/Admin/Projects/dataset_APIS/dataset"
    ]
    
    target_path = None
    for p in possible_paths:
        if os.path.exists(p) and os.path.isdir(p):
            target_path = p
            break
            
    if target_path:
        explore_cpaisd_dataset(target_path)
        # Uncomment to run full analysis (might be slow)
        # analyze_all_masks(target_path) 
    else:
        print("Could not find dataset directory in common locations.")
