#!/usr/bin/env python3
"""
Download and prepare the Chest X-Ray Pneumonia dataset.

Usage:
    python download_dataset.py

This script provides two methods:
  1. Kaggle API (requires kaggle credentials)
  2. Manual download instructions

After download, verifies directory structure is correct.
"""
import os
import sys
import zipfile
import shutil

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chest_xray')
EXPECTED_STRUCTURE = {
    'train': ['NORMAL', 'PNEUMONIA'],
    'val':   ['NORMAL', 'PNEUMONIA'],
    'test':  ['NORMAL', 'PNEUMONIA'],
}


def download_kaggle():
    """Download via Kaggle API."""
    try:
        import kaggle
        print('[*] Downloading from Kaggle...')
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'paultimothymooney/chest-xray-pneumonia',
            path='.', unzip=False,
        )
        zip_path = 'chest-xray-pneumonia.zip'
        if os.path.exists(zip_path):
            print('[*] Extracting...')
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall('.')
            os.remove(zip_path)
            # Handle nested directory structure
            nested = os.path.join('chest_xray', 'chest_xray')
            if os.path.exists(nested):
                for item in os.listdir(nested):
                    src = os.path.join(nested, item)
                    dst = os.path.join('chest_xray', item)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                shutil.rmtree(nested)
            return True
        else:
            print('[!] Download file not found.')
            return False
    except Exception as e:
        print(f'[!] Kaggle download failed: {e}')
        return False


def verify_structure():
    """Verify the dataset directory structure is correct."""
    if not os.path.isdir(DATA_DIR):
        return False

    all_ok = True
    total_images = 0
    print(f'\n{"Split":<10} {"Class":<12} {"Count":>6}')
    print('-' * 32)

    for split, classes in EXPECTED_STRUCTURE.items():
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.isdir(split_dir):
            print(f'  [MISSING] {split}/')
            all_ok = False
            continue
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                print(f'  [MISSING] {split}/{cls}/')
                all_ok = False
                continue
            count = len([f for f in os.listdir(cls_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            total_images += count
            print(f'{split:<10} {cls:<12} {count:>6}')

    print('-' * 32)
    print(f'{"Total":<23} {total_images:>6}')
    return all_ok and total_images > 0


def main():
    print('=' * 50)
    print('  Chest X-Ray Pneumonia Dataset Setup')
    print('=' * 50)

    if verify_structure():
        print('\n[OK] Dataset already exists and looks correct!')
        return

    print('\n[*] Dataset not found. Attempting download...\n')

    # Method 1: Kaggle API
    print('--- Method 1: Kaggle API ---')
    if download_kaggle():
        if verify_structure():
            print('\n[OK] Dataset downloaded and verified!')
            return

    # Method 2: Manual instructions
    print('\n--- Manual Download Instructions ---')
    print('''
1. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Click "Download" (you may need a Kaggle account)
3. Extract the zip file into this project directory
4. Ensure the following structure exists:

   chest_xray/
   ├── train/
   │   ├── NORMAL/       (~1,341 images)
   │   └── PNEUMONIA/    (~3,875 images)
   ├── val/
   │   ├── NORMAL/       (8 images)
   │   └── PNEUMONIA/    (8 images)
   └── test/
       ├── NORMAL/       (~234 images)
       └── PNEUMONIA/    (~390 images)

5. Run this script again to verify: python download_dataset.py
''')

    # Alternative: Kaggle CLI setup
    print('--- Alternative: Setup Kaggle CLI ---')
    print('''
pip install kaggle
# Place your kaggle.json in ~/.kaggle/kaggle.json
# Get it from: https://www.kaggle.com/settings -> API -> Create New Token
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip
''')


if __name__ == '__main__':
    main()
