"""
Quick verification script for data loaders.
Run this to verify all datasets load correctly.
"""

import sys
sys.path.insert(0, 'F:/Thesis')

from src.config import ensure_dirs, MVTEC_CATEGORIES, NEU_CATEGORIES
from src.data import MVTecDataset, KolektorDataset, NEUDataset

def test_mvtec():
    """Test MVTec dataset loading."""
    print("=" * 50)
    print("Testing MVTec AD Dataset")
    print("=" * 50)
    
    # Test one category
    category = 'bottle'
    print(f"\nLoading {category} category...")
    
    train_dataset = MVTecDataset(category=category, split='train')
    test_dataset = MVTecDataset(category=category, split='test', return_mask=True)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Test loading a sample
    img, label = train_dataset[0]
    print(f"  Train image shape: {img.shape}")
    print(f"  Train label: {label}")
    
    img, mask, label = test_dataset[0]
    print(f"  Test image shape: {img.shape}")
    print(f"  Test mask shape: {mask.shape}")
    print(f"  Test label: {label}")
    
    print(f"\n✅ MVTec AD dataset loading successful!")
    return True


def test_kolektor():
    """Test Kolektor dataset loading."""
    print("\n" + "=" * 50)
    print("Testing KolektorSDD2 Dataset")
    print("=" * 50)
    
    train_dataset = KolektorDataset(split='train')
    test_dataset = KolektorDataset(split='test')
    
    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Test loading a sample
    img, mask, label = train_dataset[0]
    print(f"  Train image shape: {img.shape}")
    print(f"  Train mask shape: {mask.shape}")
    print(f"  Train label: {label}")
    
    print(f"\n✅ KolektorSDD2 dataset loading successful!")
    return True


def test_neu():
    """Test NEU dataset loading."""
    print("\n" + "=" * 50)
    print("Testing NEU Surface Defect Dataset")
    print("=" * 50)
    
    train_dataset = NEUDataset(split='train')
    val_dataset = NEUDataset(split='validation')
    
    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Number of classes: {train_dataset.get_num_classes()}")
    print(f"  Categories: {NEU_CATEGORIES}")
    
    # Count samples per category
    from collections import Counter
    category_counts = Counter(train_dataset.category_names)
    print(f"  Samples per category (train):")
    for cat, count in sorted(category_counts.items()):
        print(f"    - {cat}: {count}")
    
    # Test loading a sample
    img, label = train_dataset[0]
    print(f"\n  Train image shape: {img.shape}")
    print(f"  Train label: {label} ({train_dataset.get_category_name(label)})")
    
    print(f"\n✅ NEU Surface Defect dataset loading successful!")
    return True


def main():
    """Run all dataset tests."""
    print("\n" + "#" * 60)
    print("# Industrial Defect Detection - Data Loader Verification")
    print("#" * 60)
    
    # Ensure output directories exist
    ensure_dirs()
    
    all_passed = True
    
    try:
        all_passed &= test_mvtec()
    except Exception as e:
        print(f"\n❌ MVTec test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_kolektor()
    except Exception as e:
        print(f"\n❌ Kolektor test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_neu()
    except Exception as e:
        print(f"\n❌ NEU test failed: {e}")
        all_passed = False
    
    print("\n" + "#" * 60)
    if all_passed:
        print("# ALL TESTS PASSED! ✅")
    else:
        print("# SOME TESTS FAILED ❌")
    print("#" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
