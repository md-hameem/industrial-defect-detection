# Datasets

This directory contains the datasets used for industrial defect detection research.

## 📁 Dataset Structure

```
datasets/
├── mvtec_ad/                   # MVTec Anomaly Detection Dataset
│   ├── bottle/
│   ├── cable/
│   ├── capsule/
│   ├── carpet/
│   ├── grid/
│   ├── hazelnut/
│   ├── leather/
│   ├── metal_nut/
│   ├── pill/
│   ├── screw/
│   ├── tile/
│   ├── toothbrush/
│   ├── transistor/
│   ├── wood/
│   └── zipper/
├── kolektor_sdd2/              # Kolektor Surface Defect Dataset 2
│   ├── train/
│   │   ├── img/
│   │   └── ann/
│   └── test/
│       ├── img/
│       └── ann/
└── neu_surface_defect/         # NEU Surface Defect Database
    ├── train/
    └── validation/
```

## 📥 Download Links

### 1. MVTec AD (Primary Benchmark)
- **URL**: https://www.mvtec.com/company/research/datasets/mvtec-ad
- **Size**: ~5GB
- **Categories**: 15 industrial objects and textures
- **Usage**: Unsupervised anomaly detection (CAE, VAE, DAE)

### 2. KolektorSDD2 (Generalization Testing)
- **URL**: https://www.vicos.si/resources/kolektorsdd2/
- **Size**: ~1GB
- **Description**: Real-world electrical commutator defects
- **Usage**: Cross-dataset evaluation
- **Note**: Annotations are JSON format (Supervisely)

### 3. NEU Surface Defect (Supervised Baseline)
- **URL**: http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/
- **Size**: ~200MB
- **Classes**: 6 (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches)
- **Usage**: Supervised CNN classification

## 📊 Dataset Statistics

| Dataset | Train | Test | Normal | Defect | Categories |
|---------|-------|------|--------|--------|------------|
| MVTec AD | 3,629 | 1,725 | 5,354 | 1,258 | 15 |
| KolektorSDD2 | 2,335 | 1,004 | 2,229 | 1,110 | 1 |
| NEU Surface | 1,440 | 360 | - | 1,800 | 6 |

## ⚙️ Usage

Data loaders are implemented in `src/data/`:

```python
# MVTec AD
from src.data import MVTecDataset, create_mvtec_dataloaders
train_loader, test_loader = create_mvtec_dataloaders('bottle', batch_size=16)

# Kolektor
from src.data import KolektorDataset
kolektor_test = KolektorDataset(split='test', return_mask=True)

# NEU
from src.data import NEUDataset
neu_train = NEUDataset(split='train')
```

## 📝 Notes

- All images are resized to 256×256
- Normalization uses ImageNet statistics
- MVTec and Kolektor include pixel-level masks
- NEU is classification only (no pixel masks)
