# Datasets

This directory contains the datasets used for industrial defect detection research.

## Required Datasets

### 1. MVTec AD
- **Download**: https://www.mvtec.com/company/research/datasets/mvtec-ad
- **Structure**: Extract to `datasets/mvtec_ad/`
- **Categories**: 15 (bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper)

### 2. KolektorSDD2
- **Download**: https://www.vicos.si/resources/kolektorsdd2/
- **Structure**: Extract to `datasets/kolektor_sdd2/`
- **Format**: train/test splits with img/ann folders

### 3. NEU Surface Defect Database
- **Download**: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
- **Structure**: Extract to `datasets/neu_surface_defect/`
- **Categories**: 6 (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches)

## Directory Structure

```
datasets/
├── mvtec_ad/
│   ├── bottle/
│   ├── cable/
│   └── ... (15 categories)
├── kolektor_sdd2/
│   ├── train/
│   └── test/
└── neu_surface_defect/
    ├── train/
    └── validation/
```

## Note

Datasets are excluded from git due to their large size. Download them separately and place them in this directory.
