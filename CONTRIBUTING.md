# Contributing

Thank you for your interest in contributing to the Industrial Defect Detection project!

## 🚀 Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/md-hameem/industrial-defect-detection.git
cd industrial-defect-detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Download Datasets
See [`datasets/README.md`](datasets/README.md) for download links and setup instructions.

## 📁 Project Structure

```
src/
├── config.py       # Configuration settings
├── data/           # Dataset loaders
├── models/         # Model architectures (CAE, VAE, DAE, CNN)
├── training/       # Training utilities
└── evaluation/     # Metrics & visualization

notebooks/          # Jupyter notebooks for training/analysis
outputs/            # Generated models, logs, figures
tests/              # Unit tests
```

## 🎨 Code Style

- Follow **PEP 8** guidelines
- Use **type hints** for function signatures
- Write **docstrings** for all public functions
- Keep lines under **100 characters**
- Use **snake_case** for variables and functions
- Use **PascalCase** for classes

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_loading.py -v
```

## 📓 Working with Notebooks

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Notebook order:
   - `00_data_exploration.ipynb` - Explore datasets
   - `01_train_cae.ipynb` - Train CAE models
   - `02_train_vae.ipynb` - Train VAE models
   - `03_train_denoising_ae.ipynb` - Train Denoising AE
   - `04_train_cnn_classifier.ipynb` - Train CNN classifier
   - `05_analysis_visualization.ipynb` - Compare models
   - `06_cross_dataset_evaluation.ipynb` - Generalization testing
   - `07_thesis_figures.ipynb` - Generate thesis figures

## 🔧 Pull Request Process

1. **Fork** the repository
2. Create a **feature branch**: `git checkout -b feature/your-feature`
3. Make your changes
4. **Run tests**: `pytest tests/ -v`
5. **Commit** with clear messages: `git commit -m "Add: feature description"`
6. **Push**: `git push origin feature/your-feature`
7. Open a **Pull Request**

## 📝 Commit Message Format

Use clear, descriptive commit messages:
- `Add: new feature description`
- `Fix: bug description`
- `Update: changed component`
- `Docs: documentation update`
- `Refactor: code improvement`

## ❓ Questions?

- Open an [issue](https://github.com/md-hameem/industrial-defect-detection/issues) for bugs or feature requests
- Check existing issues before creating a new one

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.
