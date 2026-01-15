# Contributing

Thank you for your interest in contributing to this project!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/industrial-defect-detection.git
cd industrial-defect-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

4. Download datasets (see `datasets/README.md`)

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings for all public functions
- Keep lines under 100 characters

## Running Tests

```bash
pytest tests/ -v
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests
5. Commit with clear messages
6. Push and open a PR

## Questions?

Open an issue for any questions or suggestions.
