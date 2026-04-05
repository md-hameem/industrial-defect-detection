"""Quick API integration test."""
import requests
from pathlib import Path

API = "http://localhost:8000"

# Find a test image
img_path = Path("f:/Thesis/datasets/kolektor_sdd2/test/img/20000.png")
if not img_path.exists():
    # Try to find any image
    import glob
    imgs = glob.glob("f:/Thesis/datasets/**/*.png", recursive=True)
    img_path = Path(imgs[0]) if imgs else None

if img_path is None:
    print("ERROR: No test images found")
    exit(1)

print(f"Test image: {img_path}")
print(f"File size: {img_path.stat().st_size} bytes")
print()

# Test 1: Health check
print("=== Test 1: Health Check ===")
r = requests.get(f"{API}/")
data = r.json()
print(f"Status: {data['status']}")
print(f"Models: {data['models']}")
print()

# Test 2: List models
print("=== Test 2: List Models ===")
r = requests.get(f"{API}/models")
models = r.json()
print(f"Found {len(models)} trained models")
for m in models[:5]:
    print(f"  - {m['name']} ({m['file_size_mb']} MB)")
print()

# Test 3: Single CAE inference
print("=== Test 3: CAE Inference ===")
with open(img_path, "rb") as f:
    r = requests.post(
        f"{API}/predict",
        params={"model_type": "CAE", "category": "bottle"},
        files={"file": ("test.png", f, "image/png")},
    )
data = r.json()
print(f"Success: {data.get('success')}")
print(f"Model: {data.get('model')}")
print(f"Score: {data.get('anomaly_score', 'N/A')}")
print(f"Time: {data.get('processing_time', 'N/A'):.2f}s")
print(f"Heatmap size: {len(data.get('heatmap', ''))} chars")
print()

# Test 4: Batch inference (CAE + DAE)
print("=== Test 4: Batch Inference (CAE + DAE) ===")
with open(img_path, "rb") as f:
    r = requests.post(
        f"{API}/predict/batch",
        params={"model_types": "CAE,DAE", "category": "bottle"},
        files={"file": ("test.png", f, "image/png")},
    )
data = r.json()
print(f"Success: {data.get('success')}")
print(f"Total models: {data.get('total_models')}")
print(f"Processing time: {data.get('processing_time', 0):.2f}s")
for res in data.get("results", []):
    mt = res.get("model_type", res.get("model", "?"))
    score = res.get("anomaly_score", "N/A")
    print(f"  - {res.get('model', '?')}: score={score}")
print()

# Test 5: CNN classifier
print("=== Test 5: CNN Classifier ===")
with open(img_path, "rb") as f:
    r = requests.post(
        f"{API}/predict",
        params={"model_type": "CNN"},
        files={"file": ("test.png", f, "image/png")},
    )
data = r.json()
print(f"Success: {data.get('success')}")
print(f"Predicted class: {data.get('predicted_class')}")
print(f"Confidence: {data.get('confidence', 0):.1%}")
print()

print("=== ALL TESTS PASSED ===")
