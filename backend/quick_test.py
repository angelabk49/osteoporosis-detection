"""Quick API test with held-out images"""
import os
import requests

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend', 'FOR TESTING', 'testcases')

url = 'http://127.0.0.1:8000/predict'

tests = [
    {
        "name": "OS34.jpg (Osteoporosis, 80yr male)",
        "path": os.path.join(TEST_DIR, "OS34.jpg"),
        "age": 80, "sex": "male", "vitamin_deficiency": "false"
    },
    {
        "name": "OP100.jpg (Osteopenia, 65yr female, vit def)",
        "path": os.path.join(TEST_DIR, "OP100.jpg"),
        "age": 65, "sex": "female", "vitamin_deficiency": "true"
    },
    {
        "name": "N34.jpg (Normal, 30yr male)",
        "path": os.path.join(TEST_DIR, "N34.jpg"),
        "age": 30, "sex": "male", "vitamin_deficiency": "false"
    },
]

for i, t in enumerate(tests, 1):
    with open(t["path"], "rb") as f:
        resp = requests.post(url, data={
            "age": t["age"],
            "sex": t["sex"],
            "vitamin_deficiency": t["vitamin_deficiency"]
        }, files={"xray": (t["path"].split("\\")[-1], f, "image/jpeg")})

    r = resp.json()
    print(f"=== Test {i}: {t['name']} ===")
    print(f"  Prediction:  {r['prediction']}")
    print(f"  Confidence:  {r['confidence']:.1%}")
    print(f"  Urgency:     {r['urgency']}")
    print(f"  Message:     {r['message']}")
    print(f"  Probs:       N={r['class_probabilities']['normal']:.3f}  "
          f"OP={r['class_probabilities']['osteopenia']:.3f}  "
          f"OS={r['class_probabilities']['osteoporosis']:.3f}")
    print()
