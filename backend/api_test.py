"""
API Test Suite - Osteoporosis Knee X-ray Classification API
============================================================
Tests every endpoint and edge case, generates an HTML test report.

Run:
    pip install requests
    python api_test.py
"""

import requests
import json
import os
import sys
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

# Paths to test images (one per class)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test")
NORMAL_IMG = os.path.join(DATA_DIR, "normal", "N1.JPEG")
OSTEOPENIA_IMG = os.path.join(DATA_DIR, "osteopenia", "OP1.JPEG")
OSTEOPOROSIS_IMG = os.path.join(DATA_DIR, "osteoporosis", "OS1.JPEG")

results = []


def log_test(test_id, name, method, endpoint, params, status_code, response_body, expected, passed):
    """Record a test result."""
    results.append({
        "id": test_id,
        "name": name,
        "method": method,
        "endpoint": endpoint,
        "params": params,
        "status_code": status_code,
        "response": response_body,
        "expected": expected,
        "passed": passed,
    })
    icon = "PASS" if passed else "FAIL"
    print(f"  [{icon}]  Test {test_id}: {name}  [HTTP {status_code}]")


# ============================================================
# TEST 1 - GET / Health Check (Happy Path)
# ============================================================
def test_1_health_check():
    r = requests.get(f"{BASE_URL}/")
    body = r.json()
    passed = r.status_code == 200 and body.get("status") == "ok"
    log_test(1, "GET / - Health Check", "GET", "/", "None",
             r.status_code, json.dumps(body, indent=2),
             "Status 200, status='ok'", passed)


# ============================================================
# TEST 2 - POST /predict - Valid Normal X-ray
# ============================================================
def test_2_predict_normal():
    with open(NORMAL_IMG, "rb") as f:
        files = {"xray": ("N1.JPEG", f, "image/jpeg")}
        data = {"age": 45, "sex": "Male", "vitamin_deficiency": "false"}
        r = requests.post(f"{BASE_URL}/predict", data=data, files=files)

    body = r.json()
    passed = (r.status_code == 200
              and "prediction" in body
              and "confidence" in body
              and "class_probabilities" in body
              and "urgency" in body)
    log_test(2, "POST /predict - Valid Normal X-ray (Male, 45, no vitamin def)",
             "POST", "/predict",
             "age=45, sex=Male, vitamin_deficiency=false, xray=N1.JPEG",
             r.status_code, json.dumps(body, indent=2),
             "Status 200, full prediction response", passed)


# ============================================================
# TEST 3 - POST /predict - Valid Osteopenia X-ray
# ============================================================
def test_3_predict_osteopenia():
    with open(OSTEOPENIA_IMG, "rb") as f:
        files = {"xray": ("OP1.JPEG", f, "image/jpeg")}
        data = {"age": 62, "sex": "Female", "vitamin_deficiency": "true"}
        r = requests.post(f"{BASE_URL}/predict", data=data, files=files)

    body = r.json()
    passed = (r.status_code == 200
              and "prediction" in body
              and "urgency" in body)
    log_test(3, "POST /predict - Valid Osteopenia X-ray (Female, 62, vitamin def)",
             "POST", "/predict",
             "age=62, sex=Female, vitamin_deficiency=true, xray=OP1.JPEG",
             r.status_code, json.dumps(body, indent=2),
             "Status 200, full prediction response", passed)


# ============================================================
# TEST 4 - POST /predict - Valid Osteoporosis X-ray
# ============================================================
def test_4_predict_osteoporosis():
    with open(OSTEOPOROSIS_IMG, "rb") as f:
        files = {"xray": ("OS1.JPEG", f, "image/jpeg")}
        data = {"age": 72, "sex": "Female", "vitamin_deficiency": "true"}
        r = requests.post(f"{BASE_URL}/predict", data=data, files=files)

    body = r.json()
    passed = (r.status_code == 200
              and "prediction" in body
              and "urgency" in body)
    log_test(4, "POST /predict - Valid Osteoporosis X-ray (Female, 72, vitamin def)",
             "POST", "/predict",
             "age=72, sex=Female, vitamin_deficiency=true, xray=OS1.JPEG",
             r.status_code, json.dumps(body, indent=2),
             "Status 200, full prediction response", passed)


# ============================================================
# TEST 5 - POST /predict - Missing xray file
# ============================================================
def test_5_missing_xray():
    data = {"age": 45, "sex": "Male", "vitamin_deficiency": "false"}
    r = requests.post(f"{BASE_URL}/predict", data=data)
    try:
        body = json.dumps(r.json(), indent=2)
    except Exception:
        body = r.text
    passed = r.status_code == 422  # FastAPI validation error
    log_test(5, "POST /predict - Missing xray file (validation error)",
             "POST", "/predict",
             "age=45, sex=Male, vitamin_deficiency=false, xray=MISSING",
             r.status_code, body,
             "Status 422 (Validation Error)", passed)


# ============================================================
# TEST 6 - POST /predict - Missing age field
# ============================================================
def test_6_missing_age():
    with open(NORMAL_IMG, "rb") as f:
        files = {"xray": ("N1.JPEG", f, "image/jpeg")}
        data = {"sex": "Male", "vitamin_deficiency": "false"}
        r = requests.post(f"{BASE_URL}/predict", data=data, files=files)
    try:
        body = json.dumps(r.json(), indent=2)
    except Exception:
        body = r.text
    passed = r.status_code == 422
    log_test(6, "POST /predict - Missing age field",
             "POST", "/predict",
             "age=MISSING, sex=Male, vitamin_deficiency=false, xray=N1.JPEG",
             r.status_code, body,
             "Status 422 (Validation Error)", passed)


# ============================================================
# TEST 7 - POST /predict - Missing sex field
# ============================================================
def test_7_missing_sex():
    with open(NORMAL_IMG, "rb") as f:
        files = {"xray": ("N1.JPEG", f, "image/jpeg")}
        data = {"age": 45, "vitamin_deficiency": "false"}
        r = requests.post(f"{BASE_URL}/predict", data=data, files=files)
    try:
        body = json.dumps(r.json(), indent=2)
    except Exception:
        body = r.text
    passed = r.status_code == 422
    log_test(7, "POST /predict - Missing sex field",
             "POST", "/predict",
             "age=45, sex=MISSING, vitamin_deficiency=false, xray=N1.JPEG",
             r.status_code, body,
             "Status 422 (Validation Error)", passed)


# ============================================================
# TEST 8 - POST /predict - Invalid file type (txt file)
# ============================================================
def test_8_invalid_file_type():
    dummy = b"This is not an image"
    files = {"xray": ("test.txt", dummy, "text/plain")}
    data = {"age": 45, "sex": "Male", "vitamin_deficiency": "false"}
    r = requests.post(f"{BASE_URL}/predict", data=data, files=files)
    try:
        body = json.dumps(r.json(), indent=2)
    except Exception:
        body = r.text
    passed = r.status_code == 400  # Custom HTTPException for bad file type
    log_test(8, "POST /predict - Invalid file type (text/plain)",
             "POST", "/predict",
             "age=45, sex=Male, vitamin_deficiency=false, xray=test.txt (text/plain)",
             r.status_code, body,
             "Status 400 (Invalid file type)", passed)


# ============================================================
# TEST 9 - POST /predict - Edge case: elderly female with vitamin def
# ============================================================
def test_9_edge_elderly_female():
    with open(NORMAL_IMG, "rb") as f:
        files = {"xray": ("N1.JPEG", f, "image/jpeg")}
        data = {"age": 80, "sex": "Female", "vitamin_deficiency": "true"}
        r = requests.post(f"{BASE_URL}/predict", data=data, files=files)
    body = r.json()
    passed = (r.status_code == 200
              and body.get("urgency") in ["High", "Critical"]
              and body["patient"]["age"] == 80)
    log_test(9, "POST /predict - Elderly female (80) with vitamin deficiency",
             "POST", "/predict",
             "age=80, sex=Female, vitamin_deficiency=true, xray=N1.JPEG",
             r.status_code, json.dumps(body, indent=2),
             "Status 200, urgency=High or Critical", passed)


# ============================================================
# TEST 10 - POST /predict - Young male, no risk factors
# ============================================================
def test_10_low_risk():
    with open(NORMAL_IMG, "rb") as f:
        files = {"xray": ("N1.JPEG", f, "image/jpeg")}
        data = {"age": 25, "sex": "Male", "vitamin_deficiency": "false"}
        r = requests.post(f"{BASE_URL}/predict", data=data, files=files)
    body = r.json()
    passed = (r.status_code == 200 and "urgency" in body)
    log_test(10, "POST /predict - Young male (25), no risk factors",
             "POST", "/predict",
             "age=25, sex=Male, vitamin_deficiency=false, xray=N1.JPEG",
             r.status_code, json.dumps(body, indent=2),
             "Status 200, risk assessment returned", passed)


# ============================================================
# TEST 11 - GET /predict - Wrong HTTP method
# ============================================================
def test_11_wrong_method():
    r = requests.get(f"{BASE_URL}/predict")
    try:
        body = json.dumps(r.json(), indent=2)
    except Exception:
        body = r.text
    passed = r.status_code == 405  # Method Not Allowed
    log_test(11, "GET /predict - Wrong HTTP method (should be POST)",
             "GET", "/predict", "None",
             r.status_code, body,
             "Status 405 (Method Not Allowed)", passed)


# ============================================================
# TEST 12 - GET /nonexistent - 404 Not Found
# ============================================================
def test_12_not_found():
    r = requests.get(f"{BASE_URL}/nonexistent")
    try:
        body = json.dumps(r.json(), indent=2)
    except Exception:
        body = r.text
    passed = r.status_code == 404
    log_test(12, "GET /nonexistent - 404 Not Found",
             "GET", "/nonexistent", "None",
             r.status_code, body,
             "Status 404 (Not Found)", passed)


# ============================================================
# HTML Report Generation
# ============================================================
def generate_html_report():
    rows = ""
    for r in results:
        status_class = "pass" if r["passed"] else "fail"
        status_text = "PASS" if r["passed"] else "FAIL"
        resp_escaped = r["response"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        rows += f"""
        <tr class="{status_class}">
            <td class="test-id">{r['id']}</td>
            <td>
                <div class="test-name">{r['name']}</div>
                <div class="test-meta"><span class="method {r['method'].lower()}">{r['method']}</span> <code>{r['endpoint']}</code></div>
            </td>
            <td><code>{r['params']}</code></td>
            <td class="status-code">{r['status_code']}</td>
            <td><pre class="response-body">{resp_escaped}</pre></td>
            <td><code>{r['expected']}</code></td>
            <td class="status {status_class}">{status_text}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Test Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #fff; color: #222;
            padding: 1.5rem;
        }}
        table {{
            width: 100%; border-collapse: collapse;
            border: 1px solid #d0d0d0;
        }}
        th {{
            background: #f5f5f5; color: #333;
            padding: .6rem .8rem; text-align: left;
            font-size: .8rem; text-transform: uppercase;
            letter-spacing: .03em; border-bottom: 2px solid #ccc;
        }}
        td {{
            padding: .6rem .8rem;
            border-bottom: 1px solid #e5e5e5;
            vertical-align: top; font-size: .85rem;
        }}
        tr:hover {{ background: #fafafa; }}
        .test-id {{ font-weight: 700; text-align: center; }}
        .test-name {{ font-weight: 600; margin-bottom: .15rem; }}
        .test-meta {{ font-size: .8rem; color: #666; }}
        .method {{
            display: inline-block; padding: 1px 6px;
            border-radius: 3px; font-size: .7rem;
            font-weight: 700; color: #fff;
        }}
        .method.get {{ background: #2e7d32; }}
        .method.post {{ background: #1565c0; }}
        .status-code {{ font-weight: 700; text-align: center; }}
        .response-body {{
            font-size: .72rem; max-height: 120px;
            overflow-y: auto; white-space: pre-wrap;
            background: #f8f8f8; padding: .4rem;
            border-radius: 4px; color: #333;
            border: 1px solid #e0e0e0;
            font-family: Consolas, monospace;
        }}
        .status {{ font-weight: 700; text-align: center; }}
        .status.pass {{ color: #2e7d32; }}
        .status.fail {{ color: #c62828; }}
        tr.fail {{ background: #fff5f5; }}
        code {{
            background: #f0f0f0; padding: 1px 4px;
            border-radius: 3px; font-size: .78rem;
            font-family: Consolas, monospace;
        }}
    </style>
</head>
<body>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Test Case</th>
                <th>Parameters</th>
                <th>HTTP Code</th>
                <th>Response Body</th>
                <th>Expected</th>
                <th>Result</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</body>
</html>"""

    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api_test_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nHTML Report saved to: {os.path.abspath(report_path)}")
    return os.path.abspath(report_path)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  OsteoScan AI - API Test Suite")
    print("  Testing all endpoints at " + BASE_URL)
    print("=" * 60)
    print()

    # Verify server is up
    try:
        requests.get(BASE_URL, timeout=5)
    except requests.ConnectionError:
        print("ERROR: Cannot connect to the backend server!")
        print("  Make sure uvicorn is running:")
        print("  uvicorn app:app --host 127.0.0.1 --port 8000")
        sys.exit(1)

    # Verify test images exist
    for path, label in [(NORMAL_IMG, "Normal"), (OSTEOPENIA_IMG, "Osteopenia"), (OSTEOPOROSIS_IMG, "Osteoporosis")]:
        if not os.path.exists(path):
            print(f"Warning: {label} test image not found at {path}")
        else:
            print(f"  Found {label} image: {os.path.basename(path)}")

    print()
    print("-" * 60)
    print("  Running 12 test cases...")
    print("-" * 60)

    test_1_health_check()
    test_2_predict_normal()
    test_3_predict_osteopenia()
    test_4_predict_osteoporosis()
    test_5_missing_xray()
    test_6_missing_age()
    test_7_missing_sex()
    test_8_invalid_file_type()
    test_9_edge_elderly_female()
    test_10_low_risk()
    test_11_wrong_method()
    test_12_not_found()

    print()
    print("-" * 60)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("-" * 60)

    report_path = generate_html_report()
    print(f"\nOpen the report in your browser:")
    print(f"  {report_path}")
    print()
