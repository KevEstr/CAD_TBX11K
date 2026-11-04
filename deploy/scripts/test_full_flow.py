import json, urllib.request, io, time, numpy as np, cv2

def test(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))

BASE = "http://localhost:8081"
all_ok = True

# --- Test 1: Health endpoint ---
try:
    r = json.load(urllib.request.urlopen(f"{BASE}/health"))
    test("Health endpoint", r.get("status") == "ok" and r.get("model_loaded"))
except Exception as e:
    test("Health endpoint", False, str(e))
    all_ok = False

# --- Test 2: Model info ---
try:
    r = json.load(urllib.request.urlopen(f"{BASE}/api/v1/model/info"))
    ok = r.get("model") == "XGBoost" and r.get("features_selected") == 2000
    test("Model info", ok, f"model={r.get('model')} feat={r.get('features_selected')}")
except Exception as e:
    test("Model info", False, str(e))
    all_ok = False

# --- Test 3-4: Predict with synthetic image ---
try:
    img = np.ones((3000, 3000), dtype=np.uint8) * 180
    img[500:2500, 400:2600] = 120
    img[600:2400, 500:2550] = 100
    img[700:2300, 600:2500] = 80
    img += np.random.randint(0, 30, (3000, 3000), dtype=np.uint8).astype(np.uint8)
    _, buf = cv2.imencode('.png', img)
    data = buf.tobytes()

    boundary = '----TestBoundary'
    body = b''
    body += b'--' + boundary.encode() + b'\r\n'
    body += b'Content-Disposition: form-data; name="file"; filename="test.png"\r\n'
    body += b'Content-Type: image/png\r\n\r\n'
    body += data
    body += b'\r\n--' + boundary.encode() + b'--\r\n'

    req = urllib.request.Request(
        f'{BASE}/api/v1/predict', data=body,
        headers={'Content-Type': 'multipart/form-data; boundary=' + boundary}
    )
    r = json.load(urllib.request.urlopen(req, timeout=120))
    session_id = r.get("session_id")

    # Test 3: Prediction has 3 probs summing to 1.0
    prob_sum = sum(r["probabilities"].values())
    test("Predict: probs sum to 1.0", abs(prob_sum - 1.0) < 0.001, f"sum={prob_sum:.4f}")

    # Test 4: Region heatmap not empty
    test("Predict: region heatmap present", bool(r.get("level1", {}).get("region_heatmap_base64", "")))

    # Test 5: HOG grid 7x7
    hog_grid = r.get("level2", {}).get("hog_heatmap_7x7", [])
    test("HOG grid 7x7", len(hog_grid) == 7 and len(hog_grid[0]) == 7)

    # Test 6: Top features present
    test("Top features", len(r.get("level1", {}).get("top_features", [])) > 0)

    # Test 7: Families present
    test("Families present", bool(r.get("level1", {}).get("families", {})))

    # Test 8: Inference time < 15s
    t = r.get("metadata", {}).get("inference_time_ms", 99999)
    test("Inference time < 15s", t < 15000, f"{t:.0f}ms")

except Exception as e:
    test("Predict pipeline", False, str(e))
    all_ok = False

# --- Test 9: Invalid file ---
try:
    boundary = '----TestBoundary'
    body = b''
    body += b'--' + boundary.encode() + b'\r\n'
    body += b'Content-Disposition: form-data; name="file"; filename="test.txt"\r\n'
    body += b'Content-Type: text/plain\r\n\r\n'
    body += b'not an image\r\n--' + boundary.encode() + b'--\r\n'
    req = urllib.request.Request(
        f'{BASE}/api/v1/predict', data=body,
        headers={'Content-Type': 'multipart/form-data; boundary=' + boundary}
    )
    try:
        urllib.request.urlopen(req)
        test("Invalid file reject", False, "should have returned 400")
        all_ok = False
    except urllib.error.HTTPError as e:
        test("Invalid file rejected", e.code == 400, f"HTTP {e.code}")

except Exception as e:
    test("Invalid file test", False, str(e))
    all_ok = False

# --- Test 10: Session cleanup ---
try:
    if session_id:
        req = urllib.request.Request(f'{BASE}/api/v1/session/{session_id}', method='DELETE')
        r = json.load(urllib.request.urlopen(req))
        test("Session cleanup", r.get("status") == "cleaned", f"session={session_id}")
except Exception as e:
    test("Session cleanup", False, str(e))
    all_ok = False

# --- Test 11: Non-existent session ---
try:
    req = urllib.request.Request(f'{BASE}/api/v1/session/{session_id}', method='DELETE')
    r = json.load(urllib.request.urlopen(req))
    test("Double session cleanup", r.get("status") == "cleaned")
except Exception as e:
    test("Double session cleanup", False, str(e))
    all_ok = False

print()
print(f"{'='*50}")
print(f"ALL TESTS PASSED: {all_ok}")
print(f"{'='*50}")