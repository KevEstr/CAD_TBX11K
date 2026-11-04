import json, urllib.request, io, numpy as np, cv2

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
body += b'Content-Disposition: form-data; name="file"; filename="synthetic_xray.png"\r\n'
body += b'Content-Type: image/png\r\n\r\n'
body += data
body += b'\r\n--' + boundary.encode() + b'--\r\n'

req = urllib.request.Request(
    'http://localhost:8081/api/v1/predict', data=body,
    headers={'Content-Type': 'multipart/form-data; boundary=' + boundary}
)
resp = urllib.request.urlopen(req, timeout=120)
result = json.load(resp)
print('Prediction:', result['prediction'])
print('Probabilities:', result['probabilities'])
print('Level1 regions:', result['level1']['regions'])
print('Level2 HOG grid shape:', len(result['level2']['hog_heatmap_7x7']), 'x', len(result['level2']['hog_heatmap_7x7'][0]))
print('Top-3 features:', result['level1']['top_features'][:3])
print('Inference time:', result['metadata']['inference_time_ms'], 'ms')
print()
print('SUCCESS: Pipeline completed without errors')