import json, urllib.request, time, numpy as np, cv2

BASE = 'http://127.0.0.1:8097'

img = (np.ones((512, 512), dtype=np.uint8) * 160).astype(np.uint8)
cv2.ellipse(img, (150, 200), (90, 130), 0, 0, 360, 110, -1)
cv2.ellipse(img, (360, 200), (90, 130), 0, 0, 360, 100, -1)
cv2.rectangle(img, (320, 90), (430, 200), 70, -1)
img += np.random.randint(0, 20, (512, 512), dtype=np.uint8).astype(np.uint8)
_, b = cv2.imencode('.png', img)
data = b.tobytes()

bd = '----T'
body = b''
body += b'--' + bd.encode() + b'\r\n'
body += b'Content-Disposition: form-data; name="file"; filename="t.png"\r\n'
body += b'Content-Type: image/png\r\n\r\n'
body += data + b'\r\n--' + bd.encode() + b'--\r\n'

req = urllib.request.Request(BASE + '/api/v1/predict', data=body,
    headers={'Content-Type': 'multipart/form-data; boundary=' + bd})
t0 = time.time()
r = json.load(urllib.request.urlopen(req, timeout=120))
print(f'Predict in {(time.time()-t0)*1000:.0f}ms')
print('Prediction:', r['prediction']['class'], 'conf:', r['prediction']['confidence'])
print('Probabilities (calibradas):', r['probabilities'])
print()
print('Regiones:', {k: f"{v*100:.1f}%" for k, v in r['level1']['regions'].items()})
print('Top features:', [(f['name'], round(f['importance'],3), f['direction']) for f in r['level1']['top_features'][:5]])
print('Familias:', {k: f"{v*100:.0f}%" for k, v in r['level1']['families'].items()})
print()
sid = r['session_id']

# Oclusion
urllib.request.urlopen(urllib.request.Request(BASE + f'/api/v1/predict/{sid}/occlude', data=b'', method='POST'))
t0 = time.time()
for i in range(40):
    time.sleep(2)
    rr = json.load(urllib.request.urlopen(BASE + f'/api/v1/predict/{sid}/progress'))
    if rr['status'] == 'complete':
        print(f'Oclusion COMPLETE in {rr["result"]["time_ms"]/1000:.1f}s')
        print('Overlay chars:', len(rr['result']['overlay']))
        hm = rr['result']['heatmap']
        print('Heatmap:', len(hm), 'x', len(hm[0]))
        print('Max drop:', max(max(row) for row in hm))
        break
    p = rr.get('progress', {})
    print(f'  {p.get("current")}/{p.get("total")} in {(time.time()-t0):.0f}s')
else:
    print('TIMEOUT')

urllib.request.urlopen(urllib.request.Request(BASE + f'/api/v1/session/{sid}', method='DELETE'))