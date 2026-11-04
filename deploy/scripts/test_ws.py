import json, urllib.request, asyncio, numpy as np, cv2, io

BASE = "http://localhost:8082"

# 1. Predict to get session_id
img = (np.ones((3000, 3000), dtype=np.uint8) * 180).astype(np.uint8)
img[500:2500, 400:2600] = 120
img += np.random.randint(0, 30, (3000, 3000), dtype=np.uint8).astype(np.uint8)
_, buf = cv2.imencode('.png', img)
data = buf.tobytes()

boundary = '----Test'
body = b''
body += b'--' + boundary.encode() + b'\r\n'
body += b'Content-Disposition: form-data; name="file"; filename="test.png"\r\n'
body += b'Content-Type: image/png\r\n\r\n' + data
body += b'\r\n--' + boundary.encode() + b'--\r\n'
req = urllib.request.Request(f'{BASE}/api/v1/predict', data=body,
    headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
r = json.load(urllib.request.urlopen(req, timeout=60))
sid = r['session_id']
print("Session:", sid)
print("Prediction:", r['prediction'])

# 2. Connect WebSocket and wait for complete
async def test_ws():
    import websockets
    uri = f"ws://localhost:8082/ws/{sid}"
    print(f"Connecting to {uri}")
    timeout = 120
    try:
        async with websockets.connect(uri, ping_interval=30, close_timeout=10) as ws:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                data = json.loads(msg)
                if data['type'] == 'progress':
                    print(f"  Progress: {data['percent']}% - {data['detail']}")
                elif data['type'] == 'complete':
                    print(f"  COMPLETE! Time: {data['occlusion_time_ms']/1000:.1f}s")
                    print(f"  Heatmap 6x6: {len(data['occlusion_heatmap_6x6'])}x{len(data['occlusion_heatmap_6x6'][0])}")
                    print(f"  Overlay: {len(data['occlusion_overlay_base64'])} chars")
                    return True
                elif data['type'] == 'error':
                    print(f"  ERROR: {data['detail']}")
                    return False
    except asyncio.TimeoutError:
        print("  TIMEOUT waiting for WS messages")
        return False
    except Exception as e:
        print(f"  WS ERROR: {e}")
        return False

result = asyncio.run(test_ws())
print(f"\nWS Test {'PASSED' if result else 'FAILED'}")