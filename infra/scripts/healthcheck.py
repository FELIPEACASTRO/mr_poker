from urllib.request import urlopen

with urlopen("http://127.0.0.1:8000/health", timeout=3) as response:
    body = response.read().decode("utf-8")
    if 'ok' not in body:
        raise SystemExit(1)
