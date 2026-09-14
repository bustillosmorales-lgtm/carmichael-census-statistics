"""Download the census of Pinch into data/ and check the SHA-256 checksums of the files used.

The files were retrieved on 20 August 2026 and again on 13 September 2026 with identical checksums.
"""
import hashlib
import os
import urllib.request

from common import DATA

BASE = "http://www.s369624816.websitehome.co.uk/rgep/"
SHA256 = {
    "carmichael-16.gz": "f6aff9810006d4077f5d52ab300ab2abd04268277abce31340aeb596234182b4",
    "carmichael17.gz": "1e3fcc81922f0c30cf5c07c8f2c5932fcd1b2031944a9e28f60a9a98885ea55b",
    "carmichael18.gz": "21a9e3a713e38b396f688df5225ea9fc87634b2a3d68739bec3625a7c6fbde1c",
    "car3-18.gz": "c72261044235f0cfc891d122aa9d15a19ee7ad621e20dfc2fff656d654bb375c",
}

os.makedirs(DATA, exist_ok=True)
ok = True
for name, digest in SHA256.items():
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print("downloading", name)
        urllib.request.urlretrieve(BASE + name, path)
    with open(path, "rb") as fh:
        h = hashlib.sha256(fh.read()).hexdigest()
    match = h == digest
    ok &= match
    print(f"{name:18s} sha256 {'OK' if match else 'MISMATCH ' + h}")
print("all checksums match:", ok)
if not ok:
    raise SystemExit(1)
