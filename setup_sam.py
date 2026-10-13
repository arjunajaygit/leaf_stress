"""Download the official SAM ViT-B checkpoint; never run during a web request."""
import hashlib
from pathlib import Path
import urllib.request

URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
TARGET = Path(__file__).resolve().parent / 'models' / 'sam_vit_b_01ec64.pth'
SHA256 = 'ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912'


def verify_checkpoint(path=TARGET):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise ValueError('SAM checkpoint checksum mismatch. Move the damaged file aside and rerun setup_sam.py.')
    return digest.hexdigest()


def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        verify_checkpoint()
        print('Checkpoint verified:', TARGET)
        return
    partial = TARGET.with_suffix('.part')
    print('Downloading official SAM ViT-B checkpoint (~375 MB).', flush=True)
    digest, total = hashlib.sha256(), 0
    with urllib.request.urlopen(URL, timeout=60) as response, partial.open('wb') as stream:
        expected = int(response.headers.get('Content-Length', 0))
        while chunk := response.read(1024 * 1024):
            stream.write(chunk); digest.update(chunk); total += len(chunk)
            if total % (32 * 1024 * 1024) == 0:
                print(f'{total / 1024**2:.0f} MiB downloaded', flush=True)
    if (expected and total != expected) or total < 300_000_000:
        raise SystemExit('Incomplete checkpoint download; rerun setup_sam.py to retry.')
    if digest.hexdigest() != SHA256:
        raise SystemExit('Checkpoint checksum mismatch; partial download was not activated. Rerun setup_sam.py to retry.')
    partial.replace(TARGET)
    print('Checkpoint:', TARGET)
    print('SHA-256 (local download record):', digest.hexdigest())


if __name__ == '__main__':
    main()
