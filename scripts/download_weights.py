from __future__ import annotations

from autocus.weights import download_registered_weights, verify_registered_weights


if __name__ == "__main__":
    downloaded = download_registered_weights()
    print({"downloaded": [str(p) for p in downloaded], "status": verify_registered_weights()})
