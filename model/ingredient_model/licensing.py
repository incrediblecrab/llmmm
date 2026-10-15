"""The one definition of the license on everything published to Hugging Face: the model, the dataset and the Space.

The repository's own code stays under the MIT License in the root `LICENSE`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

LICENSE_NAME = "CC BY-NC 4.0"
LICENSE_TITLE = "Creative Commons Attribution-NonCommercial 4.0 International License"
LICENSE_HUB_ID = "cc-by-nc-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"
# Creative Commons' plain-text legal code, fetched from https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt
# on October 15, 2026; every published repository ships it unchanged as `LICENSE`.
LICENSE_TEXT = Path(__file__).resolve().parent / "licenses" / "CC-BY-NC-4.0.txt"
LICENSE_TEXT_SHA256 = "41003d4a74749c0220e33dd415042164b5a1093ed401f36277234f772d22d3d0"
LICENSE_USE = "It permits non-commercial use, such as education and research, with attribution."

WEIGHTS_LICENSE_NOTE = f"The license covers the weights and configuration files in this repository. {LICENSE_USE}"
DATASET_LICENSE_NOTE = f"The license covers every file in this dataset. {LICENSE_USE}"
SPACE_LICENSE_NOTE = f"The license covers every file in this Space. {LICENSE_USE}"


def license_bytes() -> bytes:
    data = LICENSE_TEXT.read_bytes()
    if hashlib.sha256(data).hexdigest() != LICENSE_TEXT_SHA256:
        raise ValueError(f"{LICENSE_TEXT}: the license text differs from the recorded legal code")
    return data
