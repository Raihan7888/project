from pathlib import Path
from urllib.request import urlopen


BASE_DIR = Path(
    __file__
).resolve().parent

VENDOR_DIR = (
    BASE_DIR
    / "static"
    / "vendor"
)

VENDOR_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FILES = {

    "ort.min.js":
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js",

    "ort-wasm-simd.wasm":
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort-wasm-simd.wasm",
}


def download(
    filename,
    url,
):

    destination = (
        VENDOR_DIR
        / filename
    )

    if destination.exists():

        print(
            f"{filename} already exists."
        )

        return

    print(
        f"Downloading {filename}..."
    )

    with urlopen(
        url,
        timeout=60,
    ) as response:

        data = response.read()

    destination.write_bytes(
        data
    )

    print(
        f"Saved {destination}"
    )


def main():

    for filename, url in FILES.items():

        download(
            filename,
            url,
        )

    print()
    print(
        "Browser runtime prepared."
    )


if __name__ == "__main__":
    main()