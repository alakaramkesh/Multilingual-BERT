from pathlib import Path
import requests
from ud_utils import read_yaml


def download_file(url, out_path):
    # This is simple on purpose: one URL gives one conllu file.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_text(r.text, encoding="utf-8")


def main():
    params = read_yaml("params.yaml")
    raw_dir = Path(params["paths"]["raw_dir"])
    for lang, cfg in params["languages"].items():
        # Download train, dev, and test files for each selected language.
        for split in ["train", "dev", "test"]:
            filename = cfg[split]
            url = f"{cfg['base_url']}/{filename}"
            out_path = raw_dir / lang / filename
            if out_path.exists():
                continue
            download_file(url, out_path)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
