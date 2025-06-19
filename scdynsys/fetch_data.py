import requests
import tqdm
import os

def main():
    """
    Download files from a Zenodo record.
    Replace `record_id` with the ID of the Zenodo record you want to download.
    """
    record_id = "14201749"  # FIXME: Replace with correct Zenodo record ID

    url = f"https://zenodo.org/api/records/{record_id}"
    r = requests.get(url)
    files = r.json()["files"]

    print(f"Found {len(files)} files in Zenodo record {record_id}.")
    print("files are:")
    for f in files:
        print(f"- {f['key']}")

    print("Downloading files ...")

    target_dir = "zenodo_data" ## FIXME: this should be just "data"    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for f in tqdm.tqdm(files):
        url = f["links"]["self"]
        name = f["key"]
        out_path = os.path.join(target_dir, name)
        with requests.get(url, stream=True) as resp:
            with open(out_path, "wb") as out:
                for chunk in resp.iter_content(chunk_size=8192):
                    out.write(chunk)


if __name__ == "__main__":
    main()