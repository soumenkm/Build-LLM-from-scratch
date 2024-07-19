import torch, requests, tqdm, zipfile, tarfile, shutil
from pathlib import Path

def download_file(
    src_url: str, 
    destination_file_path: Path, 
    is_text: bool=False
) -> None:
    
    response = requests.request(method="get", url=src_url, stream=True)

    status = response.status_code
    if status != 200:
        raise ValueError(f"Download unsuccessful. Status: {status}")
    
    headers = response.headers
    file_size = int(headers["Content-Length"]) if "Content-Length" in headers.keys() else 0
    chunk_size = 4096
    
    if Path.exists(destination_file_path):
        if destination_file_path.stat().st_size == file_size:
            print(f"File already downloaded and matches the expected size ({file_size / (1024 ** 2):.2f} MB). Skipping download.")
            return None
        elif file_size == 0:
            print(f"File already downloaded but cannot be matched with the expected size ({file_size / (1024 ** 2):.2f} MB). Skipping download. "+
                  "Delete the file before starting fresh download")
            return None
        else:
            Path.unlink(destination_file_path)
            print(f"File already downloaded but does not match the expected size ({file_size / (1024 ** 2):.2f} MB). Starting fresh download.")
    
    if is_text:
        encoding = response.encoding
        mode = "w"
    else:
        encoding = None
        mode = "wb"

    with open(file=destination_file_path, mode=mode, encoding=encoding) as f: 
        with tqdm.tqdm(response.iter_content(chunk_size=chunk_size, decode_unicode=is_text),
                       desc="Downloading...",
                       total=file_size//chunk_size if file_size > 0 else None,
                       unit=f" chunk (1 chunk = {chunk_size//1024} KB)",
                       colour="green") as pbar:
            for i, chunk in enumerate(pbar):
                if chunk:
                    f.write(chunk) # to ensire that None chunks are filtered out
                pbar.set_postfix(downloaded=f"{chunk_size * i/1024**2:.2f} MB")
    
    print(f"Download is successfully completed! Total disk space used: {file_size / (1024 ** 2):.2f} MB")
    return None

def uncompress_file(
    src_file_path: Path, 
    dest_dir_path: Path
) -> None:
    
    if Path.exists(dest_dir_path):
        dir_content = [i for i in Path.iterdir(dest_dir_path)]
        if dir_content:
            print(f"{dest_dir_path} is not empty so unzipping process is skipped")
            return None
    else:
        Path.mkdir(dest_dir_path)
        
    file_type = src_file_path.suffix
    if file_type == ".zip":
        with zipfile.ZipFile(file=src_file_path, mode="r") as f:
            f.extractall(path=dest_dir_path)
        print(f"Extracted {src_file_path} to {dest_dir_path}")
            
    elif file_type in [".tar", ".gz", ".bz2", ".xz"]:
        with tarfile.open(name=src_file_path, mode="r:*") as f:
            f.extractall(path=dest_dir_path)
        print(f"Extracted {src_file_path} to {dest_dir_path}")
        
    else:
        raise ValueError(f"Unsupported file extension: {file_type}")

    return None
      
if __name__ == "__main__":
    
    url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz"
    cwd = pathlib.Path.cwd()
    destination = pathlib.Path(cwd, "fine_tuning/datasets/data.tar.gz")
    download_file(src_url=url, destination_file_path=destination)