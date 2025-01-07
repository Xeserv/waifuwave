from multiprocessing import Pool
from typing import Generator, Iterable, List
from urllib.parse import urlparse

import os
import boto3


models = [
    "checkpoints/counterfeitV30_v30.safetensors",
    "embeddings/7dirtywords.pt",
    "embeddings/easynegative.safetensors",
    "embeddings/negative_hand-neg.pt",
    "loras/pastelMixStylizedAnime_pastelMixLoraVersion.safetensors",
    "loras/ligne_claire_anime.safetensors",
    "vae/sdVAEForAnime_v10.pt",
]


def batcher(iterable: Iterable, batch_size: int) -> Generator[List, None, None]:
    """Batch an iterator. The last item might be of smaller len than batch_size.

    Args:
        iterable (Iterable): Any iterable that should be batched
        batch_size (int): Len of the generated lists

    Yields:
        Generator[List, None, None]: List of items in iterable
    """
    batch = []
    counter = 0
    for i in iterable:
        batch.append(i)
        counter += 1
        if counter % batch_size == 0:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def download_batch(batch) -> int:
    s3 = boto3.client("s3")
    n = 0
    for line in batch:
        line, destdir = line
        url = urlparse(line)
        url_path = url.path.lstrip("/")

        folder, basename = os.path.split(url_path)

        dir = os.path.join(destdir, folder)
        os.makedirs(dir, exist_ok=True)
        filepath = os.path.join(dir, basename)

        if os.path.exists(filepath):
            print(f"{line} already exists")
            continue

        print(f"{line} -> {filepath}")
        s3.download_file(url.netloc, url_path, filepath)
        n += 1
    return n


def copy_from_tigris(
        models: List[str] = models,
        bucket_name: str = os.getenv("BUCKET_NAME", "comfyui"),
        destdir: str = "/opt/comfyui",
        n_cpus: int = os.cpu_count()
    ):
    """Copy files from Tigris to the destination folder. This will be done in parallel.

    Args:
        models (List[str]): List of models to download. Defaults to the list of models in this file.
        bucket_name (str): Tigris bucket to query. Defaults to envvar $BUCKET_NAME.
        destdir (str): path to store the files.
        n_cpus (int): number of simultaneous batches. Defaults to the number of cpus in the computer.
    """

    model_files = [ (f"s3://{bucket_name}/models/{x}", destdir) for x in models ]

    print(f"using {n_cpus} cpu cores for downloads")
    n_cpus = min(len(model_files), n_cpus)
    batch_size = len(model_files) // n_cpus
    with Pool(processes=n_cpus) as pool:
        for n in pool.imap_unordered(
            download_batch, batcher(model_files, batch_size)
        ):
            pass


if __name__ == "__main__":
    copy_from_tigris(n_cpus=999)