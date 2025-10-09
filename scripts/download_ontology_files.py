import asyncio
from pathlib import Path

import httpx

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DATA_DIR = REPO_ROOT.joinpath("ontology")
DATA_DIR.mkdir(exist_ok=True, parents=True)


async def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(120.0, connect=10.0)
    ) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            try:
                with dest.open("wb") as file:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):  # 1 MB
                        file.write(chunk)
            except Exception:
                try:
                    dest.unlink()
                except Exception:  # pylint: disable=broad-except
                    pass
                raise


async def download_file_mapper(file_name: str, url: str) -> None:
    dest = DATA_DIR.joinpath(file_name)
    if dest.exists():
        print(f"{dest} already exists, skipping download.")
        return
    print(f"Downloading {url} to {dest}...")
    await download_file(url, dest)
    print(f"Downloaded {dest}.")


async def main() -> None:
    print("Downloading ontology files...")

    download_files = [
        {
            # Cellosaurus
            # https://ftp.expasy.org/databases/cellosaurus/
            # https://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo
            # then convert to owl using robot
            # docker run -v $PWD:/work -w /work --rm -it obolibrary/robot robot convert -i ./cellosaurus.obo -o ./cellosaurus.owl --format owl
            "file_name": "cellosaurus.obo",
            "url": "https://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo",
        },
        {
            # Cell Ontology
            # https://purl.obolibrary.org/obo/cl/cl-base.owl
            "file_name": "cell_ontology.owl",
            "url": "https://purl.obolibrary.org/obo/cl/cl-base.owl",
        },
        {
            # UBERON
            # https://purl.obolibrary.org/obo/uberon/uberon-base.owl
            "file_name": "uberon.owl",
            "url": "https://purl.obolibrary.org/obo/uberon/uberon-base.owl",
        },
        {
            # MONDO
            # https://purl.obolibrary.org/obo/mondo.owl
            "file_name": "mondo.owl",
            "url": "https://purl.obolibrary.org/obo/mondo.owl",
        }
    ]

    for file_info in download_files:
        await download_file_mapper(file_info["file_name"], file_info["url"])

    print("Finished downloading ontology files.")


if __name__ == "__main__":
    asyncio.run(main())
