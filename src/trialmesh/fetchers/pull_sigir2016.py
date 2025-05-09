#!/usr/bin/env python3
# src/trialmesh/data/pull_sigir2016.py

import os
import sys
import requests
import tarfile
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_file(url, destination_path):
    """Download a file from url to destination_path with progress bar.

    This function handles file downloads with proper progress reporting,
    error handling, and directory creation.

    Args:
        url: URL to download from
        destination_path: Path to save the downloaded file

    Returns:
        Path to the downloaded file

    Raises:
        Exception: If download fails due to connection issues,
                  permission errors, or other problems
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise exception for HTTP errors

    # Get file size for progress bar if available
    file_size = int(response.headers.get('content-length', 0))
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up progress bar
    desc = f"Downloading {destination_path.name}"
    progress = tqdm(total=file_size, unit='B', unit_scale=True, desc=desc)

    # Download the file
    with open(destination_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                file.write(chunk)
                progress.update(len(chunk))
    progress.close()

    return destination_path


def download_paper(data_dir):
    """Download the reference paper.

    This function downloads the SIGIR2016 clinical trials collection
    reference paper, which contains important information about the
    dataset structure, format, and evaluation methodology.

    Args:
        data_dir: Directory to save the paper

    Returns:
        True if download succeeded, False otherwise
    """
    paper_url = "https://bevankoopman.github.io/papers/sigir2016_clinicaltrials_collection.pdf"
    paper_path = data_dir / "sigir2016_clinicaltrials_collection.pdf"

    print("Downloading reference paper...")
    try:
        download_file(paper_url, paper_path)
        print(f"Paper successfully downloaded to {paper_path}")
    except Exception as e:
        print(f"Error downloading paper: {e}")
        return False

    return True


def process_download_links_file(links_file_path, data_dir):
    """Parse the download links file and return a list of URL and destination pairs.

    This function extracts download information from the CSIRO Data Portal
    download links file, mapping each URL to its appropriate destination path.

    Args:
        links_file_path: Path to the file containing download links
        data_dir: Base data directory

    Returns:
        List of (url, destination_path) tuples for all files to download
    """
    if not links_file_path.exists():
        print(f"Error: Download links file does not exist at {links_file_path}")
        return []

    downloads = []

    with open(links_file_path, 'r') as f:
        url = None
        directory = None

        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            if line.startswith('http'):
                url = line
            elif line.startswith('dir='):
                directory = line.replace('dir=', '').strip()

                if url and directory:
                    # Get the filename from content-disposition in URL
                    filename = url.split('filename%3D%22')[1].split('%22')[0] if 'filename%3D%22' in url else \
                    url.split('/')[-1].split('?')[0]

                    # Create destination path relative to data_dir
                    rel_dir = Path(directory.lstrip('./')).relative_to(
                        '000017152v004') if '000017152v004' in directory else Path(directory.lstrip('./'))
                    dest_dir = data_dir / "sigir2016" / rel_dir
                    dest_path = dest_dir / filename

                    downloads.append((url, dest_path))
                    url = None
                    directory = None

    return downloads


def download_dataset(links_file_path, data_dir):
    """Download all files from the dataset using the links file.

    This function handles the downloading and extraction of the SIGIR2016
    clinical trials dataset. It:
    1. Processes a file containing download links
    2. Downloads all specified files in parallel
    3. Extracts compressed archives when found
    4. Reports on success and failure rates

    Args:
        links_file_path: Path to file containing download links
        data_dir: Directory to save downloaded files

    Returns:
        True if at least one file was successfully downloaded
    """
    print(f"Processing download links from {links_file_path}...")
    downloads = process_download_links_file(links_file_path, data_dir)

    if not downloads:
        print("No valid download links found.")
        return False

    print(f"Found {len(downloads)} files to download.")

    successful_downloads = 0

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_file, url, dest_path): (url, dest_path) for url, dest_path in downloads}

        for future in as_completed(futures):
            url, dest_path = futures[future]
            try:
                future.result()
                successful_downloads += 1
            except Exception as e:
                print(f"Error downloading {dest_path.name}: {e}")

    print(f"Successfully downloaded {successful_downloads} of {len(downloads)} files.")

    # Extract the tarball if it exists
    tarball_path = list(data_dir.glob("**/clinicaltrials.gov-*.tgz"))
    if tarball_path:
        tarball_path = tarball_path[0]
        print(f"Extracting {tarball_path}...")
        try:
            with tarfile.open(tarball_path, 'r:gz') as tar:
                tar.extractall(path=tarball_path.parent)
            print(f"Successfully extracted {tarball_path}")
        except Exception as e:
            print(f"Error extracting {tarball_path}: {e}")

    return successful_downloads > 0


def main():
    """Main entry point for the script.

    This function handles:
    1. Parsing command-line arguments
    2. Creating necessary directories
    3. Downloading the reference paper
    4. Downloading and extracting the dataset if links are provided
    5. Providing instructions if links aren't provided
    """
    parser = argparse.ArgumentParser(description='Download SIGIR 2016 Clinical Trials dataset')
    parser.add_argument('--links_file', type=str,
                        help='Path to the download links file (.txt) generated from the CSIRO Data Portal')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to store the downloaded dataset files (default: ./data)')
    args = parser.parse_args()

    # Create data directory at the specified location (or default to ./data)
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using data directory: {data_dir}")

    print("SIGIR 2016 Clinical Trials Dataset Downloader")
    print("---------------------------------------------")

    # Download reference paper
    download_paper(data_dir)

    # Handle the dataset download
    if args.links_file:
        links_file_path = Path(args.links_file)
        download_dataset(links_file_path, data_dir)
    else:
        print("\nTo download the dataset, please follow these steps:")
        print("1. Go to https://data.csiro.au/collection/csiro:17152")
        print("2. Click 'Download all files' on the webpage")
        print("3. Download the generated text file (will be named something like '17152v004.txt')")
        print("4. Run this script again with the path to the downloaded file:")
        print(f"   trialmesh-download-sigir2016 --links_file=/path/to/17152v004.txt")


def cli_main():
    """Entry point for the console script"""
    main()


if __name__ == "__main__":
    main()