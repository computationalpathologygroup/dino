import random
import argparse
import numpy as np

from pathlib import Path


def extract_file(tarball_path, start_offset, end_offset):
    with open(tarball_path, "rb") as f:
        f.seek(start_offset)
        return f.read(end_offset - start_offset)


def compare_files(original_path, extracted_data):
    with open(original_path, "rb") as f:
        original_data = f.read()
    return original_data == extracted_data


def test_tarball_entries(
    image_root, tarball_path, entries_path, file_indices_path, subset_size
):
    entries = np.load(entries_path)
    file_indices = np.load(file_indices_path, allow_pickle=True).item()

    ntotal = len(entries)
    nsample = min(subset_size, ntotal)
    print(f"Randomly sampling {nsample}/{ntotal} images")
    sampled_indices = random.sample(range(ntotal), nsample)
    subset_entries = entries[sampled_indices]

    mismatch = 0
    for entry in subset_entries:
        _, img_idx, start_offset, end_offset = entry
        img_name = file_indices[img_idx]
        original_path = Path(image_root, img_name)
        assert original_path.is_file(), f"{original_path} doesn't exist! aborting..."
        extracted_data = extract_file(tarball_path, start_offset, end_offset)

        if not compare_files(original_path, extracted_data):
            mismatch += 1
            # print(f"Mismatch found in file: {img_name}")
            # return False

    if mismatch:
        print(f"{mismatch}/{subset_size} mismatch found.")
    else:
        print("All files in the random subset match successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test if tarball and entries files were created correctly for pretraining dataset."
    )
    parser.add_argument(
        "-i",
        "--image_root",
        type=str,
        required=True,
        help="Path to the directory containing images.",
    )
    parser.add_argument(
        "-t",
        "--tarball_path",
        type=str,
        required=True,
        help="Path to the tarball file.",
    )
    parser.add_argument(
        "-e",
        "--entries_path",
        type=str,
        required=True,
        help="Path to the entries file.",
    )
    parser.add_argument(
        "-f",
        "--file_indices_path",
        type=str,
        required=True,
        help="Path to the file_indices file.",
    )
    parser.add_argument(
        "-s",
        "--sample_size",
        type=int,
        default=1000,
        help="Number of images to test randomly.",
    )

    args = parser.parse_args()

    test_tarball_entries(
        args.image_root,
        args.tarball_path,
        args.entries_path,
        args.file_indices_path,
        args.sample_size,
    )


if __name__ == "__main__":
    main()
