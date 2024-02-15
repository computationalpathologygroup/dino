import tqdm
import argparse
import numpy as np

from pathlib import Path
from typing import Optional


def concat_entries(
    entries_paths,
    output_root,
    suffix: Optional[str] = None,
):
    try:
        cohort_indices = {}  # store cohort names with an index
        concat_entries = np.load(entries_paths[0])
        cohort_index_column = np.full((concat_entries.shape[0], 1), 0)
        concat_entries = np.hstack([concat_entries, cohort_index_column])
        cohort_name = entries_paths[0].stem.split("_entries")[0]
        cohort_indices[0] = cohort_name

        # load and concatenate the rest of the files
        with tqdm.tqdm(
            entries_paths[1:],
            desc="Concatenating entries",
            unit=" file",
            initial=1,
            total=len(entries_paths),
            leave=True,
        ) as t:
            for i, e in enumerate(t):
                data = np.load(e)
                cohort_index_column = np.full((data.shape[0], 1), i + 1)
                data = np.hstack([data, cohort_index_column])
                cohort_name = e.stem.split("_entries")[0]
                cohort_indices[i + 1] = cohort_name
                concat_entries = np.concatenate((concat_entries, data))

        # save cohort indices
        cohort_indices_filepath = Path(output_root, "cohort_indices.npy")
        np.save(cohort_indices_filepath, cohort_indices)

        # save concatenated entries
        if suffix:
            entries_filepath = Path(output_root, f"pretrain_entries_{suffix}.npy")
        else:
            entries_filepath = Path(output_root, "pretrain_entries.npy")
        np.save(entries_filepath, np.array(concat_entries, dtype=np.uint64))
        print(f"Concatenated entries saved to: {entries_filepath}")

    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate entriy files from multiple datasets."
    )
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        required=True,
        help="Folder where entry files to concatenate are located.",
    )
    parser.add_argument(
        "-o",
        "--output_root",
        type=str,
        required=True,
        help="Path to the output directory where the concatenated entry will be saved.",
    )
    parser.add_argument(
        "-s",
        "--suffix",
        type=str,
        help="Suffix to append at the end of the concatenated entries file name.",
    )

    args = parser.parse_args()
    suffix = f"{args.suffix}" if args.suffix else None

    # grab all entries
    entries_paths = sorted(
        [fp for fp in Path(args.root).glob("*.npy") if "entries" in str(fp.name)]
    )
    assert len(entries_paths) > 0, f"0 entry file found under {args.root}"
    print(f"{len(entries_paths)} entry files found!")

    concat_entries(entries_paths, args.output_root, suffix)


if __name__ == "__main__":
    # python scripts/concat_entries.py --root /root/data --output_root /root/data
    main()
