import argparse
import concurrent.futures
import math
import os
import random

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

def load_file(path):
    """Read parquet file into Arrow Table (only text col)."""
    pf = pq.ParquetFile(path)
    batches = [b for b in pf.iter_batches(columns=["text"])]
    return pa.Table.from_batches(batches)


def main():
    parser = argparse.ArgumentParser(description="Globally shuffle parquet shards containing a `text` column.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input parquet shards.")
    parser.add_argument("--output_dir", required=True, help="Directory to write shuffled parquet shards.")
    parser.add_argument("--max_shards", type=int, default=50, help="Maximum number of output shards.")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel workers used for parquet loading.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = ds.dataset(args.input_dir, format="parquet")
    files = dataset.files
    print(f"Found {len(files)} files")

    tables = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for table in tqdm(executor.map(load_file, files), total=len(files), desc="Loading files"):
            tables.append(table)

    full_table = pa.concat_tables(tables, promote=True)
    full_table = full_table.set_column(
        0,
        full_table.column_names[0],
        full_table[0].cast(pa.large_string()),
    )
    print(f"Total rows: {full_table.num_rows}, type: {full_table.schema}")

    random.seed(args.seed)
    indices = list(range(full_table.num_rows))
    random.shuffle(indices)
    shuffled = full_table.take(pa.array(indices))
    print("Shuffling done.")

    rows_per_shard = math.ceil(shuffled.num_rows / args.max_shards)
    for i in tqdm(range(args.max_shards), desc="Writing shards"):
        start = i * rows_per_shard
        end = min((i + 1) * rows_per_shard, shuffled.num_rows)
        if start >= end:
            break
        shard = shuffled.slice(start, end - start)
        pq.write_table(shard, os.path.join(args.output_dir, f"train-{i:05d}.parquet"))


if __name__ == "__main__":
    main()
