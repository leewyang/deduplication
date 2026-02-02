import glob
import os

import fire
import pyarrow.parquet as pq
import pyarrow as pa


def convert_arrow_to_parquet(arrow_file, parquet_file):
    # HuggingFace datasets use Arrow IPC stream format, not Feather format
    # Use RecordBatchStreamReader to read the file
    with pa.memory_map(arrow_file, 'r') as source:
        reader = pa.ipc.open_stream(source)
        table = reader.read_all()

    # Write the Table to a Parquet file
    pq.write_table(table, parquet_file, compression='snappy')

    print(f"Successfully converted {arrow_file} to {parquet_file}")


def main(arrow_dir, parquet_dir):
    """Convert all Arrow files in a directory to Parquet files.

    Args:
        arrow_dir: Directory containing Arrow files.
        parquet_dir: Directory to save Parquet files.
    """
    os.makedirs(parquet_dir, exist_ok=True)
    arrow_files = glob.glob(f'{arrow_dir}/**/*.arrow', recursive=True)

    for arrow_file in arrow_files:
        stem = os.path.splitext(os.path.basename(arrow_file))[0]
        parquet_file = os.path.join(parquet_dir, f"{stem}.parquet")
        convert_arrow_to_parquet(arrow_file, parquet_file)


if __name__ == "__main__":
    fire.Fire(main)
