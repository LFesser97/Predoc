"""
subset_mag.py

Created on Mon Mar 18 2024

@author: Lukas

This script is used to split the MAG papers dataset into smaller chunks.
"""

import bz2
import os


def split_bz2_file(input_file, chunk_size, output_prefix):
    # Open the input .bz2 file for reading
    with bz2.open(input_file, 'rb') as f:
        # Initialize chunk variables
        total_size = os.path.getsize(input_file)
        total_chunks = total_size // chunk_size
        current_chunk = 0
        
        # Read data from the input file and write to smaller output files
        while True:
            print(f"Processing chunk {current_chunk + 1} of {total_chunks + 1}")
            # Read a chunk of data from the input file
            chunk = f.read(chunk_size)
            
            # If no more data, break the loop
            if not chunk:
                break
            
            # Write the chunk to a new output file
            output_file = f"{output_prefix}_{current_chunk + 1}.bz2"
            with open(output_file, 'wb') as output:
                output.write(chunk)
            
            # Move to the next chunk
            current_chunk += 1


if __name__ == "__main__":
    # Define the input file and chunk size
    input_file = "10.Papers.nt.bz2"
    chunk_size = 10 * 1024 * 1024 * 1024  # 10 GB
    output_prefix = "papers_10GB_chunk"

    # Split the input file into smaller chunks
    split_bz2_file(input_file, chunk_size, output_prefix)