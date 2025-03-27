import gzip
import pickle

# Step 2.1: Define file paths
original_file = "vectorizer.pkl"  # Your original vectorizer file
compressed_file = "vectorizer_compressed.pkl.gz"  # Compressed output file

# Step 2.2: Compress the file
with open(original_file, 'rb') as f_in:
    with gzip.open(compressed_file, 'wb') as f_out:
        f_out.writelines(f_in)

print("✅ Vectorizer file compressed successfully!")
