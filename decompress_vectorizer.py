import gzip
import pickle

# Step 1.1: Define paths
compressed_file_path = "vectorizer_compressed.pkl.gz"  # Your compressed file
decompressed_file_path = "vectorizer.pkl"  # Output file after decompression

# Step 1.2: Decompress the file
with gzip.open(compressed_file_path, 'rb') as f_in:
    with open(decompressed_file_path, 'wb') as f_out:
        f_out.write(f_in.read())

print("✅ Vectorizer decompressed successfully!")

# Step 1.3: Load the vectorizer (to check if it's working)
with open(decompressed_file_path, 'rb') as f:
    vectorizer = pickle.load(f)

print("✅ Vectorizer loaded successfully!")
