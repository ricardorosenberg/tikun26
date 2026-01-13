import librosa
import numpy as np

# Load the fire alarm audio file
filename = "alarmeincendio.wav"

print(f"Loading {filename}...")
audio, sample_rate = librosa.load(filename, sr=None)

# Display basic information
print(f"\n--- Audio Information ---")
print(f"Filename: {filename}")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {len(audio) / sample_rate:.2f} seconds")
print(f"Number of samples: {len(audio)}")

# Extract MFCC features (these are useful for sound recognition)
print(f"\n--- Extracting Features ---")
mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
print(f"MFCC features shape: {mfccs. shape}")
print(f"MFCC features extracted successfully!")

# Calculate some statistics
print(f"\n--- Audio Statistics ---")
print(f"Max amplitude: {np.max(audio):.4f}")
print(f"Min amplitude: {np.min(audio):.4f}")
print(f"Mean amplitude: {np. mean(audio):.4f}")
