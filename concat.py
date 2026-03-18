import os
import torch
import torchaudio

input_dir = 'snare_padded'
output_dir = 'snare_concat'

# Create the new folder for the long files
os.makedirs(output_dir, exist_ok=True)

# Grab all the wav files and sort them so they group neatly
files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
files.sort()

group_size = 20

print(f"Found {len(files)} files. Grouping into chunks of {group_size}...")

for i in range(0, len(files), group_size):
    chunk = files[i:i + group_size]
    tensors = []
    
    # Load each file in the group
    for f in chunk:
        filepath = os.path.join(input_dir, f)
        try:
            wav, sr = torchaudio.load(filepath)
            tensors.append(wav)
            sample_rate = sr
        except Exception as e:
            print(f"Skipping broken file {f}: {e}")
            
    if not tensors:
        continue
        
    # Stitch them together end-to-end along the time dimension
    concatenated_wav = torch.cat(tensors, dim=1)
    
    # Save the new 30-second file safely
    out_filename = f"snare_group_{i//group_size + 1:03d}.wav"
    out_filepath = os.path.join(output_dir, out_filename)
    
    torchaudio.save(out_filepath, concatenated_wav, sample_rate, encoding="PCM_S", bits_per_sample=16)
    
print(f"Done! Saved combined files to the '{output_dir}' folder.")
