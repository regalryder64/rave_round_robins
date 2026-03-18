import os
import torch
import torchaudio

input_dir = 'rmod_samples/snare_wav'
output_dir = 'snare_padded'

# Create the new safe folder
os.makedirs(output_dir, exist_ok=True)

for f in os.listdir(input_dir):
    if f.endswith('.wav'):
        in_path = os.path.join(input_dir, f)
        out_path = os.path.join(output_dir, f)
        
        try:
            # 1. Load the raw audio using PyTorch
            wav, sr = torchaudio.load(in_path)
            
            # 2. Calculate exactly 1.5 seconds of samples
            target_samples = int(sr * 1.5)
            current_samples = wav.shape[1]
            
            if current_samples < target_samples:
                # Add pure silence to the end if it's too short
                pad_amount = target_samples - current_samples
                wav_padded = torch.nn.functional.pad(wav, (0, pad_amount))
            else:
                # Trim the tail if it's somehow longer than 1.5s
                wav_padded = wav[:, :target_samples]
                
            # 3. Save as a strict, universally accepted 16-bit PCM WAV
            torchaudio.save(out_path, wav_padded, sr, encoding="PCM_S", bits_per_sample=16)
            
        except Exception as e:
            print(f"Skipping corrupted file: {f}")

print("Padding complete! All files saved safely to the 'snare_padded' folder.")
