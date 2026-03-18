import streamlit as st
import torch
import torchaudio
import io

# 1. New Helper Function: Normalize audio to a specific dB target
def normalize_audio(wav_tensor, target_db=-6.0):
    # Find the loudest peak in the audio wave
    peak = wav_tensor.abs().max()
    
    # Safety check to prevent dividing by zero if the audio is completely silent
    if peak == 0:
        return wav_tensor
    
    # Convert the -6 dB target into a linear amplitude multiplier
    target_amplitude = 10 ** (target_db / 20.0)
    
    # Scale the wave so the peak is exactly 1.0, then scale to -6 dB
    return (wav_tensor / peak) * target_amplitude

# 2. Define your local models here
MODELS = {
    "model_4_final": "models/model_4.ts",
    "model_4_endphase1": "models/m4_endphase1.ts",
    "model_4_midphase2": "models/m4_midphase2.ts",
    "model_3.2": "models/SundayMorning2.ts", 
}

# Cache the model so it doesn't reload every time you click a button
@st.cache_resource
def load_model(model_name):
    return torch.jit.load(MODELS[model_name], map_location='cpu')

st.title("RAVE Latent Space Explorer")
st.markdown("Upload a snare sample, inject variance into the latent space, and generate new local variations.")

# 3. Build the Sidebar UI
with st.sidebar:
    st.header("Controls")
    selected_model = st.selectbox("Select Model", list(MODELS.keys()))
    variance = st.slider("Variance (Latent Noise)", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
    n_samples = st.number_input("Number of Samples (N)", min_value=1, max_value=20, value=4)

# 4. Main Interface
uploaded_file = st.file_uploader("Upload Audio File (WAV)", type=['wav'])

if uploaded_file is not None:
    st.subheader("Original Sample (Normalized to -6 dB)")
    
    # Read the uploaded audio
    wav, sr = torchaudio.load(uploaded_file)
    
    # Force mono (RAVE expects 1 channel)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        
    # NORMALIZE THE INPUT AUDIO TO -6 dB
    wav = normalize_audio(wav, target_db=-6.0)
    
    # Save the normalized input to a buffer so you can hear the volume change in the UI
    input_buffer = io.BytesIO()
    torchaudio.save(input_buffer, wav, sr, format="wav")
    input_buffer.seek(0)
    st.audio(input_buffer, format='audio/wav')
    
    # Add batch dimension for the model: shape becomes (1, 1, Length)
    wav = wav.unsqueeze(0)
    
    # 1. Initialize Session State memory (Put this right above your Generate button)
    if 'generated_tracks' not in st.session_state:
        st.session_state.generated_tracks = []

    # THE GENERATION BLOCK
    if st.button("Generate Variations"):
        with st.spinner("Crunching latent math..."):
            
            # Clear the old tracks from memory before making new ones
            st.session_state.generated_tracks = []
            
            model = load_model(selected_model)
            
            with torch.no_grad():
                z = model.encode(wav)
                
                for i in range(int(n_samples)):
                    noise = torch.randn_like(z)
                    z_varied = z + (noise * variance)
                    generated_wav = model.decode(z_varied).squeeze(0)
                    generated_wav = normalize_audio(generated_wav, target_db=-6.0)
                    
                    # Save to buffer and extract the raw bytes
                    buffer = io.BytesIO()
                    torchaudio.save(buffer, generated_wav, sr, format="wav")
                    
                    # Store the raw audio bytes and the filename in session state
                    st.session_state.generated_tracks.append({
                        "bytes": buffer.getvalue(),
                        "filename": f"snare_var_{variance}_{i+1}.wav"
                    })

    # THE DISPLAY BLOCK (Notice this is moved completely OUTSIDE the button's indentation!)
    if len(st.session_state.generated_tracks) > 0:
        st.subheader("Generated Variations (Normalized to -6 dB)")
        cols = st.columns(2) 
        
        for i, track_data in enumerate(st.session_state.generated_tracks):
            with cols[i % 2]:
                st.write(f"Variation {i+1}")
                # Play the audio from the saved bytes
                st.audio(track_data["bytes"], format='audio/wav')
                
                # Download button using the saved bytes and filename
                st.download_button(
                    label="⬇️ Download WAV",
                    data=track_data["bytes"],
                    file_name=track_data["filename"],
                    mime="audio/wav",
                    key=f"download_btn_{i}" # Keys ensure Streamlit doesn't confuse multiple buttons
                )