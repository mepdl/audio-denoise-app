import streamlit as st
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
import io
import tempfile


def denoise_array(y, sr, noise_duration=0.5, prop_decrease=0.8):
    n_noise_samples = int(noise_duration * sr)
    if n_noise_samples >= len(y):
        n_noise_samples = max(1, int(0.2 * len(y)))

    noise_clip = y[:n_noise_samples]

    reduced_noise = nr.reduce_noise(
        y=y,
        y_noise=noise_clip,
        sr=sr,
        prop_decrease=prop_decrease,
    )

    max_abs = np.max(np.abs(reduced_noise))
    if max_abs > 1:
        reduced_noise = reduced_noise / max_abs

    return reduced_noise


st.title("🧼 Removedor de Ruído de Áudio (noisereduce + librosa)")

uploaded_file = st.file_uploader("Envie um arquivo de áudio", type=["wav", "mp3", "ogg", "flac"])

noise_duration = st.slider(
    "Duração do trecho inicial considerado ruído (segundos)",
    min_value=0.1,
    max_value=3.0,
    value=0.5,
    step=0.1,
)

prop_decrease = st.slider(
    "Intensidade da redução de ruído",
    min_value=0.1,
    max_value=1.0,
    value=0.8,
    step=0.05,
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    if st.button("Remover ruído"):
        with st.spinner("Processando áudio..."):
            # Salvar temporariamente para ler com librosa
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            y, sr = librosa.load(temp_path, sr=None, mono=True)
            reduced = denoise_array(
                y,
                sr,
                noise_duration=noise_duration,
                prop_decrease=prop_decrease,
            )

            # Salvar em buffer
            buf = io.BytesIO()
            sf.write(buf, reduced, sr, format="WAV")
            buf.seek(0)

            st.success("Áudio processado com sucesso!")
            st.audio(buf, format="audio/wav")
            st.download_button(
                label="⬇️ Baixar áudio sem ruído",
                data=buf,
                file_name="audio_denoised.wav",
                mime="audio/wav",
            )
