<<<<<<< HEAD
import io
import tempfile

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import streamlit as st


def remove_silence_edges(y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """
    Remove silêncio do início e do fim do áudio usando librosa.effects.trim.
    """
    yt, idx = librosa.effects.trim(y, top_db=top_db)
    st.write(
        f"Silêncio removido nas bordas. Amostras originais: {len(y)}, "
        f"após corte: {len(yt)} (índices {idx[0]}:{idx[1]})."
    )
    return yt


def denoise_array(
    y: np.ndarray,
    sr: int,
    noise_duration: float = 0.5,
    prop_decrease: float = 0.8,
    trim_silence: bool = True,
    trim_top_db: float = 30.0,
) -> np.ndarray:
    """
    Processa um array de áudio: remove ruído e, opcionalmente, silêncio das bordas.
    """
    n_noise_samples = int(noise_duration * sr)
    if n_noise_samples >= len(y):
        n_noise_samples = max(1, int(0.2 * len(y)))
        st.write(
            "⚠️ Áudio curto. Ajustando amostra de ruído para "
            f"{n_noise_samples / sr:.2f} s (≈ 20% do áudio)."
        )

    noise_clip = y[:n_noise_samples]

    st.write("Aplicando redução de ruído...")
    reduced_noise = nr.reduce_noise(
        y=y,
        y_noise=noise_clip,
        sr=sr,
        prop_decrease=prop_decrease,
    )

    if trim_silence:
        st.write("Removendo espaços vazios (silêncio) do início/fim...")
        reduced_noise = remove_silence_edges(reduced_noise, top_db=trim_top_db)

    max_abs = float(np.max(np.abs(reduced_noise)))
    if max_abs > 1e-9:
        reduced_noise = reduced_noise / max_abs

    return reduced_noise


# ----------------- INTERFACE STREAMLIT ----------------- #

st.set_page_config(
    page_title="Removedor de Ruído e Silêncio",
    page_icon="🧼",
    layout="centered",
)

st.title("🧼 Removedor de Ruído + Cortador de Silêncio")
st.write(
    "Envie um arquivo de áudio, o app vai **remover ruído de fundo** "
    "e, opcionalmente, **cortar espaços vazios do início e do fim**."
)

uploaded_file = st.file_uploader(
    "Envie um arquivo de áudio",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
)

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

trim_silence = st.checkbox(
    "Remover espaços vazios (silêncio) do início/fim do áudio",
    value=True,
)

trim_top_db = st.slider(
    "Sensibilidade do corte de silêncio (dB) – menor = mais agressivo",
    min_value=10,
    max_value=60,
    value=30,
    step=2,
)

st.markdown("---")

if uploaded_file is not None:
    # Limite de tamanho opcional (ex.: 20 MB)
    if uploaded_file.size > 20 * 1024 * 1024:
        st.error("Arquivo muito grande. Envie um áudio de até 20 MB.")
    else:
        # Ler bytes uma única vez (para evitar problemas de ponteiro)
        audio_bytes = uploaded_file.read()

        st.subheader("Áudio original")
        st.audio(audio_bytes)

        if st.button("🚀 Processar áudio (remover ruído e espaços vazios)"):
            with st.spinner("Processando áudio..."):

                # Salvar temporariamente para o librosa ler
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f"_{uploaded_file.name}",
                ) as tmp:
                    tmp.write(audio_bytes)
                    temp_path = tmp.name

                # Carregar áudio com librosa
                y, sr = librosa.load(temp_path, sr=None, mono=True)

                # Processar (ruído + silêncio)
                reduced = denoise_array(
                    y,
                    sr,
                    noise_duration=noise_duration,
                    prop_decrease=prop_decrease,
                    trim_silence=trim_silence,
                    trim_top_db=trim_top_db,
                )

                # Salvar em buffer em memória
                buf = io.BytesIO()
                sf.write(buf, reduced, sr, format="WAV")
                buf.seek(0)

                st.success("Áudio processado com sucesso!")

                st.subheader("Áudio processado")
                st.audio(buf, format="audio/wav")

                st.download_button(
                    label="⬇️ Baixar áudio processado",
                    data=buf,
                    file_name="audio_denoised_trimmed.wav",
                    mime="audio/wav",
                )
else:
    st.info("Envie um arquivo de áudio para começar.")
=======
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
>>>>>>> dc9aa0b33e2f37055e73869652c2da4a6901cefe
