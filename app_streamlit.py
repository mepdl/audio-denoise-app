import io
import os
import tempfile

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import streamlit as st
from moviepy.editor import VideoFileClip


# ----------------- FUNÇÕES DE PROCESSAMENTO ----------------- #


def remove_silence_segments(
    y: np.ndarray,
    sr: int,
    top_db: float = 30.0,
    max_silence_sec: float = 0.3,
    min_segment_sec: float = 0.1,
) -> np.ndarray:
    """
    Remove (quase) todos os silêncios do áudio usando librosa.effects.split.

    - top_db: sensibilidade do que é considerado "som" vs "silêncio".
      Quanto MENOR, mais agressivo (remove mais coisa).
    - max_silence_sec: pausa máxima mantida entre frases (segundos).
    - min_segment_sec: descarta trechos muito curtos (ruídos, clicks).
    """
    intervals = librosa.effects.split(y, top_db=top_db)

    if len(intervals) == 0:
        st.write(
            "⚠️ Não foi possível detectar segmentos acima do nível de ruído. "
            "Mantendo o áudio original."
        )
        return y

    segments = []
    gap_samples = int(max_silence_sec * sr) if max_silence_sec > 0 else 0
    gap = np.zeros(gap_samples, dtype=y.dtype) if gap_samples > 0 else None

    for i, (start, end) in enumerate(intervals):
        dur = (end - start) / sr
        if dur < min_segment_sec:
            # trecho muito curto, provavelmente ruído → ignora
            continue

        segment = y[start:end]
        segments.append(segment)

        # adiciona silêncio curto entre trechos para não ficar "robô"
        if gap is not None and i < len(intervals) - 1:
            segments.append(gap)

    if not segments:
        st.write(
            "⚠️ Todos os segmentos detectados eram muito curtos. "
            "Mantendo o áudio original."
        )
        return y

    y_out = np.concatenate(segments)
    st.write(
        f"Silêncio removido em todo o áudio. "
        f"Original: {len(y) / sr:.2f}s → Novo: {len(y_out) / sr:.2f}s"
    )
    return y_out


def denoise_array(
    y: np.ndarray,
    sr: int,
    noise_duration: float = 0.5,
    prop_decrease: float = 0.8,
    trim_silence: bool = True,
    trim_top_db: float = 30.0,
    max_silence_sec: float = 0.3,
    min_segment_sec: float = 0.1,
) -> np.ndarray:
    """
    Processa um array de áudio: remove ruído e, opcionalmente,
    remove silêncios ao longo de todo o áudio.
    """
    # Definir trecho inicial usado como amostra de ruído
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
        st.write("Removendo espaços vazios (silêncio) em todo o áudio...")
        reduced_noise = remove_silence_segments(
            reduced_noise,
            sr,
            top_db=trim_top_db,
            max_silence_sec=max_silence_sec,
            min_segment_sec=min_segment_sec,
        )

    # Normalizar para evitar clipping
    max_abs = float(np.max(np.abs(reduced_noise)))
    if max_abs > 1e-9:
        reduced_noise = reduced_noise / max_abs

    return reduced_noise


def extract_audio_from_video(video_path: str) -> str:
    """
    Extrai o áudio de um arquivo de vídeo para um .wav temporário
    e retorna o caminho desse .wav.
    """
    st.write("Extraindo áudio do vídeo...")
    clip = VideoFileClip(video_path)
    audio = clip.audio

    if audio is None:
        raise ValueError("O vídeo não possui trilha de áudio.")

    tmp_audio = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav",
    )
    tmp_audio_path = tmp_audio.name
    tmp_audio.close()

    # write_audiofile salva o áudio em wav
    audio.write_audiofile(tmp_audio_path, logger=None)
    clip.close()

    st.write("Áudio extraído com sucesso.")
    return tmp_audio_path


def is_video_file(filename: str) -> bool:
    ext = os.path.splitext(filename.lower())[1]
    return ext in [".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm"]


# ----------------- INTERFACE STREAMLIT ----------------- #


st.set_page_config(
    page_title="Removedor de Ruído e Silêncio",
    page_icon="🧼",
    layout="centered",
)

st.title("🧼 Removedor de Ruído + Cortador de Silêncio (Áudio e Vídeo)")
st.write(
    "Envie um **áudio** ou **vídeo**. O app vai extrair o **áudio**, "
    "remover ruído de fundo e cortar espaços vazios ao longo do áudio, "
    "mantendo apenas pausas curtas entre as falas."
)

uploaded_file = st.file_uploader(
    "Envie um arquivo de áudio ou vídeo",
    type=[
        "wav",
        "mp3",
        "ogg",
        "flac",
        "m4a",
        "mp4",
        "mov",
        "mkv",
        "avi",
        "wmv",
        "flv",
        "webm",
    ],
    key="uploader_arquivo",
)

noise_duration = st.slider(
    "Duração do trecho inicial considerado ruído (segundos)",
    min_value=0.1,
    max_value=3.0,
    value=0.5,
    step=0.1,
    key="slider_noise_duration",
)

prop_decrease = st.slider(
    "Intensidade da redução de ruído",
    min_value=0.1,
    max_value=1.0,
    value=0.8,
    step=0.05,
    key="slider_prop_decrease",
)

trim_silence = st.checkbox(
    "Remover espaços vazios (silêncio) ao longo do áudio",
    value=True,
    key="checkbox_trim_silence",
)

trim_top_db = st.slider(
    "Sensibilidade do corte de silêncio (dB) – menor = mais agressivo",
    min_value=10,
    max_value=60,
    value=30,
    step=2,
    key="slider_trim_top_db",
)

max_silence_sec = st.slider(
    "Pausa máxima entre trechos (segundos)",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
    key="slider_max_silence_sec",
)

min_segment_sec = st.slider(
    "Duração mínima de um trecho mantido (segundos)",
    min_value=0.05,
    max_value=0.5,
    value=0.1,
    step=0.05,
    key="slider_min_segment_sec",
)

st.markdown("---")

if uploaded_file is not None:
    # Limite de tamanho opcional (ex.: 100 MB pra vídeo)
    if uploaded_file.size > 100 * 1024 * 1024:
        st.error("Arquivo muito grande. Envie um arquivo de até 100 MB.")
    else:
        # Ler bytes uma única vez
        file_bytes = uploaded_file.read()

        # Tipo de preview: se for áudio, usamos st.audio; se for vídeo, st.video.
        ext = os.path.splitext(uploaded_file.name.lower())[1]
        st.subheader("Pré-visualização do arquivo enviado")
        if is_video_file(uploaded_file.name):
            st.video(file_bytes)
        else:
            st.audio(file_bytes)

        if st.button(
            "🚀 Processar (extrair áudio, remover ruído e silêncio)",
            key="btn_processar",
        ):
            with st.spinner("Processando áudio..."):

                # Salvar o upload em arquivo temporário
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f"_{uploaded_file.name}",
                ) as tmp:
                    tmp.write(file_bytes)
                    temp_path = tmp.name

                # Se for vídeo → extrai áudio para wav temporário
                if is_video_file(uploaded_file.name):
                    audio_path = extract_audio_from_video(temp_path)
                else:
                    audio_path = temp_path

                # Carregar áudio com librosa
                y, sr = librosa.load(audio_path, sr=None, mono=True)

                # Processar (ruído + silêncio)
                reduced = denoise_array(
                    y,
                    sr,
                    noise_duration=noise_duration,
                    prop_decrease=prop_decrease,
                    trim_silence=trim_silence,
                    trim_top_db=trim_top_db,
                    max_silence_sec=max_silence_sec,
                    min_segment_sec=min_segment_sec,
                )

                # Salvar em buffer em memória (WAV)
                buf = io.BytesIO()
                sf.write(buf, reduced, sr, format="WAV")
                buf.seek(0)

                st.success("Áudio processado com sucesso!")

                st.subheader("Áudio processado (somente áudio)")
                st.audio(buf, format="audio/wav")

                st.download_button(
                    label="⬇️ Baixar áudio processado (WAV)",
                    data=buf,
                    file_name="audio_denoised_trimmed.wav",
                    mime="audio/wav",
                    key="btn_download",
                )
else:
    st.info("Envie um arquivo de áudio ou vídeo para começar.")
