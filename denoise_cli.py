<<<<<<< HEAD
import os
import argparse
from typing import Optional

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf


def remove_silence_edges(y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """
    Remove silêncio do início e do fim do áudio usando librosa.effects.trim.

    top_db: quanto menor, mais sensível (remove mais partes baixas).
    """
    yt, idx = librosa.effects.trim(y, top_db=top_db)
    print(
        f"Silêncio removido nas bordas. Amostras originais: {len(y)}, "
        f"após corte: {len(yt)} (índices {idx[0]}:{idx[1]})."
    )
    return yt


def denoise_file(
    input_path: str,
    output_path: Optional[str] = None,
    noise_duration: float = 0.5,
    prop_decrease: float = 0.8,
    trim_silence: bool = True,
    trim_top_db: float = 30.0,
) -> str:
    """
    Remove ruído e silêncio (opcional) de um arquivo de áudio.

    Parâmetros:
        input_path: caminho do arquivo de entrada (wav, mp3, etc.).
        output_path: caminho do arquivo de saída. Se None, gera automaticamente.
        noise_duration: segundos iniciais usados como amostra de ruído.
        prop_decrease: intensidade da redução de ruído (0–1). 0.8 = ~80% do ruído.
        trim_silence: se True, remove silêncio do início/fim.
        trim_top_db: sensibilidade do corte de silêncio (dB).

    Retorna:
        Caminho do arquivo gerado.
    """

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

    print(f"Carregando áudio: {input_path}")
    # sr=None para manter a taxa original; mono=True para unificar canais
    y, sr = librosa.load(input_path, sr=None, mono=True)
    print(f"Áudio carregado. Duração: {len(y) / sr:.2f} s | Sample rate: {sr} Hz")

    if len(y) == 0:
        raise ValueError("O arquivo de áudio está vazio ou não pôde ser lido.")

    # 1) Redução de ruído
    n_noise_samples = int(noise_duration * sr)
    if n_noise_samples >= len(y):
        # Se o áudio for muito curto, usar 20% do áudio como "ruído"
        n_noise_samples = max(1, int(0.2 * len(y)))
        print(
            "⚠️ Áudio curto. Ajustando amostra de ruído para "
            f"{n_noise_samples / sr:.2f} s (≈ 20% do áudio)."
        )

    noise_clip = y[:n_noise_samples]
    print(
        f"Usando {n_noise_samples} amostras iniciais "
        f"({n_noise_samples / sr:.2f} s) como ruído."
    )

    print("Aplicando redução de ruído...")
    y_denoised = nr.reduce_noise(
        y=y,
        y_noise=noise_clip,
        sr=sr,
        prop_decrease=prop_decrease,
    )

    # 2) Remoção de silêncio nas bordas (opcional)
    if trim_silence:
        print("Removendo espaços vazios (silêncio) do início/fim...")
        y_denoised = remove_silence_edges(y_denoised, top_db=trim_top_db)

    # 3) Normalização de amplitude para evitar clipping
    max_abs = float(np.max(np.abs(y_denoised)))
    if max_abs > 1e-9:
        y_denoised = y_denoised / max_abs

    # 4) Definir nome de saída, se não informado
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        if not ext:
            ext = ".wav"
        if trim_silence:
            output_path = f"{base}_denoised_trimmed{ext}"
        else:
            output_path = f"{base}_denoised{ext}"

    # 5) Salvar arquivo processado
    sf.write(output_path, y_denoised, sr)
    print(f"✅ Áudio processado salvo em: {output_path}")

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ferramenta de remoção de ruído e silêncio em áudio "
            "usando noisereduce + librosa."
        )
    )

    parser.add_argument(
        "input",
        help="Caminho do arquivo de áudio de entrada (ex.: audio.wav)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Caminho do arquivo de saída (opcional). "
            "Se não informado, será gerado *_denoised[_trimmed].ext*."
        ),
        default=None,
    )
    parser.add_argument(
        "-n",
        "--noise-duration",
        type=float,
        default=0.5,
        help="Duração (em segundos) do trecho inicial usado como ruído. Padrão: 0.5",
    )
    parser.add_argument(
        "-p",
        "--prop-decrease",
        type=float,
        default=0.8,
        help="Intensidade da redução de ruído (0–1). Padrão: 0.8",
    )
    parser.add_argument(
        "--no-trim-silence",
        action="store_true",
        help="Não remover silêncio do início/fim do áudio.",
    )
    parser.add_argument(
        "--trim-top-db",
        type=float,
        default=30.0,
        help="Sensibilidade para corte de silêncio (dB). Menor = mais agressivo. Padrão: 30.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        denoise_file(
            input_path=args.input,
            output_path=args.output,
            noise_duration=args.noise_duration,
            prop_decrease=args.prop_decrease,
            trim_silence=not args.no_trim_silence,
            trim_top_db=args.trim_top_db,
        )
    except FileNotFoundError as e:
        print(f"❌ Erro: {e}")
    except Exception as e:
        print("❌ Ocorreu um erro durante o processamento:")
        print(e)


if __name__ == "__main__":
    main()
=======
import os
import argparse
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np


def denoise_file(
    input_path: str,
    output_path: str | None = None,
    noise_duration: float = 0.5,
    prop_decrease: float = 0.8,
) -> str:
    """
    Remove ruído de um arquivo de áudio usando noisereduce + librosa.

    Parâmetros:
        input_path: caminho do arquivo de entrada (wav, mp3, etc.).
        output_path: caminho do arquivo de saída. Se None, gera automaticamente.
        noise_duration: segundos iniciais usados como amostra de ruído.
        prop_decrease: intensidade da redução (0–1). 0.8 = remove ~80% do ruído.

    Retorna:
        Caminho do arquivo gerado.
    """

    # 1. Verificar se o arquivo existe
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

    print(f"Carregando áudio: {input_path}")

    # 2. Carregar áudio com librosa (mantendo a taxa original)
    y, sr = librosa.load(input_path, sr=None, mono=True)
    print(f"Áudio carregado. Duração: {len(y) / sr:.2f} s | Sample rate: {sr} Hz")

    if len(y) == 0:
        raise ValueError("O arquivo de áudio está vazio ou não pôde ser lido.")

    # 3. Definir amostra de ruído (primeiros X segundos)
    n_noise_samples = int(noise_duration * sr)

    if n_noise_samples >= len(y):
        # Se o áudio for muito curto, usa até 20% do áudio como ruído
        n_noise_samples = max(1, int(0.2 * len(y)))
        print(
            "⚠️ Áudio curto. Ajustando amostra de ruído para "
            f"{n_noise_samples / sr:.2f} s (≈ 20% do áudio)."
        )

    noise_clip = y[:n_noise_samples]

    print(
        f"Usando {n_noise_samples} amostras iniciais "
        f"({n_noise_samples / sr:.2f} s) como ruído."
    )

    # 4. Aplicar redução de ruído
    print("Aplicando redução de ruído... (isso pode levar alguns segundos)")
    reduced_noise = nr.reduce_noise(
        y=y,
        y_noise=noise_clip,
        sr=sr,
        prop_decrease=prop_decrease,
    )

    # 5. Normalizar e garantir faixa válida para salvar
    max_abs = np.max(np.abs(reduced_noise))
    if max_abs > 1:
        reduced_noise = reduced_noise / max_abs

    # 6. Definir nome de saída, se não informado
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        if ext == "":
            ext = ".wav"
        output_path = f"{base}_denoised{ext}"

    # 7. Salvar com soundfile
    sf.write(output_path, reduced_noise, sr)
    print(f"✅ Áudio sem ruído salvo em: {output_path}")

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ferramenta simples de remoção de ruído em áudio "
                    "usando noisereduce + librosa."
    )
    parser.add_argument(
        "input",
        help="Caminho do arquivo de áudio de entrada (ex.: audio.wav)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Caminho do arquivo de saída (opcional). "
             "Se não informado, será gerado *_denoised.ext*.",
        default=None,
    )
    parser.add_argument(
        "-n",
        "--noise-duration",
        type=float,
        default=0.5,
        help="Duração (em segundos) do trecho inicial usado como ruído. Padrão: 0.5",
    )
    parser.add_argument(
        "-p",
        "--prop-decrease",
        type=float,
        default=0.8,
        help="Intensidade da redução de ruído (0–1). Padrão: 0.8",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        denoise_file(
            input_path=args.input,
            output_path=args.output,
            noise_duration=args.noise_duration,
            prop_decrease=args.prop_decrease,
        )
    except FileNotFoundError as e:
        print(f"❌ Erro: {e}")
    except Exception as e:
        print("❌ Ocorreu um erro durante o processamento:")
        print(e)


if __name__ == "__main__":
    main()
>>>>>>> dc9aa0b33e2f37055e73869652c2da4a6901cefe
