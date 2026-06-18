"""
Bot de Geração Automática de Vídeos - Fala Brasil App
======================================================
Pipeline: GPT (script) → OpenAI TTS (narração) → Runway ML (vídeo) → YouTube (upload)

Dependências:
    pip install openai runwayml google-api-python-client google-auth-oauthlib moviepy requests python-dotenv
"""

import os
import time
import random
import requests
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI
import runwayml
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
RUNWAY_API_KEY   = os.getenv("RUNWAY_API_KEY")
YOUTUBE_CLIENT_SECRETS = "client_secrets.json"   # baixe do Google Cloud Console
OUTPUT_DIR       = Path("output_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────
# TEMAS PROMOCIONAIS DO FALA BRASIL
# ─────────────────────────────────────────────
TEMAS = [
    # App Fala Brasil
    "Fala Brasil: o app de mensagens mais seguro do Brasil",
    "Como o Fala Brasil protege suas mensagens com criptografia",
    "Fala Brasil vs WhatsApp: qual é mais seguro para você?",
    "Por que brasileiros estão migrando para o Fala Brasil",
    "Fala Brasil: comunicação 100% brasileira e privada",

    # Tecnologia
    "As maiores tendências de tecnologia para 2026",
    "Como a tecnologia está mudando a forma de nos comunicar",
    "5G, IA e apps: o futuro da comunicação no Brasil",
    "Por que escolher apps nacionais de tecnologia",
    "O crescimento das fintechs e apps brasileiros",

    # Inteligência Artificial
    "Como a IA está revolucionando os apps de mensagem",
    "IA no Fala Brasil: mensagens mais inteligentes e seguras",
    "O que a inteligência artificial pode fazer pela sua privacidade",
    "ChatGPT, Gemini e o futuro da IA no Brasil",
    "Como a IA detecta golpes e protege suas conversas",

    # Segurança Cibernética
    "Como se proteger de golpes no WhatsApp e redes sociais",
    "Segurança cibernética: o que você precisa saber em 2026",
    "Seus dados estão seguros? Entenda a LGPD e privacidade digital",
    "Os maiores erros de segurança digital que as pessoas cometem",
    "Por que criptografia ponta a ponta é essencial nos apps de mensagem",
]

# ─────────────────────────────────────────────
# PASSO 1: GERAR SCRIPT COM GPT
# ─────────────────────────────────────────────
def gerar_script(tema: str) -> dict:
    """Retorna {titulo, descricao, script, prompt_video}"""
    print(f"[1/4] Gerando script para: {tema}")

    resposta = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um roteirista de vídeos curtos do YouTube (60s) para um canal "
                    "sobre tecnologia, inteligência artificial e segurança cibernética. "
                    "O canal também promove o 'Fala Brasil', um app de mensagens brasileiro "
                    "seguro e privado. Seja direto, informativo e persuasivo."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Crie um vídeo promocional sobre: {tema}\n\n"
                    "Responda em JSON com:\n"
                    "- titulo: título chamativo para o YouTube (max 70 chars)\n"
                    "- descricao: descrição do vídeo com hashtags (max 300 chars)\n"
                    "- script: narração do vídeo (max 150 palavras, voz animada)\n"
                    "- prompt_video: prompt em inglês para geração de vídeo com IA "
                    "(cenas visuais relacionadas ao tema, sem texto na tela)"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    import json
    dados = json.loads(resposta.choices[0].message.content)
    print(f"   Título: {dados['titulo']}")
    return dados


# ─────────────────────────────────────────────
# PASSO 2: GERAR NARRAÇÃO COM OPENAI TTS
# ─────────────────────────────────────────────
def gerar_narracao(script: str, caminho_audio: Path) -> Path:
    """Gera MP3 com narração do script."""
    print("[2/4] Gerando narração com OpenAI TTS...")

    vozes = ["nova", "shimmer", "alloy"]  # vozes femininas/neutras
    voz = random.choice(vozes)

    resposta = openai_client.audio.speech.create(
        model="tts-1-hd",
        voice=voz,
        input=script,
        response_format="mp3",
    )

    resposta.stream_to_file(caminho_audio)
    print(f"   Áudio salvo em: {caminho_audio}")
    return caminho_audio


# ─────────────────────────────────────────────
# PASSO 3: GERAR VÍDEO COM EFEITO KEN BURNS
# ─────────────────────────────────────────────
def gerar_video_ken_burns(prompt: str, caminho_video: Path) -> Path:
    """Gera vídeo animado com efeito Ken Burns a partir de imagem DALL-E."""
    print("[3/4] Gerando imagem com DALL-E + efeito Ken Burns...")

    import base64, io
    import numpy as np
    from PIL import Image
    from moviepy import VideoClip

    # Gerar imagem com DALL-E
    img_resposta = openai_client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1536x1024",
        quality="medium",
        n=1,
    )
    imagem_bytes = base64.b64decode(img_resposta.data[0].b64_json)
    img = Image.open(io.BytesIO(imagem_bytes)).convert("RGB")
    img_array = np.array(img)
    print("   Imagem gerada, aplicando animação...")

    W, H = 1280, 720
    duracao = 15
    fps = 24
    ih, iw = img_array.shape[:2]

    # Zoom lento 100% → 120% com pan suave
    def make_frame(t):
        progresso = t / duracao
        zoom = 1.0 + 0.2 * progresso
        cw = int(iw / zoom)
        ch = int(ih / zoom)
        x0 = int((iw - cw) * progresso * 0.5)
        y0 = int((ih - ch) * 0.5)
        recorte = img_array[y0:y0+ch, x0:x0+cw]
        return np.array(Image.fromarray(recorte).resize((W, H), Image.LANCZOS))

    clip = VideoClip(make_frame, duration=duracao)
    clip.write_videofile(str(caminho_video), fps=fps, codec="libx264",
                         audio=False, logger=None)
    print(f"   Vídeo gerado: {caminho_video}")
    return caminho_video


# ─────────────────────────────────────────────
# PASSO 3B: COMBINAR VÍDEO + ÁUDIO
# ─────────────────────────────────────────────
def combinar_video_audio(caminho_video: Path, caminho_audio: Path, saida: Path) -> Path:
    """Une o vídeo do Runway com a narração TTS."""
    from moviepy import VideoFileClip, AudioFileClip

    print("   Combinando vídeo + áudio...")
    video = VideoFileClip(str(caminho_video))
    audio = AudioFileClip(str(caminho_audio))

    # Ajusta duração do vídeo ao áudio (loop se necessário)
    if audio.duration > video.duration:
        from moviepy import vfx
        video = video.with_effects([vfx.Loop(duration=audio.duration)])

    video_final = video.with_audio(audio.subclipped(0, video.duration))
    video_final.write_videofile(str(saida), codec="libx264", audio_codec="aac")
    print(f"   Vídeo final: {saida}")
    return saida


# ─────────────────────────────────────────────
# PASSO 4: UPLOAD NO YOUTUBE
# ─────────────────────────────────────────────
def autenticar_youtube():
    """Autentica via OAuth2 e retorna serviço YouTube."""
    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    creds = None
    token_path = Path("youtube_token.pickle")

    if token_path.exists():
        with open(token_path, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                YOUTUBE_CLIENT_SECRETS, SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(token_path, "wb") as f:
            pickle.dump(creds, f)

    return build("youtube", "v3", credentials=creds)


def upload_youtube(youtube, caminho_video: Path, titulo: str, descricao: str):
    """Faz upload do vídeo no YouTube."""
    print("[4/4] Fazendo upload no YouTube...")

    body = {
        "snippet": {
            "title": titulo,
            "description": descricao + "\n\n#FalaBrasil #AprenderPortugues #AppPortugues",
            "tags": ["Fala Brasil", "português", "aprender português", "app", "idioma"],
            "categoryId": "27",  # Educação
            "defaultLanguage": "pt",
        },
        "status": {
            "privacyStatus": "public",   # ou "private" para revisar antes
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(str(caminho_video), chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"   Upload: {int(status.progress() * 100)}%")

    video_id = response["id"]
    print(f"   Publicado! https://youtube.com/watch?v={video_id}")
    return video_id


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def gerar_e_publicar_video(tema: str = None):
    if tema is None:
        tema = random.choice(TEMAS)

    print(f"\n{'='*60}")
    print(f"GERANDO VÍDEO: {tema}")
    print(f"{'='*60}\n")

    # Nomes de arquivo únicos por timestamp
    ts = int(time.time())
    audio_path  = OUTPUT_DIR / f"audio_{ts}.mp3"
    video_raw   = OUTPUT_DIR / f"video_raw_{ts}.mp4"
    video_final = OUTPUT_DIR / f"video_final_{ts}.mp4"

    try:
        # 1. Script
        dados = gerar_script(tema)

        # 2. Narração
        gerar_narracao(dados["script"], audio_path)

        # 3. Vídeo Ken Burns
        gerar_video_ken_burns(dados["prompt_video"], video_raw)

        # 3b. Combinar
        combinar_video_audio(video_raw, audio_path, video_final)

        # 4. YouTube
        youtube = autenticar_youtube()
        video_id = upload_youtube(
            youtube,
            video_final,
            dados["titulo"],
            dados["descricao"],
        )

        print(f"\nSucesso! Vídeo publicado: https://youtube.com/watch?v={video_id}\n")
        return video_id

    finally:
        # Limpa arquivos temporários
        for f in [audio_path, video_raw]:
            if f.exists():
                f.unlink()


# ─────────────────────────────────────────────
# MODO DE AGENDAMENTO (loop diário)
# ─────────────────────────────────────────────
def rodar_agendado(videos_por_dia: int = 1, hora_publicacao: int = 10):
    """
    Publica `videos_por_dia` vídeos todo dia às `hora_publicacao`:00.
    Rode em background: python fala_brasil_bot.py --agendar
    """
    import schedule

    def job():
        for _ in range(videos_por_dia):
            try:
                gerar_e_publicar_video()
                time.sleep(300)  # 5 min entre vídeos
            except Exception as e:
                print(f"Erro: {e}")

    schedule.every().day.at(f"{hora_publicacao:02d}:00").do(job)
    print(f"Bot agendado: {videos_por_dia} vídeo(s)/dia às {hora_publicacao}h")

    while True:
        schedule.run_pending()
        time.sleep(60)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if "--agendar" in sys.argv:
        rodar_agendado(videos_por_dia=1, hora_publicacao=10)
    else:
        # Gera um vídeo agora
        tema_customizado = " ".join(sys.argv[1:]) or None
        gerar_e_publicar_video(tema_customizado)
