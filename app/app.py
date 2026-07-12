"""Demo web — Monitoreo de Biodiversidad Aviar en Chile (Hito 5).

Sube un audio (canto) y/o una foto de un ave chilena; el sistema identifica la especie
(fusión multimodal) y muestra su mapa de distribución estimada (SDM). Interfaz con la
identidad visual de la UC.

    uv run streamlit run app/app.py
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components
import torch
import torchaudio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biomonitor.audio.dataset import CLIP_SAMPLES, HOP, N_FFT, N_MELS, SR  # noqa: E402
from biomonitor.audio.model import build_audio_model  # noqa: E402
from biomonitor.vision.dataset import _eval_tf  # noqa: E402
from biomonitor.vision.model import build_image_model  # noqa: E402

MODELS = ROOT / "models"
DEVICE = torch.device("cpu")

UC = dict(celeste="#0176DE", azul="#173F8A", azul_osc="#03122E", amarillo="#FEC60D",
          verde="#0DA400", gris="#707070", gris_claro="#F6F6F6", gris_borde="#E3E8EF")


def svg(inner: str, size: int = 22, color: str = UC["celeste"], sw: float = 1.9) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align:middle;flex:0 0 auto;">{inner}</svg>')


# Íconos de línea (estilo Feather), monocromáticos en color institucional.
IC_MIC = ('<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>'
          '<path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/>'
          '<line x1="8" y1="23" x2="16" y2="23"/>')
IC_CAM = ('<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>'
          '<circle cx="12" cy="13" r="4"/>')
IC_FEATHER = ('<path d="M20.24 12.24a6 6 0 0 0-8.49-8.49L5 10.5V19h8.5z"/>'
              '<line x1="16" y1="8" x2="2" y2="22"/><line x1="17.5" y1="15" x2="9" y2="15"/>')
IC_PIN = ('<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>')


# =============================== Carga y predicción (cacheadas) ===============================
@st.cache_resource(show_spinner=False)
def load_all():
    a_labels = json.loads((MODELS / "audio_efficientnet_b0_labels.json").read_text())
    i_labels = json.loads((MODELS / "image_efficientnet_b3_labels.json").read_text())
    a_model = build_audio_model("efficientnet_b0", num_classes=len(a_labels), pretrained=False)
    a_model.load_state_dict(torch.load(MODELS / "audio_efficientnet_b0.pt", map_location=DEVICE))
    a_model.eval()
    i_model = build_image_model("efficientnet_b3", num_classes=len(i_labels), pretrained=False)
    i_model.load_state_dict(torch.load(MODELS / "image_efficientnet_b3.pt", map_location=DEVICE))
    i_model.eval()
    sdm = joblib.load(MODELS / "sdm_rf.pkl")
    grid = pd.read_csv(ROOT / "data/processed/chile_grid.csv")
    species = sorted(a_labels, key=lambda s: a_labels[s])
    return a_model, i_model, a_labels, i_labels, sdm, grid, species


@st.cache_data(show_spinner=False)
def logo_b64() -> str:
    return base64.b64encode((ROOT / "marca_uc" / "UC BLANCO-01.png").read_bytes()).decode()


def _to_canonical(probs, labels, species):
    idx2sp = {v: k for k, v in labels.items()}
    out = np.zeros(len(species), dtype=np.float32)
    canon = {s: i for i, s in enumerate(species)}
    for j, p in enumerate(probs):
        if idx2sp[j] in canon:
            out[canon[idx2sp[j]]] = p
    return out


_mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
_todb = torchaudio.transforms.AmplitudeToDB(top_db=80)


@st.cache_data(show_spinner=False)
def predict_audio(audio_bytes: bytes) -> np.ndarray:
    a_model, _, a_labels, _, _, _, species = load_all()
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = torchaudio.functional.resample(torch.from_numpy(y), sr, SR).numpy()
    n = CLIP_SAMPLES
    if len(y) < n:
        y = np.tile(y, int(np.ceil(n / max(len(y), 1))))[:n]
    starts = np.linspace(0, max(len(y) - n, 0), num=min(12, max(len(y) // n, 1)), dtype=int)
    wins = torch.stack([torch.from_numpy(y[s:s + n]) for s in starts])
    mel = _todb(_mel(wins))
    mel = (mel - mel.mean(dim=(1, 2), keepdim=True)) / (mel.std(dim=(1, 2), keepdim=True) + 1e-6)
    with torch.no_grad():
        probs = a_model(mel.unsqueeze(1)).softmax(1).mean(0).numpy()
    return _to_canonical(probs, a_labels, species)


@st.cache_data(show_spinner=False)
def predict_image(image_bytes: bytes) -> np.ndarray:
    _, i_model, _, i_labels, _, _, species = load_all()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with torch.no_grad():
        probs = i_model(_eval_tf()(img).unsqueeze(0)).softmax(1)[0].numpy()
    return _to_canonical(probs, i_labels, species)


@st.cache_data(show_spinner=False)
def map_html(species_name: str, month: int) -> str | None:
    import folium
    from folium.plugins import HeatMap

    _, _, _, _, sdm, grid, _ = load_all()
    model = sdm["models"].get(species_name)
    if model is None:
        return None
    feat = grid[["ndvi", "temp", "precip", "elev"]].copy()
    feat["sin_m"] = np.sin(2 * np.pi * month / 12)
    feat["cos_m"] = np.cos(2 * np.pi * month / 12)
    prob = model.predict_proba(feat[sdm["features"]])[:, 1]
    m = folium.Map(location=[-38, -71], zoom_start=4, tiles="CartoDB positron")
    HeatMap(list(zip(grid["lat"], grid["lon"], prob)), radius=9, blur=7, min_opacity=0.35,
            gradient={0.3: UC["celeste"], 0.6: UC["amarillo"], 1.0: "#F24F4F"}).add_to(m)
    return m._repr_html_()


# =============================== Estilos e UI (identidad UC) ===============================
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    html, body, [class*="css"], .stMarkdown, p, div, span, label, button {{ font-family:'Roboto',sans-serif; }}
    .stApp {{ background:{UC['gris_claro']}; }}
    #MainMenu, footer {{ visibility:hidden; }}
    .block-container {{ padding-top:1.2rem; max-width:1100px; }}
    /* Contenedores nativos con borde → tarjetas */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background:#fff; border:1px solid {UC['gris_borde']}!important; border-radius:14px;
        padding:8px 26px 20px; box-shadow:0 2px 12px rgba(3,18,46,.06); margin-bottom:16px; }}
    .uc-header {{ background:linear-gradient(120deg,{UC['azul_osc']} 0%,{UC['azul']} 68%,{UC['celeste']} 135%);
        border-radius:16px; padding:24px 34px; display:flex; align-items:center; gap:26px;
        box-shadow:0 6px 22px rgba(3,18,46,.18); margin-bottom:20px; }}
    .uc-header img {{ height:66px; }}
    .uc-header h1 {{ color:#fff; font-size:1.5rem; font-weight:700; margin:0; line-height:1.15; }}
    .uc-header p {{ color:#cfe1f6; font-size:.92rem; margin:5px 0 0; font-weight:300; }}
    .sec {{ color:{UC['azul_osc']}; font-size:1.05rem; font-weight:700; margin:6px 0 2px;
        display:flex; align-items:center; gap:9px; }}
    .sub {{ color:{UC['gris']}; font-size:.85rem; margin:0 0 12px; }}
    .up-head {{ display:flex; align-items:center; gap:9px; color:{UC['azul_osc']}; font-weight:500;
        font-size:.95rem; margin:2px 0 8px; }}
    .card {{ background:#fff; border:1px solid {UC['gris_borde']}; border-radius:14px; padding:22px 26px;
        box-shadow:0 2px 12px rgba(3,18,46,.06); margin-bottom:16px; }}
    .hero {{ border-left:6px solid {UC['celeste']}; background:linear-gradient(120deg,#fff,{UC['gris_claro']}); }}
    .sp-name {{ color:{UC['azul_osc']}; font-size:1.85rem; font-weight:700; font-style:italic; margin:2px 0 8px; }}
    .sp-conf {{ display:inline-block; background:{UC['celeste']}; color:#fff; font-weight:700; font-size:1rem;
        padding:4px 14px; border-radius:20px; }}
    .chip {{ display:inline-block; background:{UC['gris_claro']}; color:{UC['azul']}; border:1px solid {UC['gris_borde']};
        font-size:.78rem; font-weight:500; padding:3px 11px; border-radius:14px; margin-left:8px; }}
    .bar-row {{ display:flex; align-items:center; gap:12px; margin:8px 0; }}
    .bar-name {{ width:220px; font-size:.9rem; color:{UC['azul_osc']}; font-style:italic; text-align:right;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .bar-track {{ flex:1; background:{UC['gris_borde']}; border-radius:8px; height:18px; overflow:hidden; }}
    .bar-fill {{ height:100%; background:linear-gradient(90deg,{UC['azul']},{UC['celeste']}); border-radius:8px; }}
    .bar-val {{ width:52px; font-size:.85rem; font-weight:700; color:{UC['azul']}; }}
    .uc-footer {{ text-align:center; color:{UC['gris']}; font-size:.8rem; margin-top:22px; padding-top:12px; }}
    [data-testid="stFileUploaderDropzone"] {{ background:{UC['gris_claro']}; border:2px dashed {UC['celeste']}; }}
    </style>
    """, unsafe_allow_html=True)


def bars_html(probs, species, k=5) -> str:
    html = ""
    for i in np.argsort(probs)[::-1][:k]:
        pct = probs[i] * 100
        html += (f'<div class="bar-row"><div class="bar-name">{species[i]}</div>'
                 f'<div class="bar-track"><div class="bar-fill" style="width:{max(pct,2):.0f}%"></div></div>'
                 f'<div class="bar-val">{pct:.0f}%</div></div>')
    return html


st.set_page_config(page_title="Monitoreo de Aves · UC", page_icon="🐦", layout="wide")
inject_css()
st.markdown(f"""
<div class="uc-header">
  <img src="data:image/png;base64,{logo_b64()}"/>
  <div>
    <h1>Monitoreo de Biodiversidad Aviar en Chile</h1>
    <p>Identificación por canto y foto · Distribución espaciotemporal · Magíster en Inteligencia Artificial</p>
  </div>
</div>""", unsafe_allow_html=True)

_, _, _, _, _, _, species = load_all()

with st.container(border=True):
    st.markdown('<div class="sec">1 · Sube tu registro del ave</div>'
                '<div class="sub">Un audio del canto, una foto, o ambos para mayor precisión (fusión multimodal).</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="up-head">{svg(IC_MIC, 20)}<span>Audio del canto</span></div>',
                    unsafe_allow_html=True)
        audio_file = st.file_uploader("Audio del canto", type=["wav", "mp3", "ogg", "flac"],
                                      label_visibility="collapsed")
        if audio_file:
            st.audio(audio_file)
    with c2:
        st.markdown(f'<div class="up-head">{svg(IC_CAM, 20)}<span>Foto del ave</span></div>',
                    unsafe_allow_html=True)
        image_file = st.file_uploader("Foto del ave", type=["jpg", "jpeg", "png"],
                                      label_visibility="collapsed")
        if image_file:
            st.image(image_file, use_container_width=True)

if audio_file or image_file:
    with st.spinner("Analizando…"):
        probs, used = None, []
        if audio_file:
            probs = predict_audio(audio_file.getvalue()); used.append("canto")
        if image_file:
            pi = predict_image(image_file.getvalue())
            probs = pi if probs is None else 0.5 * probs + 0.5 * pi
            used.append("foto")
    best_i = int(np.argmax(probs))
    best = species[best_i]

    st.markdown(
        f'<div class="card hero"><div class="sec">2 · Especie identificada</div>'
        f'<div class="sp-name">{best}</div>'
        f'<span class="sp-conf">{probs[best_i]:.0%} de confianza</span>'
        f'<span class="chip">fuente: {" + ".join(used)}</span></div>'
        f'<div class="card"><div class="sec">Top 5 candidatas</div>'
        f'<div class="sub">Probabilidad estimada por el modelo.</div>{bars_html(probs, species)}</div>',
        unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(f'<div class="sec">{svg(IC_PIN, 20)}<span>3 · Distribución estimada de '
                    f'<i>{best}</i></span></div>'
                    '<div class="sub">Probabilidad de presencia en Chile según el modelo (SDM), por mes.</div>',
                    unsafe_allow_html=True)
        meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        month = st.select_slider("Mes del año", options=list(range(1, 13)),
                                 format_func=lambda m: meses[m - 1], value=1)
        html = map_html(best, month)
        if html:
            components.html(html, height=520, scrolling=False)
        else:
            st.info("Esta especie no tiene modelo de distribución (pocos registros de presencia).")
else:
    st.markdown('<div class="card" style="text-align:center; padding:48px 44px;">'
                f'{svg(IC_FEATHER, 50, UC["celeste"], 1.4)}'
                '<div class="sec" style="justify-content:center; font-size:1.2rem; margin-top:14px;">'
                'Comienza subiendo un audio o una foto</div>'
                '<div class="sub">El sistema reconoce 55 especies de aves chilenas y estima dónde habitan.</div>'
                '</div>', unsafe_allow_html=True)

st.markdown('<div class="uc-footer">Proyecto de titulación · Magíster en Inteligencia Artificial · '
            'Pontificia Universidad Católica de Chile</div>', unsafe_allow_html=True)
