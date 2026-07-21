"""Visual system for the Streamlit research dashboard."""

import streamlit as st


THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
  --vt-bg: #07101f;
  --vt-panel: rgba(16, 29, 50, .82);
  --vt-panel-strong: #101d32;
  --vt-line: rgba(148, 180, 214, .15);
  --vt-text: #edf6ff;
  --vt-muted: #91a7bf;
  --vt-cyan: #36d9d2;
  --vt-blue: #4f8cff;
  --vt-violet: #9b7bff;
  --vt-amber: #ffbd66;
  --vt-green: #5de0a3;
}

.stApp {
  background:
    radial-gradient(circle at 9% 4%, rgba(54,217,210,.11), transparent 24rem),
    radial-gradient(circle at 91% 6%, rgba(155,123,255,.11), transparent 25rem),
    linear-gradient(180deg, #081321 0%, var(--vt-bg) 38%, #060d18 100%);
  color: var(--vt-text);
}
.stApp, .stApp button, .stApp input, .stApp textarea { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, .vt-brand { font-family: 'Manrope', sans-serif !important; letter-spacing: -.025em; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
.block-container { max-width: 1480px; padding: 2rem 2.5rem 5rem; }
[data-testid="stSidebar"] { background: rgba(6,14,26,.96); border-right: 1px solid var(--vt-line); }
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
[data-testid="stSidebar"] hr { border-color: var(--vt-line); }

.vt-brand { font-size: 1.25rem; font-weight: 800; color: var(--vt-text); }
.vt-brand-dot { color: var(--vt-cyan); }
.vt-kicker { color: var(--vt-cyan); font-size: .72rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
.vt-hero {
  position: relative; overflow: hidden; padding: 2rem 2.2rem; margin: .25rem 0 1.25rem;
  border: 1px solid var(--vt-line); border-radius: 24px;
  background: linear-gradient(125deg, rgba(18,38,61,.94), rgba(14,25,44,.78));
  box-shadow: 0 24px 70px rgba(0,0,0,.23);
}
.vt-hero:after { content:""; position:absolute; width:270px; height:270px; right:-80px; top:-120px; border-radius:50%; background:rgba(54,217,210,.12); filter:blur(4px); }
.vt-hero h1 { max-width: 760px; margin: .38rem 0 .65rem; font-size: clamp(2rem,4vw,3.35rem); line-height: 1.02; color: var(--vt-text); }
.vt-hero p { max-width: 680px; color: #a9bdd1; font-size: 1rem; line-height: 1.65; margin: 0; }
.vt-hero-meta { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.35rem; }
.vt-pill { display:inline-flex; align-items:center; gap:.4rem; padding:.42rem .72rem; border-radius:999px; border:1px solid var(--vt-line); background:rgba(5,14,26,.48); color:#c8d8e8; font-size:.78rem; font-weight:600; }
.vt-dot { width:.45rem; height:.45rem; border-radius:50%; background:var(--vt-green); box-shadow:0 0 12px var(--vt-green); }
.vt-safety { display:flex; gap:.8rem; align-items:flex-start; padding:.85rem 1rem; margin-bottom:1.25rem; border:1px solid rgba(255,189,102,.32); border-radius:14px; background:rgba(255,189,102,.08); color:#ffdbad; font-size:.82rem; }
.vt-safety strong { color:#ffe8c9; }
.vt-section { margin: 2.4rem 0 1rem; }
.vt-section h2 { color:var(--vt-text); font-size:1.45rem; margin:.25rem 0; }
.vt-section p { color:var(--vt-muted); margin:0; font-size:.9rem; }
.vt-eyebrow { color:var(--vt-cyan); text-transform:uppercase; letter-spacing:.15em; font-weight:700; font-size:.68rem; }

.vt-card { border:1px solid var(--vt-line); border-radius:18px; padding:1.1rem 1.15rem; background:var(--vt-panel); box-shadow:0 12px 30px rgba(0,0,0,.12); }
.vt-metric { min-height:126px; position:relative; overflow:hidden; }
.vt-metric:after { content:""; position:absolute; height:3px; left:1rem; right:1rem; top:0; border-radius:5px; background:linear-gradient(90deg,var(--vt-cyan),var(--vt-violet)); }
.vt-metric-label { color:var(--vt-muted); font-size:.75rem; font-weight:600; text-transform:uppercase; letter-spacing:.08em; }
.vt-metric-value { color:var(--vt-text); font:800 1.8rem/1.2 'Manrope',sans-serif; margin-top:.45rem; }
.vt-metric-unit { color:var(--vt-muted); font-size:.75rem; margin-left:.2rem; font-weight:500; }
.vt-metric-note { color:#7890aa; font-size:.72rem; margin-top:.32rem; }
.vt-status { display:flex; justify-content:space-between; gap:1rem; padding:.9rem 1rem; margin:.55rem 0; border:1px solid var(--vt-line); border-radius:14px; background:rgba(12,24,42,.72); }
.vt-status-label { color:#c9d8e7; font-size:.84rem; }
.vt-badge { padding:.2rem .55rem; border-radius:999px; font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.07em; }
.vt-badge-warn { color:#ffd9a3; background:rgba(255,189,102,.13); border:1px solid rgba(255,189,102,.25); }
.vt-badge-good { color:#9df0c5; background:rgba(93,224,163,.1); border:1px solid rgba(93,224,163,.24); }
.vt-report { border-left:3px solid var(--vt-cyan); padding:1rem 1.2rem; background:rgba(54,217,210,.045); border-radius:0 14px 14px 0; color:#c8d8e8; line-height:1.65; }
.vt-evidence { padding:1.1rem 1.2rem; margin:.7rem 0; border:1px solid var(--vt-line); border-radius:16px; background:rgba(15,28,48,.72); }
.vt-evidence-title { color:var(--vt-text); font-weight:700; margin-bottom:.4rem; }
.vt-evidence-text { color:#a8bbcf; font-size:.88rem; line-height:1.55; }
.vt-source { color:var(--vt-cyan); font-size:.72rem; margin-top:.7rem; overflow-wrap:anywhere; }
.vt-step { display:flex; gap:.7rem; align-items:flex-start; color:#b7c9da; font-size:.82rem; padding:.38rem 0; }
.vt-step-num { display:grid; place-items:center; flex:0 0 1.45rem; height:1.45rem; border-radius:50%; color:var(--vt-cyan); background:rgba(54,217,210,.1); border:1px solid rgba(54,217,210,.2); font-size:.66rem; font-weight:700; }

[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.4rem; padding:.35rem; border:1px solid var(--vt-line); border-radius:14px; background:rgba(5,14,26,.5); }
[data-testid="stTabs"] [data-baseweb="tab"] { height:2.7rem; border-radius:10px; padding:0 1rem; color:var(--vt-muted); }
[data-testid="stTabs"] [aria-selected="true"] { background:rgba(54,217,210,.1); color:var(--vt-text); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--vt-cyan); }
[data-testid="stImage"] { border-radius:18px; overflow:hidden; border:1px solid var(--vt-line); background:#030812; }
[data-testid="stImage"] img { width:100%; }
[data-testid="stImageCaption"] { color:var(--vt-muted); padding:.35rem; }
[data-testid="stExpander"] { border:1px solid var(--vt-line); border-radius:14px; background:rgba(13,26,44,.55); }
[data-testid="stDataFrame"] { border:1px solid var(--vt-line); border-radius:14px; overflow:hidden; }
.stTextInput input { border:1px solid var(--vt-line) !important; background:rgba(7,17,31,.85) !important; color:var(--vt-text) !important; border-radius:12px !important; }
.stTextInput input:focus { border-color:var(--vt-cyan) !important; box-shadow:0 0 0 2px rgba(54,217,210,.12) !important; }
.stButton button { border-radius:12px; border:1px solid rgba(54,217,210,.3); background:rgba(54,217,210,.1); color:#c9fffc; }
a { color:var(--vt-cyan) !important; }

@media (max-width: 800px) {
  .block-container { padding:1.25rem 1rem 3rem; }
  .vt-hero { padding:1.45rem 1.25rem; border-radius:19px; }
  .vt-hero h1 { font-size:2.15rem; }
  [data-testid="stTabs"] [data-baseweb="tab"] { padding:0 .62rem; font-size:.76rem; }
}
@media (prefers-reduced-motion: no-preference) {
  .vt-card, [data-testid="stImage"] { transition:transform .18s ease, border-color .18s ease; }
  .vt-card:hover { transform:translateY(-2px); border-color:rgba(54,217,210,.26); }
}
</style>
"""


def apply_theme() -> None:
    """Install the dashboard theme and document metadata."""
    st.set_page_config(
        page_title="VascuTrace AI — Research Workspace",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(THEME_CSS, unsafe_allow_html=True)
