import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Topic Modeling Bencana Banjir",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d1117;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}
[data-testid="stSidebarNav"] {
    display: none;
}

/* Headers */
h1, h2, h3 { color: #e6edf3 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 28px !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #30363d;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e !important;
    border-radius: 8px;
    font-weight: 500;
    font-size: 13px;
}
.stTabs [aria-selected="true"] {
    background: #1f6feb !important;
    color: #ffffff !important;
}

/* Selectbox & multiselect */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}

/* Divider */
hr { border-color: #30363d !important; }

/* Section card */
.section-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Topic badge */
.topic-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 2px;
}

/* Scrollable table */
.dataframe-container {
    overflow-x: auto;
    border-radius: 10px;
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 30% 50%, rgba(31,111,235,0.12) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(88,166,255,0.08) 0%, transparent 50%);
    pointer-events: none;
}
.header-title {
    font-size: 28px;
    font-weight: 800;
    color: #e6edf3;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.header-sub {
    font-size: 14px;
    color: #8b949e;
    margin: 0;
    font-weight: 400;
}
.header-accent {
    color: #58a6ff;
}
</style>
""", unsafe_allow_html=True)

# ─── TOPIC COLORS ─────────────────────────────────────────────────────────────
TOPIC_COLORS = {
    "Kehilangan & Trauma Keluarga":       "#f85149",
    "Ketakutan & Dampak Banjir":          "#388bfd",
    "Kerusakan Rumah & Bantuan":          "#e3b341",
    "Kebersihan & Aktivitas Pemulihan":   "#3fb950",
    "Dampak Pendidikan & Tempat Tidur":   "#bc8cff",
    "Bencana & Kehilangan Tempat Tinggal":"#79c0ff",
    "Kebutuhan Makan & Aktivitas Fisik":  "#ffa657",
    "Keluhan Fisik & Kesehatan":          "#ff7b72",
}
COLOR_LIST = list(TOPIC_COLORS.values())

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#e6edf3", size=12),
    legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d", borderwidth=1),
    margin=dict(t=40, b=40, l=20, r=20),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#30363d"),
)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("hasil_topic_modeling.csv")

    umur_map = {
        "8 sampai 11 tahun":  "8–11 tahun",
        "12 sampai 14 tahun": "12–14 tahun",
        "12 sampai 15 tahun": "12–15 tahun",
        "15 sampai 17 tahun": "15–17 tahun",
        "tidak mengisi":      "Tidak Diisi",
    }
    df["Kelompok_Umur"] = df["Umur"].map(umur_map).fillna(df["Umur"])

    def conf(p):
        if p >= 0.75:   return "Tinggi (≥0.75)"
        elif p >= 0.50: return "Sedang (0.50–0.74)"
        else:           return "Rendah (<0.50)"
    df["Confidence_Level"] = df["topic_probability"].apply(conf)
    return df

df = load_data()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌊 Filter Data")
    st.markdown("---")

    all_wilayah  = sorted(df["Wilayah"].unique())
    all_gender   = sorted(df["Jenis Kelamin"].unique())
    all_umur     = sorted(df["Kelompok_Umur"].unique())
    all_topics   = sorted(df["topic_label"].unique())

    sel_wilayah = st.multiselect("📍 Wilayah", all_wilayah, default=all_wilayah)
    sel_gender  = st.multiselect("👤 Jenis Kelamin", all_gender, default=all_gender)
    sel_umur    = st.multiselect("🎂 Kelompok Umur", all_umur,  default=all_umur)
    sel_topics  = st.multiselect("🏷️ Topik", all_topics, default=all_topics)

    prob_min, prob_max = st.slider(
        "📊 Rentang Probabilitas", 0.0, 1.0, (0.0, 1.0), step=0.05
    )

    st.markdown("---")
    st.markdown("#### ℹ️ Tentang Data")
    st.markdown(f"""
    <div style='font-size:12px; color:#8b949e; line-height:1.7'>
    📄 <b style='color:#e6edf3'>1.337</b> total responden<br>
    🏷️ <b style='color:#e6edf3'>8</b> topik hasil modeling<br>
    📍 <b style='color:#e6edf3'>11</b> wilayah terdampak<br>
    🔬 Metode: <b style='color:#e6edf3'>LDA</b>
    </div>
    """, unsafe_allow_html=True)

# ─── FILTER DATA ─────────────────────────────────────────────────────────────
fdf = df[
    df["Wilayah"].isin(sel_wilayah) &
    df["Jenis Kelamin"].isin(sel_gender) &
    df["Kelompok_Umur"].isin(sel_umur) &
    df["topic_label"].isin(sel_topics) &
    df["topic_probability"].between(prob_min, prob_max)
].copy()

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <p class="header-title">🌊 Analisis <span class="header-accent">Topic Modeling</span> Bencana Banjir</p>
  <p class="header-sub">Dashboard interaktif respons masyarakat terdampak · 8 topik utama · 11 wilayah</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI CARDS ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Responden",    f"{len(fdf):,}")
k2.metric("Topik Aktif",        f"{fdf['topic_label'].nunique()}")
k3.metric("Wilayah Terpilih",   f"{fdf['Wilayah'].nunique()}")
k4.metric("Avg Probabilitas",   f"{fdf['topic_probability'].mean():.3f}" if not fdf.empty else "0.000")
high_conf = (fdf["topic_probability"] >= 0.75).sum()
k5.metric("High Confidence",    f"{high_conf:,}", f"{high_conf/len(fdf)*100:.1f}%" if len(fdf) > 0 else "0.0%")

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview Topik",
    "🔍 Kualitas Model",
    "👥 Demografi",
    "🗺️ Analisis Wilayah",
    "📝 Detail Respons",
    "🎯 Ringkasan Eksekutif",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW TOPIK
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Distribusi 8 Topik Utama")
    
    if not fdf.empty:
        topic_counts = (
            fdf.groupby("topic_label")
            .agg(Jumlah=("topic_label","count"), Avg_Prob=("topic_probability","mean"))
            .reset_index()
            .sort_values("Jumlah", ascending=False)
        )
        topic_counts["Persen"] = (topic_counts["Jumlah"] / topic_counts["Jumlah"].sum() * 100).round(1)
        topic_counts["Warna"]  = topic_counts["topic_label"].map(TOPIC_COLORS)

        col_bar, col_donut = st.columns([3, 2])

        with col_bar:
            fig_bar = go.Figure(go.Bar(
                y=topic_counts["topic_label"],
                x=topic_counts["Jumlah"],
                orientation="h",
                marker_color=topic_counts["Warna"],
                text=[f"{r['Jumlah']} ({r['Persen']}%)" for _, r in topic_counts.iterrows()],
                textposition="outside",
                textfont=dict(size=11, color="#e6edf3"),
                hovertemplate="<b>%{y}</b><br>Responden: %{x}<extra></extra>",
            ))
            
            fig_bar.update_layout(**PLOTLY_LAYOUT)
            fig_bar.update_layout(
                height=420,
                title=dict(text="Jumlah Responden per Topik", font=dict(size=14))
            )
            fig_bar.update_xaxes(title_text="Jumlah Responden")
            fig_bar.update_yaxes(autorange="reversed")
            
            st.plotly_chart(fig_bar, width="stretch", key="chart_bar_overview")

        with col_donut:
            fig_pie = go.Figure(go.Pie(
                labels=topic_counts["topic_label"],
                values=topic_counts["Jumlah"],
                hole=0.55,
                marker_colors=topic_counts["Warna"].tolist(),
                textinfo="percent",
                hovertemplate="<b>%{label}</b><br>%{value} responden (%{percent})<extra></extra>",
                textfont=dict(size=11),
            ))
            fig_pie.add_annotation(
                text=f"<b>{len(fdf)}</b><br><span style='font-size:10px'>Responden</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#e6edf3"),
            )
            
            fig_pie.update_layout(**PLOTLY_LAYOUT)
            fig_pie.update_layout(
                height=420,
                title=dict(text="Proporsi Topik (%)", font=dict(size=14)),
                showlegend=False,
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_pie, width="stretch", key="chart_pie_overview")
    else:
        st.info("Tidak ada data untuk ditampilkan pada filter ini.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — KUALITAS MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Evaluasi Kualitas & Kepercayaan Model")

    if not fdf.empty:
        qc1, qc2 = st.columns(2)

        with qc1:
            prob_topic = (
                fdf.groupby("topic_label")["topic_probability"]
                .mean().reset_index().sort_values("topic_probability")
            )
            prob_topic["color"] = prob_topic["topic_label"].map(TOPIC_COLORS)

            fig_prob = go.Figure(go.Bar(
                y=prob_topic["topic_label"],
                x=prob_topic["topic_probability"],
                orientation="h",
                marker=dict(
                    color=prob_topic["topic_probability"],
                    colorscale=[[0,"#f85149"],[0.5,"#e3b341"],[1,"#3fb950"]],
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Prob", font=dict(color="#e6edf3")),
                        tickfont=dict(color="#e6edf3")
                    )
                ),
                text=[f"{v:.3f}" for v in prob_topic["topic_probability"]],
                textposition="outside",
                textfont=dict(color="#e6edf3", size=11),
                hovertemplate="<b>%{y}</b><br>Avg Probability: %{x:.4f}<extra></extra>",
            ))
            
            fig_prob.update_layout(**PLOTLY_LAYOUT)
            fig_prob.update_layout(
                height=380,
                title=dict(text="Rata-rata Probabilitas per Topik", font=dict(size=14))
            )
            fig_prob.update_xaxes(range=[0, 0.85])
            
            st.plotly_chart(fig_prob, width="stretch", key="chart_prob_qc")

        with qc2:
            conf_level = (
                fdf.groupby(["topic_label", "Confidence_Level"])
                .size().reset_index(name="n")
            )
            conf_level_pivot = conf_level.pivot(index="topic_label", columns="Confidence_Level", values="n").fillna(0)
            conf_level_pivot["total"] = conf_level_pivot.sum(axis=1)
            for col in conf_level_pivot.columns[conf_level_pivot.columns != "total"]:
                conf_level_pivot[col] = conf_level_pivot[col] / conf_level_pivot["total"] * 100

            CONF_COLORS = {"Tinggi (≥0.75)": "#3fb950", "Sedang (0.50–0.74)": "#e3b341", "Rendah (<0.50)": "#f85149"}

            fig_conf = go.Figure()
            for cl in ["Tinggi (≥0.75)", "Sedang (0.50–0.74)", "Rendah (<0.50)"]:
                if cl in conf_level_pivot.columns:
                    sub = conf_level_pivot.sort_values("Tinggi (≥0.75)")
                    fig_conf.add_trace(go.Bar(
                        name=cl, y=sub.index, x=sub[cl],
                        orientation="h", marker_color=CONF_COLORS[cl],
                        text=[f"{v:.0f}%" for v in sub[cl]],
                        textposition="inside",
                        textfont=dict(size=10, color="#0d1117"),
                        hovertemplate=f"<b>%{{y}}</b><br>{cl}: %{{x:.1f}}%<extra></extra>",
                    ))
            
            fig_conf.update_layout(**PLOTLY_LAYOUT)
            fig_conf.update_layout(
                height=380,
                title=dict(text="Distribusi Level Kepercayaan (%)", font=dict(size=14)),
                barmode="stack",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            fig_conf.update_xaxes(title_text="Persentase (%)")
            
            st.plotly_chart(fig_conf, width="stretch", key="chart_conf_qc")

        st.markdown("---")
        st.markdown("#### Sebaran Probabilitas & Ukuran Topik")
        s1, s2 = st.columns([3, 5])
        with s1:
            topic_size = fdf.groupby('topic_label').agg(Size=('topic_label','count'), Avg_Prob=('topic_probability','mean')).reset_index()
            fig_scatter = go.Figure(go.Scatter(
                x=topic_size["Size"],
                y=topic_size["Avg_Prob"],
                mode="markers+text",
                marker=dict(
                    size=topic_size["Size"],
                    sizemode="area",
                    sizeref=2.*max(topic_size["Size"])/(40.**2) if max(topic_size["Size"]) > 0 else 1,
                    sizemin=4,
                    color=[TOPIC_COLORS.get(t, "#58a6ff") for t in topic_size["topic_label"]],
                ),
                text=[l.split(" ")[0] for l in topic_size["topic_label"]],
                textposition="top center",
                textfont=dict(size=10),
                hovertemplate="<b>%{text}</b><br>Responden: %{x}<br>Avg Prob: %{y:.3f}<extra></extra>",
            ))

            fig_scatter.add_hline(
                y=fdf["topic_probability"].mean(), line_dash="dash",
                line_color="#8b949e", annotation_text="Rata-rata global",
                annotation_font_color="#8b949e"
            )
            
            fig_scatter.update_layout(**PLOTLY_LAYOUT)
            fig_scatter.update_layout(
                height=360,
                title=dict(text="Ukuran Topik vs Kepercayaan Model", font=dict(size=13)),
                showlegend=False
            )
            fig_scatter.update_xaxes(title_text="Jumlah Responden")
            fig_scatter.update_yaxes(title_text="Rata-rata Probabilitas")
            
            st.plotly_chart(fig_scatter, width="stretch", key="chart_scatter_qc")

        with s2:
            fig_box = go.Figure()
            for t in sorted(fdf["topic_label"].unique()):
                hex_color = TOPIC_COLORS.get(t, "#58a6ff")
                hex_clean = hex_color.lstrip('#')
                r, g, b = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
                rgba_fill = f"rgba({r}, {g}, {b}, 0.2)"

                fig_box.add_trace(go.Box(
                    y=fdf[fdf["topic_label"] == t]["topic_probability"],
                    name=t[:25] + ("..." if len(t) > 25 else ""),
                    boxpoints="outliers",
                    marker_color=hex_color,
                    line_color=hex_color,
                    fillcolor=rgba_fill,
                    boxmean=True,
                    hovertemplate="<b>%{x}</b><br>%{y:.3f}<extra></extra>",
                ))
            
            fig_box.update_layout(**PLOTLY_LAYOUT)
            fig_box.update_layout(
                height=400,
                title=dict(text="Sebaran Probabilitas per Topik", font=dict(size=14)),
                showlegend=False
            )
            fig_box.update_xaxes(tickangle=-20)
            fig_box.update_yaxes(title_text="Probabilitas")
            
            st.plotly_chart(fig_box, width="stretch", key="chart_box_qc")
    else:
        st.info("Tidak ada data untuk ditampilkan pada filter ini.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEMOGRAFI
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Analisis Demografi Responden")

    if not fdf.empty:
        # ── Jenis Kelamin ─────────────────────────────────────────────────────────
        st.markdown("#### 👤 Berdasarkan Jenis Kelamin")
        gender_topic = (
            fdf.groupby(["topic_label", "Jenis Kelamin"])
            .size().reset_index(name="n")
        )

        gc1, gc2 = st.columns(2)
        with gc1:
            fig_gbar = go.Figure()
            for g, color in [("Perempuan", "#ff7b72"), ("Laki laki", "#388bfd")]:
                sub = gender_topic[gender_topic["Jenis Kelamin"] == g]
                fig_gbar.add_trace(go.Bar(
                    name=g, y=sub["topic_label"], x=sub["n"],
                    orientation="h", marker_color=color,
                    hovertemplate=f"<b>%{{y}}</b><br>{g}: %{{x}}<extra></extra>",
                ))
            
            fig_gbar.update_layout(**PLOTLY_LAYOUT)
            fig_gbar.update_layout(
                height=380, barmode="group",
                title=dict(text="Jumlah Responden: Topik × Jenis Kelamin", font=dict(size=13)),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            fig_gbar.update_xaxes(title_text="Jumlah")
            fig_gbar.update_yaxes(autorange="reversed")
            
            st.plotly_chart(fig_gbar, width="stretch", key="chart_gbar_demo")

        with gc2:
            gender_total = fdf["Jenis Kelamin"].value_counts().reset_index()
            gender_total.columns = ["Jenis Kelamin", "n"]
            fig_gpie = go.Figure(go.Pie(
                labels=gender_total["Jenis Kelamin"],
                values=gender_total["n"],
                hole=0.5,
                marker_colors=["#ff7b72", "#388bfd"],
                textinfo="label+percent",
                textfont=dict(size=12),
            ))
            fig_gpie.update_layout(**PLOTLY_LAYOUT)
            fig_gpie.update_layout(
                height=380,
                title=dict(text="Proporsi Jenis Kelamin (Total)", font=dict(size=13)),
                showlegend=False,
            )
            st.plotly_chart(fig_gpie, width="stretch", key="chart_gpie_demo")

        # 100% stacked
        gender_pct = gender_topic.copy()
        tot = gender_pct.groupby("topic_label")["n"].transform("sum")
        gender_pct["pct"] = gender_pct["n"] / tot * 100

        fig_g100 = go.Figure()
        for g, color in [("Perempuan", "#ff7b72"), ("Laki laki", "#388bfd")]:
            sub = gender_pct[gender_pct["Jenis Kelamin"] == g]
            fig_g100.add_trace(go.Bar(
                name=g, y=sub["topic_label"], x=sub["pct"],
                orientation="h", marker_color=color,
                text=[f"{v:.0f}%" for v in sub["pct"]],
                textposition="inside", textfont=dict(size=11, color="#0d1117"),
            ))
            
        fig_g100.update_layout(**PLOTLY_LAYOUT)
        fig_g100.update_layout(
            height=350, barmode="stack",
            title=dict(text="Proporsi 100% Jenis Kelamin per Topik", font=dict(size=13)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        fig_g100.update_xaxes(title_text="Persentase (%)", range=[0, 100])
        st.plotly_chart(fig_g100, width="stretch", key="chart_g100_demo")

        st.markdown("---")

        # ── Kelompok Umur ─────────────────────────────────────────────────────────
        st.markdown("#### 🎂 Berdasarkan Kelompok Umur")
        umur_topic = (
            fdf.groupby(["topic_label", "Kelompok_Umur"])
            .size().reset_index(name="n")
        )

        uc1, uc2 = st.columns(2)
        UMUR_COLORS = {
            "8–11 tahun":   "#58a6ff",
            "12–14 tahun":  "#3fb950",
            "12–15 tahun":  "#e3b341",
            "15–17 tahun":  "#bc8cff",
            "Tidak Diisi":  "#8b949e",
        }

        with uc1:
            fig_ubar = go.Figure()
            for u in sorted(umur_topic["Kelompok_Umur"].unique()):
                sub = umur_topic[umur_topic["Kelompok_Umur"] == u]
                fig_ubar.add_trace(go.Bar(
                    name=u, x=sub["topic_label"], y=sub["n"],
                    marker_color=UMUR_COLORS.get(u, "#8b949e"),
                    hovertemplate=f"<b>%{{x}}</b><br>{u}: %{{y}}<extra></extra>",
                ))
                
            fig_ubar.update_layout(**PLOTLY_LAYOUT)
            fig_ubar.update_layout(
                height=380, barmode="stack",
                title=dict(text="Distribusi Kelompok Umur per Topik", font=dict(size=13)),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            fig_ubar.update_xaxes(tickangle=-30)
            fig_ubar.update_yaxes(title_text="Jumlah Responden")
            st.plotly_chart(fig_ubar, width="stretch", key="chart_ubar_demo")

        with uc2:
            umur_total = fdf["Kelompok_Umur"].value_counts().reset_index()
            umur_total.columns = ["Kelompok_Umur", "n"]
            fig_upie = go.Figure(go.Pie(
                labels=umur_total["Kelompok_Umur"],
                values=umur_total["n"],
                hole=0.45,
                marker_colors=[UMUR_COLORS.get(u, "#8b949e") for u in umur_total["Kelompok_Umur"]],
                textinfo="label+percent",
                textfont=dict(size=11),
            ))
            fig_upie.update_layout(**PLOTLY_LAYOUT)
            fig_upie.update_layout(
                height=380,
                title=dict(text="Proporsi Kelompok Umur (Total)", font=dict(size=13)),
                showlegend=False,
            )
            st.plotly_chart(fig_upie, width="stretch", key="chart_upie_demo")

        # Matrix umur x topik
        st.markdown("#### 📋 Matrix Kelompok Umur × Topik")
        matrix_umur = pd.crosstab(fdf["Kelompok_Umur"], fdf["topic_label"])
        fig_heat_umur = go.Figure(go.Heatmap(
            z=matrix_umur.values,
            x=[c[:20]+"…" if len(c)>20 else c for c in matrix_umur.columns],
            y=matrix_umur.index,
            colorscale=[[0,"#0d1117"],[0.3,"#1f6feb"],[0.7,"#388bfd"],[1,"#79c0ff"]],
            text=matrix_umur.values,
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Jumlah: %{z}<extra></extra>",
        ))
        
        fig_heat_umur.update_layout(**PLOTLY_LAYOUT)
        fig_heat_umur.update_layout(
            height=320,
            title=dict(text="Heatmap: Kelompok Umur × Topik", font=dict(size=13))
        )
        fig_heat_umur.update_xaxes(tickangle=-25)
        st.plotly_chart(fig_heat_umur, width="stretch", key="chart_heat_umur_demo")
    else:
        st.info("Tidak ada data untuk ditampilkan pada filter ini.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALISIS WILAYAH
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Analisis Berdasarkan Wilayah")

    if not fdf.empty:
        wt_matrix = pd.crosstab(fdf["Wilayah"], fdf["topic_label"])
        labels_short = [c[:18]+"…" if len(c)>18 else c for c in wt_matrix.columns]

        # Heatmap
        fig_heat = go.Figure(go.Heatmap(
            z=wt_matrix.values,
            x=labels_short,
            y=wt_matrix.index,
            colorscale=[[0,"#0d1117"],[0.2,"#0d419d"],[0.5,"#1f6feb"],[0.8,"#388bfd"],[1,"#79c0ff"]],
            text=wt_matrix.values,
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br><b>%{x}</b><br>Responden: %{z}<extra></extra>",
        ))
        
        fig_heat.update_layout(**PLOTLY_LAYOUT)
        fig_heat.update_layout(
            height=420,
            title=dict(text="⭐ Heatmap Intensitas Topik per Wilayah", font=dict(size=14)),
            margin=dict(t=50, b=80, l=200, r=20)
        )
        fig_heat.update_xaxes(tickangle=-30, gridcolor="rgba(0,0,0,0)")
        fig_heat.update_yaxes(gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_heat, width="stretch", key="chart_heat_wilayah")

        wc1, wc2 = st.columns(2)

        with wc1:
            # Stacked bar per wilayah
            wilayah_topic = (
                fdf.groupby(["Wilayah", "topic_label"])
                .size().reset_index(name="n")
            )
            fig_wbar = go.Figure()
            for t in sorted(fdf["topic_label"].unique()):
                sub = wilayah_topic[wilayah_topic["topic_label"] == t]
                fig_wbar.add_trace(go.Bar(
                    name=t[:22], y=sub["Wilayah"], x=sub["n"],
                    orientation="h",
                    marker_color=TOPIC_COLORS.get(t, "#58a6ff"),
                    hovertemplate=f"<b>%{{y}}</b><br>{t}: %{{x}}<extra></extra>",
                ))
            
            fig_wbar.update_layout(**PLOTLY_LAYOUT)
            fig_wbar.update_layout(
                height=420, barmode="stack",
                title=dict(text="Komposisi Topik per Wilayah", font=dict(size=13)),
                showlegend=False
            )
            fig_wbar.update_xaxes(title_text="Jumlah Responden")
            st.plotly_chart(fig_wbar, width="stretch", key="chart_wbar_wilayah")

        with wc2:
            # Top wilayah bar
            wilayah_total = fdf["Wilayah"].value_counts().reset_index()
            wilayah_total.columns = ["Wilayah", "n"]
            fig_wt = go.Figure(go.Bar(
                y=wilayah_total["Wilayah"],
                x=wilayah_total["n"],
                orientation="h",
                marker=dict(
                    color=wilayah_total["n"],
                    colorscale=[[0,"#1f6feb"],[1,"#79c0ff"]],
                ),
                text=wilayah_total["n"],
                textposition="outside",
                textfont=dict(color="#e6edf3", size=11),
            ))
            
            fig_wt.update_layout(**PLOTLY_LAYOUT)
            fig_wt.update_layout(
                height=420,
                title=dict(text="Total Responden per Wilayah", font=dict(size=13))
            )
            fig_wt.update_xaxes(title_text="Jumlah Responden")
            fig_wt.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_wt, width="stretch", key="chart_wt_wilayah")

        # Wilayah selector detail
        st.markdown("---")
        st.markdown("#### 🔎 Detail Satu Wilayah")
        sel_one = st.selectbox("Pilih wilayah untuk detail:", sorted(fdf["Wilayah"].unique()))
        one_df  = fdf[fdf["Wilayah"] == sel_one]
        one_tc  = one_df["topic_label"].value_counts().reset_index()
        one_tc.columns = ["topic_label","n"]
        one_tc["warna"] = one_tc["topic_label"].map(TOPIC_COLORS)

        d1, d2, d3 = st.columns(3)
        d1.metric("Responden", len(one_df))
        d2.metric("Topik Dominan", one_tc.iloc[0]["topic_label"].split(" & ")[0] if not one_tc.empty else "N/A")
        d3.metric("Avg Probabilitas", f"{one_df['topic_probability'].mean():.3f}")

        fig_one = go.Figure(go.Bar(
            x=one_tc["topic_label"],
            y=one_tc["n"],
            marker_color=one_tc["warna"],
            text=one_tc["n"],
            textposition="outside",
            textfont=dict(color="#e6edf3"),
        ))
        
        fig_one.update_layout(**PLOTLY_LAYOUT)
        fig_one.update_layout(
            height=320,
            title=dict(text=f"Distribusi Topik — {sel_one}", font=dict(size=13))
        )
        fig_one.update_xaxes(tickangle=-25)
        fig_one.update_yaxes(title_text="Jumlah Responden")
        st.plotly_chart(fig_one, width="stretch", key="chart_one_wilayah")
    else:
        st.info("Tidak ada data untuk ditampilkan pada filter ini.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DETAIL RESPONS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 📝 Tabel Detail Respons")

    fc1, fc2 = st.columns(2)
    with fc1:
        topik_filter = st.selectbox("Filter Topik:", ["Semua"] + sorted(fdf["topic_label"].unique()))
    with fc2:
        conf_filter  = st.selectbox("Filter Confidence:", ["Semua"] + sorted(fdf["Confidence_Level"].unique()))

    view_df = fdf.copy()
    if topik_filter != "Semua":
        view_df = view_df[view_df["topic_label"] == topik_filter]
    if conf_filter != "Semua":
        view_df = view_df[view_df["Confidence_Level"] == conf_filter]

    view_df = view_df.sort_values("topic_probability", ascending=False)

    if not view_df.empty:
        disp = view_df[[
            "Wilayah","Jenis Kelamin","Kelompok_Umur",
            "topic_label","topic_probability","Confidence_Level","Tanggapan_Final"
        ]].rename(columns={
            "Wilayah":"Wilayah",
            "Jenis Kelamin":"Gender",
            "Kelompok_Umur":"Umur",
            "topic_label":"Topik",
            "topic_probability":"Probabilitas",
            "Confidence_Level":"Confidence",
            "Tanggapan_Final":"Respons",
        })

        st.markdown(f"**{len(disp):,} respons** ditampilkan")
        st.dataframe(
            disp.reset_index(drop=True),
            width="stretch",
            height=480,
            column_config={
                "Probabilitas": st.column_config.ProgressColumn(
                    "Probabilitas", min_value=0, max_value=1, format="%.3f"
                ),
                "Respons": st.column_config.TextColumn("Respons", width="large"),
            }
        )

        # Download button
        csv_bytes = disp.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Tabel (CSV)",
            data=csv_bytes,
            file_name="detail_respons_filtered.csv",
            mime="text/csv",
        )
    else:
        st.info("Tidak ada data untuk ditampilkan pada filter ini.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RINGKASAN EKSEKUTIF
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 🎯 Ringkasan Eksekutif — 8 Topik Utama")

    if not fdf.empty:
        summary = (
            fdf.groupby("topic_label")
            .agg(
                Jumlah=("topic_label","count"),
                Avg_Prob=("topic_probability","mean"),
                High_Conf=("topic_probability", lambda x: (x >= 0.75).sum()),
            )
            .reset_index()
            .sort_values("Jumlah", ascending=False)
        )
        summary["Persen"]    = (summary["Jumlah"] / summary["Jumlah"].sum() * 100).round(1)
        summary["High_Pct"]  = (summary["High_Conf"] / summary["Jumlah"] * 100).round(1)

        st.markdown("#### Ringkasan Statistik per Topik")
        st.dataframe(
            summary.style.format({
                "Avg_Prob": "{:.3f}",
                "Persen": "{:.1f}%",
                "High_Pct": "{:.1f}%",
            }).background_gradient(cmap="Blues", subset=["Jumlah"]),
            width="stretch"
        )

        st.markdown("#### Radar Chart Profil Dampak")
        radar_df = summary.copy()
        radar_df["Jumlah_norm"] = (radar_df["Jumlah"] / radar_df["Jumlah"].max() * 100)
        radar_df = radar_df.sort_values("topic_label")

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df["Jumlah_norm"],
            theta=[l.replace(" & ", " &<br>") for l in radar_df["topic_label"]],
            fill='toself',
            fillcolor="rgba(31,111,235,0.2)",
            line_color="#388bfd",
            line_width=2,
            name="Jumlah Responden (normalized)",
        ))
        
        fig_radar.update_layout(**PLOTLY_LAYOUT)
        fig_radar.update_layout(
            height=400,
            title=dict(text="Radar Chart — Profil 8 Dimensi Dampak Bencana", font=dict(size=13)),
            showlegend=False,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,100], color="#8b949e", gridcolor="#21262d"),
                angularaxis=dict(color="#e6edf3", gridcolor="#21262d"),
            )
        )
        st.plotly_chart(fig_radar, width="stretch", key="chart_radar_summary")

        st.markdown("#### Rekomendasi & Insight Utama")
        st.markdown("""
        <div class="section-card">
        <ul style="line-height:1.8; padding-left:20px">
            <li><b style="color:#f85149">Kehilangan & Trauma Keluarga</b> adalah topik paling dominan, menandakan dampak psikologis mendalam yang membutuhkan intervensi kesehatan mental.</li>
            <li>Topik <b style="color:#388bfd">Ketakutan & Dampak Banjir</b> juga sangat signifikan, menunjukkan perlunya program pemulihan trauma dan edukasi kesiapsiagaan bencana.</li>
            <li>Wilayah <b>Bantaran Sungai</b> menunjukkan intensitas tertinggi di hampir semua topik, menjadikannya area prioritas untuk bantuan dan program pemulihan.</li>
            <li>Tingkat kepercayaan model (probabilitas) cukup tinggi, dengan rata-rata <b>di atas 0.65</b> untuk sebagian besar topik, mengindikasikan hasil topic modeling yang reliabel.</li>
            <li>Kelompok umur <b>12-15 tahun</b> menjadi responden terbesar, menyoroti pentingnya program yang berfokus pada remaja dalam manajemen pasca-bencana.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Tidak ada data untuk ditampilkan pada ringkasan.")
