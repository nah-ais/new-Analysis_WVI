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
k4.metric("Avg Probabilitas",   f"{fdf['topic_probability'].mean():.3f}")
high_conf = (fdf["topic_probability"] >= 0.75).sum()
k5.metric("High Confidence",    f"{high_conf:,}", f"{high_conf/len(fdf)*100:.1f}%")

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
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=420,
            title=dict(text="Jumlah Responden per Topik", font=dict(size=14)),
            xaxis_title="Jumlah Responden", yaxis_title="",
            yaxis=dict(gridcolor="#21262d", autorange="reversed"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

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
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=420,
            title=dict(text="Proporsi Topik (%)", font=dict(size=14)),
            showlegend=False,
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Ranking cards
    st.markdown("#### 🏆 Ranking Topik")
    rank_cols = st.columns(4)
    medals = ["🥇", "🥈", "🥉", "4️⃣"]
    for i, (_, row) in enumerate(topic_counts.head(4).iterrows()):
        color = row["Warna"]
        with rank_cols[i]:
            st.markdown(f"""
            <div style='background:#161b22;border:1px solid {color};border-radius:12px;padding:16px;text-align:center'>
              <div style='font-size:24px'>{medals[i]}</div>
              <div style='font-size:11px;color:#8b949e;margin:6px 0 4px 0;line-height:1.4'>{row["topic_label"]}</div>
              <div style='font-size:22px;font-weight:800;color:{color}'>{row["Jumlah"]}</div>
              <div style='font-size:10px;color:#8b949e'>responden · {row["Persen"]}%</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — KUALITAS MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Kualitas & Kepercayaan Model")

    col_l, col_r = st.columns(2)

    with col_l:
        # Avg probability bar
        prob_topic = (
            fdf.groupby("topic_label")["topic_probability"]
            .mean().reset_index()
            .sort_values("topic_probability", ascending=True)
        )
        fig_prob = go.Figure(go.Bar(
            y=prob_topic["topic_label"],
            x=prob_topic["topic_probability"],
            orientation="h",
            marker=dict(
                color=prob_topic["topic_probability"],
                colorscale=[[0,"#f85149"],[0.5,"#e3b341"],[1,"#3fb950"]],
                showscale=True,
                colorbar=dict(title="Prob", tickfont=dict(color="#e6edf3"), titlefont=dict(color="#e6edf3")),
            ),
            text=[f"{v:.3f}" for v in prob_topic["topic_probability"]],
            textposition="outside",
            textfont=dict(color="#e6edf3", size=11),
            hovertemplate="<b>%{y}</b><br>Avg Probability: %{x:.4f}<extra></extra>",
        ))
        fig_prob.update_layout(**PLOTLY_LAYOUT, height=380,
            title=dict(text="Rata-rata Probabilitas per Topik", font=dict(size=14)),
            xaxis=dict(range=[0, 0.85], gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    with col_r:
        # Confidence level stacked bar
        conf_order = ["Tinggi (≥0.75)", "Sedang (0.50–0.74)", "Rendah (<0.50)"]
        conf_colors = {"Tinggi (≥0.75)": "#3fb950", "Sedang (0.50–0.74)": "#e3b341", "Rendah (<0.50)": "#f85149"}

        conf_df = (
            fdf.groupby(["topic_label", "Confidence_Level"])
            .size().reset_index(name="n")
        )
        total_per_topic = conf_df.groupby("topic_label")["n"].sum()
        conf_df["pct"] = conf_df.apply(lambda r: r["n"]/total_per_topic[r["topic_label"]]*100, axis=1)

        fig_conf = go.Figure()
        for cl in conf_order:
            sub = conf_df[conf_df["Confidence_Level"] == cl]
            fig_conf.add_trace(go.Bar(
                name=cl,
                y=sub["topic_label"],
                x=sub["pct"],
                orientation="h",
                marker_color=conf_colors[cl],
                text=[f"{v:.0f}%" for v in sub["pct"]],
                textposition="inside",
                textfont=dict(size=10, color="#0d1117"),
                hovertemplate=f"<b>%{{y}}</b><br>{cl}: %{{x:.1f}}%<extra></extra>",
            ))
        fig_conf.update_layout(**PLOTLY_LAYOUT, height=380,
            title=dict(text="Distribusi Level Kepercayaan (%)", font=dict(size=14)),
            barmode="stack",
            xaxis=dict(title="Persentase (%)", gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    # Scatter: Jumlah vs Avg Probability
    scatter_df = (
        fdf.groupby("topic_label")
        .agg(Jumlah=("topic_label","count"), Avg_Prob=("topic_probability","mean"))
        .reset_index()
    )
    scatter_df["Warna"] = scatter_df["topic_label"].map(TOPIC_COLORS)

    fig_scatter = go.Figure()
    for _, row in scatter_df.iterrows():
        fig_scatter.add_trace(go.Scatter(
            x=[row["Jumlah"]],
            y=[row["Avg_Prob"]],
            mode="markers+text",
            marker=dict(size=max(row["Jumlah"]/10, 16), color=row["Warna"], opacity=0.85,
                        line=dict(color="#0d1117", width=2)),
            text=[row["topic_label"].split(" & ")[0]],
            textposition="top center",
            textfont=dict(size=10, color="#e6edf3"),
            name=row["topic_label"],
            hovertemplate=f"<b>{row['topic_label']}</b><br>Responden: {row['Jumlah']}<br>Avg Prob: {row['Avg_Prob']:.3f}<extra></extra>",
        ))

    fig_scatter.add_hline(y=fdf["topic_probability"].mean(), line_dash="dash",
                          line_color="#8b949e", annotation_text="Rata-rata global",
                          annotation_font_color="#8b949e")
    fig_scatter.update_layout(**PLOTLY_LAYOUT, height=360,
        title=dict(text="Ukuran Topik vs Kepercayaan Model (ukuran bubble = jumlah responden)", font=dict(size=13)),
        xaxis_title="Jumlah Responden",
        yaxis_title="Rata-rata Probabilitas",
        showlegend=False,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Distribution histogram
    st.markdown("#### Distribusi Probabilitas per Topik (Box Plot)")
    topic_order = fdf["topic_label"].value_counts().index.tolist()
    fig_box = go.Figure()
    for t in topic_order:
        sub = fdf[fdf["topic_label"] == t]["topic_probability"]
        fig_box.add_trace(go.Box(
            y=sub, name=t,
            marker_color=TOPIC_COLORS.get(t, "#58a6ff"),
            line_color=TOPIC_COLORS.get(t, "#58a6ff"),
            fillcolor=TOPIC_COLORS.get(t, "#58a6ff") + "33",
            boxmean=True,
            hovertemplate="<b>%{x}</b><br>%{y:.3f}<extra></extra>",
        ))
    fig_box.update_layout(**PLOTLY_LAYOUT, height=400,
        title=dict(text="Sebaran Probabilitas per Topik", font=dict(size=14)),
        yaxis_title="Probabilitas",
        showlegend=False,
        xaxis=dict(tickangle=-20, gridcolor="#21262d"),
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEMOGRAFI
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Analisis Demografi Responden")

    # ── Jenis Kelamin ─────────────────────────────────────────────────────────
    st.markdown("#### 👤 Berdasarkan Jenis Kelamin")
    gc1, gc2 = st.columns(2)

    gender_topic = (
        fdf.groupby(["topic_label", "Jenis Kelamin"])
        .size().reset_index(name="n")
    )

    with gc1:
        fig_gbar = go.Figure()
        for g, color in [("Perempuan", "#ff7b72"), ("Laki laki", "#388bfd")]:
            sub = gender_topic[gender_topic["Jenis Kelamin"] == g]
            fig_gbar.add_trace(go.Bar(
                name=g, y=sub["topic_label"], x=sub["n"],
                orientation="h", marker_color=color,
                hovertemplate=f"<b>%{{y}}</b><br>{g}: %{{x}}<extra></extra>",
            ))
        fig_gbar.update_layout(**PLOTLY_LAYOUT, height=380, barmode="group",
            title=dict(text="Jumlah Responden: Topik × Jenis Kelamin", font=dict(size=13)),
            xaxis_title="Jumlah", yaxis=dict(gridcolor="#21262d", autorange="reversed"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_gbar, use_container_width=True)

    with gc2:
        total_gender = gender_topic.groupby("Jenis Kelamin")["n"].sum().reset_index()
        fig_gpie = go.Figure(go.Pie(
            labels=total_gender["Jenis Kelamin"],
            values=total_gender["n"],
            hole=0.5,
            marker_colors=["#ff7b72", "#388bfd"],
            textinfo="label+percent",
            textfont=dict(size=12),
        ))
        fig_gpie.update_layout(**PLOTLY_LAYOUT, height=380,
            title=dict(text="Proporsi Jenis Kelamin (Total)", font=dict(size=13)),
            showlegend=False,
        )
        st.plotly_chart(fig_gpie, use_container_width=True)

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
    fig_g100.update_layout(**PLOTLY_LAYOUT, height=350, barmode="stack",
        title=dict(text="Proporsi 100% Jenis Kelamin per Topik", font=dict(size=13)),
        xaxis=dict(title="Persentase (%)", range=[0,100], gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_g100, use_container_width=True)

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
        fig_ubar.update_layout(**PLOTLY_LAYOUT, height=380, barmode="stack",
            title=dict(text="Distribusi Kelompok Umur per Topik", font=dict(size=13)),
            yaxis_title="Jumlah Responden",
            xaxis=dict(tickangle=-30, gridcolor="#21262d"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_ubar, use_container_width=True)

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
        fig_upie.update_layout(**PLOTLY_LAYOUT, height=380,
            title=dict(text="Proporsi Kelompok Umur (Total)", font=dict(size=13)),
            showlegend=False,
        )
        st.plotly_chart(fig_upie, use_container_width=True)

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
    fig_heat_umur.update_layout(**PLOTLY_LAYOUT, height=320,
        title=dict(text="Heatmap: Kelompok Umur × Topik", font=dict(size=13)),
        xaxis=dict(tickangle=-25),
    )
    st.plotly_chart(fig_heat_umur, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALISIS WILAYAH
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Analisis Berdasarkan Wilayah")

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
    fig_heat.update_layout(**PLOTLY_LAYOUT, height=420,
        title=dict(text="⭐ Heatmap Intensitas Topik per Wilayah", font=dict(size=14)),
        xaxis=dict(tickangle=-30, gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, b=80, l=200, r=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

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
        fig_wbar.update_layout(**PLOTLY_LAYOUT, height=420, barmode="stack",
            title=dict(text="Komposisi Topik per Wilayah", font=dict(size=13)),
            xaxis_title="Jumlah Responden",
            yaxis=dict(gridcolor="#21262d"),
            showlegend=False,
        )
        st.plotly_chart(fig_wbar, use_container_width=True)

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
        fig_wt.update_layout(**PLOTLY_LAYOUT, height=420,
            title=dict(text="Total Responden per Wilayah", font=dict(size=13)),
            xaxis_title="Jumlah Responden",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_wt, use_container_width=True)

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
    d2.metric("Topik Dominan", one_tc.iloc[0]["topic_label"].split(" & ")[0])
    d3.metric("Avg Probabilitas", f"{one_df['topic_probability'].mean():.3f}")

    fig_one = go.Figure(go.Bar(
        x=one_tc["topic_label"],
        y=one_tc["n"],
        marker_color=one_tc["warna"],
        text=one_tc["n"],
        textposition="outside",
        textfont=dict(color="#e6edf3"),
    ))
    fig_one.update_layout(**PLOTLY_LAYOUT, height=320,
        title=dict(text=f"Distribusi Topik — {sel_one}", font=dict(size=13)),
        xaxis=dict(tickangle=-25, gridcolor="#21262d"),
        yaxis_title="Jumlah Responden",
    )
    st.plotly_chart(fig_one, use_container_width=True)


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
        use_container_width=True,
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RINGKASAN EKSEKUTIF
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 🎯 Ringkasan Eksekutif — 8 Topik Utama")

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

    # KPI per topik (2×4 grid)
    rows = [summary.iloc[:4], summary.iloc[4:]]
    for row_group in rows:
        cols = st.columns(4)
        for i, (_, r) in enumerate(row_group.iterrows()):
            color = TOPIC_COLORS.get(r["topic_label"], "#58a6ff")
            with cols[i]:
                st.markdown(f"""
                <div style='background:#161b22;border:1px solid {color};border-left:4px solid {color};
                            border-radius:10px;padding:16px;margin-bottom:10px;min-height:140px'>
                  <div style='font-size:10px;color:#8b949e;font-weight:600;text-transform:uppercase;
                              letter-spacing:0.5px;margin-bottom:6px'>
                    Topik {summary[summary["topic_label"]==r["topic_label"]].index[0]+1}
                  </div>
                  <div style='font-size:11px;color:#e6edf3;font-weight:600;line-height:1.4;margin-bottom:10px'>
                    {r["topic_label"]}
                  </div>
                  <div style='font-size:26px;font-weight:800;color:{color}'>{r["Jumlah"]}</div>
                  <div style='font-size:10px;color:#8b949e'>responden · {r["Persen"]}%</div>
                  <div style='margin-top:8px;font-size:10px;color:#8b949e'>
                    Avg Prob: <span style='color:{color};font-weight:600'>{r["Avg_Prob"]:.3f}</span> &nbsp;
                    High Conf: <span style='color:#3fb950;font-weight:600'>{r["High_Pct"]}%</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # Radar chart
    rc1, rc2 = st.columns([3, 2])

    with rc1:
        radar_df = summary.copy()
        cats = radar_df["topic_label"].str.split(" & ").str[0].tolist()
        vals = radar_df["Jumlah"].tolist()
        vals_norm = [v/max(vals)*100 for v in vals]
        cats_closed = cats + [cats[0]]
        vals_closed  = vals_norm + [vals_norm[0]]

        theta = np.linspace(0, 2*np.pi, len(cats), endpoint=False)
        theta_closed = np.append(theta, theta[0])
        xs = [np.cos(t)*v for t,v in zip(theta_closed, vals_closed)]
        ys = [np.sin(t)*v for t,v in zip(theta_closed, vals_closed)]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_norm + [vals_norm[0]],
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(31,111,235,0.2)",
            line_color="#388bfd",
            line_width=2,
            name="Jumlah Responden (normalized)",
        ))
        fig_radar.update_layout(**PLOTLY_LAYOUT, height=400,
            title=dict(text="Radar Chart — Profil 8 Dimensi Dampak Bencana", font=dict(size=13)),
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,100], color="#8b949e", gridcolor="#21262d"),
                angularaxis=dict(color="#e6edf3", gridcolor="#21262d"),
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with rc2:
        st.markdown("#### 📊 Tabel Ringkasan")
        disp_sum = summary[["topic_label","Jumlah","Persen","Avg_Prob","High_Pct"]].copy()
        disp_sum.columns = ["Topik","Responden","%","Avg Prob","% High"]
        st.dataframe(
            disp_sum.reset_index(drop=True),
            use_container_width=True,
            height=380,
            column_config={
                "Avg Prob": st.column_config.ProgressColumn("Avg Prob", min_value=0, max_value=1, format="%.3f"),
                "% High":   st.column_config.NumberColumn("% High Conf", format="%.1f%%"),
            }
        )

    # Insight text
    top1 = summary.iloc[0]
    top_prob = summary.loc[summary["Avg_Prob"].idxmax()]
    st.markdown("---")
    st.markdown("#### 💡 Insight Otomatis")
    st.info(f"""
    **Topik Paling Banyak Dibahas:** {top1['topic_label']} — {top1['Jumlah']} responden ({top1['Persen']}%)

    **Topik dengan Kepercayaan Model Tertinggi:** {top_prob['topic_label']} — Avg Prob {top_prob['Avg_Prob']:.3f}

    **Total High Confidence:** {high_conf:,} dari {len(fdf):,} responden ({high_conf/len(fdf)*100:.1f}%) memiliki skor probabilitas ≥ 0.75

    **Wilayah dengan Responden Terbanyak:** {fdf['Wilayah'].value_counts().index[0]} ({fdf['Wilayah'].value_counts().iloc[0]} responden)
    """)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#8b949e;font-size:12px;padding:10px 0'>
  🌊 Dashboard Analisis Topic Modeling Bencana Banjir &nbsp;·&nbsp;
  Dibangun dengan Streamlit & Plotly &nbsp;·&nbsp;
  <span style='color:#58a6ff'>1.337 Responden · 8 Topik · 11 Wilayah</span>
</div>
""", unsafe_allow_html=True)
