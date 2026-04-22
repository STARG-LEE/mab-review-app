"""
mAb-GATED Review System (Streamlit)

리뷰어가 GPT 해석을 검토하고 적절/부적절 판정 + 피드백을 남기는 앱.
"""

import json
import math
import os

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine, text


# ─── Config ───
DB_URL = None
try:
    if hasattr(st, "secrets") and "DB_URL" in st.secrets:
        DB_URL = st.secrets["DB_URL"]
except Exception:
    pass
if not DB_URL:
    DB_URL = os.environ.get("DB_URL")
if not DB_URL:
    st.error("DB_URL이 설정되지 않았습니다. Streamlit Cloud의 Secrets 또는 환경변수 DB_URL을 설정하세요.")
    st.stop()


CATEGORY_COLORS = {
    'formulation':     '#E74C3C',   # red
    'stress':          '#F39C12',   # orange
    'sequence':        '#1ABC9C',   # teal
    'structure':       '#16A085',   # dark teal
    'stability':       '#9B59B6',   # purple
    'quality_outcome': '#E91E63',   # pink
}

CATEGORY_ORDER = ['formulation', 'stress', 'sequence', 'structure', 'stability', 'quality_outcome']

RELATION_COLORS = {
    # Positive (green)
    'stabilizes': '#27AE60', 'inhibits': '#27AE60', 'prevents': '#27AE60',
    'decreases': '#27AE60', 'shields': '#27AE60',
    # Negative (red)
    'destabilizes': '#C0392B', 'promotes': '#C0392B', 'increases': '#C0392B',
    'induces': '#C0392B', 'aggregates': '#C0392B', 'oxidizes': '#C0392B',
    'deamidates': '#C0392B', 'isomerizes': '#C0392B', 'fragments': '#C0392B',
    'unfolds': '#C0392B', 'adsorbs': '#C0392B', 'precipitates': '#C0392B',
    'degrades': '#C0392B',
    # Neutral (gray)
    'correlates': '#7F8C8D', 'modifies': '#7F8C8D', 'binds': '#7F8C8D',
    'requires': '#7F8C8D',
}

RELATION_GROUP = {
    **{r: 'positive' for r in ['stabilizes', 'inhibits', 'prevents', 'decreases', 'shields']},
    **{r: 'negative' for r in ['destabilizes', 'promotes', 'increases', 'induces', 'aggregates',
                               'oxidizes', 'deamidates', 'isomerizes', 'fragments', 'unfolds',
                               'adsorbs', 'precipitates', 'degrades']},
    **{r: 'neutral' for r in ['correlates', 'modifies', 'binds', 'requires']},
}


# ─── DB helpers ───
@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

@st.cache_data(ttl=60)
def load_all_samples():
    engine = get_engine()
    df = pd.read_sql("""
        SELECT sample_id, target_node, target_subcat, record_json, gpt_interpretation
        FROM mab_review_samples
        WHERE gpt_interpretation IS NOT NULL AND gpt_interpretation != ''
        ORDER BY target_node, sample_id
    """, engine)
    return df

@st.cache_data(ttl=30)
def load_reviews(reviewer=None):
    engine = get_engine()
    q = "SELECT * FROM mab_review_results"
    params = {}
    if reviewer:
        q += " WHERE reviewer = :r"
        params['r'] = reviewer
    df = pd.read_sql(text(q), engine, params=params)
    return df

def save_review(sample_id, reviewer, verdict, feedback):
    engine = get_engine()
    with engine.connect() as conn:
        existing = conn.execute(
            text("SELECT review_id FROM mab_review_results WHERE sample_id = :s AND reviewer = :r"),
            {'s': int(sample_id), 'r': reviewer}
        ).fetchone()
        if existing:
            conn.execute(
                text("UPDATE mab_review_results SET verdict = :v, feedback = :f, reviewed_at = NOW() WHERE review_id = :id"),
                {'v': verdict, 'f': feedback, 'id': existing[0]}
            )
        else:
            conn.execute(
                text("INSERT INTO mab_review_results (sample_id, reviewer, verdict, feedback) VALUES (:s, :r, :v, :f)"),
                {'s': int(sample_id), 'r': reviewer, 'v': verdict, 'f': feedback}
            )
        conn.commit()
    st.cache_data.clear()


# ─── Interactive Plotly graph ───
def draw_record_plotly(record):
    """Clean interactive plotly graph: target center, neighbors clustered by category."""
    target = record['target']
    cats_present = [c for c in CATEGORY_ORDER if record['neighbors'].get(c)]
    n_cats = len(cats_present)

    # ── Layout: 각 카테고리별로 clear sector 할당, 노드는 그 안에서 정렬 ──
    pos = {target: (0.0, 0.0)}
    meta = {target: {'cat': 'TARGET', 'label': target}}

    R_CLUSTER = 3.2
    CLUSTER_RADIUS = 1.1

    for c_idx, cat in enumerate(cats_present):
        neighbors = record['neighbors'][cat]
        # Cluster center
        cat_angle = 2 * math.pi * c_idx / n_cats - math.pi / 2
        cx, cy = R_CLUSTER * math.cos(cat_angle), R_CLUSTER * math.sin(cat_angle)

        n = len(neighbors)
        for i, nb in enumerate(neighbors):
            # Place around cluster center in small circle
            if n == 1:
                x, y = cx, cy
            else:
                a = 2 * math.pi * i / n
                x = cx + CLUSTER_RADIUS * math.cos(a)
                y = cy + CLUSTER_RADIUS * math.sin(a)
            nid = f"{cat}::{i}::{nb['node']}"
            pos[nid] = (x, y)
            meta[nid] = {
                'cat': cat, 'label': nb['node'], 'rel': nb['relationship'],
                'freq': nb['frequency'], 'papers': nb['num_papers'],
            }

    # ── Edges (grouped by relation polarity) ──
    traces = []
    grp_style = {
        'positive': ('#27AE60', '긍정 효과 (stabilizes·inhibits·prevents·decreases·shields)'),
        'negative': ('#C0392B', '부정 효과 (promotes·increases·induces·destabilizes 등)'),
        'neutral':  ('#85929E', '중립 (modifies·correlates·binds·requires)'),
    }
    for grp, (color, name) in grp_style.items():
        xs, ys = [], []
        for nid, m in meta.items():
            if nid == target: continue
            if RELATION_GROUP.get(m['rel'], 'neutral') != grp: continue
            x1, y1 = pos[nid]
            xs += [x1, 0.0, None]
            ys += [y1, 0.0, None]
        if xs:
            traces.append(go.Scatter(
                x=xs, y=ys, mode='lines',
                line=dict(color=color, width=1.8),
                hoverinfo='skip', name=name, opacity=0.45,
            ))

    # ── Target node ──
    traces.append(go.Scatter(
        x=[0], y=[0], mode='markers+text',
        marker=dict(size=68, color='#1B4F72', line=dict(color='#FCD434', width=4), symbol='circle'),
        text=[f"<b>{target}</b>"],
        textposition='middle center',
        textfont=dict(color='#FFFFFF', size=13),
        hovertext=[f"<b>🎯 TARGET</b><br>{target}"],
        hoverinfo='text',
        name=f'🎯 {target}',
    ))

    # ── Neighbor nodes per category ──
    for cat in cats_present:
        items = [(nid, m) for nid, m in meta.items() if m.get('cat') == cat and nid != target]
        if not items: continue
        xs, ys, labels, sizes, hovers = [], [], [], [], []
        for nid, m in items:
            x, y = pos[nid]
            xs.append(x); ys.append(y)
            lab = m['label']
            labels.append(lab if len(lab) <= 22 else lab[:20] + '…')
            sizes.append(min(22 + 5 * math.sqrt(m['freq']), 50))
            hovers.append(
                f"<b>{m['label']}</b><br>"
                f"카테고리: <b>{cat}</b><br>"
                f"관계: <b>{m['rel']}</b> → {target}<br>"
                f"빈도: {m['freq']} | 논문: {m['papers']}편"
            )
        traces.append(go.Scatter(
            x=xs, y=ys, mode='markers+text',
            marker=dict(size=sizes, color=CATEGORY_COLORS[cat],
                        line=dict(color='white', width=2)),
            text=labels,
            textposition='bottom center',
            textfont=dict(color='#1B2631', size=10, family='Arial'),
            hovertext=hovers, hoverinfo='text',
            name=f'● {cat}',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation='v', yanchor='top', y=1, xanchor='left', x=1.02,
            bgcolor='rgba(255,255,255,0.95)', bordercolor='#BDC3C7', borderwidth=1,
            font=dict(size=10, color='#2C3E50'),
            itemsizing='constant',
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-5.2, 5.2]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-5.2, 5.2], scaleanchor='x', scaleratio=1),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        margin=dict(l=10, r=10, t=20, b=10),
        height=640, hovermode='closest',
        font=dict(color='#2C3E50'),
    )
    return fig


# ─── App ───
st.set_page_config(
    page_title="mAb-GATED Review",
    layout="wide",
    page_icon="🔬"
)

# Custom CSS — 모든 커스텀 HTML 블록 가시성 보장 (light/dark 테마 모두)
st.markdown("""
<style>
    .stApp { max-width: 1400px; margin: auto; }
    h1 { color: #1B4F72 !important; border-bottom: 3px solid #2E86C1; padding-bottom: 10px; }
    h2 { color: #2E86C1 !important; }
    .stButton > button { border-radius: 8px; font-weight: 600; }

    /* Force dark text on all custom HTML content blocks (avoid white-on-white) */
    .custom-block, .custom-block * {
        color: #2C3E50 !important;
    }
    .custom-block pre, .custom-block code {
        color: #2C3E50 !important;
        background: transparent !important;
    }
    .custom-block small { color: #566573 !important; }
    .custom-block b, .custom-block strong { color: #1B4F72 !important; }

    /* Light-theme-safe expander content */
    .streamlit-expanderContent {
        background: #FFFFFF !important;
    }
    .streamlit-expanderContent, .streamlit-expanderContent * {
        color: #2C3E50 !important;
    }

    /* Banner text always white (dark background) */
    .banner, .banner * { color: #FFFFFF !important; }
    .banner .highlight { color: #FCD434 !important; }

    /* Nav (sample number) bar */
    .nav-bar {
        background: #EBF5FB !important;
        color: #1B4F72 !important;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        padding: 8px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 mAb-GATED Interpretation Review")
st.caption("AI(GPT-5-mini)가 생성한 mAb 안정성 인과 해석을 도메인 전문가가 검토합니다.")

# Reviewer info
with st.sidebar:
    st.header("👤 리뷰어 정보")
    reviewer = st.text_input("이름/ID", value=st.session_state.get('reviewer', ''))
    if reviewer:
        st.session_state['reviewer'] = reviewer

    st.divider()

    if reviewer:
        my_reviews = load_reviews(reviewer)
        total_samples = len(load_all_samples())
        done = len(my_reviews) if not my_reviews.empty else 0
        st.metric("내 리뷰 진행", f"{done} / {total_samples}")
        if total_samples:
            st.progress(done / total_samples)

        if done > 0:
            agree = (my_reviews['verdict'] == 'appropriate').sum()
            disagree = (my_reviews['verdict'] == 'inappropriate').sum()
            col_a, col_b = st.columns(2)
            col_a.metric("✅ 적절", agree)
            col_b.metric("❌ 부적절", disagree)

    st.divider()
    st.caption("💡 그래프는 **인터랙티브**입니다. 노드에 마우스를 올리면 상세 정보가 표시됩니다.")

if not reviewer:
    st.warning("먼저 왼쪽 사이드바에 리뷰어 이름을 입력하세요.")
    st.stop()

df_samples = load_all_samples()
if df_samples.empty:
    st.error("아직 리뷰 샘플이 생성되지 않았습니다. `generate_review_samples.ipynb`를 먼저 실행해주세요.")
    st.info(f"DB URL (앞 30자): `{DB_URL[:30]}...`")
    st.stop()


# Filter controls
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    subcat_filter = st.selectbox(
        "Subcategory 필터",
        options=['전체'] + sorted(df_samples['target_subcat'].unique().tolist()),
    )
with col2:
    df_filtered = df_samples if subcat_filter == '전체' else df_samples[df_samples['target_subcat'] == subcat_filter]
    target_filter = st.selectbox(
        "Target 노드 필터",
        options=['전체'] + sorted(df_filtered['target_node'].unique().tolist()),
    )
with col3:
    show_only_unreviewed = st.checkbox("리뷰 전만", value=True)

df_view = df_samples.copy()
if subcat_filter != '전체':
    df_view = df_view[df_view['target_subcat'] == subcat_filter]
if target_filter != '전체':
    df_view = df_view[df_view['target_node'] == target_filter]

my_reviews = load_reviews(reviewer)
reviewed_sample_ids = set(my_reviews['sample_id'].tolist()) if not my_reviews.empty else set()
if show_only_unreviewed:
    df_view = df_view[~df_view['sample_id'].isin(reviewed_sample_ids)]

df_view = df_view.reset_index(drop=True)

st.caption(f"📋 현재 조건에 맞는 샘플: **{len(df_view)}개**")

if df_view.empty:
    st.success("🎉 조건에 해당하는 모든 샘플을 리뷰했습니다!")
    st.stop()


if 'current_idx' not in st.session_state:
    st.session_state['current_idx'] = 0
if st.session_state['current_idx'] >= len(df_view):
    st.session_state['current_idx'] = 0

col_prev, col_info, col_next = st.columns([1, 3, 1])
with col_prev:
    if st.button("⬅️ 이전", disabled=(st.session_state['current_idx'] == 0), use_container_width=True):
        st.session_state['current_idx'] -= 1
        st.rerun()
with col_info:
    st.markdown(f"<div class='nav-bar'>샘플 {st.session_state['current_idx']+1} / {len(df_view)}</div>", unsafe_allow_html=True)
with col_next:
    if st.button("다음 ➡️", disabled=(st.session_state['current_idx'] >= len(df_view) - 1), use_container_width=True):
        st.session_state['current_idx'] += 1
        st.rerun()

current_sample = df_view.iloc[st.session_state['current_idx']]
record = json.loads(current_sample['record_json'])

st.divider()

# Target banner
st.markdown(f"""
<div class='banner' style='background:linear-gradient(135deg,#1B4F72,#2E86C1);padding:16px 24px;border-radius:10px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,0.1)'>
    <h3 style='margin:0;color:#FFFFFF'>🎯 Target: <span class='highlight'>{record['target']}</span></h3>
    <small>Subcategory: <b style='color:#FFFFFF'>{current_sample['target_subcat']}</b> &nbsp; | &nbsp; Sample ID: {current_sample['sample_id']}</small>
</div>
""", unsafe_allow_html=True)

# Two-column layout
col_graph, col_interp = st.columns([1.1, 1])

with col_graph:
    st.subheader("📊 학습 레코드 그래프")
    fig = draw_record_plotly(record)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with st.expander("📋 이웃 노드 전체 리스트"):
        for cat in CATEGORY_ORDER:
            neighbors = record['neighbors'].get(cat, [])
            if not neighbors:
                continue
            color = CATEGORY_COLORS.get(cat, '#BDC3C7')
            st.markdown(
                f"<div class='custom-block' style='background:{color}33;padding:8px 12px;border-left:4px solid {color};margin:8px 0;border-radius:4px'>"
                f"<b style='color:{color}'>{cat}</b> <span style='color:#2C3E50'>({len(neighbors)}개)</span></div>",
                unsafe_allow_html=True
            )
            for n in neighbors:
                grp = RELATION_GROUP.get(n['relationship'], 'neutral')
                rel_color = {'positive': '#27AE60', 'negative': '#C0392B', 'neutral': '#7F8C8D'}[grp]
                st.markdown(
                    f"<div class='custom-block' style='padding:4px 16px;font-size:14px'>"
                    f"• <code style='background:#F4F6F7;color:#1B4F72;padding:1px 6px;border-radius:3px'>{n['node']}</code> "
                    f"<span style='background:{rel_color};color:#FFFFFF;padding:2px 10px;border-radius:10px;font-size:12px;font-weight:700'>{n['relationship']}</span> "
                    f"<span style='color:#2C3E50'>→ {record['target']}</span> "
                    f"<small style='color:#7F8C8D'>(freq={n['frequency']}, papers={n['num_papers']})</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )

with col_interp:
    st.subheader("🤖 GPT-5-mini 해석")
    # HTML escape the interpretation text to avoid breaking layout
    import html as _html
    interp_escaped = _html.escape(current_sample['gpt_interpretation'])
    st.markdown(f"""
    <div class='custom-block' style='background:#FFFFFF;padding:18px;border-radius:8px;border:1px solid #D5DBDB;border-left:4px solid #3498DB;max-height:620px;overflow-y:auto;box-shadow:0 1px 3px rgba(0,0,0,0.05)'>
        <pre style='white-space:pre-wrap;font-family:"Apple SD Gothic Neo","Malgun Gothic",sans-serif;font-size:14px;line-height:1.7;margin:0;color:#2C3E50 !important;background:transparent !important'>{interp_escaped}</pre>
    </div>
    """, unsafe_allow_html=True)


st.divider()
st.subheader("🧑‍🔬 당신의 리뷰")

existing = my_reviews[my_reviews['sample_id'] == current_sample['sample_id']] if not my_reviews.empty else pd.DataFrame()
existing_verdict = existing['verdict'].iloc[0] if not existing.empty else None
existing_feedback = existing['feedback'].iloc[0] if not existing.empty else ''

col_v, col_fb = st.columns([1, 3])
with col_v:
    verdict = st.radio(
        "해석 적절성",
        options=['appropriate', 'inappropriate'],
        index=0 if existing_verdict == 'appropriate' else (1 if existing_verdict == 'inappropriate' else 0),
        format_func=lambda x: '✅ 적절' if x == 'appropriate' else '❌ 부적절',
        key=f"verdict_{current_sample['sample_id']}",
    )

with col_fb:
    feedback = st.text_area(
        "피드백 (선택, 부적절한 경우 이유 작성)",
        value=existing_feedback or '',
        height=120,
        placeholder="예: 'thermal stress가 aggregation을 유발한다는 설명은 맞지만, polysorbate 80이 직접 단백질 변성을 막는다는 부분은 부정확함. 실제로는 공기-물 계면 흡착 방지가 주 메커니즘.'",
        key=f"feedback_{current_sample['sample_id']}",
    )

col_save, col_skip, col_status = st.columns([1, 1, 3])
with col_save:
    if st.button("💾 저장 & 다음", type="primary", use_container_width=True):
        save_review(current_sample['sample_id'], reviewer, verdict, feedback)
        if st.session_state['current_idx'] < len(df_view) - 1:
            st.session_state['current_idx'] += 1
        st.rerun()
with col_skip:
    if st.button("⏭️ 건너뛰기", use_container_width=True):
        if st.session_state['current_idx'] < len(df_view) - 1:
            st.session_state['current_idx'] += 1
            st.rerun()
with col_status:
    if not existing.empty:
        st.info(f"이미 리뷰함 ({existing['verdict'].iloc[0]}). 저장 시 업데이트됩니다.")
