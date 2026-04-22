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
    """Draw an interactive plotly graph: target in center, neighbors by category sector."""
    target = record['target']
    cats_present = [c for c in CATEGORY_ORDER if record['neighbors'].get(c)]
    n_cats = len(cats_present)

    # Compute positions
    pos = {target: (0.0, 0.0)}
    node_meta = {target: {'cat': 'TARGET', 'label': target, 'freq': 0}}

    R_INNER = 1.0   # inner ring for category labels
    R_NODES = 2.2   # radius for nodes

    for c_idx, cat in enumerate(cats_present):
        neighbors = record['neighbors'][cat]
        # Category sector angular range
        sector_start = 2 * math.pi * c_idx / n_cats - math.pi / 2
        sector_end = 2 * math.pi * (c_idx + 1) / n_cats - math.pi / 2
        sector_mid = (sector_start + sector_end) / 2
        sector_width = (sector_end - sector_start) * 0.8

        n = len(neighbors)
        for i, nb in enumerate(neighbors):
            # Spread neighbors within sector
            if n == 1:
                angle = sector_mid
            else:
                angle = sector_start + sector_width * (i + 0.5) / n + (sector_end - sector_start - sector_width) / 2
            # Alternating radii to reduce overlap
            r = R_NODES + (i % 3) * 0.3
            x, y = r * math.cos(angle), r * math.sin(angle)
            nid = f"{cat}::{nb['node']}"
            pos[nid] = (x, y)
            node_meta[nid] = {
                'cat': cat,
                'label': nb['node'],
                'rel': nb['relationship'],
                'freq': nb['frequency'],
                'papers': nb['num_papers'],
            }

    # Build edge traces (one per relation group to have distinct colors)
    edge_traces = []
    edge_groups = {'positive': [], 'negative': [], 'neutral': []}
    for nid, meta in node_meta.items():
        if nid == target:
            continue
        grp = RELATION_GROUP.get(meta.get('rel'), 'neutral')
        edge_groups[grp].append((nid, meta))

    group_styles = {
        'positive': {'color': '#27AE60', 'name': '긍정적 효과 (stabilizes 등)'},
        'negative': {'color': '#C0392B', 'name': '부정적 효과 (promotes, increases 등)'},
        'neutral':  {'color': '#7F8C8D', 'name': '중립 관계 (modifies, correlates 등)'},
    }

    for grp, edges in edge_groups.items():
        if not edges:
            continue
        xs, ys = [], []
        hover_texts = []
        for nid, meta in edges:
            x0, y0 = pos[target]
            x1, y1 = pos[nid]
            xs += [x1, x0, None]
            ys += [y1, y0, None]
        edge_traces.append(go.Scatter(
            x=xs, y=ys,
            mode='lines',
            line=dict(color=group_styles[grp]['color'], width=1.5),
            hoverinfo='skip',
            name=group_styles[grp]['name'],
            showlegend=True,
            opacity=0.55,
        ))

    # Node traces (one per category + target)
    traces = [*edge_traces]

    # Target node
    traces.append(go.Scatter(
        x=[pos[target][0]], y=[pos[target][1]],
        mode='markers+text',
        marker=dict(size=55, color='#2C3E50', line=dict(color='#FCD434', width=3),
                    symbol='circle'),
        text=[target],
        textposition='middle center',
        textfont=dict(color='white', size=13, family='Arial Black'),
        hovertext=[f"<b>TARGET</b><br>{target}"],
        hoverinfo='text',
        name='🎯 Target',
        showlegend=True,
    ))

    # Neighbor nodes per category
    for cat in cats_present:
        cat_nodes = [(nid, m) for nid, m in node_meta.items() if m['cat'] == cat]
        if not cat_nodes:
            continue
        xs, ys = [], []
        labels = []
        sizes = []
        hover_texts = []
        for nid, m in cat_nodes:
            x, y = pos[nid]
            xs.append(x); ys.append(y)
            labels.append(m['label'][:25])
            # size proportional to sqrt(frequency)
            size = 20 + 6 * math.sqrt(m['freq'])
            sizes.append(min(size, 55))
            hover_texts.append(
                f"<b>{m['label']}</b><br>"
                f"카테고리: {cat}<br>"
                f"관계: {m['rel']}<br>"
                f"빈도: {m['freq']}<br>"
                f"논문 수: {m['papers']}"
            )
        traces.append(go.Scatter(
            x=xs, y=ys,
            mode='markers+text',
            marker=dict(
                size=sizes, color=CATEGORY_COLORS[cat],
                line=dict(color='white', width=2), symbol='circle',
            ),
            text=labels,
            textposition='bottom center',
            textfont=dict(color='#2C3E50', size=10),
            hovertext=hover_texts,
            hoverinfo='text',
            name=f'📌 {cat}',
            showlegend=True,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top", y=1, xanchor="left", x=1.02,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#CCC', borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5, 3.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3.5, 3.5],
                   scaleanchor='x', scaleratio=1),
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white',
        margin=dict(l=10, r=10, t=30, b=10),
        height=620,
        hovermode='closest',
    )
    return fig


# ─── App ───
st.set_page_config(
    page_title="mAb-GATED Review",
    layout="wide",
    page_icon="🔬"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { max-width: 1400px; margin: auto; }
    h1 { color: #1B4F72; border-bottom: 3px solid #2E86C1; padding-bottom: 10px; }
    h2 { color: #2E86C1; }
    .stButton > button { border-radius: 8px; font-weight: 600; }
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
    st.markdown(f"<div style='text-align:center;font-weight:bold;font-size:18px;padding:8px;background:#EBF5FB;border-radius:6px'>샘플 {st.session_state['current_idx']+1} / {len(df_view)}</div>", unsafe_allow_html=True)
with col_next:
    if st.button("다음 ➡️", disabled=(st.session_state['current_idx'] >= len(df_view) - 1), use_container_width=True):
        st.session_state['current_idx'] += 1
        st.rerun()

current_sample = df_view.iloc[st.session_state['current_idx']]
record = json.loads(current_sample['record_json'])

st.divider()

# Target banner
st.markdown(f"""
<div style='background:linear-gradient(135deg,#1B4F72,#2E86C1);color:white;padding:16px 24px;border-radius:10px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,0.1)'>
    <h3 style='margin:0;color:white'>🎯 Target: <span style='color:#FCD434'>{record['target']}</span></h3>
    <small style='opacity:0.9'>Subcategory: <b>{current_sample['target_subcat']}</b> &nbsp; | &nbsp; Sample ID: {current_sample['sample_id']}</small>
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
            st.markdown(f"<div style='background:{color}20;padding:8px 12px;border-left:4px solid {color};margin:8px 0;border-radius:4px'><b>{cat}</b> ({len(neighbors)}개)</div>", unsafe_allow_html=True)
            for n in neighbors:
                grp = RELATION_GROUP.get(n['relationship'], 'neutral')
                rel_color = {'positive': '#27AE60', 'negative': '#C0392B', 'neutral': '#7F8C8D'}[grp]
                st.markdown(f"""  • `{n['node']}` <span style='background:{rel_color}25;color:{rel_color};padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600'>{n['relationship']}</span> → {record['target']}  <small style='color:#888'>(freq={n['frequency']}, papers={n['num_papers']})</small>""",
                unsafe_allow_html=True)

with col_interp:
    st.subheader("🤖 GPT-5-mini 해석")
    st.markdown(f"""
    <div style='background:#F4F6F7;padding:18px;border-radius:8px;border-left:4px solid #3498DB;max-height:620px;overflow-y:auto'>
        <pre style='white-space:pre-wrap;font-family:"Apple SD Gothic Neo","Malgun Gothic",sans-serif;font-size:14px;line-height:1.6;margin:0'>{current_sample['gpt_interpretation']}</pre>
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
