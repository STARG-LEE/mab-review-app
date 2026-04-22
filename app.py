"""
mAb-GATED Review System (Streamlit)

리뷰어가 GPT 해석을 검토하고 적절/부적절 판정 + 피드백을 남기는 앱.
"""

import json
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from sqlalchemy import create_engine, text
import random

# ─── Config ───
# DB_URL은 반드시 secrets 또는 환경변수로 제공 (하드코딩 금지)
import os
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
    'formulation':     '#FF6B6B',
    'stress':          '#FFB142',
    'sequence':        '#4ECDC4',
    'structure':       '#95E1D3',
    'stability':       '#AA96DA',
    'quality_outcome': '#FCBAD3',
}

RELATION_COLORS = {
    # Positive (green family)
    'stabilizes': '#27AE60', 'inhibits': '#27AE60', 'prevents': '#27AE60',
    'decreases': '#27AE60', 'shields': '#27AE60',
    # Negative (red family)
    'destabilizes': '#E74C3C', 'promotes': '#E74C3C', 'increases': '#E74C3C',
    'induces': '#E74C3C', 'aggregates': '#E74C3C', 'oxidizes': '#E74C3C',
    'deamidates': '#E74C3C', 'isomerizes': '#E74C3C', 'fragments': '#E74C3C',
    'unfolds': '#E74C3C', 'adsorbs': '#E74C3C', 'precipitates': '#E74C3C',
    'degrades': '#E74C3C',
    # Neutral (gray family)
    'correlates': '#95A5A6', 'modifies': '#95A5A6', 'binds': '#95A5A6',
    'requires': '#95A5A6',
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
        # Upsert: if reviewer already reviewed this sample, update; else insert
        existing = conn.execute(
            text("SELECT review_id FROM mab_review_results WHERE sample_id = :s AND reviewer = :r"),
            {'s': sample_id, 'r': reviewer}
        ).fetchone()
        if existing:
            conn.execute(
                text("UPDATE mab_review_results SET verdict = :v, feedback = :f, reviewed_at = NOW() WHERE review_id = :id"),
                {'v': verdict, 'f': feedback, 'id': existing[0]}
            )
        else:
            conn.execute(
                text("INSERT INTO mab_review_results (sample_id, reviewer, verdict, feedback) VALUES (:s, :r, :v, :f)"),
                {'s': sample_id, 'r': reviewer, 'v': verdict, 'f': feedback}
            )
        conn.commit()
    st.cache_data.clear()


# ─── Graph visualization ───
def draw_record_graph(record):
    """Draw a star-like graph: target in center, neighbors around by category."""
    target = record['target']
    fig, ax = plt.subplots(figsize=(11, 8))

    G = nx.DiGraph()
    G.add_node(target, cat='TARGET')

    pos = {target: (0, 0)}

    categories = list(record['neighbors'].keys())
    cats_with_data = [c for c in categories if record['neighbors'][c]]
    n_cats = len(cats_with_data)

    import math
    for c_idx, cat in enumerate(cats_with_data):
        neighbors = record['neighbors'][cat]
        # Angular position for this category
        cat_angle = 2 * math.pi * c_idx / n_cats
        r_cat = 4.0
        # Inner offset for neighbor spread
        for n_idx, n in enumerate(neighbors):
            nid = f"{cat}::{n['node']}"
            G.add_node(nid, cat=cat, label=n['node'])
            G.add_edge(nid, target, rel=n['relationship'])

            # Spread neighbors in an arc within the category sector
            n_count = len(neighbors)
            spread = 0.35 * (n_idx - (n_count - 1) / 2) / max(n_count, 1)
            angle = cat_angle + spread
            r = r_cat + (n_idx % 2) * 0.5  # slight radius variation
            pos[nid] = (r * math.cos(angle), r * math.sin(angle))

    # Draw edges
    for u, v, d in G.edges(data=True):
        rel = d['rel']
        color = RELATION_COLORS.get(rel, '#7F8C8D')
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            edge_color=color, arrows=True, arrowsize=12, alpha=0.6, width=1.5
        )

    # Draw nodes
    for node, d in G.nodes(data=True):
        cat = d['cat']
        if cat == 'TARGET':
            nx.draw_networkx_nodes(G, pos, nodelist=[node], ax=ax,
                                    node_color='#2C3E50', node_size=2200, alpha=0.95)
        else:
            color = CATEGORY_COLORS.get(cat, '#BDC3C7')
            nx.draw_networkx_nodes(G, pos, nodelist=[node], ax=ax,
                                    node_color=color, node_size=900, alpha=0.85)

    # Labels
    target_labels = {target: target}
    neighbor_labels = {n: G.nodes[n].get('label', n) for n in G.nodes if n != target}
    nx.draw_networkx_labels(G, pos, labels=target_labels, ax=ax,
                            font_size=12, font_weight='bold', font_color='white')
    nx.draw_networkx_labels(G, pos, labels=neighbor_labels, ax=ax,
                            font_size=7.5, font_color='#2C3E50')

    # Category legend
    legend_items = [
        patches.Patch(color='#2C3E50', label=f'TARGET: {target}')
    ] + [
        patches.Patch(color=c, label=cat)
        for cat, c in CATEGORY_COLORS.items() if any(cat in nid for nid in G.nodes if '::' in nid)
    ]
    ax.legend(handles=legend_items, loc='upper left', fontsize=8,
              bbox_to_anchor=(0.0, 1.0), framealpha=0.9)

    ax.axis('off')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    plt.tight_layout()
    return fig


# ─── App ───
st.set_page_config(
    page_title="mAb-GATED Review",
    layout="wide",
    page_icon="🔬"
)

st.title("🔬 mAb-GATED Interpretation Review")
st.caption("AI(GPT-5-mini)가 생성한 mAb 안정성 인과 해석을 도메인 전문가가 검토합니다.")

# Reviewer info (sidebar)
with st.sidebar:
    st.header("👤 리뷰어 정보")
    reviewer = st.text_input("이름/ID", value=st.session_state.get('reviewer', ''))
    if reviewer:
        st.session_state['reviewer'] = reviewer

    st.divider()

    # Progress stats
    if reviewer:
        my_reviews = load_reviews(reviewer)
        total_samples = len(load_all_samples())
        done = len(my_reviews)
        st.metric("내 리뷰 진행", f"{done} / {total_samples}")
        st.progress(done / total_samples if total_samples else 0)

        if done > 0:
            agree = (my_reviews['verdict'] == 'appropriate').sum()
            disagree = (my_reviews['verdict'] == 'inappropriate').sum()
            st.caption(f"✅ 적절 {agree} / ❌ 부적절 {disagree}")

if not reviewer:
    st.warning("먼저 왼쪽 사이드바에 리뷰어 이름을 입력하세요.")
    st.stop()


# Load samples
df_samples = load_all_samples()
if df_samples.empty:
    st.error("아직 리뷰 샘플이 생성되지 않았습니다. generate_review_samples.py를 먼저 실행해주세요.")
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

# Apply filters
df_view = df_samples.copy()
if subcat_filter != '전체':
    df_view = df_view[df_view['target_subcat'] == subcat_filter]
if target_filter != '전체':
    df_view = df_view[df_view['target_node'] == target_filter]

# Filter unreviewed
my_reviews = load_reviews(reviewer)
reviewed_sample_ids = set(my_reviews['sample_id'].tolist()) if not my_reviews.empty else set()
if show_only_unreviewed:
    df_view = df_view[~df_view['sample_id'].isin(reviewed_sample_ids)]

st.caption(f"📋 현재 조건에 맞는 샘플: **{len(df_view)}개**")

if df_view.empty:
    st.success("🎉 조건에 해당하는 모든 샘플을 리뷰했습니다!")
    st.stop()


# Pick current sample
if 'current_idx' not in st.session_state:
    st.session_state['current_idx'] = 0

# Ensure idx is within filtered range
if st.session_state['current_idx'] >= len(df_view):
    st.session_state['current_idx'] = 0

# Navigation
col_prev, col_info, col_next = st.columns([1, 3, 1])
with col_prev:
    if st.button("⬅️ 이전", disabled=(st.session_state['current_idx'] == 0)):
        st.session_state['current_idx'] -= 1
        st.rerun()
with col_info:
    st.markdown(f"<div style='text-align:center;font-weight:bold;font-size:18px'>샘플 {st.session_state['current_idx']+1} / {len(df_view)}</div>", unsafe_allow_html=True)
with col_next:
    if st.button("다음 ➡️", disabled=(st.session_state['current_idx'] >= len(df_view) - 1)):
        st.session_state['current_idx'] += 1
        st.rerun()

current_sample = df_view.iloc[st.session_state['current_idx']]
record = json.loads(current_sample['record_json'])

# ─── Display sample ───
st.divider()

# Target info banner
st.markdown(f"""
<div style='background:#1B4F72;color:white;padding:12px 20px;border-radius:8px;margin-bottom:12px'>
    <h3 style='margin:0'>🎯 Target: <span style='color:#FCD434'>{record['target']}</span></h3>
    <small>Subcategory: {current_sample['target_subcat']} &nbsp; | &nbsp; Sample ID: {current_sample['sample_id']}</small>
</div>
""", unsafe_allow_html=True)

# Two-column layout: Graph + Interpretation
col_graph, col_interp = st.columns([1, 1])

with col_graph:
    st.subheader("📊 학습 레코드 그래프")
    fig = draw_record_graph(record)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("📋 이웃 노드 리스트 (카테고리별)"):
        for cat, neighbors in record['neighbors'].items():
            if not neighbors: continue
            color = CATEGORY_COLORS.get(cat, '#BDC3C7')
            st.markdown(f"<div style='background:{color}20;padding:6px;border-left:4px solid {color};margin:4px 0'><b>{cat}</b></div>", unsafe_allow_html=True)
            for n in neighbors:
                st.markdown(f"  • `{n['node']}` --[{n['relationship']}]--> {record['target']}  ({n['frequency']}건, {n['num_papers']}편)")

with col_interp:
    st.subheader("🤖 GPT-5-mini 해석")
    st.markdown(f"""
    <div style='background:#F4F6F7;padding:16px;border-radius:6px;border-left:4px solid #3498DB;max-height:600px;overflow-y:auto'>
        <pre style='white-space:pre-wrap;font-family:inherit;font-size:14px;margin:0'>{current_sample['gpt_interpretation']}</pre>
    </div>
    """, unsafe_allow_html=True)


# ─── Review form ───
st.divider()
st.subheader("🧑‍🔬 당신의 리뷰")

# Pre-load existing review if any
existing = my_reviews[my_reviews['sample_id'] == current_sample['sample_id']]
existing_verdict = existing['verdict'].iloc[0] if not existing.empty else None
existing_feedback = existing['feedback'].iloc[0] if not existing.empty else ''

col_v, col_fb = st.columns([1, 3])
with col_v:
    verdict = st.radio(
        "해석 적절성",
        options=['appropriate', 'inappropriate'],
        index=0 if existing_verdict == 'appropriate' else (1 if existing_verdict == 'inappropriate' else None),
        format_func=lambda x: '✅ 적절' if x == 'appropriate' else '❌ 부적절',
        key=f"verdict_{current_sample['sample_id']}",
    )

with col_fb:
    feedback = st.text_area(
        "피드백 (선택, 부적절한 경우 이유 작성)",
        value=existing_feedback,
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
