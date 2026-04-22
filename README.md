# mAb-GATED Review System

mAb 안정성 인과 지식 그래프 해석 리뷰 시스템 (Streamlit 기반).

## 구조

- **mab_review_samples** (DB 테이블): 300개 사전 생성된 리뷰 샘플 + GPT-5-mini 해석
- **mab_review_results** (DB 테이블): 리뷰어 판정 저장
- **app.py**: Streamlit UI (리뷰어가 사용)
- **generate_review_samples.py**: 300개 샘플 사전 생성 (OpenAI API 필요, 관리자만 실행)

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud 배포

1. 이 폴더를 GitHub 레포에 푸시
2. https://share.streamlit.io 에서 레포 연결
3. 배포 — 즉시 공개 URL 생성

## 리뷰어 사용법

1. 사이드바에 이름 입력
2. Subcategory / Target 노드 필터 선택 (선택)
3. 샘플을 보면서 GPT 해석의 적절성 판단
4. 적절 / 부적절 선택 + 피드백 작성
5. 저장 & 다음

## DB 테이블 스키마

### mab_review_samples
```sql
sample_id           BIGINT PRIMARY KEY
target_node         VARCHAR(500)
target_subcat       VARCHAR(50)    -- physical/chemical/biological
record_json         MEDIUMTEXT     -- {target, neighbors: {category: [...]}}
gpt_interpretation  MEDIUMTEXT     -- GPT-5-mini 생성 해석
created_at          TIMESTAMP
```

### mab_review_results
```sql
review_id    BIGINT PRIMARY KEY
sample_id    BIGINT    -- FK to mab_review_samples
reviewer     VARCHAR(100)
verdict      VARCHAR(20)   -- 'appropriate' / 'inappropriate'
feedback     TEXT
reviewed_at  TIMESTAMP
```
