from __future__ import annotations

from typing import Dict

from ...rag.prompt import build_rag_prompt
from ...rag.retriever import retrieve


def rag_review_node(state: Dict) -> Dict:
    print("=== RAG_REVIEW_NODE DEBUG ===")
    print(f"전체 state: {state}")

    query: str = state["input"]
    print(f"검색 쿼리: '{query}'")

    # 리뷰 검색
    print("리뷰 검색 시작...")
    retrieve_results = retrieve(query, k=4)
    print(f"검색 결과 수: {len(retrieve_results)}")

    contexts = [t for t, _ in retrieve_results]
    print("=== 검색된 리뷰 컨텍스트 ===")
    for i, context in enumerate(contexts):
        print(f"컨텍스트 {i+1}: {context[:100]}...")
    print("=== 컨텍스트 끝 ===")

    # 프롬프트 생성하여 state에 저장
    prepared_prompt = build_rag_prompt(query, contexts, state)
    state["prepared_prompt"] = prepared_prompt
    
    print("=== RAG 프롬프트 생성 완료 ===")
    print(f"프롬프트 길이: {len(prepared_prompt)}")
    print("=== RAG DEBUG END ===\n")

    # Chat Node로 복귀
    state["next_node"] = "chat"
    return state
