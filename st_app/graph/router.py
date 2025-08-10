from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes.chat_node import chat_node
from .nodes.rag_review_node import rag_review_node
from .nodes.subject_info_node import subject_info_node
from ..utils.state import ConversationState


def route_decision(state):
    """chat_node에서 결정한 다음 노드로 라우팅"""
    return state.get("next_node", END)


def build_graph() -> StateGraph:
    """
    LangGraph 기반 멀티 노드 라우팅 시스템
    
    구조도:
    ┌─────────────────┐
    │   사용자 입력    │
    │ {"input": "..."} │
    └─────────┬───────┘
              │
    ┌─────────▼───────┐
    │   chat_node     │ ← 진입점 & 라우터
    │                 │
    │ _decide_route() │
    │   ├─ "chat"     │ → 내부 채팅 처리 → END
    │   ├─ "subject"  │ → subject_info_node
    │   └─ "rag"      │ → rag_review_node  
    └─────────┬───────┘
              │
        ┌─────┴─────┐
        │     │     │
    ┌───▼───┐ │ ┌───▼───┐
    │subject│ │ │  rag  │
    │_info  │ │ │review │
    │_node  │ │ │_node  │
    └───┬───┘ │ └───┬───┘
        │     │     │
        └─────┼─────┘
              │
          ┌───▼───┐
          │  END  │
          └───────┘
    
    특징:
    - chat_node: 라우팅 결정 + 일반 채팅 처리
    - 모든 경로가 END로 수렴
    - 히스토리 자동 관리: update_state() 함수 사용
    """
    graph = StateGraph(ConversationState)

    # chat 노드만 추가 (내부에서 라우팅 처리)
    graph.add_node("chat", chat_node)
    graph.add_node("subject_info", subject_info_node)
    graph.add_node("rag_review", rag_review_node)

    # chat을 진입점으로 설정
    graph.set_entry_point("chat")
    graph.add_conditional_edges(
        "chat",
        route_decision,
        {
            "subject_info": "subject_info",
            "rag_review": "rag_review",
            "END": END,
        },
    )

    graph.add_edge("subject_info", END)
    graph.add_edge("rag_review", END)

    return graph


_GRAPH = None


def get_or_create_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph().compile()
    return _GRAPH
