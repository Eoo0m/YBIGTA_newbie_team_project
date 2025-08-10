from __future__ import annotations

from typing import List, Dict, TypedDict


class Message(TypedDict):
    role: str  # "user" or "assistant"
    content: str


class ConversationState(TypedDict, total=False):
    input: str
    output: str
    next_node: str  # 라우터에서 결정된 다음 노드
    history: List[Message]  # 대화 히스토리
    prepared_prompt: str  # 다른 노드에서 준비한 프롬프트


def add_to_history(state: Dict, user_input: str, assistant_output: str) -> Dict:
    """히스토리에 새로운 대화 추가"""
    if "history" not in state:
        state["history"] = []
    
    state["history"].append({"role": "user", "content": user_input})
    state["history"].append({"role": "assistant", "content": assistant_output})
    
    return state


def get_history_messages(state: Dict, max_messages: int = 10) -> List[Message]:
    """히스토리에서 최근 메시지들을 가져와서 LLM에 전달할 형태로 반환"""
    history = state.get("history", [])
    # 최근 max_messages개의 메시지만 반환
    return history[-max_messages:] if history else []


def update_state(state: Dict, output: str, next_node: str = "END") -> Dict:
    """state의 네 가지 요소를 모두 업데이트
    
    Args:
        state: 현재 state (input 포함)
        output: LLM 응답
        next_node: 다음 노드 ("END", "subject_info", "rag_review" 등)
    
    Returns:
        업데이트된 state (input, output, history, next_node 모두 포함)
    """
    # 1. input은 이미 state에 있음 (그대로 유지)
    user_input = state.get("input", "")
    
    # 2. output 설정
    state["output"] = output
    
    # 3. history에 현재 대화 추가
    if user_input and output:
        if "history" not in state:
            state["history"] = []
        state["history"].append({"role": "user", "content": user_input})
        state["history"].append({"role": "assistant", "content": output})
    
    # 4. next_node 설정 (라우팅용)
    state["next_node"] = next_node
    
    return state


