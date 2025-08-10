from __future__ import annotations

from typing import Dict

SYSTEM_RAG = "당신은 영화 리뷰 요약/참조를 도와주는 조수입니다. 주어진 컨텍스트만을 근거로 한국어로 간결히 답하세요."


def build_rag_prompt(query: str, contexts: list[str], state: Dict = None) -> str:
    joined = "\n- " + "\n- ".join(contexts) if contexts else ""

    # state에서 대화 히스토리 추출
    history_text = ""
    if state and "history" in state:
        history = state["history"][-6:]  # 최근 6개 메시지만 (3턴)
        if history:
            history_items = []
            for msg in history:
                role = "사용자" if msg["role"] == "user" else "AI"
                history_items.append(f"{role}: {msg['content']}")
            history_text = f"\n[대화 히스토리]\n" + "\n".join(history_items) + "\n"

    return f"""
당신은 영화 리뷰 요약/참조를 도와주는 조수입니다. 주어진 컨텍스트만을 근거로 답변해주세요.
{history_text}
사용자 질문: {query}

[컨텍스트]{joined}

답변 가이드라인:
1. 컨텍스트만을 근거로 답변
2. 대화 히스토리를 참고하여 맥락을 이해
3. 필요 시 참고 문장을 인용
4. 간결하고 정확한 정보 제공
5. 한국어로 답변
"""


def build_subject_info_context(item: Dict) -> str:
    """주제 정보를 기반으로 컨텍스트를 구성합니다."""
    return f"""
주제 정보:
- 이름: {item['name']}
- 타입: {item['type']}
- 요약: {item.get('summary', '정보 없음')}
- 상세 정보: {', '.join(f'{k}: {v}' for k, v in item.get('spec', {}).items())}
"""


def build_subject_info_prompt(context: str, user_input: str, state: Dict) -> str:
    """주제 정보 노드용 프롬프트를 구성합니다."""

    # state에서 대화 히스토리 추출
    history_text = ""
    if state and "history" in state:
        history = state["history"][-6:]  # 최근 6개 메시지만 (3턴)
        if history:
            history_items = []
            for msg in history:
                role = "사용자" if msg["role"] == "user" else "AI"
                history_items.append(f"{role}: {msg['content']}")
            history_text = f"\n[대화 히스토리]\n" + "\n".join(history_items) + "\n"

    return f"""
다음 주제 정보를 바탕으로 사용자 질문에 자연스럽게 답변해주세요.

{context}
{history_text}
사용자 질문: {user_input}

답변 가이드라인:
1. 친근하고 자연스러운 톤으로 답변
2. 질문에 적절한 수준의 정보 제공
3. 대화 히스토리를 참고하여 맥락을 이해
4. 마크다운 포맷 사용 가능
5. 한국어로 답변
"""


def build_chat_prompt(user_input: str, state: Dict) -> str:
    """기본 채팅 노드용 프롬프트를 구성합니다."""

    # state에서 대화 히스토리 추출
    history_text = ""
    if state and "history" in state:
        history = state["history"][-10:]  # 최근 10개 메시지만 (5턴)
        if history:
            history_items = []
            for msg in history:
                role = "사용자" if msg["role"] == "user" else "AI"
                history_items.append(f"{role}: {msg['content']}")
            history_text = f"\n[대화 히스토리]\n" + "\n".join(history_items) + "\n"

    return f"""
당신은 도움이 되는 AI 어시스턴트입니다. 사용자와 자연스럽고 친근한 대화를 나누며 질문에 답변해주세요.
{history_text}
사용자 질문: {user_input}

답변 가이드라인:
1. 친근하고 자연스러운 톤으로 답변
2. 대화 히스토리를 참고하여 맥락을 이해
3. 도움이 되는 정보를 제공
4. 마크다운 포맷 사용 가능
5. 한국어로 답변
"""
