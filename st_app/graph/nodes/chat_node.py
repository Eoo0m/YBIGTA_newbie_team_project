from __future__ import annotations

from typing import Dict

from ...rag.llm import get_llm
from ...rag.prompt import build_chat_prompt
from ...utils.state import get_history_messages, update_state


SYSTEM_DECIDER = (
    "당신은 라우팅 분류기입니다. 사용자의 입력을 분석하여 어떤 노드로 라우팅할지 결정하세요:\n\n"
    "- chat: 일반적인 대화, 인사, 잡담 등 특별한 정보가 필요하지 않은 경우\n"
    "- subject_info: 영화나 제품의 기본 정보, 스펙, 감독, 배우 등에 대한 문의\n"
    "- rag_review: 리뷰 내용, 평점, 관객 반응, 리뷰 요약/분석에 대한 문의\n\n"
    "반드시 다음 중 하나만 출력하세요: chat, subject_info, rag_review\n"
    "다른 텍스트나 설명은 포함하지 마세요."
)


def _decide_route(user_text: str) -> str:
    """LLM을 사용한 지능형 라우팅 결정"""
    try:
        llm = get_llm(temperature=0.0)
        msgs = [
            {"role": "system", "content": SYSTEM_DECIDER},
            {"role": "user", "content": f"사용자 입력: {user_text}"},
        ]
        print("라우팅 LLM 호출 중...")
        response = llm.invoke(msgs).content.strip().lower()
        print(f"LLM 라우팅 응답: '{response}'")

        # 유효한 라우트인지 확인
        if response in {"chat", "subject_info", "rag_review"}:
            print(f"유효한 라우트 반환: {response}")
            return response
        else:
            print(f"유효하지 않은 라우트: '{response}', 기본값 사용")

    except Exception as e:
        print(f"라우팅 결정 중 오류: {e}")

    # 폴백: 기본값 반환
    print("키워드 기반 라우팅으로 폴백")
    print("기본값 'chat' 반환")
    return "chat"


def chat_node(state: Dict) -> Dict:

    user_input = state["input"]
    print(f"사용자 입력: '{user_input}'")

    # 라우팅 결정

    route = _decide_route(user_input)

    # 다른 노드로 라우팅
    if route == "subject_info":
        state["next_node"] = "subject_info"
        return {"next_node": "subject_info"}
    elif route == "rag_review":
        state["next_node"] = "rag_review"
        return {"next_node": "rag_review"}

    # 일반 채팅 처리
    print("일반 채팅으로 처리")
    print("=== CHAT_NODE DEBUG ===")
    llm = get_llm()

    # 채팅 프롬프트 생성
    prompt = build_chat_prompt(user_input, state)

    print("=== CHAT 프롬프트 ===")
    print(prompt)
    print("=== CHAT 프롬프트 끝 ===")

    print("LLM 호출 중...")
    resp = llm.invoke([{"role": "user", "content": prompt}])
    output = resp.content
    print(f"LLM 응답: {output}")
    print("=== CHAT DEBUG END ===\n")

    return update_state(state, output, "END")
