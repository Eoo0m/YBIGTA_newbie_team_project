from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from st_app.rag.prompt import build_subject_info_context, build_subject_info_prompt, build_chat_prompt


def subject_info_node(state: Dict) -> Dict:
    print("=== SUBJECT_INFO_NODE DEBUG ===")

    query: str = state["input"]
    print(f"입력 쿼리: '{query}' (소문자: '{query.lower()}')")

    db = json.loads(
        Path("st_app/db/subject_information/subjects.json").read_text(encoding="utf-8")
    )
    print(f"DB에서 로드된 아이템 수: {len(db)}")

    # 매칭되는 주제 찾기
    query_lower = query.lower()
    for item in db:
        print(
            f"매칭 체크: '{item['name'].lower()}' in '{query_lower}' or '{item['id'].lower()}' in '{query_lower}'"
        )
        if item["name"].lower() in query_lower or item["id"].lower() in query_lower:
            print(f"매칭된 아이템: {item}")

            # 컨텍스트 정보 구성
            context = build_subject_info_context(item)

            # 프롬프트 구성하여 state에 저장
            prepared_prompt = build_subject_info_prompt(context, query, state)
            state["prepared_prompt"] = prepared_prompt
            
            print("=== SUBJECT_INFO 프롬프트 생성 완료 ===")
            print(f"프롬프트 길이: {len(prepared_prompt)}")
            print("=== DEBUG END ===\n")

            # Chat Node로 복귀
            state["next_node"] = "chat"
            return state
    
    print("매칭되는 주제를 찾지 못함")
    
    # 매칭 실패 시 일반 채팅 프롬프트로 대체
    fallback_prompt = build_chat_prompt(query + " (요청하신 주제 정보를 찾지 못했습니다)", state)
    state["prepared_prompt"] = fallback_prompt
    
    print("=== 주제 정보 못 찾음, 일반 채팅 프롬프트 생성 ===")
    print("=== DEBUG END ===\n")

    # Chat Node로 복귀
    state["next_node"] = "chat"
    return state
