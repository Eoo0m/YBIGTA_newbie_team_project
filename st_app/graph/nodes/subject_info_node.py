from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from st_app.rag.llm import get_llm
from st_app.rag.prompt import build_subject_info_context, build_subject_info_prompt
from st_app.utils.state import get_history_messages, update_state


def subject_info_node(state: Dict) -> Dict:
    print("=== SUBJECT_INFO_NODE DEBUG ===")

    query: str = state["input"].lower()
    print(f"입력 쿼리: '{state['input']}' (소문자: '{query}')")

    db = json.loads(
        Path("st_app/db/subject_information/subjects.json").read_text(encoding="utf-8")
    )
    print(f"DB에서 로드된 아이템 수: {len(db)}")

    # 매칭되는 주제 찾기
    for item in db:
        print(
            f"매칭 체크: '{item['name'].lower()}' in '{query}' or '{item['id'].lower()}' in '{query}'"
        )
        if item["name"].lower() in query or item["id"].lower() in query:
            # 주제 정보를 state에 저장하여 다른 노드에서 활용할 수 있도록 함
            subject_info = {
                "id": item["id"],
                "name": item["name"],
                "type": item["type"],
                "spec": item.get("spec", {}),
                "summary": item.get("summary", ""),
            }

            print(f"매칭된 아이템: {item}")

            # LLM을 사용하여 자연스러운 답변 생성
            llm = get_llm()

            # 컨텍스트 정보 구성
            context = build_subject_info_context(item)

            # 프롬프트 구성
            prompt = build_subject_info_prompt(context, query, state)

            print("=== SUBJECT_INFO 프롬프트 ===")
            print(prompt)
            print("=== SUBJECT_INFO 프롬프트 끝 ===")

            response = llm.invoke([{"role": "user", "content": prompt}]).content
            print(f"LLM 응답: {response}")
            print("=== DEBUG END ===\n")

            return update_state(state, response)
    print("매칭되는 주제를 찾지 못함")
    print("=== DEBUG END ===\n")
    output = "해당 주제에 대한 정보를 찾지 못했습니다."

    return update_state(state, output)
