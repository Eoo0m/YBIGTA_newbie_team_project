import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    st.set_page_config(page_title="RAG Agent Demo", page_icon="🤖", layout="wide")
    st.title("RAG Agent Demo (LangGraph · Streamlit)")
    st.caption(
        "항상 Chat Node를 시작점으로 실행하여, 필요 시 내부에서 Subject Info나 RAG Review로 라우팅 후 다시 Chat Node로 복귀"
    )

    st.sidebar.header("환경 설정")
    st.sidebar.info(
        "Upstage/OpenAI 등 LLM 키는 환경변수로 설정해주세요. 예: OPENAI_API_KEY"
    )

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("graph_state", {"history": []})

    # 그래프 불러오기 (항상 Chat Node가 START)
    from st_app.graph.router import get_or_create_graph

    graph = get_or_create_graph()
    state = st.session_state["graph_state"]

    user_input = st.chat_input(
        "메시지를 입력하세요… 예: 리뷰 내용 알려줘, 영화 정보 알려줘"
    )
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        state["input"] = user_input

        # 항상 chat_node부터 실행
        # graph.stream()이 아니라 graph.invoke()로 한 턴 실행
        # Chat Node 내부에서 라우팅 → 복귀 후 output 반환
        state = graph.invoke(state)
        st.session_state["graph_state"] = state  # 업데이트된 state 저장
        output = state.get("output", "")

        if output:
            st.session_state["messages"].append(
                {"role": "assistant", "content": output}
            )

    # 대화 렌더링
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


if __name__ == "__main__":
    main()
