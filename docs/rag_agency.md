# RAG Agency

## 개요

이 프로젝트는 영화 리뷰 기반 RAG(Retrieval-Augmented Generation) 시스템을 구현하며, 대화 맥락을 유지하는 메모리 관리 기능을 포함합니다.

## 핵심 설계 원칙

### 1. State 기반 메모리 관리

기존의 대화 내역을 직접 매개변수로 넘기는 방식 대신, **State 객체**를 통한 통합적인 상태 관리를 구현했습니다.

```python
class ConversationState(TypedDict, total=False):
    input: str
    output: str
    next_node: str
    history: List[Message]  # 대화 히스토리
    prepared_prompt: str          # 다른 노드에서 준비한 프롬프트
```

**장점:**
- 후에 툴 콜이나 다양한 기능 추가 시에도 수월함
- 툴, 챗 등 여러 가지를 개별적으로 넘기는 것은 복잡하지만, State 클래스에 저장해두면 접근 시 편리함
- 일관된 인터페이스로 모든 노드에서 동일한 방식으로 상태 접근 가능

### 2. 모듈화

질문, 대화내역, 출력 등을 State에 업데이트하는 로직을 함수로 분리했습니다.

```python
def update_state(state: Dict, output: str, next_node: str = "END") -> Dict:
    """state의 세 가지 요소를 업데이트"""
    # 1. input은 이미 state에 있음 (그대로 유지)
    # 2. output 설정
    # 3. history에 현재 대화 추가  
    # 4. next_node 설정
```

**장점:**
- 코드 재사용성 향상
- 상태 업데이트 로직의 일관성 보장
- 유지보수 편의성

### 3. 프롬프트에 기존 대화를 넣는 방식

각 노드의 프롬프트 함수에서 State로부터 대화 히스토리를 추출하여 맥락 정보를 포함합니다.

```python
def build_rag_prompt(query: str, contexts: list[str], state: Dict = None) -> str:
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
    [지시] 다음 사용자 질문에 대해 컨텍스트만 근거로 답하세요. 대화 히스토리를 참고하여 맥락을 이해하고 필요 시 참고 문장을 인용하세요.
    {history_text}
    [질문] {query}
    [컨텍스트]{joined}
    """
```

**예시:**

```python
당신은 영화 리뷰 요약/참조를 도와주는 조수입니다. 주어진 컨텍스트만을 근거로 답변해주세요.

[대화 히스토리]
사용자: 안녕하세요
AI: 안녕하세요! 저는 업스테이지의 AI 챗봇 솔라입니다. 어떤 도움이 필요하신가요?

사용자 질문: 기생충 리뷰는 어때

[컨텍스트]
- 금방 보고 나왔는데...봉준호감독 영화인데 왜 박찬욱감독 영화를 본 느낌이지.어쨌든 최고다. 봉드로 역시 천재!!이정은님 연기상 줘야한다 진짜
- 시나리오, 연출, 배우 모두 좋았다.소름끼치게...
- 말이 필요없는 영화 모두 꼭 한번쯤은 봐야할법한 영화입니다. 그리고 조여정이 개인적으로 크게 한껀 한것같습니다
- 영화가 끝나고 돌아오는 지하철을 타고..

답변 가이드라인:
1. 컨텍스트만을 근거로 답변
2. 대화 히스토리를 참고하여 맥락을 이해
3. 필요 시 참고 문장을 인용
4. 간결하고 정확한 정보 제공
5. 한국어로 답변
```


**특징:**
- 최근 3턴의 대화만 포함하여 토큰 효율성 확보
- 모든 노드(RAG, Subject Info, Chat)에서 일관된 방식으로 대화 맥락 활용
- 프롬프트 내에서 명시적으로 대화 히스토리 섹션을 구분

## 아키텍처

### LangGraph 기반 노드 순회 구조

```
           START
             │
             ▼
      ┌─────────────┐
      │ Chat Node   │ ◄─── 라우터 + 최종 응답 생성
      │             │
      │ 1. 라우팅     │       ┌─────────────┐
      │ 2. 최종응답   │ ◄──── ┤Subject Info │
      └──────┬──────┘       │    Node     │
             │              │             │
        ┌────┼────┐         │컨텍스트수집    │
        │    │    │         │프롬프트생성    │
    "chat"  "rag" "subject" └─────────────┘
        │    │    │              │
        ▼    ▼    │              │
      ┌───┐ ┌──────────┐         │
      │직접│ │RAG Review│         │
      │응답│ │   Node   │         │
      └─┬─┘ │          │         │
        │   │컨텍스트수집 │         │
        │   │프롬프트생성 │         │
        │   └────┬─────┘         │
        │        │               │
        │        └───────────────┘
        │                │
        └────────────────┼
                         ▼
                        END
```




### 메모리 흐름
1. 사용자 입력 → State에 저장
2. Chat Node에서 라우팅 결정 → 전용 노드로 이동
3. 전용 노드에서 컨텍스트 수집 및 프롬프트 생성 → `prepared_prompt`에 저장
4. Chat Node 복귀 → 준비된 프롬프트로 최종 응답 생성
5. LLM 응답 → State 업데이트 (출력 및 히스토리에 추가)
5. 다음 대화에서 누적된 히스토리 활용
