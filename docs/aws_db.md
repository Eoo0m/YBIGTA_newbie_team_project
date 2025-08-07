

# 비공개 설정 & 로드밸런서 구성

[클라이언트] → 80포트 → [AWS ALB] → 8000포트 → [EC2 FastAPI]


## 1. RDS 비공개 설정 (퍼블릭 액세스 차단)

### 구성 개요
- RDS 생성 시 `Public Access = No`로 설정하여 외부 인터넷에서 직접 접속 불가
- EC2와 RDS를 **동일한 VPC**에 배치해 내부 IP로만 통신
- 보안 그룹에서 **MySQL(3306)** 포트를 EC2 보안 그룹만 허용

### 설정 효과
- DB 엔드포인트가 외부에 노출되지 않아 보안 강화
- 오직 EC2를 거쳐서만 RDS 접근 가능

### 설정

<img width="600" alt="Screenshot 2025-08-06 at 10 27 32 PM" src="https://github.com/user-attachments/assets/06b4ce95-279e-4a8e-8f35-bf57850a88ac" />


---

## 2. 로드밸런서를 통한 포트 비공개 설정

### 구성 개요
- 외부 사용자는 HTTP 80 포트(로드밸런서)로만 접근 가능
- 로드밸런서가 Target Group을 통해 내부적으로 EC2의 8000 포트로 요청 전달
- EC2 8000 포트는 로드밸런서 보안 그룹만 접근 가능
- Target Group Health Check Path를 `/health`로 설정하여 정상 상태 확인

### 헬스 체크 엔드포인트 코드
`/app/app/main.py`
```python
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}
```



### 경로 /health로 설정하여 로드밸런서가 서버가 살아 있는지 주기적으로 확인할 때 http://<서버주소>/health 경로로 요청을 보내도록 설정
<img width="600" alt="image" src="https://github.com/user-attachments/assets/3791e1d7-ddf0-4296-b9a4-d9b590653afb" />


### 리스너 설정
<img width="600" alt="image" src="https://github.com/user-attachments/assets/d208ac42-ff4f-4e5a-9b57-db6191f79eab" />


### 대상 설정
<img width="600" alt="image" src="https://github.com/user-attachments/assets/5c4794ef-64f9-4652-9a74-5cfbb6efeacc" />


### 로드밸런서 통해 문서 접속
<img width="600" alt="image" src="https://github.com/user-attachments/assets/076e1b19-e152-4d51-9ac6-72b6f28985ca" /> 



