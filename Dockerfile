FROM python:3

WORKDIR /app

# 시스템 패키지 설치
ENV JAVA_HOME=/usr/lib/jvm/java-1.7-openjdk/jre
RUN apt-get update && apt-get install -y g++ default-jdk

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY . .

# 작업 디렉토리 설정
WORKDIR /app

EXPOSE 8000

# 전처리 스크립트 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]