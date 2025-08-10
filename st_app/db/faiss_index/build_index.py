from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

import faiss
import numpy as np
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 상위 디렉토리를 path에 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from st_app.rag.embedder import encode_texts


def build_faiss_index():
    """FAISS 인덱스를 빌드합니다."""
    current_dir = Path(__file__).parent
    meta_file = current_dir / "meta.json"

    # 메타데이터 로드
    with open(meta_file, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    print(f"총 {len(reviews)}개의 리뷰 데이터를 로드했습니다.")

    # 리뷰 텍스트 추출
    texts = [review["review"] for review in reviews]

    # 임베딩 생성
    print("임베딩을 생성하고 있습니다...")
    embeddings = encode_texts(texts)
    embeddings_np = np.array(embeddings, dtype=np.float32)

    # FAISS 인덱스 생성
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    # 인덱스 저장
    index_file = current_dir / "index.faiss"
    faiss.write_index(index, str(index_file))

    print(f"FAISS 인덱스가 생성되었습니다: {index_file}")
    print(f"임베딩 차원: {dimension}")
    print(f"인덱스된 문서 수: {index.ntotal}")


if __name__ == "__main__":
    build_faiss_index()
