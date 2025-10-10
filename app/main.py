import pandas as pd
from fastapi import FastAPI, HTTPException
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import datetime
from sentence_transformers import SentenceTransformer
import os
import re
import traceback
import json

# --- 모델 및 클라이언트 초기화 ---
embedding_model = SentenceTransformer("nlpai-lab/KURE-v1")
VECTOR_DIMENSIONS = 1024

# --- FastAPI 및 Elasticsearch 초기화 ---
app = FastAPI()
es = Elasticsearch("http://localhost:9200")


def create_index_if_not_exists(index_name: str):
    """
    Elasticsearch에 특정 인덱스가 존재하지 않으면, 새 인덱스를 생성.
    """
    
    #해당 이름의 인덱스가 이미 존재하는지 확인
    if not es.indices.exists(index=index_name):
        print(f"✨ '{index_name}' 인덱스가 없어 새로 생성합니다.")

        mappings = {
            "properties": {
                "user_id": {"type": "keyword"}, # 사용자 ID
                "timestamp": {"type": "date"},
                "qa_pairs": { 
                    "type": "nested", # 질문-응답 쌍들을 독립적인 객체 배열로 처리하기 위한 타입
                    "properties": {
                        "q_code": {"type": "keyword"}, # 질문 코드 (예: 'Q1', 'Q5_1')
                        "q_text": {"type": "text", "analyzer": "nori"}, # 질문 텍스트
                        "q_type": {"type": "keyword"},  # 질문 유형 (예: 'SINGLE', 'MULTI')
                        "answer_text": {"type": "text", "analyzer": "nori"}, # 답변 텍스트
                        "embedding_text": {
                            "type": "text",
                            "analyzer": "nori",
                            "index": False, # 임베딩 생성에만 사용, 검색 대상에서는 제외하여 저장 공간 절약
                        },
                        "answer_vector": {
                            "type": "dense_vector", # 벡터 데이터를 저장하기 위한 타입
                            "dims": VECTOR_DIMENSIONS,
                        },
                    },
                },
            }
        }
        try:
            # 정의된 매핑으로 새로운 인덱스를 생성
            es.indices.create(index=index_name, mappings=mappings)
            print(f"👍 '{index_name}' 인덱스 생성 완료")
        except Exception as e:
            # 인덱스 생성 중 오류가 발생하면 500 서버 오류를 발생
            raise HTTPException(
                status_code=500, detail=f"'{index_name}' 인덱스 생성 실패: {e}"
            )
            
            
# --- 4. 질문 메타데이터 파싱 함수 ---
def parse_question_metadata(file_path: str) -> dict:
    """
    csv 파일을 읽어, 각 질문 코드에 대한 정보(질문 내용, 유형, 선택지)를
    딕셔너리 형태로 가공하여 반환.
    """
    metadata = {} # 모든 질문 정보를 담을 빈 딕셔너리
    current_q_code = None # 현재 처리 중인 질문 코드를 추적하기 위한 변수
    with open(file_path, "r", encoding="utf-8-sig") as f:
        for line in f:  
            line = line.strip() # 각 줄의 앞뒤 공백 제거
            if not line:  # 빈 줄이면 건너뛰기.
                continue
            
            # 정규표현식을 사용하여 "코드,텍스트,타입" 형식의 새로운 질문 행
            match = re.match(r"^([a-zA-Z0-9_]+),([^,]+),([^,]+)$", line)
            if match:
                
                # 매칭에 성공하면, 그룹으로 나눠 각 변수에 할당합니다.
                q_code, q_text, q_type = match.groups()
                current_q_code = q_code.strip() # 현재 질문 코드를 업데이트합니다.
                # 메타데이터 딕셔너리에 현재 질문에 대한 정보를 저장할 새로운 항목을 만듭니다.
                metadata[current_q_code] = {
                    "text": q_text.strip(),
                    "type": q_type.strip(),
                    "options": {}, # 이 질문에 대한 선택지들을 저장할 빈 딕셔너리
                }
            # 숫자로 시작하고 콤마가 있는 행은 이전 질문에 대한 선택지(옵션) 행으로 간주
            elif current_q_code and re.match(r"^\d+,", line):
                parts = line.split(",", 2)  # 콤마를 기준으로 최대 2번만 분리
                option_code = parts[0].strip()
                option_text = parts[1].strip()
                if option_code: # 선택지 코드가 비어있지 않다면
                    # 현재 처리 중인 질문의 'options' 딕셔너리에 선택지 정보를 추가
                    metadata[current_q_code]["options"][option_code] = option_text
    return metadata

# --- 5. FastAPI 라우트(API 엔드포인트) 정의 ---
@app.get("/")
def read_root():
    return {"message": "FAST API서버 실행 중"}


@app.post("/index-survey-data") 
def index_survey_data_by_user():
    """
    설문 데이터 전체를 색인하는 메인 로직 실행.
    """
    question_file = "./data/question_list.csv" # 질문 목록 파일 경로
    response_file = "./data/response_list_300.csv" # 응답 데이터 파일 경로
    index_name = "survey_responses" # 사용할 인덱스 이름

    try:
        # Elasticsearch 서버가 활성 상태인지 확인
        if not es.ping():
            raise HTTPException(
                status_code=503, detail="Elasticsearch 서버에 연결할 수 없습니다."
            )

        print("\n--- 🚀 데이터 색인 작업 시작")
        print(f"- 대상 인덱스: {index_name}\n")
        
        # 만약 인덱스가 이미 존재한다면 삭제
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
            print(f"🗑️  기존 '{index_name}' 인덱스를 삭제했습니다.")
        
        # 인덱스 생성 함수를 호출하여 인덱스 생성
        create_index_if_not_exists(index_name)

        # 질문 메타데이터 파일을 파싱하여 딕셔너리로
        questions_meta = parse_question_metadata(question_file)
        print("✅ 질문 메타데이터 파싱 완료.")

        df_responses = pd.read_csv(response_file, encoding="utf-8-sig")
        
        # pandas의 결측값(NaN)을 파이썬의 None 객체로 변환하여 일관성을 유지
        df_responses = df_responses.astype(object).where(pd.notnull(df_responses), None)
        print(f"✅ 응답 데이터 로드 완료. (총 {len(df_responses)}명)")

        actions = [] # Elasticsearch bulk API에 사용할 작업들을 담을 리스트
        user_count = 0 # 처리 중인 사용자 수를 세기 위한 변수
        total_users = len(df_responses) # 전체 사용자 수

        # 데이터프레임의 한 행(row)씩, 즉 사용자 한 명씩 순회
        for _, row in df_responses.iterrows():
            user_count += 1
            
            # 10명 단위, 첫 번째, 마지막 사용자에 대해 진행 상황을 출력
            if user_count % 10 == 0 or user_count == 1 or user_count == total_users:
                print(f"🔄 사용자 데이터 처리 중... ({user_count}/{total_users})")

            user_id = row.get("mb_sn")
            if not user_id:
                continue

            # 해당 사용자의 모든 질문-응답 쌍을 담을 리스트
            all_qa_pairs_for_user = []

            # 사용자의 각 응답(컬럼)을 순회
            for q_code, raw_answer in row.items():

                # 'mb_sn' 컬럼이거나 응답 값이 없으면 처리x
                if q_code == "mb_sn" or raw_answer is None:
                    continue
                
                # 질문 메타데이터에서 현재 질문 코드(q_code)에 대한 정보 가져오기
                q_info = questions_meta.get(q_code)
                if not q_info:
                    continue
                
                # 질문 텍스트와 유형 가져오기
                q_text, q_type = q_info["text"], q_info["type"]
                # 최종 답변 텍스트들을 저장할 리스트
                answers_text_list = []

                # 질문 유형이 'MULTI'(다중 응답)인 경우
                if q_type == "MULTI":
                    # 응답 문자열을 콤마(,)를 기준으로 분리하여 코드 리스트 생성
                    answer_codes = str(raw_answer).split(",")
                    for code in answer_codes:
                        code = code.strip()
                        if code: # 코드가 비어있지 않다면
                            
                            # 메타데이터의 'options'에서 코드에 해당하는 텍스트를 찾아 추가
                            # 만약 해당하는 텍스트가 없으면 "없는 코드"라고 표시
                            answers_text_list.append(
                                q_info["options"].get(code, f"없는 코드: {code}")
                            )
                            
                # 질문 유형이 'SINGLE'(단일 응답)인 경우
                elif q_type == "SINGLE":
                    # 응답 코드에 해당하는 텍스트를 찾아 추가

                    # 만약 해당하는 텍스트가 없으면 원래 응답값(raw_answer)을 그대로 사용. (주관식 응답 등)
                    answers_text_list.append(
                        q_info["options"].get(str(raw_answer).strip(), raw_answer)
                    )

                # 그 외 유형(예: Numeric, String)은 응답값을 그대로 사용.
                else:
                    answers_text_list.append(str(raw_answer))

                # 각 답변 텍스트를 nested 객체로 변환
                for answer_text in answers_text_list:
                    # 답변 텍스트가 비어있으면 처리x.
                    if answer_text is None or str(answer_text).strip() == "":
                        continue
                    
                    
                    embedding_text = f"{q_text} 질문에 '{answer_text}'라고 답변"
                    vector = embedding_model.encode(embedding_text).tolist()

                    # Elasticsearch의 nested 필드에 저장될 하나의 질문-응답 객체를 생성
                    qa_pair_doc = {
                        "q_code": q_code,
                        "q_text": q_text,
                        "q_type": q_type,
                        "answer_text": answer_text,
                        "embedding_text": embedding_text,
                        "answer_vector": vector,
                    }
                    all_qa_pairs_for_user.append(qa_pair_doc)

            # 처리된 질문-응답 쌍이 있을 경우에만 최종 사용자 문서를 생성
            if all_qa_pairs_for_user:
                final_user_document = {
                    "user_id": user_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "qa_pairs": all_qa_pairs_for_user,
                }
                actions.append(
                    {"_index": index_name, "_id": user_id, "_source": final_user_document}
                )

        if not actions:
            print("⚠️  처리할 문서가 없습니다. 작업을 중단합니다.")
            return {"message": "처리할 데이터가 없습니다."}

        print(
            f"\n✅ 총 {len(actions)}개의 문서를 생성했습니다. (사용자 단위로 그룹화)"
        )
        print("--- 📄 첫 번째 사용자 문서 샘플 ---")
        print(json.dumps(actions[0]["_source"], indent=2, ensure_ascii=False))
        print("--------------------------------\n")

        print("⏳ Elasticsearch에 데이터 대량 삽입(bulk)을 시작합니다...")
        success, failed = bulk(es, actions, raise_on_error=False, refresh=True)

        print(f"🎉 작업 완료! 성공: {success}, 실패: {len(failed)}")

        return {
            "message": "데이터 색인 작업 완료.",
            "성공": success,
            "실패": len(failed),
        }

    except Exception as e:
        print("예상치 못한 오류 발생")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"처리 중 예상치 못한 오류가 발생했습니다: {str(e)}"
        )