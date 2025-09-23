import pandas as pd
from fastapi import FastAPI, HTTPException
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import datetime
from sentence_transformers import SentenceTransformer
import anthropic  # <-- Gemini 대신 Claude 라이브러리 임포트
import os
import time       # <-- API 속도 제어를 위한 라이브러리 임포트
from dotenv import load_dotenv  # <-- 1. 이 줄을 추가하세요

# --- 모델 및 클라이언트 초기화 ---
load_dotenv()  # <-- 2. 이 줄을 추가하여 .env 파일을 로드합니다.



# --- 모델 및 클라이언트 초기화 ---
# 컨테이너 내부에 연결된 로컬 경로에서 모델을 즉시 불러옵니다.
embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
# --- ▼▼▼ Gemini -> Claude 변경 부분 (1) ▼▼▼ ---
claude_api_key = os.getenv("ANTHROPIC_API_KEY")
if not claude_api_key:
    raise ValueError("ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.")

# Claude 클라이언트를 초기화합니다.
claude_client = anthropic.Anthropic(api_key=claude_api_key)
# --- ▲▲▲ 여기까지 변경 ▲▲▲ ---

app = FastAPI()
es = Elasticsearch("http://localhost:9200")

# (create_index_if_not_exists, read_root 함수는 변경 없음)
def create_index_if_not_exists(index_name: str):
    if not es.indices.exists(index=index_name):
        print(f"✨ '{index_name}' 인덱스가 없어 새로 생성합니다.")
        vector_dimensions = 768
        mappings = { "properties": { "response_id": {"type": "keyword"}, "respondent_id": {"type": "keyword"}, "survey_name": {"type": "keyword"}, "submitted_at": {"type": "date"}, "answers": {"type": "object", "dynamic": True}, "search_assistance": { "properties": { "activity_text": {"type": "text", "analyzer": "nori"}, "activity_vector": {"type": "dense_vector", "dims": vector_dimensions} } }, "respondent_info": {"type": "object"} } }
        try:
            es.indices.create(index=index_name, mappings=mappings)
            print(f"👍 '{index_name}' 인덱스 생성 완료.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"'{index_name}' 인덱스 생성 실패: {e}")

@app.get("/")
def read_root():
    return {"message": "FastAPI가 Elasticsearch와 함께 실행 중입니다!"}

@app.post("/index-survey-data")
def index_survey_data():
    csv_file_path = "./data/cleaned_user_data.csv"
    index_name = "survey_responses_final" # 새 인덱스 이름 사용

    try:
        # --- (필수) 기존 인덱스가 있다면 삭제하여 초기화 ---
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
            print(f"🗑️ 기존 '{index_name}' 인덱스를 삭제했습니다.")
        
        create_index_if_not_exists(index_name)
        df = pd.read_csv(csv_file_path, encoding='utf-8-sig')

        # --- ▼▼▼ 가장 중요한 수정 부분 ▼▼▼ ---
        # 1. DataFrame의 모든 열 이름에서 특수문자를 '_'로 변경하여 안전하게 만듭니다.
        df.columns = [
            col.replace('.', '_').replace('/', '_').replace('(', '').replace(')', '').replace(' ', '_')
            for col in df.columns
        ]
        # --- ▲▲▲ 여기까지 수정 ▲▲▲ ---

        df = df.astype(object).where(pd.notnull(df), None)

        actions = []
        for index, row in df.iterrows():
            try:
                # 이제 row.to_dict()는 이미 정제된 열 이름을 사용합니다.
                answers_text_parts = [ f"- {col_name}: {value}" for col_name, value in row.items() if value is not None and col_name != "mb_sn" ]
                raw_data_text = "\n".join(answers_text_parts)
                prompt = f"다음 설문조사 응답 데이터를 바탕으로, 이 사람의 특징을 설명하는 자연스러운 한두 문장의 서술형 문장을 생성해주세요.\n\n<data>\n{raw_data_text}\n</data>\n\n예시: '4인 가족의 기혼자로, 자녀가 2명 있으며 사무직에 종사하는 사용자'"

                # --- ▼▼▼ 모델 이름 수정 ▼▼▼ ---
                message = claude_client.messages.create(
                    model="claude-3-7-sonnet-latest", # 접근 가능한 Sonnet 모델로 변경
                    max_tokens=1024,
                    messages=[ {"role": "user", "content": prompt} ]
                ).content[0].text
                summary_text = message.strip()
                # --- ▲▲▲ 여기까지 수정 ▲▲▲ ---

                vector = embedding_model.encode(summary_text).tolist()
                
                doc = {
                    "response_id": f"resp_claude_{row.get('mb_sn')}_{index}",
                    "respondent_id": row.get('mb_sn'),
                    "survey_name": "claude_summary_survey_v1",
                    "submitted_at": datetime.datetime.now().isoformat(),
                    "answers": row.to_dict(), # 이제 안전한 키를 가진 딕셔너리가 됩니다.
                    "search_assistance": {"activity_text": summary_text, "activity_vector": vector},
                    "respondent_info": None
                }
                action = {"_index": index_name, "_id": doc["response_id"], "_source": doc}
                actions.append(action)

                time.sleep(1) # API Rate Limit 방지

            except Exception as e:
                print(f"행 {index} 처리 중 오류 발생: {e}")
                continue

        if not actions:
            return {"message": "처리할 데이터가 없거나 모든 행에서 오류가 발생했습니다."}
        
        success, failed = bulk(es, actions, raise_on_error=False)
        
        failed_reasons = []
        for item in failed:
            action_type = list(item.keys())[0]
            error_details = item[action_type].get('error', {})
            failed_reasons.append({ "document_id": item[action_type].get('_id'), "reason": error_details.get('reason', '알 수 없는 이유') })

        return { "message": "데이터 색인 작업이 완료되었습니다.", "success_count": success, "failed_count": len(failed), "failures": failed_reasons }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"'{csv_file_path}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"전체 프로세스 중 오류 발생: {str(e)}")
