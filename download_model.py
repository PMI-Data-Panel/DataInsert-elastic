from sentence_transformers import SentenceTransformer

model_name = 'nlpai-lab/KURE-v1'
local_model_path = './kure-v1-model' # 모델이 저장될 폴더 이름

print(f"'{model_name}' 모델을 '{local_model_path}' 폴더에 다운로드합니다...")
# 모델을 다운로드하고 지정된 폴더에 저장합니다.
model = SentenceTransformer(model_name)
model.save(local_model_path)
print("✅ 모델 다운로드 및 저장 완료!")