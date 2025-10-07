# Elasticsearch 기반 설문 데이터 분석 및 검색 프로젝트

본 프로젝트는 FastAPI를 사용하여 설문조사 응답 데이터를 Elasticsearch에 색인하고, AI 임베딩 모델(`nlpai-lab/KURE-v1`)을 Elasticsearch에 직접 내장하여 의미 기반 검색 및 분석을 수행하는 API 서버입니다.

## 🏛️ 아키텍처
* **Backend**: FastAPI, Python 3.11
* **Database**: Elasticsearch 8.x
* **Infrastructure**: Docker, Docker Compose
* **AI Model Deployment**: Eland(추가예정)
* **AI Model**: `nlpai-lab/KURE-v1` (via `sentence-transformers`)
* **Monitoring/UI**: Kibana 8.x

---

## 🏁 시작하기

### 사전 준비 사항

로컬 개발 환경에 아래 프로그램들이 설치되어 있어야 합니다.

* [Git](https://git-scm.com/)
* [Python 3.11](https://www.python.org/downloads/release/python-3110/) (라이브러리 호환성을 위해 **3.11 버전 권장**)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)

---

### ⚙️ 로컬 환경 설정 (Installation)

아래 순서대로 따라 하면 프로젝트를 로컬 환경에서 실행할 수 있습니다.

**1. 프로젝트 클론**
```bash
git clone [저장소_URL]
cd [프로젝트_폴더명]
```

**3. 데이터베이스 실행 (Docker-compose)**

Dockerfile을 기반으로 Elasticsearch 이미지를 빌드하고   
Kibana 이미지도 docker-compose 명령어를 통해  
Elasticsearch & Kibana 두개 이미지를 하나의 컨테이너로 백그라운드에서 실행합니다.  
```
docker-compose up -d --build
```

**4. 파이썬 가상환경 설정 및 패키지 설치**

## 1. 가상환경 생성(루트파일에서)
```
python -m venv venv
```
## 2. 가상환경 활성화
### Windows
```
.\venv\Scripts\activate
```
### macOS / Linux
```
source venv/bin/activate
```
## 3. requirements.txt에 명시된 모든 패키지 설치
```
pip install -r requirements.txt
```

## ▶️ 애플리케이션 실행

**1. FastAPI 서버 시작**
```
uvicorn app.main:app --reload
```
**2. API 문서 확인**
서버가 실행되면, 웹 브라우저에서 아래 주소로 접속하여 자동 생성된 API 문서를 확인할 수 있습니다.

Swagger UI: http://127.0.0.1:8000/docs

🛠️ 초기 데이터 및 임베딩 생성

**1. CSV 데이터 Elasticsearch에 삽입**
data 폴더의 CSV 파일들을 읽어 Elasticsearch 데이터베이스에 삽입합니다.
<img width="2822" height="902" alt="image" src="https://github.com/user-attachments/assets/4a339968-3708-4b78-a3a4-8bc1ec15622b" />
excute를 실행시켜주세요.

