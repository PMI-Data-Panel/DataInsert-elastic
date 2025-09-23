# 1. 엘라스틱서치 공식 이미지를 기반으로 시작합니다.
FROM docker.elastic.co/elasticsearch/elasticsearch:8.14.1

# 2. 'nori' 한국어 형태소 분석기 플러그인을 설치합니다.
RUN elasticsearch-plugin install analysis-nori
