# Running scripts for yamyam-lab

`scripts/` contains various scripts used when building recommender system in `yamyam-lab` repository.


| Python script                          | Description                                                        |
|----------------------------------------|--------------------------------------------------------------------|
| `scripts/create_google_drive_token.py` | Used when creating token.json in ci                                |
| `scripts/download_result.py`           | Used when downloading candidate generation or trained model result |
| `scripts/generate_candidate.py`        | Used when generating candidates from trained model                 |
| `scripts/build_regions.py`             | Used when generating region cluster                                |
| `scripts/prepare_diner_embedding_data.py` | Used when preparing data for diner embedding model              |
| `scripts/evaluate_diner_similarity.py` | Used for qualitative evaluation of diner embedding model           |
## How to download candidate generation or trained model result

Here, we run `scripts/download_result.py` python file to download results.

Note that you could directly download candidate results or trained model result in [google drive](https://drive.google.com/drive/u/0/folders/1kjoSmJ8bn3NIWbzJlPFXZkt6IFrcWIz4).

To download it using python code, please follow below steps.

1. Place credential file to authenticate google drive api to `credentials/` directory.
   - Refer to [this discussion](https://github.com/LearningnRunning/yamyam-lab/discussions/118#discussioncomment-12590729) about how to download credential json file from gcp.

2. Run following command depending on the embedding model and what you want to download, which is either of `candidates` or `models`.

    - If you want to download candidate generation results, place `--download_file_type` argument as `candidates`.
    - If you want to download trained model results with torch weight, logs, metric plot etc, place `--download_file_type` argument as `models`.
    - Currently, this script supports downloading latest result, which is denoted as `--latest` argument.

```bash
$ poetry run python3 scripts/download_result.py \
  --model_name "node2vec" \
  --download_file_type "candidates" \
  --latest \
  --credential_file_path_from_gcloud_console "PATH/TO/CREDENTIALS.json" \
  --reusable_token_path "PATH/TO/TOKEN.json"
```

Refer to description of each parameter.

| Parameter name                             | Description                                                                                |
|--------------------------------------------|--------------------------------------------------------------------------------------------|
| `model_name`                               | Name of embedding model (`node2vec` / `metapath2vec` / `graphsage` are allowed)            |
| `download_file_type`                       | File type you want to download (`candidates` / `models` are allowed)                       |
| `latest`                                   | Indicator whether downloading latest candidate generation results in selected model or not |
| `credential_file_path_from_gcloud_console` | Path to credential json file from gcp                                                      |
| `reusable_token_path`                      | Path to reusable token path                                                                |


After running script, check whether zip file is downloaded in `candidates/{model_name}` or `trained_models/{model_name}` successfully.

Latest version of each embedding model is given below. Note that identical version is applied both of candidate generation result and training result. (If zip file name is identical, they are generated from same training pipeline)

| Model name   | Latest version   |
|--------------|------------------|
| node2vec     | 202504070010.zip |
| metapath2vec | 202504011954.zip |
| graphsage    | 202504122051.zip |


## How to generate candidates from trained embedding model

Here, we run `scripts/generate_candidate.py` python file to generate candidates.

There are two required files when generating candidates from trained embedding model.

- weight.pt
- data_object.pkl

You could directly download those files in [google drive](https://drive.google.com/drive/u/0/folders/1zdqZldExdYZ2eH-Gfabnh8rHkWamPnVG) unzipping training results.

Or you could download them running `scripts/download_result.py` script. Please refer to above `How to download candidate generation or trained model result` section for more details.

You should specify path for trained pytorch weight and data object when running `scripts/generate_candidate.py`.

Depending on the embedding model you want, different arguments are required.

Refer to description of each parameter.

| Parameter name                     | Description                                                                     |
|------------------------------------|---------------------------------------------------------------------------------|
| `model`                            | Name of embedding model (`node2vec` / `metapath2vec` / `graphsage` / `lightgcn` are allowed) |
| `data_obj_path`                    | Path to data_object.pkl                                                         |
| `model_pt_path`                    | Path to weight.pt                                                               |
| `candidate_top_k`                  | Number of candidates to generate                                                |
| `reusable_token_path`              | Path to reusable token path                                                     |
| `upload_candidate_to_google_drive` | Indicator value whether to upload generated candidates to google drive or not   |

### node2vec

```bash
$ poetry run python3 scripts/generate_candidate.py \
  --model node2vec \
  --data_obj_path /PATH/TO/NODE2VEC/data_object.pkl \
  --model_pt_path /PATH/TO/NODE2VEC/weight.pt \
  --candidate_top_k 100 \
  --reusable_token_path PATH/TO/token.json
```

### metapath2vec
```bash
$ poetry run python3 scripts/generate_candidate.py \
  --model metapath2vec \
  --data_obj_path /PATH/TO/METAPATH2VEC/data_object.pkl \
  --model_pt_path /PATH/TO/METAPATH2VEC/weight.pt \
  --candidate_top_k 100 \
  --reusable_token_path PATH/TO/token.json
```


### graphsage

```bash
$ poetry run python3 scripts/generate_candidate.py \
  --model graphsage \
  --data_obj_path /PATH/TO/GRAPHSAGE/data_object.pkl \
  --model_pt_path /PATH/TO/GRAPHSAGE/weight.pkl \
  --candidate_top_k 100 \
  --reusable_token_path PATH/TO/token.json
```


## Region preprocessing (walking regions)

아래 내용은 `src/preprocess/region/README.md`의 핵심을 종합하여 `scripts/` 관점에서 정리한 것입니다. 서울 등 특정 지역을 H3 해상도 10으로 커버한 뒤, 도보 거리 기반으로 연결/클러스터링하여 추천에 적합한 권역을 생성합니다.

### 주요 기능

- 🏙️ OSMnx로 특정 지역(예: 서울시) 행정경계 자동 획득
- 📍 H3 해상도 10 기반 전체 커버리지 생성
- 🚶 OSRM 도보 거리 계산 (Haversine 백업)
- 🔗 그래프 기반 연결 요소 분석
- 📁 CSV/GeoJSON 결과 출력 및 중간 분석 산출물 옵션 제공
- 🍽️ 음식점 데이터 통합 (yamyam-lab DataLoader 연동)

### 디렉터리 구조

```
src/preprocess/region/
├── __init__.py          # 모듈 초기화 및 공개 API
├── builder.py           # 핵심 권역 생성 로직
└── README.md            # 상세 문서
```

### 1) CLI 스크립트 사용 (권장)

권역 생성은 `scripts/build_regions.py`로 실행합니다.

```bash
# 영등포구 테스트 (빠른 실행)
poetry run python scripts/build_regions.py --region "영등포구"

# 서울시 전체 실행
poetry run python scripts/build_regions.py --region "서울특별시"

# 음식점 데이터 없이 실행
poetry run python scripts/build_regions.py --region "영등포구" --no_restaurant_data

# 세부 설정 조정 예시
poetry run python scripts/build_regions.py \
    --region "중구" \
    --threshold_m 400 \
    --max_region_distance_m 1500 \
    --min_cells 20 \
    --max_cells 25 \
    --out_dir data/processed/regions/test
```

주요 인자 요약:

- `--region`: 행정구역명 (예: "영등포구", "서울특별시")
- `--threshold_m`: 도보 연결 임계값(미터)
- `--max_region_distance_m`: 권역 내 최대 연결 거리(미터)
- `--min_cells`/`--max_cells`: 권역 크기 하한/상한(H3 셀 수)
- `--no_restaurant_data`: 음식점 데이터 미사용 플래그
- `--out_dir`: 결과 저장 디렉터리

### 2) Python 코드에서 직접 사용

스크립트 대신 라이브러리처럼 사용할 수도 있습니다.

```python
from data.config import DataConfig
from preprocess.region import build_walking_regions

# 데이터 설정 로드
data_config = DataConfig.from_yaml("config/data/default.yaml")

# 권역 생성 실행
result_df = build_walking_regions(
    data_config=data_config,
    region_name="영등포구",
    walking_threshold_m=500,
    max_region_distance_m=2000,
    out_dir="data/processed/regions",
    use_restaurant_data=True,
)

print(f"생성된 권역 수: {result_df['region_id'].nunique()}")
```

### 출력 파일

권역 생성 완료 후 아래와 같은 파일들이 생성됩니다.

- `{region}_walking_regions_{timestamp}.csv`: 권역 정보 CSV
- `{region}_walking_regions_{timestamp}.geojson`: 시각화용 GeoJSON

선택적 중간 산출물(옵션):

- `{prefix}_graph.pkl`: NetworkX 그래프 객체
- `{prefix}_distance_cache.pkl`: 거리 계산 캐시
- `{prefix}_edges.csv`: 엣지 리스트
- `{prefix}_nodes.csv`: 노드 통계
- `{prefix}_region_stats.json`: 권역 통계

### 설정 파일

기본 설정은 `config/preprocess/region.yaml`에서 관리합니다. 주요 항목은 다음과 같습니다.

- H3 해상도
- 도보 거리 임계값
- 최대 권역 거리
- OSRM 설정
- 캐시 설정
- 출력 설정

### 성능 최적화 팁

- OSRM 캐시: 거리 계산 결과를 `data/cache/osrm/osrm_distance_cache_{region}.pkl`에 캐시하여 재실행 시 속도를 크게 향상합니다.
- H3 K-ring 전략: k-ring 이웃에 대해서만 거리 계산하여 O(n²) → O(n) 수준으로 줄입니다. 기본 k=1 권장.

### 추천 시스템에서의 활용 예

```python
import pandas as pd
import h3

# 권역 데이터 로드
# TODO 모듈로 쓸 수 있게 추가 예정
regions_df = pd.read_csv("data/processed/regions/Seoul_walking_regions_latest.csv")

# 특정 좌표의 권역 찾기
def find_region(lat, lon, regions_df):
    cell_id = h3.latlng_to_cell(lat, lon, 10)
    region_info = regions_df[regions_df['cell_id'] == cell_id]
    return region_info['region_id'].iloc[0] if not region_info.empty else -1

# 권역 기반 추천 필터링
user_region = find_region(user_lat, user_lon, regions_df)
nearby_restaurants = restaurants_df[restaurants_df['region_id'] == user_region]
```

보다 상세한 배경과 구현 설명은 `src/preprocess/region/README.md`와 `src/preprocess/region/builder.py`를 참고하세요.


## Diner Embedding Model

The diner embedding model creates 128-dimensional embeddings for restaurants where dot-product similarity returns semantically similar diners. This section covers data preparation and qualitative evaluation scripts.

### Data Preparation

Use `scripts/prepare_diner_embedding_data.py` to preprocess raw data for training.

```bash
# Full data preparation
poetry run python scripts/prepare_diner_embedding_data.py \
    --local_data_dir data/ \
    --output_dir data/processed

# Test mode (subset of data for quick testing)
poetry run python scripts/prepare_diner_embedding_data.py \
    --local_data_dir data/ \
    --output_dir data/processed \
    --test
```

| Parameter | Description |
|-----------|-------------|
| `--local_data_dir` | Directory containing raw CSV files (diner.csv, review.csv, menu_df.csv, diner_category.csv) |
| `--output_dir` | Directory to save processed parquet files |
| `--kobert_model_name` | HuggingFace model for Korean BERT (default: `klue/bert-base`) |
| `--test` | Run in test mode with subset of data |
| `--val_ratio` | Ratio of pairs for validation (default: 0.1) |
| `--test_ratio` | Ratio of pairs for test (default: 0.1) |

Output files:
- `diner_features.parquet` - Preprocessed features for all diners
- `training_pairs.parquet` - Positive training pairs
- `val_pairs.parquet` - Validation pairs
- `test_pairs.parquet` - Test pairs
- `category_mapping.parquet` - Category mapping for hard negative mining

### Training

After data preparation, train the model:

```bash
poetry run python -m yamyam_lab.train \
    --model diner_embedding \
    --epochs 50 \
    --device cuda
```

### Qualitative Evaluation

Use `scripts/evaluate_diner_similarity.py` to inspect top-N similar diners for a given anchor diner with names, categories, and similarity scores.

#### Search diners by name

```bash
poetry run python scripts/evaluate_diner_similarity.py \
    --model_path result/untest/diner_embedding/<timestamp>/weight.pt \
    --search "버거"
```

Output:
```
Found 2 diners matching '버거':
 diner_idx        diner_name
 354918976   로우로우 버거샵
 183570751  파이브가이즈 용산
```

#### Show similar diners for a specific diner_idx

```bash
poetry run python scripts/evaluate_diner_similarity.py \
    --model_path result/untest/diner_embedding/<timestamp>/weight.pt \
    --diner_idx 354918976 \
    --top_n 10
```

Output:
```
================================================================================
Anchor Diner (idx=354918976):
  Name: 로우로우 버거샵
  Category: 양식 > 햄버거 > nan
================================================================================

Top-10 Similar Diners:
--------------------------------------------------------------------------------
 1. [0.9523] 파이브가이즈 용산
    Category: 양식 > 햄버거 > nan ✓✓
 2. [0.8912] 브루클린버거
    Category: 양식 > 햄버거 > nan ✓✓
 ...
--------------------------------------------------------------------------------
```

The ✓ marks indicate category matches:
- ✓ = same large category as anchor
- ✓✓ = same large AND middle category as anchor

#### Interactive mode

```bash
poetry run python scripts/evaluate_diner_similarity.py \
    --model_path result/untest/diner_embedding/<timestamp>/weight.pt \
    --interactive
```

Commands in interactive mode:
- `search <query>` - Search diners by name
- `show <diner_idx>` - Show similar diners for a diner_idx
- `quit` - Exit

#### Parameters

| Parameter | Description |
|-----------|-------------|
| `--model_path` | Path to trained model weights (.pt file) - **required** |
| `--features_path` | Path to preprocessed features parquet (default: `data/processed/diner_features.parquet`) |
| `--diner_csv` | Path to diner.csv (default: `data/diner.csv`) |
| `--category_csv` | Path to diner_category.csv (default: `data/diner_category.csv`) |
| `--diner_idx` | Anchor diner index to query |
| `--search` | Search for diners by name |
| `--top_n` | Number of similar diners to show (default: 10) |
| `--device` | Device to use: cpu or cuda (default: cpu) |
| `--interactive` | Run in interactive mode |
