# Data Description

### diner.csv

| 컬럼명                     | 설명                                         |
| -------------------------- | -------------------------------------------- |
| diner_idx                  | 음식점 고유 식별자                           |
| diner_name                 | 음식점 이름                                  |
| diner_category_large       | 대분류 카테고리                              |
| diner_category_middle      | 중분류 카테고리                              |
| diner_category_small       | 소분류 카테고리                              |
| diner_category_detail      | 상세 카테고리                                |
| diner_tag                  | 음식점별 특징 (혼밥, 혼술, 제로페이 등)      |
| diner_menu                 | 메뉴 정보                                    |
| diner_menu_name            | 메뉴 이름                                    |
| diner_menu_price           | 메뉴 가격                                    |
| diner_review_cnt           | 리뷰 수                                      |
| diner_blog_review_cnt      | 블로그 리뷰 수                               |
| diner_review_avg           | 평균 평점                                    |
| diner_review_tags          | 리뷰 태그                                    |
| diner_address              | 주소                                         |
| diner_phone                | 전화번호                                     |
| diner_lat                  | 위도                                         |
| diner_lon                  | 경도                                         |
| diner_url                  | 음식점 URL                                   |
| diner_open_time            | 영업 시간                                    |
| diner_address_constituency | 행정구역                                     |
| real_good_review_cnt       | 긍정적 리뷰 수                               |
| real_bad_review_cnt        | 부정적 리뷰 수                               |
| all_review_cnt             | 전체 리뷰 수                                 |
| real_good_review_percent   | 긍정적 리뷰 비율                             |
| real_bad_review_percent    | 부정적 리뷰 비율                             |
| is_small_category_missing  | 소분류 카테고리 누락 여부(0: null이었던 row) |
| bayesian_score             | 베이지안 조정 점수                           |
| rank                       | 음식점 랭킹                                  |

---

### review.csv

| 컬럼명                | 설명                                          |
| --------------------- | --------------------------------------------- |
| diner_idx             | 리뷰가 작성된 음식점 고유 식별자              |
| review_id             | 리뷰 고유 식별자                              |
| reviewer_id           | 리뷰어 고유 식별자                            |
| reviewer_review_score | 리뷰 평점                                     |
| reviewer_review       | 리뷰 내용                                     |
| reviewer_review_date  | 리뷰 작성 날짜                                |
| date_weight           | 날짜 가중치 (리뷰 날짜 기반으로 가중치 부여)  |
| date_scaled           | 날짜 기반으로 스케일링된 값                   |
| score_scaled          | 평점 기반으로 스케일링된 값                   |
| combined_score        | 조합 점수 (날짜 및 평점의 가중치를 반영한 값) |

---

### reviewer.csv

| 컬럼명             | 설명                          |
| ------------------ | ----------------------------- |
| reviewer_id        | 리뷰어 고유 식별자            |
| reviewer_avg       | 리뷰어의 평균 평점            |
| badge_grade        | 리뷰어의 배지 등급            |
| badge_level        | 리뷰어의 배지 레벨            |
| reviewer_user_name | 리뷰어의 공개 닉네임/사용자명 |

## 부가설명
- **`diner_tag`**: 음식점별 특징을 나타내며, 혼밥, 혼술, 제로페이 등 태그를 포함합니다.
- **리스트 형식으로 변환**:
  - `diner_tag`, `diner_review_tags`, `diner_menu_price`, `diner_menu_name`은 리스트 형식으로 변경되었습니다.
- **`real_good_review_cnt` 기준 설명**:
  - 긍정적 리뷰 기준에 대한 자세한 내용은 [블로그 포스팅 참조](https://learningnrunning.github.io/post/tech/review/2024-10-03-What-to-eat-today/#%EC%A3%BC%EC%9A%94-%ED%8A%B9%EC%A7%95).
- **`combined_score`**:
  - 머먹의 음식점 지수에 대한 상세 설명은 [블로그 포스팅 참조](https://learningnrunning.github.io/post/tech/review/2024-10-25-what2eat-upgrade/).

- **`badge_grade`와 `badge_level`**:
  - 리뷰어의 활동성과 신뢰도를 나타내는 지표입니다.
  - ![](https://blog.kakaocdn.net/dn/LbARw/btsczLDWyrc/zp7bToMZOCvFYj2iMJFWck/img.png)
- **`reviewer_user_name`**:
  - 리뷰어의 공개 닉네임 또는 사용자명(기존 `reviewer_id`와 별도 관리).
- **`review_id`**:
  - 각 리뷰를 구분하는 고유 식별자가 아닙니다. 아직 활용할 부분은 없지만 보존하고 있는 상태입니다.
- **`reviewer_collected_review_cnt`**:
  - 해당 리뷰어의 실제로 수집된 리뷰 수를 나타냅니다.

---