# NARKMA: Non-autoregressive Korean Morphological Analyzer

비자동회귀 한국어 형태소 분석기입니다.

O(1)의 속도로 형태소 분석 결과를 생성합니다.

동시에 좋은 성능을 보여줍니다.

혹시나... 형태소 분석기가 필요한 사람들이 있을까봐 석사 과정 때 연구한 내용을 바탕으로 제작하였습니다.

비자동회귀 형태소 분석기에 관심이 있으시면 읽어주세요!
- [**어절 정보를 활용한 비자동회귀 한국어 형태소 분석**](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11495791)

- [**비자동회귀 다중 디코더 기반 한국어 형태소 분석**](https://koreascience.kr/article/CFKO202226455347146.page)

---

# 사용 방법

## Prepare
```
pip install -r requirements.txt
```
```
apt-get install python-is-python3
```

It worked on (nvcc --version : 11.8) and (CUDA version : 12.2)

## Train
```
bash scripts/train.sh
```


## Inference
```
bash scripts/infer.sh
```


# 모델 다운 및 설명

[**Enc1NARDec2(100epochs)**](https://drive.google.com/file/d/1Af-YW4DYyCjNoqjSnWbOiWFz0Qp5-zv2/view?usp=drive_link)

원하는 모델을 다운로드 하셔서 압축해제 하고 NARKMA 안에 넣어주세요

---

# 실험 결과(wip)

|      Model     | Epochs |         ACC        |         F1         |
|:--------------:|:------:|:------------------:|:------------------:|
| Enc1ARDec1     |        |                    |                    |
| Enc1ARDec1_CRF |        |                    |                    |
| Enc1NARDec2    | 100    | 0.9456928372383118 | 0.9673664569854736 |
|                | 400    |                    |                    |

---

# 한계

### 아직 띄어쓰기에 강건하지 않습니다!!!

띄어쓰기에 강건한 한국어 형태소 분석기에 관심이 있으시다면

박천음 교수님의 논문을 참고 바랍니다.
- [**Robust Multi-task Learning-based Korean POS Tagging to Overcome Word Spacing Errors**](https://dl.acm.org/doi/10.1145/3591206)


