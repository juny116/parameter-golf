# Parameter Golf

작은 용량의 언어모델을 학습하고 양자화·압축하는 코드입니다. [OpenAI Parameter Golf](https://github.com/openai/parameter-golf)를 기반으로, 학습 변형과 체크포인트 재압축 도구를 추가했습니다.

## 코드 구조

| 파일 | 역할 |
|---|---|
| `train_gpt.py` | PyTorch 분산 학습 기본 구현 |
| `train_gpt_mlx.py` | Apple Silicon용 MLX 학습 구현 |
| `train_gpt_improved.py` | n-gram 해시 임베딩, 양자화 및 학습 변형 |
| `train_gpt_submit.py` | QAT, SWA, 혼합 정밀도 양자화를 포함한 학습 구현 |
| `recompress.py` | 체크포인트에 pruning과 양자화를 적용해 압축 크기 비교 |
| `data/cached_challenge_fineweb.py` | FineWeb 학습 데이터 다운로드 |
| `data/download_hf_docs_and_tokenize.py` | 데이터 다운로드와 토큰화 |

## 사용

```bash
pip install -r requirements.txt
```

데이터 준비 방법은 [data/README.md](data/README.md)를 참고합니다. PyTorch 학습에는 CUDA 환경이 필요하고, MLX 버전은 Apple Silicon에서 실행합니다.

```bash
torchrun --standalone --nproc_per_node=1 train_gpt_improved.py
```

학습 스크립트의 `Hyperparameters`에서 지원하는 환경변수로 데이터 경로, 모델 크기, 학습 시간을 조정합니다. `recompress.py`는 루트의 `final_model.pt`를 읽으며 `train_gpt_improved.py`의 양자화 함수를 사용합니다.

기반 코드의 라이선스는 [LICENSE](LICENSE), 외부 코드 표기는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 있습니다.
