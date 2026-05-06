# Tic-Tac-Toe Policy Gradient RL

Python 표준 라이브러리만으로 구현한 tic-tac-toe policy gradient 예제입니다.

## 학습

```bash
python3 train.py --episodes 80000 --output model.json
```

학습은 REINFORCE 스타일의 Monte Carlo policy gradient를 사용합니다. 두 플레이어가 같은 정책으로 self-play를 하며, 각 상태에서 가능한 수에 대한 softmax 정책을 업데이트합니다.

## 모델과 대국하기

```bash
python3 serve.py
```

브라우저에서 `http://127.0.0.1:8000`을 열면 학습된 모델과 tic-tac-toe를 둘 수 있습니다.

## 파일

- `tictactoe_rl.py`: 게임 로직, 정책, 학습 루프
- `train.py`: CLI 학습 스크립트
- `serve.py`: 로컬 웹 서버
- `web/index.html`: 평가 UI
- `web/app.js`: 브라우저용 게임 및 모델 추론 로직

