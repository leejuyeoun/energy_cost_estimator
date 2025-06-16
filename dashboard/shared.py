from pathlib import Path
import pandas as pd

# 현재 파일 위치 기준으로 데이터 경로 설정
app_dir = Path(__file__).parent

train = pd.read_csv(app_dir / "./data/train.csv")
train['측정일시'] = pd.to_datetime(train['측정일시'])
train['월'] = train['측정일시'].dt.month

test = pd.read_csv(app_dir / "./data/test.csv")

streaming_df = pd.read_csv(app_dir / "./data/streaming_df.csv")
streaming_df.columns = streaming_df.columns.str.strip()  # 공백 제거
streaming_df = streaming_df.rename(columns={"작업유형": "작업유형", "예측_전기요금": "예측_전기요금", "예측_전력사용량" : "예측_전력사용량"})
streaming_df["측정일시"] = pd.to_datetime(streaming_df["측정일시"])



