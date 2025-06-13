from pathlib import Path

import pandas as pd

app_dir = Path(__file__).parent
train = pd.read_csv(app_dir / "./data/train.csv")
test = pd.read_csv(app_dir / "./data/test.csv")
