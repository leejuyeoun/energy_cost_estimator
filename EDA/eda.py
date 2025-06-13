import pandas as pd
import matplotlib.pyplot as plt

# 한글 설정하고 시작
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

df= pd.read_csv('../data/train.csv')


plt.scatter(df['지상무효전력량(kVarh)'], df['전기요금(원)'], alpha=0.5)
plt.xlabel('지상무효전력량(kVarh)')
plt.ylabel('전기요금(원)')
plt.title('지상무효전력량과 전기요금의 관계')
plt.tight_layout()
plt.show()

plt.scatter(df['진상무효전력량(kVarh)'], df['전기요금(원)'], alpha=0.5)
plt.xlabel('진상무효전력량(kVarh)')
plt.ylabel('전기요금(원)')
plt.title('진상무효전력량과 전기요금의 관계')
plt.tight_layout()
plt.show()

plt.scatter(df['지상역률(%)'], df['전기요금(원)'], alpha=0.5)
plt.xlabel('지상역률(%)')
plt.ylabel('전기요금(원)')
plt.title('지상역률과 전기요금의 관계')
plt.tight_layout()
plt.show()

plt.scatter(df['진상역률(%)'], df['전기요금(원)'], alpha=0.5)
plt.xlabel('진상역률(%)')
plt.ylabel('전기요금(원)')
plt.title('진상역률과 전기요금의 관계')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(df['전력사용량(kWh)'], df['탄소배출량(tCO2)'], alpha=0.5)
plt.xlabel('전력사용량(kWh)')
plt.ylabel('탄소배출량(tCO₂)')
plt.title('전력사용량과 탄소배출량의 관계')
plt.tight_layout()
plt.show()



df['사용량구간2'] = pd.cut(
    df['전력사용량(kWh)'], 
    bins=[0, 50, 100, 200, 300, 2000], 
    labels=['0-50', '50-100', '100-200', '200-300', '300+']
)
print(df['사용량구간2'].value_counts())

plt.figure(figsize=(8,5))
plt.scatter(df['진상역률(%)'], df['단가'], alpha=0.6)
plt.xlabel('진상역률(%)')
plt.ylabel('단가(원/kWh)')
plt.title('역률과 단가(원/kWh)의 관계 (전체 데이터)')
plt.tight_layout()
plt.show()


test= pd.read_csv('../data/test.csv')


