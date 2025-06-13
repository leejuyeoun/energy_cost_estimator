import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shared import train
from shiny import App, render, ui
import matplotlib as mpl
from matplotlib import font_manager
from pathlib import Path
from shinywidgets import output_widget, render_widget
import tempfile

# Windows에서 한글 폰트 설정 (예: 맑은 고딕)
mpl.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 깨짐 방지
mpl.rcParams['axes.unicode_minus'] = False

# 1. 데이터 전처리
train['측정일시'] = pd.to_datetime(train['측정일시'])
train['월'] = train['측정일시'].dt.month

# 2. 요약 지표 계산
total_usage_val = train['전력사용량(kWh)'].sum()
total_cost_val = train['전기요금(원)'].sum()
avg_unit_price_val = total_cost_val / total_usage_val if total_usage_val > 0 else 0
peak_month_val = train.groupby('월')['전기요금(원)'].sum().idxmax()





#######################################################
# 3. UI 구성
#######################################################
from pathlib import Path
from shiny import ui

app_ui = ui.TagList(
    # 1. CSS 먼저 포함
    ui.include_css(Path(__file__).parent / "styles.css"),

    # 2. 페이지 전체를 page_navbar로 정의
    ui.page_navbar(
        # [탭1] 1~11월 전기요금 분석
        ui.nav_panel(
            "1~11월 전기요금 분석",

            # A. 요약 카드
            ui.input_date_range("기간", "기간 선택", start="2024-01-01", end="2024-11-30"),
            ui.layout_column_wrap(
                ui.card("총 전력 사용량 (kWh)", ui.output_text("range_usage")),
                ui.card("총 전기요금 (원)", ui.output_text("range_cost")),
                ui.card("일평균 전력 사용량 (kWh)", ui.output_text("avg_usage")),
                ui.card("일평균 전기요금 (원)", ui.output_text("avg_cost")),
                width=1/4,
                gap="20px"
            ),
            ui.hr(),

            # B + C
            ui.layout_columns(
                ui.card(
                    ui.h4("[B] 전력 사용량 및 전기요금 추이 (분석 단위별)"),
                    ui.layout_columns(
                        ui.input_select("선택월", "분석할 월", choices=["전체(1~11)"] + [str(i) for i in range(1, 12)], selected="전체(1~11)"),
                        ui.input_select("단위", "분석 단위", choices=["월", "주차", "일", "요일", "시간"], selected="월")
                    ),
                    ui.output_plot("usage_cost_drilldown")
                ),
                ui.card(
                    ui.h4("[C] 선택 단위별 전력사용량 / 전기요금"),
                    ui.output_ui("summary_table"),
                    style="height: 300px; overflow-y: auto;"
                )
            ),
            ui.hr(),

            # D + E

            # E 작업유형별 시각화
            ui.layout_columns(
                ui.card(
                    ui.h4("[D]월별 작업유형별 전력 사용량 (matplotlib)"),

            #  월 선택 드롭다운: 딱 여기!
            ui.input_select(
                "selected_month", "월 선택",
                choices=[str(m) for m in sorted(train['월'].unique())],
                selected="1"
            ),

                    ui.output_image("usage_by_type_matplotlib")
                ),
                ui.card(
                    ui.h4("[E] 선택 월의 작업유형별 분포"),
                    ui.input_select(  
                        "selected_day", "요일 선택",
                        choices=["월", "화", "수", "목", "금", "토", "일"],
                        selected="월"
                    ),
                    ui.output_image("usage_by_dayofweek_matplotlib"),
                    ui.output_image("usage_by_hour_matplotlib")                    
                
                ),
            ),  
        ),

        # [탭2] 12월 예측 및 모델 근거
        ui.nav_panel("12월 예측 및 모델 근거"),

        # 기본 설정
        title="피카피카",
        id="page"
    )
)





# 4. 서버 함수 정의
#####################################
#  TAB1 A
#####################################
def server(input, output, session):
    @output
    @render.text
    def range_usage():
        start, end = input.기간()
        mask = (train['측정일시'].dt.date >= start) & (train['측정일시'].dt.date <= end)
        return f"{train.loc[mask, '전력사용량(kWh)'].sum():,.2f} kWh"

    @output
    @render.text
    def range_cost():
        start, end = input.기간()
        mask = (train['측정일시'].dt.date >= start) & (train['측정일시'].dt.date <= end)
        return f"{train.loc[mask, '전기요금(원)'].sum():,.0f} 원"

    @output
    @render.text
    def avg_usage():
        start, end = input.기간()
        mask = (train['측정일시'].dt.date >= start) & (train['측정일시'].dt.date <= end)
        days = (end - start).days + 1
        val = train.loc[mask, '전력사용량(kWh)'].sum() / days
        return f"{val:,.2f} kWh"

    @output
    @render.text
    def avg_cost():
        start, end = input.기간()
        mask = (train['측정일시'].dt.date >= start) & (train['측정일시'].dt.date <= end)
        days = (end - start).days + 1
        val = train.loc[mask, '전기요금(원)'].sum() / days
        return f"{val:,.0f} 원"


#####################################
#  TAB1 B - 월별 전력 사용량 및 전기요금 추이
#####################################
    @output
    @render.plot
    def usage_cost_drilldown():
        단위 = input.단위()
        선택월 = input.선택월()

        df = train.copy()
        df['측정일시'] = pd.to_datetime(df['측정일시'])
        df['월'] = df['측정일시'].dt.month

        if 단위 != "월" and 선택월 != "전체(1~11)":
            df = df[df['월'] == int(선택월)]

        if 단위 == "시간":
            df['단위'] = df['측정일시'].dt.hour
            grouped = df.groupby(['단위', '작업유형'])[['전력사용량(kWh)', '전기요금(원)']].sum().reset_index()

            colors = {
                'Light_Load': 'skyblue',
                'Medium_Load': 'orange',
                'Maximum_Load': 'crimson'
            }
            hours = np.arange(0, 24)
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()

            bottoms = np.zeros_like(hours, dtype=float)
            for load_type in ['Light_Load', 'Medium_Load', 'Maximum_Load']:
                vals = []
                for h in hours:
                    v = grouped[(grouped['단위'] == h) & (grouped['작업유형'] == load_type)]['전력사용량(kWh)']
                    vals.append(float(v.iloc[0]) if not v.empty else 0)
                ax1.bar(hours, vals, color=colors.get(load_type, 'gray'), bottom=bottoms, label=load_type)
                bottoms += np.array(vals)

            total_by_hour = df.groupby('단위')['전기요금(원)'].sum().reindex(hours, fill_value=0)
            ax2.plot(hours, total_by_hour.values, color='red', marker='o', label='전기요금')

            ax1.set_xticks(hours)
            ax1.set_xlabel("시간")
            ax1.set_ylabel("전력 사용량 (kWh)")
            ax2.set_ylabel("전기요금 (원)")
            ax1.set_title(f"{선택월} 월 기준 시간별 전력 사용량(누적) 및 전기요금 추이")
            ax1.legend(title="작업유형")
            fig.tight_layout()
            return fig

        elif 단위 == "요일":
            요일_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
            df['단위'] = df['측정일시'].dt.dayofweek.map(요일_map)
            요일순서 = ["월", "화", "수", "목", "금", "토", "일"]
            grouped = (
                df.groupby('단위')[['전력사용량(kWh)', '전기요금(원)']]
                .sum()
                .reindex(요일순서)
            )
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.bar(요일순서, grouped['전력사용량(kWh)'], color='skyblue', label='전력 사용량')
            ax2.plot(요일순서, grouped['전기요금(원)'], color='red', marker='o', label='전기요금')
            ax1.set_ylabel("전력 사용량 (kWh)")
            ax2.set_ylabel("전기요금 (원)")
            ax1.set_xlabel("요일")
            ax1.set_title(f"{선택월} 월 기준 요일별 전력 사용량 및 전기요금 추이")
            fig.tight_layout()
            return fig

        elif 단위 == "일":
            df['일'] = df['측정일시'].dt.day
            df['요일'] = df['측정일시'].dt.dayofweek
            df['구분'] = df['요일'].apply(lambda x: '주말' if x >= 5 else '평일')

            grouped = df.groupby(['일', '구분'])[['전력사용량(kWh)', '전기요금(원)']].sum().reset_index()

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()

            color_map = {'평일': 'skyblue', '주말': 'coral'}

            for gubun in ['평일', '주말']:
                sub = grouped[grouped['구분'] == gubun]
                ax1.bar(sub['일'], sub['전력사용량(kWh)'], color=color_map[gubun], label=gubun)

            total_by_day = df.groupby('일')['전기요금(원)'].sum().reindex(range(1, 32), fill_value=0)
            ax2.plot(total_by_day.index, total_by_day.values, color='red', marker='o', label='전기요금')

            ax1.set_xlabel("일")
            ax1.set_ylabel("전력 사용량 (kWh)")
            ax2.set_ylabel("전기요금 (원)")
            ax1.set_title(f"{선택월} 월 기준 일별 전력 사용량 및 전기요금 추이")

            # 범례 병합
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2)

            fig.tight_layout()
            return fig

        else:
            if 단위 == "월":
                df['단위'] = df['월']
            elif 단위 == "주차":
                df['단위'] = df['측정일시'].dt.day // 7 + 1

            grouped = df.groupby('단위').agg({
                '전력사용량(kWh)': 'sum',
                '전기요금(원)': 'sum'
            }).reset_index()

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.bar(grouped['단위'], grouped['전력사용량(kWh)'], color='skyblue', label='전력 사용량')
            ax2.plot(grouped['단위'], grouped['전기요금(원)'], color='red', marker='o', label='전기요금')
            ax1.set_ylabel("전력 사용량 (kWh)")
            ax2.set_ylabel("전기요금 (원)")
            ax1.set_xlabel(단위)
            ax1.set_title(f"{선택월} 월 기준 {단위}별 전력 사용량 및 전기요금 추이")
            fig.tight_layout()
            return fig




#####################################
#  TAB1 C - 선택 단위별 전력사용량 / 전기요금 카드
#####################################
    @output
    @render.ui
    def summary_table():
        df = train.copy()
        df['측정일시'] = pd.to_datetime(df['측정일시'])
        df['월'] = df['측정일시'].dt.month

        선택월 = input.선택월()
        단위 = input.단위()

        # '월' 단위가 아닌 경우에만 월 필터 적용
        if 단위 != "월" and 선택월 != "전체(1~11)":
            df = df[df['월'] == int(선택월)]

        if 단위 == "월":
            df['월'] = df['측정일시'].dt.month
            df['구분'] = df['월'].astype(str) + "월"

            grouped = (
                df.groupby(['구분'])[['전력사용량(kWh)', '전기요금(원)']]
                .sum()
                .reset_index()
                .sort_values('구분', key=lambda x: x.str.replace("월", "").astype(int))
            )

        elif 단위 == "주차":
            df['구분'] = (df['측정일시'].dt.day // 7 + 1).astype(str) + " 주차"
            grouped = df.groupby('구분')[['전력사용량(kWh)', '전기요금(원)']].sum().reset_index()

        elif 단위 == "일":
            df['정렬용'] = df['측정일시'].dt.day
            df['구분'] = df['정렬용'].astype(str) + "일"
            grouped = (
                df.groupby(['정렬용', '구분'])[['전력사용량(kWh)', '전기요금(원)']]
                .sum()
                .reset_index()
                .sort_values('정렬용')
                .drop(columns='정렬용')
            )

        elif 단위 == "요일":
            요일_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
            요일순서 = ["월", "화", "수", "목", "금", "토", "일"]
            df['구분'] = df['측정일시'].dt.dayofweek.map(요일_map)
            grouped = (
                df.groupby('구분')[['전력사용량(kWh)', '전기요금(원)']]
                .sum()
                .reindex(요일순서)
                .reset_index()
            )

        elif 단위 == "시간":
            df['시간'] = df['측정일시'].dt.hour
            df['구분'] = df['시간'].astype(str) + "시"
            시간순서 = [f"{i}시" for i in range(24)]
            grouped = (
                df.groupby('구분')[['전력사용량(kWh)', '전기요금(원)']]
                .sum()
                .reindex(시간순서, fill_value=0)
                .reset_index()
            )

        # 숫자 포맷 적용
        grouped['전력사용량(kWh)'] = grouped['전력사용량(kWh)'].apply(lambda x: f"{x:,.2f}")
        grouped['전기요금(원)'] = grouped['전기요금(원)'].apply(lambda x: f"{x:,.0f}")
        grouped = grouped[['구분', '전력사용량(kWh)', '전기요금(원)']]

        # HTML 테이블 출력
        html = grouped.to_html(index=False, classes="table table-striped", escape=False, border=0)

        custom_style = """
        <style>
            .table th, .table td {
                text-align: center !important;
                vertical-align: middle !important;
            }
        </style>
        """
        return ui.HTML(custom_style + html)








#####################################
#  TAB1 D - 요일 및 날짜별 요금 패턴
#####################################
    # [D][E] 대체: matplotlib 시각화
    @output
    @render.image
    def usage_by_type_matplotlib():
        selected_month = int(input.selected_month())
        # 월-작업유형별 피벗
        monthly = train.groupby(['월', '작업유형'])['전력사용량(kWh)'].sum().unstack().fillna(0)
        months = monthly.index.tolist()
        fig, ax = plt.subplots(figsize=(8, 4))
        bottom = np.zeros(len(months))
        colors = ['#FFD700', '#FF6347', '#DB7093']

        total_usage = monthly.values.sum()  # 전체 사용량


        for idx, col in enumerate(monthly.columns):
            y = monthly[col].values
            for i, m in enumerate(months):
                month_total = monthly.iloc[i].sum()  # 현재 월의 전체 사용량
                ratio = (y[i] / month_total * 100) if month_total > 0 else 0
                edgecolor = 'royalblue' if m == int(input.selected_month()) else 'gray'
                linewidth = 3 if m == int(input.selected_month()) else 1
                alpha = 1 if m == int(input.selected_month()) else 0.4
                ax.bar(
                    m, y[i],
                    bottom=bottom[i],
                    color=colors[idx],
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=alpha,
                    label=col if i == 0 else ""
                )
                # 바 위에 값+비율 표기
                if y[i] > 0:
                    ax.text(
                        m, bottom[i] + y[i] / 2,
                        f"{int(y[i]):,}\n({ratio:.1f}%)",  # ← 월별 비율
                        ha='center', va='center',
                        fontsize=8,
                        fontweight='normal',
                        color='black' if m == int(input.selected_month()) else 'dimgray'
                    )
            bottom += y

        ax.set_title('월별 작업유형별 전력 사용량 (Stacked Bar)')
        ax.set_xlabel('월')
        ax.set_ylabel('전력사용량 (kWh)')
        ax.legend(title='작업유형')
        ax.set_xticks(months) 
        ax.set_xticklabels([str(m) for m in months]) 
        plt.tight_layout()

         # temp 파일로 저장 후 경로 리턴!
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(tmpfile, format="png")
        plt.close(fig)
        tmpfile.close()
        return {
            "src": tmpfile.name,  # 경로!
            "alt": "월별 작업유형별 전력사용량 (matplotlib)"
        }

 

    # [D][E] 대체: 월별 작업유형별 전력 사용량 및 비율 (표)
    # 
    @output
    @render.image
    def usage_by_dayofweek_matplotlib():
        selected_month = int(input.selected_month())
        df_month = train[train['월'] == selected_month].copy()
        dow_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
        df_month['요일'] = df_month['측정일시'].dt.dayofweek.map(dow_map)
        pivot = df_month.pivot_table(
            index='요일', columns='작업유형', values='전력사용량(kWh)', aggfunc='sum', fill_value=0
        ).reindex(list(dow_map.values()))
        fig, ax = plt.subplots(figsize=(7, 3))
        bottom = np.zeros(len(pivot))
        colors = ['#FFD700', '#FF6347', '#DB7093']

        for idx, col in enumerate(pivot.columns):
            bar = ax.bar(pivot.index, pivot[col], bottom=bottom, color=colors[idx], label=col)
            for i, val in enumerate(pivot[col]):
                # --- 2500 미만 사용량은 표기 생략 ---
                if val > 2500:
                    total = pivot.iloc[i].sum()
                    ratio = (val / total * 100) if total > 0 else 0
                    ax.text(
                        i, bottom[i] + val / 2, f"{int(val):,}\n({ratio:.1f}%)",
                        ha='center', va='center', fontsize=8, color='black'
                    )
            bottom += pivot[col].values
        ax.set_title(f"{selected_month}월 요일별 작업유형별 전력 사용량")
        ax.set_xlabel("요일")
        ax.set_ylabel("전력사용량 (kWh)")  
        ax.legend(title='작업유형')
        plt.tight_layout()                 # ← 이거 한 줄 추가!
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(tmpfile, format="png")
        plt.close(fig)
        tmpfile.close()
        return {"src": tmpfile.name, "alt": "요일별 작업유형별 전력사용량"}

    @output
    @render.image
    def usage_by_hour_matplotlib():
        selected_month = int(input.selected_month())
        selected_day = input.selected_day()  # 새 input 사용
        df_month = train[train['월'] == selected_month].copy()
        dow_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
        df_month['요일'] = df_month['측정일시'].dt.dayofweek.map(dow_map)
        df_month['시각'] = df_month['측정일시'].dt.hour
        df_day = df_month[df_month['요일'] == selected_day]
        pivot = df_day.pivot_table(
            index='시각', columns='작업유형', values='전력사용량(kWh)', aggfunc='sum', fill_value=0
        ).sort_index()
        fig, ax = plt.subplots(figsize=(7, 2.7))
        bottom = np.zeros(len(pivot))
        colors = ['#FFD700', '#FF6347', '#DB7093']
        for idx, col in enumerate(pivot.columns):
            ax.bar(pivot.index, pivot[col], bottom=bottom, color=colors[idx], label=col, width=0.8, alpha=0.85)
            bottom += pivot[col].values
        ax.set_title(f"{selected_month}월 {selected_day}요일 시간대별 작업유형별 전력 사용량")
        ax.set_xlabel("시각(0~23시)")
        ax.set_ylabel("전력사용량 (kWh)")
        ax.legend(title='작업유형')
        ax.set_xticks(range(0,24))
        plt.tight_layout()
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(tmpfile, format="png")
        plt.close(fig)
        tmpfile.close()
        return {"src": tmpfile.name, "alt": "시간대별 작업유형별 전력사용량"}




##############
# 5. 앱 실행
##############
app = App(app_ui, server)

