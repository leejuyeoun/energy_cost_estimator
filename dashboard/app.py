# ===============================
# 라이브러리 임포트
# ===============================
# 데이터 처리
import pandas as pd                  # 데이터프레임 처리
import numpy as np                   # 수치 계산, 배열 연산

# 경로/파일 처리
from pathlib import Path             # 경로 다루기
import tempfile                      # 임시파일 생성
import io                            # 메모리 버퍼 (PDF/이미지 등 저장용)

# 대시보드 프레임워크
from shiny import App, render, ui, reactive         # Shiny 앱 UI/서버
from shinywidgets import output_widget, render_widget # Shiny 위젯 확장

# 시각화 및 한글 폰트 설정
import matplotlib.pyplot as plt      # 데이터 시각화
import matplotlib as mpl             # 전역 폰트 등 스타일 설정
from matplotlib.dates import DateFormatter  # x축 날짜 포맷
from matplotlib import font_manager         # 폰트 관리
import matplotlib.ticker as ticker         # y축 포맷 (ex: 만원단위)
import matplotlib.ticker as mticker        # PDF 내 y축 포맷 (이름만 다름, 일부 코드에서 씀)
from matplotlib import font_manager as mpl_font_manager

# PDF 생성 관련
from reportlab.lib.utils import ImageReader         # matplotlib 이미지를 PDF로 넣기
from reportlab.pdfgen import canvas                 # 간단 PDF 생성
from reportlab.lib.pagesizes import A4              # A4 용지 사이즈
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Image, Table, TableStyle)  # PDF 구조 잡기
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle   # PDF 텍스트 스타일
from reportlab.lib import colors                    # PDF 색상 지정
from reportlab.pdfbase.ttfonts import TTFont        # 한글 폰트 등록
from reportlab.pdfbase import pdfmetrics            # 폰트 등록
from reportlab.lib.enums import TA_LEFT             # 텍스트 정렬

# 통계적 카운팅
from collections import Counter                     # 예: 부하 타입별 빈도 계산

import os
STATIC_DIR = os.path.join(os.path.dirname(__file__), "www")
from shared import streaming_df, train
from le_report import le_report
# ===============================
# 한글 폰트 설정, 마이너스 깨짐 방지
# ===============================
mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

def register_korean_font():
    # malgun.ttf 경로 설정 (www 폴더 기준)
    font_path = Path(__file__).parent / "www" / "malgun.ttf"

    # 1. matplotlib용 한글 폰트 등록
    mpl_font_manager.fontManager.addfont(str(font_path))
    malgun_name = mpl_font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["font.family"] = malgun_name
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

    # 2. reportlab용 한글 폰트 등록
    pdfmetrics.registerFont(TTFont("MalgunGothic", str(font_path)))
register_korean_font()


# 1. 요약 지표 계산
total_usage_val = train['전력사용량(kWh)'].sum()
total_cost_val = train['전기요금(원)'].sum()
avg_unit_price_val = total_cost_val / total_usage_val if total_usage_val > 0 else 0
peak_month_val = train.groupby('월')['전기요금(원)'].sum().idxmax()



# ===============================
# 실시간 스트리머 클래스 정의
# ===============================
class SimpleStreamer:
    def __init__(self, streaming_df):
        self.streaming_df = streaming_df.reset_index(drop=True)
        self.idx = 0
        self.current = pd.DataFrame(columns=streaming_df.columns)

    def get_next(self, n=1):
        if self.idx >= len(self.streaming_df):
            return None
        next_chunk = self.streaming_df.iloc[self.idx : self.idx + n]
        self.idx += n
        self.current = pd.concat([self.current, next_chunk], ignore_index=True) \
            if not self.current.empty else next_chunk
        return next_chunk

    def get_data(self):
        return self.current.copy()

    def reset(self):
        self.idx = 0
        self.current = pd.DataFrame(columns=self.streaming_df.columns)




#######################################################
# 3. UI 구성
#######################################################
from pathlib import Path
from shiny import ui

app_ui = ui.TagList(
    ui.include_css(Path(__file__).parent / "styles.css"),
    # 외부 부트스트랩 아이콘 로드 (직접 link 태그로)
    ui.tags.link(
        rel="stylesheet",
        href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"
    ),

    ui.page_navbar(
        # [탭1] 1~11월 전기요금 분석
        ui.nav_panel(
            "1~11월 전기요금 분석",

        ui.layout_columns(
                        ui.input_date_range("기간", "기간 선택", start="2024-01-01", end="2024-01-15"),

                        ui.div(
                            ui.tags.div(
                                [
                                    # "분석 월 선택 :" 라벨과 드롭다운을 수평 정렬
                                    ui.tags.div(
                                        [
                                            ui.tags.label("분석 월 선택 :", style="margin: 0 3px 15px 0; font-size: 16px;"),
                                            ui.input_select(
                                                "pdf_month", "",
                                                choices=[str(m) for m in sorted(train["월"].unique())],
                                                selected="1",
                                                width="100px"
                                            )
                                        ],
                                        style="display: flex; align-items: center; margin-right: 10px; margin-top:10px;"
                                    ),

                                    # 다운로드 버튼
                                    ui.download_button(
                                        "download_pdf", "분석보고서 PDF 다운로드",
                                        class_="btn btn-primary",
                                        style="height: 40px; font-weight: bold; padding: 6px 20px;"
                                    )
                                ],
                                style="""
                                    display: flex;
                                    align-items: center;
                                    justify-content: flex-end;
                                    flex-wrap: nowrap;
                                    gap: 20px;
                                """
                            ),
                            style="width: 100%;"
                        ),

                        col_widths=[6, 6]
                    ),

            ui.layout_column_wrap(
                # 카드 1: 총 전력 사용량
            # 카드 1: 총 전력 사용량
                ui.card(
                    ui.card_body(
                        ui.tags.div(
                            ui.tags.i(
                                class_="bi bi-plug-fill",
                                style=(
                                    "flex:0 0 6rem; font-size:5rem; display:flex; align-items:center; justify-content:center;"
                                )
                            ),
                            ui.tags.div(
                                [
                                    ui.tags.span("총 전력 사용량", class_="fw-bold", style="font-size:1.1rem;"),
                                    ui.tags.span(ui.output_text("range_usage"), class_="fw-bold", style="font-size:1.3rem;"),
                                ],
                                style=(
                                    "margin-left:0.5rem; display:flex; flex-direction:column; justify-content:center;"
                                )
                            ),
                            style="display:flex; align-items:center; height:6rem;"
                        )
                    )
                ),

                # 카드 2: 총 전기요금
                ui.card(
                    ui.card_body(
                        ui.tags.div(
                            ui.tags.i(
                                class_="bi bi-cash-stack",
                                style=(
                                    "flex:0 0 6rem; font-size:5rem; "
                                    "display:flex; align-items:center; justify-content:center;"
                                )
                            ),
                            ui.tags.div(
                                [
                                    ui.tags.span("총 전기요금", class_="fw-bold", style="font-size:1.1rem;"),
                                    ui.tags.span(ui.output_text("range_cost_amount"), class_="fw-bold", style="font-size:1.3rem;"),
                                    ui.tags.span(ui.output_text("range_cost_unit"), style="font-size:0.95rem; color:#555; margin-top:4px;")
                                ],
                                style=(
                                    "margin-left:0.5rem; display:flex; flex-direction:column; justify-content:center;"
                                )
                            ),
                            style="display:flex; align-items:center; height:6rem;"
                        )
                    )
                ),

                # 카드 3: 일평균 전력 사용량
                ui.card(
                    ui.card_body(
                        ui.tags.div(
                            ui.tags.i(
                                class_="bi bi-bar-chart-line",
                                style=(
                                    "flex:0 0 6rem; font-size:5rem; display:flex; align-items:center; justify-content:center;"
                                )
                            ),
                            ui.tags.div(
                                [
                                    ui.tags.span("일평균 전력 사용량", class_="fw-bold", style="font-size:1.1rem;"),
                                    ui.tags.span(ui.output_text("avg_usage"), class_="fw-bold", style="font-size:1.3rem;"),
                                ],
                                style=(
                                    "margin-left:0.5rem; display:flex; flex-direction:column; justify-content:center;"
                                )
                            ),
                            style="display:flex; align-items:center; height:6rem;"
                        )
                    )
                ),

                # 카드: 일평균 전기요금 (단가 줄바꿈 + 비볼드)
                ui.card(
                    ui.card_body(
                        ui.tags.div(
                            ui.tags.i(
                                class_="bi bi-receipt",
                                style=(
                                    "flex:0 0 6rem; font-size:5rem; "
                                    "display:flex; align-items:center; justify-content:center;"
                                )
                            ),
                            ui.tags.div(
                                [
                                    ui.tags.span("일평균 전기요금", class_="fw-bold", style="font-size:1.1rem;"),
                                    ui.tags.span(ui.output_text("avg_cost_amount"), class_="fw-bold", style="font-size:1.3rem;"),
                                    ui.tags.span(ui.output_text("avg_cost_unit"), style="font-size:0.95rem; color:#555; margin-top:4px;")
                                ],
                                style=(
                                    "margin-left:0.5rem; display:flex; flex-direction:column; justify-content:center;"
                                )
                            ),
                            style="display:flex; align-items:center; height:6rem;"
                        )
                    )
                ),

                width=1/4,
                gap="20px"
            ),

            ui.hr(),

            ui.card(
                ui.card_header("요금 중심 마인드맵"),
                ui.layout_columns(
                    # ────── 좌측: Mermaid 마인드맵 ──────
                    ui.output_ui("my_image"),

                    # ────── 우측: 설명 ──────
                    ui.HTML("""
                    <div style="font-size: 16px; padding: 16px;">
                        <strong>전력 관계식</strong>
                        <ul>
                        <li><strong>피상전력 관계식:</strong> S² = P² + Q²  
                            피상전력(S)은 유효전력(P)과 무효전력(Q)의 벡터 합으로, 전기설비가 실제로 부담하는 전체 전력량을 나타냅니다.</li><br>
                        
                        <li><strong>역률(Power Factor):</strong> 역률 = P / S  
                            유효전력이 전체 피상전력에서 차지하는 비율로, 1에 가까울수록 전력 사용이 효율적입니다.  
                            역률이 낮을수록 무효전력 비중이 높아져, 산업용 설비에서는 벌금 또는 기본요금 증가로 이어질 수 있습니다.</li><br>
                        
                        <li><strong>지상과 진상은 동시에 성립하지 않음:</strong>  
                            지상무효전력은 유도성 부하에서, 진상무효전력은 용량성 부하에서 발생하므로  
                            특정 시점에는 두 중 하나만 발생합니다. 전류가 전압보다 늦을 때는 지상, 빠를 때는 진상 상태입니다.</li><br>

                        <li><strong>탄소배출량과 전기 요금의 관계:</strong>  
                            탄소배출량(tCO₂)은 전기요금에 직접적으로 영향을 주는 요인에 포함되지 않습니다. 그러나 전력사용량이 증가하면 전기요금뿐 아니라 탄소배출량도 함께 증가하기 때문에, 탄소배출량과 전기요금 간에 높은 상관계수가 나타나는 착시 현상이 발생할 수 있습니다.
                        </ul>
                    </div>
                    """),

                    col_widths=[6, 6]
                )
            ),
            ui.hr(),
    
           # B: Plotly 멀티라인 차트 추가
            ui.card(
                ui.card_header("[B] 1~11월 일자별 전력 사용량 추이 (6개 시간대 구분)"),

                # ✅ 먼저 선택 체크박스를 배치
                ui.input_checkbox_group(
                    "선택시간구간", "표시할 시간대 선택",
                    choices=[
                        "00:00–04:00", "04:01–08:00", "08:01–12:00",
                        "12:01–16:00", "16:01–20:00", "20:01–24:00"
                    ],
                    selected=[
                        "00:00–04:00", "04:01–08:00", "08:01–12:00",
                        "12:01–16:00", "16:01–20:00", "20:01–24:00"
                    ],
                    inline=True
                ),

                # ✅ 그 아래에 그래프 출력
                ui.output_plot("time_bin_plot")
            ),


            ui.layout_columns(
                ui.card(
                    ui.card_header("[B] 전력 사용량 및 전기요금 추이 (분석 단위별)"),
                    ui.layout_columns(
                        ui.input_select("선택월", "분석할 월", choices=["전체(1~11)"] + [str(i) for i in range(1, 12)], selected="전체(1~11)"),
                        ui.input_select("단위", "분석 단위", choices=["월", "주차", "일", "요일", "시간"], selected="월")
                    ),
                    ui.output_plot("usage_cost_drilldown")
                ),
                ui.card(
                    ui.card_header("[C] 선택 단위별 전력사용량 / 전기요금"),
                    ui.output_ui("summary_table"),
                    style="height: 300px; overflow-y: auto;"
                )
            ),
            ui.hr(),

            ui.layout_columns(
                ui.card(
                    ui.card_header("[D]월별 작업유형별 전력 사용량 (matplotlib)"),
                    ui.input_select(
                        "selected_month", "월 선택",
                        choices=[str(m) for m in sorted(train['월'].unique())],
                        selected="1"
                    ),
                    ui.output_image("usage_by_type_matplotlib")
                ),
                ui.card(
                    ui.card_header("[E] 선택 월의 작업유형별 분포"),
                    ui.input_select(
                        "selected_day", "요일 선택",
                        choices=["월", "화", "수", "목", "금", "토", "일"],
                        selected="월"
                    ),
                    ui.output_image("usage_by_dayofweek_matplotlib"),
                    ui.output_image("usage_by_hour_matplotlib")
                )
            ),
        ),
        

        # [탭2] 12월 예측 및 모델 근거
        ui.nav_panel(
            "12월 예측 및 모델 근거",
            # ▶ 버튼 + 라디오 버튼 그룹 정렬
            ui.div(
                ui.div(
                    ui.input_action_button("start_btn", "시작", class_="btn btn-primary", style="width:100px;"),
                    ui.input_action_button("stop_btn", "멈춤", class_="btn btn-primary", style="width:100px;"),
                    ui.input_action_button("reset_btn", "리셋", class_="btn btn-primary", style="width:100px;"),
                    ui.output_text("stream_status"),
                    class_="d-flex gap-2 align-items-center",
                    style="margin-right:100px;"  # 직접 설정
                ),
                ui.input_radio_buttons(
                    "time_unit", "시간 단위 선택",
                    choices=["일별", "시간대별", "분별(15분)"],
                    selected="분별(15분)",
                    inline=True
                ),
                class_="d-flex align-items-center"  # ▶ 세로 가운데 정렬
            ),
            # [A] 요약 카드 + 진행률 바
            ui.card(
                ui.card_header("[A] 12월 실시간 요금"),
                ui.output_ui("card_a"),
                style="margin-bottom: 16px;"
            ),

            # [C] 실시간 그래프 (전력 + 요금)
            ui.layout_columns(
                ui.card(
                    ui.card_header("[C] 12월 실시간 전력사용량 및 전기요금"),
                    ui.output_ui("latest_info_tags"),
                    ui.output_plot("live_plot", height="500px")
                ),
                ui.card(
                        ui.card_header("[B] 전 기간과 비교"),  # ✅ 제목만 header에!

                        ui.div(  # ✅ 카드 본문 좌측 상단에 select 위치
                            ui.tags.label("월 선택", class_="me-2"),
                            ui.input_select(
                                "비교월", None,
                                choices=[str(i) for i in range(1, 12)],
                                selected="11",
                                width="100px"
                            ),
                            class_="d-flex align-items-center",
                            style="margin-left: 10px; margin-bottom: 10px;"  # 여백 조절
                        ),
                        ui.output_ui("card_b"),  # 그래프 등 주요 콘텐츠
                        style="margin-bottom: 10px; padding-bottom: 0px;"
                    ),
                    col_widths=[7, 5]
            ),
        ),

        # page_navbar 옵션
        title="전기요금 분석 및 예측 대시보드",
        id="page"
    )
)



# 4. 서버 함수 정의
#####################################
#  TAB1 A
#####################################
def server(input, output, session):
# PDF 다운로드 기능 (기간별 요약 리포트)
    @output
    @render.download(
        filename=lambda: f"{input.pdf_month()}월_전력사용_보고서.pdf",  
        media_type="application/pdf"
    )
    def download_pdf():
        selected_month = int(input.pdf_month())
        return le_report(train, selected_month)
    

#####################################
#  TAB1 A
#####################################

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

        total_cost = train.loc[mask, '전기요금(원)'].sum()
        total_usage = train.loc[mask, '전력사용량(kWh)'].sum()

        if total_usage > 0:
            avg_unit_price = total_cost / total_usage
            return f"{total_cost:,.0f} 원\n(단가: {avg_unit_price:,.2f} 원/kWh)"
        else:
            return f"{total_cost:,.0f} 원\n(단가: 계산불가)"

    @output
    @render.text
    def range_cost_amount():
        start, end = input.기간()
        mask = (train['측정일시'].dt.date >= start) & (train['측정일시'].dt.date <= end)
        total_cost = train.loc[mask, '전기요금(원)'].sum()
        return f"{total_cost:,.0f} 원"

    @output
    @render.text
    def range_cost_unit():
        start, end = input.기간()
        mask = (train['측정일시'].dt.date >= start) & (train['측정일시'].dt.date <= end)
        total_cost = train.loc[mask, '전기요금(원)'].sum()
        total_usage = train.loc[mask, '전력사용량(kWh)'].sum()
        if total_usage > 0:
            unit_price = total_cost / total_usage
            return f"(단가: {unit_price:,.2f} 원/kWh)"
        else:
            return "(단가: 계산불가)"
        
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

        total_cost = train.loc[mask, '전기요금(원)'].sum()
        total_usage = train.loc[mask, '전력사용량(kWh)'].sum()

        if days > 0 and total_usage > 0:
            avg_cost_val = total_cost / days
            avg_unit_price = total_cost / total_usage
            return f"{avg_cost_val:,.0f} 원\n(단가: {avg_unit_price:,.2f} 원/kWh)"
        else:
            return f"{0:,.0f} 원\n(단가: 계산불가)"

    @output
    @render.text
    def avg_cost_amount():
        start, end = input.기간()
        mask = (train['측정일시'].dt.date >= start) & (train['측정일시'].dt.date <= end)
        days = (end - start).days + 1
        total_cost = train.loc[mask, '전기요금(원)'].sum()
        avg_cost_val = total_cost / days if days > 0 else 0
        return f"{avg_cost_val:,.0f} 원"

    @output
    @render.text
    def avg_cost_unit():
        start, end = input.기간()
        mask = (train['측정일시'].dt.date >= start) & (train['측정일시'].dt.date <= end)
        total_cost = train.loc[mask, '전기요금(원)'].sum()
        total_usage = train.loc[mask, '전력사용량(kWh)'].sum()
        if total_usage > 0:
            unit_price = total_cost / total_usage
            return f"(단가: {unit_price:,.2f} 원/kWh)"
        else:
            return "(단가: 계산불가)"



#####################################
#  TAB1 F - 1~11월 일자별 전력 사용량 추이 (6개 시간대 구분) 
#####################################
    @output
    @render.plot
    def time_bin_plot():
        # 1. 데이터 로드
        data_path = Path(__file__).parent / "data" / "train.csv"
        df = pd.read_csv(data_path, parse_dates=["측정일시"])
        df["date"] = df["측정일시"].dt.floor("D")
        df["day"] = df["측정일시"].dt.day
        df["minutes"] = df["측정일시"].dt.hour * 60 + df["측정일시"].dt.minute

        holidays = pd.to_datetime([
            "2024-01-01", "2024-01-10", "2024-01-11", "2024-01-12", "2024-01-13",
            "2024-03-01", "2024-05-05", "2024-05-06", "2024-05-15", "2024-06-06",
            "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18", "2024-09-19",
            "2024-10-03", "2024-10-09"
        ])
        df = df[~df["date"].isin(holidays)]

        # 2. 시간대 구간 지정
        bins = [0, 240, 480, 720, 960, 1200, 1440]
        labels = [
            "00:00–04:00", "04:01–08:00", "08:01–12:00",
            "12:01–16:00", "16:01–20:00", "20:01–24:00"
        ]
        df["time_bin"] = pd.cut(df["minutes"], bins=bins, labels=labels, right=True, include_lowest=True)

        # ✅ 3. 선택한 구간 필터링
        selected_bins = input.선택시간구간()
        if not selected_bins:
            selected_bins = []  # 없으면 비워두기
        df = df[df["time_bin"].isin(selected_bins)]

        if df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.axis("off")
            ax.text(0.5, 0.5, "선택된 시간 구간에 해당하는 데이터가 없습니다.", ha='center', va='center', fontsize=14)
            return fig

        # 4. 그룹핑 및 피벗
        grp = df.groupby(["day", "time_bin"], observed=True)["전력사용량(kWh)"].mean().reset_index()
        pivot = (
            grp
            .pivot(index="day", columns="time_bin", values="전력사용량(kWh)")
            .reindex(columns=labels)
            .reindex(index=range(1, 32))
            .fillna(0)
        )

        # 5. 그래프
        fig, ax = plt.subplots(figsize=(10, 5))
        color_map = {
            "00:00–04:00": "#B3D7FF",
            "04:01–08:00": "#FFEB99",
            "08:01–12:00": "#FF9999",
            "12:01–16:00": "#F9C0C0",
            "16:01–20:00": "#A1E3A1",
            "20:01–24:00": "#D1C4E9"
        }

        for label in selected_bins:
            if label in pivot.columns:
                ax.plot(pivot.index, pivot[label], label=label, color=color_map.get(label, "gray"), marker="o")

        ax.set_xlabel("일자")
        ax.set_ylabel("평균 전력 사용량 (kWh)")
        ax.set_xticks(range(1, 32))
        ax.legend(title="시간 구간")
        fig.tight_layout()

        return fig



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
                "Light_Load": "#B3D7FF",     # 밝은 파랑 (color-primary의 파스텔톤)
                "Medium_Load": "#FFEB99",    # 머스터드 옐로우 (color-accent 계열)
                "Maximum_Load": "#FF9999"    # 연한 빨강 (color-danger 계열)
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
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
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
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
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

            total_by_day = df.groupby('일')['전기요금(원)'].sum().sort_index()
            ax2.plot(total_by_day.index, total_by_day.values, color='red', marker='o', label='전기요금')

            ax1.set_xlabel("일")
            ax1.set_ylabel("전력 사용량 (kWh)")
            ax2.set_ylabel("전기요금 (원)")
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
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

            # ✅ 색상 변경
            ax1.bar(grouped['단위'], grouped['전력사용량(kWh)'],
                    color='#B3D7FF', label='전력 사용량')  # pastel blue
            ax2.plot(grouped['단위'], grouped['전기요금(원)'],
                    color='#ED1C24', marker='o', label='전기요금')  # strong red

            ax1.set_ylabel("전력 사용량 (kWh)")
            ax2.set_ylabel("전기요금 (원)")
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
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

        # ✅ 단가(원/kWh) 컬럼 추가
        grouped['단가(원/kWh)'] = grouped['전기요금(원)'] / grouped['전력사용량(kWh)']
        grouped['단가(원/kWh)'] = grouped['단가(원/kWh)'].replace([float('inf'), float('nan')], 0)

        # ✅ 숫자 포맷
        grouped['전력사용량(kWh)'] = grouped['전력사용량(kWh)'].apply(lambda x: f"{x:,.2f}")
        grouped['전기요금(원)'] = grouped['전기요금(원)'].apply(lambda x: f"{x:,.0f}")
        grouped['단가(원/kWh)'] = grouped['단가(원/kWh)'].apply(lambda x: f"{x:,.2f}")

        grouped = grouped[['구분', '전력사용량(kWh)', '전기요금(원)', '단가(원/kWh)']]

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

        # ① 피벗
        monthly = train.groupby(['월', '작업유형'])['전력사용량(kWh)'].sum().unstack().fillna(0)

        # ② 순서를 명시적으로 고정
        order = ['Light_Load', 'Medium_Load', 'Maximum_Load']
        monthly = monthly[order]  # 컬럼 순서 재정렬

        # ③ 색상 매핑도 순서에 맞게
        color_map = {
            'Light_Load': '#B3D7FF',
            'Medium_Load': '#FFEB99',
            'Maximum_Load': '#FF9999'
        }

        months = monthly.index.tolist()
        fig, ax = plt.subplots(figsize=(7, 6))
        bottom = np.zeros(len(months))

        for col in order:
            y = monthly[col].values
            for i, m in enumerate(months):
                month_total = monthly.iloc[i].sum()
                ratio = (y[i] / month_total * 100) if month_total > 0 else 0
                edgecolor = 'royalblue' if m == selected_month else 'gray'
                linewidth = 3 if m == selected_month else 1
                alpha = 1 if m == selected_month else 0.4
                ax.bar(
                    m, y[i],
                    bottom=bottom[i],
                    color=color_map[col],
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=alpha,
                    label=col if i == 0 else ""  # 범례 중복 방지
                )
                if y[i] > 0:
                    ax.text(
                        m, bottom[i] + y[i]/2,
                        f"{int(y[i]):,}\n({ratio:.1f}%)",
                        ha='center', va='center',
                        fontsize=8,
                        fontweight='normal',
                        color='black' if m == selected_month else 'dimgray'
                    )
            bottom += y

        ax.set_title('월별 작업유형별 전력 사용량 (Stacked Bar)')
        ax.set_xlabel('월')
        ax.set_ylabel('전력사용량 (kWh)')
        ax.set_xticks(months)
        ax.set_xticklabels([str(m) for m in months])
        ax.legend(title='작업유형')
        fig.tight_layout()

        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(tmpfile, format="png")
        plt.close(fig)
        tmpfile.close()

        return {"src": tmpfile.name, "alt": "월별 작업유형별 전력사용량 (matplotlib)"}


 

    # [D][E] 대체: 월별 작업유형별 전력 사용량 및 비율 (표)
    # 
    @output
    @render.image
    def usage_by_dayofweek_matplotlib():
        selected_month = int(input.selected_month())
        df_month = train[train['월'] == selected_month].copy()

        dow_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
        df_month['요일'] = df_month['측정일시'].dt.dayofweek.map(dow_map)

        # ✅ 고정 순서 및 색상 설정
        load_order = ["Light_Load", "Medium_Load", "Maximum_Load"]
        color_map = {
            "Light_Load": "#B3D7FF",
            "Medium_Load": "#FFEB99",
            "Maximum_Load": "#FF9999"
        }

        # ✅ pivot 생성 및 순서 고정
        pivot = df_month.pivot_table(
            index='요일', columns='작업유형', values='전력사용량(kWh)', aggfunc='sum', fill_value=0
        ).reindex(list(dow_map.values())).fillna(0)
        pivot = pivot.reindex(columns=load_order, fill_value=0)

        # ✅ 시각화
        fig, ax = plt.subplots(figsize=(7, 3))
        bottom = np.zeros(len(pivot))

        for col in load_order:
            ax.bar(pivot.index, pivot[col], bottom=bottom, color=color_map[col], label=col)
            for i, val in enumerate(pivot[col]):
                if val > 2500:
                    total = pivot.iloc[i].sum()
                    ratio = (val / total * 100) if total > 0 else 0
                    ax.text(
                        i, bottom[i] + val / 2,
                        f"{int(val):,}\n({ratio:.1f}%)",
                        ha='center', va='center', fontsize=8, color='black'
                    )
            bottom += pivot[col].values

        ax.set_title(f"{selected_month}월 요일별 작업유형별 전력 사용량")
        ax.set_xlabel("요일")
        ax.set_ylabel("전력사용량 (kWh)")  
        ax.legend(title='작업유형')
        plt.tight_layout()

        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(tmpfile, format="png")
        plt.close(fig)
        tmpfile.close()
        return {"src": tmpfile.name, "alt": "요일별 작업유형별 전력사용량"}


    @output
    @render.image
    def usage_by_hour_matplotlib():
        selected_month = int(input.selected_month())
        selected_day = input.selected_day()

        df_month = train[train['월'] == selected_month].copy()
        dow_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
        df_month['요일'] = df_month['측정일시'].dt.dayofweek.map(dow_map)
        df_month['시각'] = df_month['측정일시'].dt.hour
        df_day = df_month[df_month['요일'] == selected_day]

        load_order = ["Light_Load", "Medium_Load", "Maximum_Load"]
        color_map = {
            "Light_Load": "#B3D7FF",
            "Medium_Load": "#FFEB99",
            "Maximum_Load": "#FF9999"
        }

        pivot = df_day.pivot_table(
            index='시각', columns='작업유형', values='전력사용량(kWh)', aggfunc='sum', fill_value=0
        ).sort_index()
        pivot = pivot.reindex(columns=load_order, fill_value=0)

        fig, ax = plt.subplots(figsize=(7, 2.7))
        bottom = np.zeros(len(pivot))

        for col in load_order:
            ax.bar(pivot.index, pivot[col], bottom=bottom,
                color=color_map[col], label=col, width=0.8, alpha=0.85)
            bottom += pivot[col].values

        ax.set_title(f"{selected_month}월 {selected_day}요일 시간대별 작업유형별 전력 사용량")
        ax.set_xlabel("시각(0~23시)")
        ax.set_ylabel("전력사용량 (kWh)")
        ax.legend(title='작업유형')
        ax.set_xticks(range(0, 24))
        plt.tight_layout()

        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(tmpfile, format="png")
        plt.close(fig)
        tmpfile.close()
        return {"src": tmpfile.name, "alt": "시간대별 작업유형별 전력사용량"}






# ===============================
# TAB2 서버 로직
# ===============================
    streamer = reactive.Value(SimpleStreamer(streaming_df))
    is_streaming = reactive.Value(False)

    def transform_time(streaming_df, time_unit):
        streaming_df = streaming_df.copy()

        # 시간 단위별로 데이터를 변환하는 함수 (일별, 시간대별, 15분 단위)
        if time_unit == "일별":
            streaming_df["단위"] = streaming_df["측정일시"].dt.floor("D")
        elif time_unit == "시간대별":
            streaming_df["단위"] = streaming_df["측정일시"].dt.floor("H")
        elif time_unit == "분별(15분)":
            streaming_df["단위"] = streaming_df["측정일시"].dt.floor("15min")
        else:
            streaming_df["단위"] = streaming_df["측정일시"]

        return streaming_df

    # 스트리밍 시작, 멈춤, 리셋 버튼
    @reactive.Effect
    @reactive.event(input.start_btn)
    def start_stream():
        is_streaming.set(True)

    @reactive.Effect
    @reactive.event(input.stop_btn)
    def stop_stream():
        is_streaming.set(False)

    @reactive.Effect
    @reactive.event(input.reset_btn)
    def reset_stream():
        is_streaming.set(False)
        streamer.get().reset()
    # 3초마다 1줄씩 데이터 추가하는 스트리밍 로직
    @reactive.Effect
    def auto_stream():
        if not is_streaming.get():
            return
        reactive.invalidate_later(3)
        next_row = streamer.get().get_next(1)
        if next_row is None:
            is_streaming.set(False)

     # 스트리밍 상태 텍스트 출력 ("스트리밍 중" 또는 "중지")
    @output
    @render.text
    def stream_status():
        return "스트리밍 중" if is_streaming.get() else "중지"
    
    ################################
    # [A] 실시간 전기요금 추이 그래프 출력
    ################################
    @output
    @render.ui
    def card_a():
        return ui.div(
            ui.layout_columns(
                ui.card(
                    ui.tags.div(
                        ui.tags.span("총 예상 전기요금", class_="fw-bold", style="font-size:1.1rem;"),
                        ui.br(),
                        ui.tags.span(ui.output_text("estimated_total_cost"), style="font-size:1.2rem;"),
                        class_="text-center"
                    )
                ),
                ui.card(
                    ui.tags.div(
                        ui.tags.span("총 예상 전력사용량", class_="fw-bold", style="font-size:1.1rem;"),
                        ui.br(),
                        ui.tags.span(ui.output_text("estimated_total_usage"), style="font-size:1.2rem;"),
                        class_="text-center"
                    )
                ),
                ui.card(
                    ui.tags.div(
                        ui.tags.span("12월 누적 전기요금", class_="fw-bold", style="font-size:1.1rem;"),
                        ui.br(),
                        ui.tags.span(ui.output_text("realtime_total_cost"), style="font-size:1.2rem;"),
                        class_="text-center"
                    )
                ),
                ui.card(
                    ui.tags.div(
                        ui.tags.span("12월 누적 전력사용량", class_="fw-bold", style="font-size:1.1rem;"),
                        ui.br(),
                        ui.tags.span(ui.output_text("realtime_total_usage"), style="font-size:1.2rem;"),
                        class_="text-center"
                    )
                ),
                col_widths=[3, 3, 3, 3],
                gap="16px"
            ),
            ui.hr(),

            ui.tags.div(
                ui.tags.p("12월 진행률", class_="fw-bold", style="font-size:1.1rem;"),
                ui.output_ui("december_progress_bar"),
                style="margin-top: 10px;"
            )
        )


    @output
    @render.text
    def realtime_total_cost():
        reactive.invalidate_later(3)
        df = streamer.get().get_data()
        if df.empty:
            return "-"
        df["날짜"] = df["측정일시"].dt.date
        df_day = df.groupby("날짜")["예측_전기요금"].sum().reset_index(name="당일요금")
        df_day["누적요금"] = df_day["당일요금"].cumsum()
        today = df_day["날짜"].max()
        current_total = df_day[df_day["날짜"] == today]["누적요금"].values[0]
        return f"{current_total:,.0f} 원"

    @output
    @render.text
    def realtime_total_usage():
        reactive.invalidate_later(3)
        try:
            df = streamer.get().get_data()
            if df.empty:
                return "-"

            df["날짜"] = df["측정일시"].dt.date
            df_day = df.groupby("날짜")["예측_전력사용량"].sum().reset_index(name="당일사용량")
            df_day["누적사용량"] = df_day["당일사용량"].cumsum()
            today = df_day["날짜"].max()
            current_total = df_day[df_day["날짜"] == today]["누적사용량"].values[0]
            return f"{current_total:,.0f} kWh"

        except Exception:
            return "-"


    @output
    @render.text
    def estimated_total_cost():
        total_cost = streaming_df["예측_전기요금"].sum()
        return f"{total_cost:,.0f} 원"
    
    @output
    @render.ui
    def december_progress_bar():
        reactive.invalidate_later(3)
        df = streamer.get().get_data()
        if df.empty:
            return ui.div("진행률 없음", class_="text-muted")
        df["날짜"] = df["측정일시"].dt.date
        today = df["날짜"].max()
        start_date = pd.to_datetime("2024-12-01").date()
        total_days = 31
        days_elapsed = (today - start_date).days + 1
        progress_ratio = int((days_elapsed / total_days) * 100)
        return ui.div(
            ui.tags.progress(value=progress_ratio, max=100, style="width:100%"),
            f"{days_elapsed}일 경과 / 총 {total_days}일 ({progress_ratio}%)"
        )
    
    @output
    @render.text
    def estimated_total_usage():
        total_usage = streaming_df["예측_전력사용량"].sum()
        return f"{total_usage:,.0f} kWh"

    ################################
    # [B] 
    ################################
    @output
    @render.ui
    def card_b():
        return ui.output_image("compare_bar")
    
    @output
    @render.image
    def compare_bar():
        reactive.invalidate_later(3)
        
        streamer_obj = streamer.get()
        df_stream = streamer_obj.get_data()

        if df_stream.empty or df_stream["측정일시"].isna().all():
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.axis("off")
            ax.text(0.5, 0.5, "스트리밍 데이터 없음", ha='center', va='center', fontsize=11, color='gray')
            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(tmpfile.name, format="png")
            plt.close(fig)
            return {"src": tmpfile.name, "alt": "스트리밍 데이터 없음"}

        latest_day = df_stream["측정일시"].max()
        current_weekday = latest_day.strftime("%A")

        weekday_map = {
            "Monday": "월", "Tuesday": "화", "Wednesday": "수",
            "Thursday": "목", "Friday": "금", "Saturday": "토", "Sunday": "일"
        }
        요일 = weekday_map.get(current_weekday, "")
        비교월 = int(input.비교월())

        # 기준 데이터: 해당 월의 동일 요일만 필터링
        df_ref = train[
            (train["월"] == 비교월) &
            (train["측정일시"].dt.dayofweek == latest_day.dayofweek)
        ].copy()

        if df_ref.empty:
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.axis("off")
            ax.text(0.5, 0.5, f"{비교월}월 {요일} 데이터 없음", ha='center', va='center', fontsize=11, color='gray')
            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            plt.savefig(tmpfile.name, format="png")
            plt.close(fig)
            return {"src": tmpfile.name, "alt": f"{비교월}월 {요일} 데이터 없음"}

        # 하루 단위 총합 후 평균
        df_ref["날짜"] = df_ref["측정일시"].dt.date
        df_grouped = df_ref.groupby("날짜")[["전기요금(원)", "전력사용량(kWh)"]].sum()

        ref_cost = df_grouped["전기요금(원)"].mean()
        ref_usage = df_grouped["전력사용량(kWh)"].mean()

        # 실시간 누적
        df_stream["날짜"] = df_stream["측정일시"].dt.date
        stream_cost = df_stream["예측_전기요금"].sum()
        stream_usage = df_stream["예측_전력사용량"].sum()

        # 비율 계산
        cost_ratio = (stream_cost / ref_cost) * 100 if ref_cost > 0 else 0
        usage_ratio = (stream_usage / ref_usage) * 100 if ref_usage > 0 else 0

        # 그래프
        fig, ax = plt.subplots(2, 2, figsize=(6.5, 4.5), gridspec_kw={'height_ratios': [3, 1]})
        colors = ["#B3D7FF", "#FF9999"]
##########################################구교빈 시작

        # ─ 1행: 막대그래프
        label_기준 = f"기준({비교월}월 {요일}요일 평균)"
        label_실시간 = "실시간"
        bars1 = ax[0, 0].bar([label_기준, label_실시간], [ref_cost, stream_cost], color=colors)
        ax[0, 0].set_title("전기요금 비교")
        ax[0, 0].set_ylabel("원")
        ax[0, 0].set_ylim(0, max(ref_cost, stream_cost) * 1.2)
        for bar in bars1:
            height = bar.get_height()
            ax[0, 0].text(bar.get_x() + bar.get_width()/2, height,
                        f"{height:,.0f}원", ha='center', va='bottom', fontsize=9)

        
        bars2 = ax[0, 1].bar([label_기준, label_실시간], [ref_usage, stream_usage], color=colors)
        ax[0, 1].set_title("전력사용량 비교")
        ax[0, 1].set_ylabel("kWh")
        ax[0, 1].set_ylim(0, max(ref_usage, stream_usage) * 1.2)
        for bar in bars2:
            height = bar.get_height()
            ax[0, 1].text(bar.get_x() + bar.get_width()/2, height,
                        f"{height:,.2f}kWh", ha='center', va='bottom', fontsize=9)

        # ─ 2행: 텍스트 (각 subplot에 글자만 표시)
        ax[1, 0].axis("off")
        ax[1, 1].axis("off")
        ax[1, 0].text(0.5, 0.5, f"현재 요금은 기준의 {cost_ratio:.1f}%", ha='center', va='center', fontsize=10)
        ax[1, 1].text(0.5, 0.5, f"현재 사용량은 기준의 {usage_ratio:.1f}%", ha='center', va='center', fontsize=10)

        fig.suptitle(f"오늘은 {요일}요일입니다", fontsize=12, y=1.02)
        fig.tight_layout()

        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(tmpfile.name, format="png", bbox_inches='tight')
        plt.close(fig)

        return {"src": tmpfile.name, "alt": "요일 비교 막대그래프",
                "style": "width: 100%; max-width: 600px; height: 400px; display: block; margin-left: auto; margin-right: auto;"}
    
    # @output
    # @render.ui
    # def card_b():
    #     return ui.output_image("compare_bar")
    
    # @output
    # @render.image
    # def compare_bar():
    #     reactive.invalidate_later(3)
        
    #     streamer_obj = streamer.get()
    #     df_stream = streamer_obj.get_data()

    #     if df_stream.empty or df_stream["측정일시"].isna().all():
    #         fig, ax = plt.subplots(figsize=(4, 2))
    #         ax.axis("off")
    #         ax.text(0.5, 0.5, "스트리밍 데이터 없음", ha='center', va='center', fontsize=11, color='gray')
    #         tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    #         plt.savefig(tmpfile.name, format="png")
    #         plt.close(fig)
    #         return {"src": tmpfile.name, "alt": "스트리밍 데이터 없음"}

    #     latest_day = df_stream["측정일시"].max()
    #     current_weekday = latest_day.strftime("%A")

    #     weekday_map = {
    #         "Monday": "월", "Tuesday": "화", "Wednesday": "수",
    #         "Thursday": "목", "Friday": "금", "Saturday": "토", "Sunday": "일"
    #     }
    #     요일 = weekday_map.get(current_weekday, "")
    #     비교월 = int(input.비교월())

    #     # 기준 데이터: 해당 월의 동일 요일만 필터링
    #     df_ref = train[
    #         (train["월"] == 비교월) &
    #         (train["측정일시"].dt.dayofweek == latest_day.dayofweek)
    #     ].copy()

    #     if df_ref.empty:
    #         fig, ax = plt.subplots(figsize=(4, 2))
    #         ax.axis("off")
    #         ax.text(0.5, 0.5, f"{비교월}월 {요일} 데이터 없음", ha='center', va='center', fontsize=11, color='gray')
    #         tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    #         plt.savefig(tmpfile.name, format="png")
    #         plt.close(fig)
    #         return {"src": tmpfile.name, "alt": f"{비교월}월 {요일} 데이터 없음"}

    #     # 하루 단위 총합 후 평균
    #     df_ref["날짜"] = df_ref["측정일시"].dt.date
    #     df_grouped = df_ref.groupby("날짜")[["전기요금(원)", "전력사용량(kWh)"]].sum()

    #     ref_cost = df_grouped["전기요금(원)"].mean()
    #     ref_usage = df_grouped["전력사용량(kWh)"].mean()

    #     # 실시간 누적
    #     df_stream["날짜"] = df_stream["측정일시"].dt.date
    #     stream_cost = df_stream["예측_전기요금"].sum()
    #     stream_usage = df_stream["예측_전력사용량"].sum()

    #     # 비율 계산
    #     cost_ratio = (stream_cost / ref_cost) * 100 if ref_cost > 0 else 0
    #     usage_ratio = (stream_usage / ref_usage) * 100 if ref_usage > 0 else 0

    #     # 그래프
    #     fig, ax = plt.subplots(2, 2, figsize=(6.5, 4.5), gridspec_kw={'height_ratios': [3, 1]})
    #     colors = ["#B3D7FF", "#FF9999"]

    #     # ─ 1행: 막대그래프
    #     label_기준 = f"기준({비교월}월 {요일}요일 평균)"
    #     label_실시간 = "실시간"
    #     ax[0, 0].bar([label_기준, label_실시간], [ref_cost, stream_cost], color=colors)
    #     ax[0, 0].set_title("전기요금 비교")
    #     ax[0, 0].set_ylabel("원")

        
    #     ax[0, 1].bar([label_기준, label_실시간], [ref_usage, stream_usage], color=colors)
    #     ax[0, 1].set_title("전력사용량 비교")
    #     ax[0, 1].set_ylabel("kWh")

    #     # ─ 2행: 텍스트 (각 subplot에 글자만 표시)
    #     ax[1, 0].axis("off")
    #     ax[1, 1].axis("off")
    #     ax[1, 0].text(0.5, 0.5, f"현재 요금은 기준의 {cost_ratio:.1f}%", ha='center', va='center', fontsize=10)
    #     ax[1, 1].text(0.5, 0.5, f"현재 사용량은 기준의 {usage_ratio:.1f}%", ha='center', va='center', fontsize=10)

    #     fig.suptitle(f"오늘은 {요일}요일입니다", fontsize=12, y=1.02)
    #     fig.tight_layout()

    #     tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    #     plt.savefig(tmpfile.name, format="png", bbox_inches='tight')
    #     plt.close(fig)

    #     return {"src": tmpfile.name, "alt": "요일 비교 막대그래프",
    #             "style": "width: 100%; max-width: 600px; height: 400px; display: block; margin-left: auto; margin-right: auto;"}
             

    
    ################################
    # [C] 실시간 전기요금 추이 그래프 출력
    ################################
    @output
    @render.plot
    def live_plot():
        reactive.invalidate_later(3)
        df = streamer.get().get_data()
        fig, ax1 = plt.subplots(figsize=(10, 3))

        if df.empty:
            ax1.text(0.5, 0.5, "시작 버튼을 눌러 데이터를 로드해주세요", ha="center", va="center", fontsize=14, color="gray")
            ax1.axis("off")
            return fig

        time_unit = input.time_unit()
        df = transform_time(df, time_unit)

        grouped = df.groupby("단위")[["예측_전력사용량", "예측_전기요금"]].sum().reset_index()
        grouped = grouped.tail(20)

        if time_unit == "일별":
            formatter = DateFormatter("%Y-%m-%d")
            xticks = sorted(grouped["단위"].drop_duplicates())
        elif time_unit == "시간대별":
            formatter = DateFormatter("%Y-%m-%d %H시")
            xticks = sorted(grouped["단위"].drop_duplicates())
        elif time_unit == "분별(15분)":
            formatter = DateFormatter("%Y-%m-%d %H:%M")
            xticks = grouped["단위"]
        else:
            formatter = DateFormatter("%Y-%m-%d %H:%M")
            xticks = grouped["단위"]

        x = grouped["단위"]
        usage = grouped["예측_전력사용량"]
        cost = grouped["예측_전기요금"]

        # 색상 변경
        usage_color = "#B3D7FF"  # 파스텔 블루
        cost_color = "#ED1C24"   # LS 레드

        # 1. 예측 전력사용량 bar
        bar_width = 5 / (24 * 60)
        ax1.bar(x, usage, width=bar_width, color=usage_color, align="center", label="예측 전력사용량")
        ax1.set_ylabel("예측 전력사용량 (kWh)")
        ax1.set_ylim(0, usage.max() * 1.2)

        # 2. 예측 전기요금 line
        ax2 = ax1.twinx()
        ax2.plot(x, cost, color=cost_color, marker="o", linestyle="-", label="예측 전기요금")
        ax2.set_ylabel("예측 전기요금 (원)")
        ax2.set_ylim(0, cost.max() * 1.2)
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

        # 3. x축 설정
        ax1.set_xticks(xticks)
        ax1.xaxis.set_major_formatter(formatter)
        ax1.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        return fig




    # # 최신 행 기준 작업유형과 예측요금을 카드 형태로 출력
    @output
    @render.ui
    def latest_info_tags():
        reactive.invalidate_later(3)
        streaming_df = streamer.get().get_data()
        if streaming_df.empty:
            return ui.div("데이터 없음", class_="text-muted", style="font-size: 14px;")

        latest = streaming_df.iloc[-1]
        작업유형 = latest.get("작업유형", "N/A")
        요금 = latest.get("예측_전기요금", "N/A")
        사용량 = latest.get("예측_전력사용량", "N/A")

        return ui.div(
            ui.div(
                ui.tags.span("작업유형", class_="fw-bold", style="font-size:1rem;"),
                ui.br(),
                ui.tags.span(str(작업유형), style="font-size:1.15rem;"),
                style="padding: 10px; min-width: 160px;"
            ),
            ui.div(
                ui.tags.span("전기요금", class_="fw-bold", style="font-size:1rem;"),
                ui.br(),
                ui.tags.span(
                    f"{요금:,.0f} 원" if pd.notna(요금) else "N/A",
                    style="font-size:1.15rem;"
                ),
                style="padding: 10px; min-width: 160px;"
            ),
            ui.div(
                ui.tags.span("전력사용량", class_="fw-bold", style="font-size:1rem;"),
                ui.br(),
                ui.tags.span(
                    f"{사용량:,.2f} kWh" if pd.notna(사용량) else "N/A",
                    style="font-size:1.15rem;"
                ),
                style="padding: 10px; min-width: 160px;"
            ),
            style="display: flex; flex-direction: row; gap: 2rem;"
        )




    @output
    @render.ui
    def my_image():
        return ui.HTML("""
        <img src="img.png" style="max-width: 100%; height: auto;">
        """)




##############
# 5. 앱 실행
##############
app = App(app_ui, server, static_assets=STATIC_DIR)

