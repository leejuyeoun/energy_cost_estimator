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

# 공유 데이터셋
from shared import streaming_df, train              # 외부에서 불러오는 데이터셋

# ===============================
# 한글 폰트 설정, 마이너스 깨짐 방지
# ===============================
mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False


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


app_ui = ui.TagList(
    ui.include_css(Path(__file__).parent / "styles.css"),

    ui.page_navbar(
        # [탭1] 1~11월 전기요금 분석
        ui.nav_panel(
            "1~11월 전기요금 분석",

            ui.layout_columns(
                ui.div(
                    ui.input_date_range(
                        "기간", "기간 선택", start="2024-01-01", end="2024-11-30",
                        )
                    ),
                # 오른쪽 영역: 월 선택 + PDF 다운로드 버튼 나란히
                ui.div(
                    ui.div(
                        ui.input_select(
                            "pdf_month", "월 선택:",
                            choices=[str(m) for m in sorted(train["월"].unique())],
                            selected="1",           
                        ),
                        style="width: 80px; margin-right: 8px;"
                    ),
                    ui.download_button(
                        "download_pdf", "PDF 다운로드",
                        class_="btn btn-warning",
                        style="display: inline-block; margin-top: 25px; width: 140px;"
                    ),
                    style="display: flex; align-items: flex-end; gap: 5px;"
                        "justify-content: flex-end; width: 100%;"
                ),
                col_widths=[6, 6],
            ),

            ui.layout_column_wrap(
                ui.card("총 전력 사용량", ui.output_text("range_usage")),
                ui.card("총 전기요금", ui.output_text("range_cost")),
                ui.card("일평균 전력 사용량", ui.output_text("avg_usage")),
                ui.card("일평균 전기요금", ui.output_text("avg_cost")),
                width=1/4,
                gap="20px"
            ),
            ui.hr(),


            ui.card(
                ui.card_header("요금 중심 마인드맵"),
                ui.layout_columns(
                    # ────── 좌측: Mermaid 마인드맵 ──────
                    ui.HTML("""
                    <div style="padding: 16px;">
                        <div class="mermaid" style="font-size: 30px;">
                        flowchart TD
                            D["지상 무효전력량(kVarh)"] --> Q(("Q: 무효전력량(kVarh)"))
                            E["진상무효전력량(kVarh)"] --> Q

                            Q --> F[지상/진상 역률]
                            B(["P: 전력사용량(kWh)"]) --> F["지상/진상 역률(%)"]
                            F -.->|역률에 따른 추가 요금 부과|A[전기요금]

                            B -->|회귀계수: 107.25| A["전기요금(원)"]
                            B --> C["탄소배출량(tCO2)"]
                            C --> A
                        </div>
                    </div>

                    <script type="module">
                    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                    mermaid.initialize({ startOnLoad: true });
                    </script>
                    """),

                    # ────── 우측: 설명 ──────
                    ui.HTML("""
                    <div style="font-size: 16px; padding: 16px;">
                        <br><br><br>
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
                        </ul>
                    </div>
                    """),

                    col_widths=[6, 6]
                )
            ),
            ui.hr(),


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
            )
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
                    style="margin-right:100px;"  #  직접 설정
                ),
                ui.input_radio_buttons(
                    "time_unit", "시간 단위 선택",
                    choices=["일별", "시간대별", "분별(15분)"],
                    selected="분별(15분)",
                    inline=True
                ),
                class_="d-flex align-items-center"  # ▶ 세로 가운데 정렬
            ),


            ui.layout_columns(
                ui.card(
                    ui.card_header("[A] 12월 실시간 요금"),
                    ui.output_ui("card_a"),
                    # style="height:220px"
                ),
                ui.card(
                    ui.card_header("[B] 전 기간과 비교"),
                    ui.output_ui("card_b"),
                    # style="height:220px"
                ),
                col_widths=[8, 4]
            ),

            ui.layout_columns(
                ui.card(
                    ui.card_header("[C] 12월 실시간 전기요금 현황"),
                    
                    # ▶ 실시간 그래프 + 실시간 카드
                    ui.div(
                        # 좌측: 실시간 그래프
                        ui.div(ui.output_plot("live_plot", height="450px"), class_="flex-fill me-3"),
                        # 우측: 실시간 카드
                        ui.div(ui.output_ui("latest_info_cards"), class_="flex-fill", style="max-width: 200px; min-width: 180px;"),
                        class_="d-flex align-items-start"
                    ),
                )
            ),
        ),

        # page_navbar 옵션
        title="피카피카",
        id="page"
    )
)







# 4. 서버 함수 정의
#####################################
#  TAB1 A
#####################################성필 pdf시작###############################297~555

def server(input, output, session):
    # PDF 다운로드 기능 (기간별 요약 리포트)

    @output
    @render.download(
        filename=lambda: f"{input.pdf_month()}월_전력사용_보고서.pdf",
        media_type="application/pdf"
    )

    def download_pdf():

        # 한글 폰트 등록
        pdfmetrics.registerFont(TTFont('MalgunGothic', 'C:/Windows/Fonts/malgun.ttf'))
        mpl.rc('font', family='Malgun Gothic')
        mpl.rcParams['axes.unicode_minus'] = False

        # 1. 데이터 필터 및 요약값
        selected_month = int(input.pdf_month())
        df_until_month = train[train['월'] <= selected_month]
        df_month = train[train['월'] == selected_month]
        if df_month.empty:
            buf = io.BytesIO()
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(buf, pagesize=A4)
            c.setFont('MalgunGothic', 14)
            c.drawString(100, 750, f"{selected_month}월 데이터가 없습니다.")
            c.save()
            buf.seek(0)
            return buf

        # 누적값 (해당월까지)
        total_usage_cum = df_until_month["전력사용량(kWh)"].sum()
        total_cost_cum = df_until_month["전기요금(원)"].sum()
        days_cum = df_until_month['측정일시'].dt.date.nunique()
        avg_usage_cum = total_usage_cum / days_cum if days_cum > 0 else 0
        avg_cost_cum = total_cost_cum / days_cum if days_cum > 0 else 0
        peak_day = df_month.groupby(df_month['측정일시'].dt.day)["전기요금(원)"].sum().idxmax()

        # 2. 좌: 표 제목, 우: 누적 요약 표
        summary_title = f"2024년 누적 전력소비 정보 현황 (1월~{selected_month}월)"
        summary_data = [
            [f"2024년 1월~{selected_month}월 누적 전력 사용량 (kWh)", f"{total_usage_cum:,.2f}"],
            [f"2024년 1월~{selected_month}월 누적 전기요금 (원)", f"{total_cost_cum:,.0f}"],
            [f"2024년 1월~{selected_month}월 일평균 전력 사용량 (kWh)", f"{avg_usage_cum:,.2f}"],
            [f"2024년 1월~{selected_month}월 일평균 전기요금 (원)", f"{avg_cost_cum:,.0f}"],
            [f"{selected_month}월 최대 요금 발생일", f"{selected_month}월 {peak_day}일"],
        ]
        table = Table(summary_data, colWidths=[230,90], hAlign='LEFT')
        table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),        # 전체 좌측 정렬
            ('ALIGN', (1,0), (1,-1), 'RIGHT'),        # 값(숫자)만 우측 정렬
            ('FONTNAME', (0,0), (-1,-1), 'MalgunGothic'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ]))
        # 3. 다단 레이아웃(좌: 표 제목, 우: 표)
        styles = getSampleStyleSheet()
        styles['Title'].fontName = 'MalgunGothic'
        styles['BodyText'].fontName = 'MalgunGothic'
        
        custom_left = ParagraphStyle(
            name='Left',
            parent=styles['BodyText'],
            alignment=TA_LEFT
        )
        summary_par = Paragraph(f"<b>{summary_title}</b>", custom_left)
        datacell = [[summary_par, table]]
        multicol_table = Table(datacell, colWidths=[120, 160], hAlign='LEFT')
        multicol_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP')
        ]))

        # 4. 요일별 전력/요금 그래프
        dow_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
        df_month['요일'] = df_month['측정일시'].dt.dayofweek.map(dow_map)


        by_dow = df_month.groupby('요일').agg({'전력사용량(kWh)': 'sum', '전기요금(원)': 'mean'}).reindex(list(dow_map.values()))
        buf1 = io.BytesIO()
        fig1, ax1 = plt.subplots(figsize=(6.4, 3.2))
        by_dow["전력사용량(kWh)"].plot(kind='bar', ax=ax1, color='skyblue', width=0.7, label="전력사용량(kWh)")

        ax2 = ax1.twinx()
        # 👇 전기요금 "만원 단위"로 변환해서 그리기!
        by_dow["전기요금(만원)"] = by_dow["전기요금(원)"] / 10000
        ax2.plot(by_dow.index, by_dow["전기요금(만원)"], color='red', marker='o', linewidth=2, label="전기요금(만원)")

        ax1.set_xlabel("요일")
        ax1.set_ylabel("전력사용량(kWh)")
        ax1.set_xticklabels(by_dow.index, rotation=0)  #  요일 라벨 세우기

        # 👇 요금축을 "만원" 단위로 축약
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}만" if x else "0"))
        ax2.set_ylabel("전기요금(만원)")
        ax2.set_ylim(0, by_dow["전기요금(만원)"].max() * 1.5)   # Y축 20% 여유

        ax1.set_title(f"{selected_month}월 요일별 전력사용량 및 전기요금")
        # 우측 상단 범례
        ax2.legend(['전기요금(만원)'], loc='upper right', bbox_to_anchor=(1, 1), fontsize=9)
        ax1.legend(['전력사용량(kWh)'], loc='upper left', bbox_to_anchor=(0, 1), fontsize=9)
        fig1.tight_layout()

        plt.savefig(buf1, format='png', dpi=150)
        plt.close(fig1)
        buf1.seek(0)

        # 5. 전월대비 증감 해설(자동)
        if selected_month == 1:
            prev_diff_text = "전월(또는 전년 동월) 데이터가 없어 증감 비교가 불가능합니다."
        else:
            prev_month = selected_month - 1
            df_prev = train[train['월'] == prev_month]
            usage_prev = df_prev["전력사용량(kWh)"].sum()
            cost_prev = df_prev["전기요금(원)"].sum()
            # 증감치/증감률 (0 division 보호)
            diff_usage = total_usage_cum - train[train['월'] <= prev_month]["전력사용량(kWh)"].sum()
            diff_cost = total_cost_cum - train[train['월'] <= prev_month]["전기요금(원)"].sum()
            diff_usage_pct = (diff_usage / usage_prev * 100) if usage_prev else 0
            diff_cost_pct = (diff_cost / cost_prev * 100) if cost_prev else 0
            prev_diff_text = (
                f"전월 대비 전력사용량 {diff_usage:+,.0f} kWh ({diff_usage_pct:+.1f}%), "
                f"전기요금 {diff_cost:+,.0f}원 ({diff_cost_pct:+.1f}%)"
            )

        # 6. 월간 특징 및 해설 (확장 가능)
        특징_문구 = [
            prev_diff_text,
            f"최대 요금 발생일은 {selected_month}월 {peak_day}일입니다.",
            "화~목요일에 사용량이 많고, 토/일요일 사용량은 낮은 편입니다."
        ]

        # 7. 두 번째 페이지: 요일×작업유형별 전력사용량 (스택드 바)
        buf2 = io.BytesIO()
        load_order = ["Light_Load", "Medium_Load", "Maximum_Load"]
        color_map = {
            "Light_Load": "#B3D7FF",
            "Medium_Load": "#FFEB99",
            "Maximum_Load": "#FF9999"
        }
        pivot = df_month.pivot_table(
            index='요일', columns='작업유형', values='전력사용량(kWh)', aggfunc='sum', fill_value=0
        ).reindex(list(dow_map.values())).fillna(0)
        pivot = pivot.reindex(columns=load_order, fill_value=0)

        fig2, ax3 = plt.subplots(figsize=(6.2, 3.0))
        bottom = np.zeros(len(pivot))

        for col in load_order:
            values = pivot[col].values
            bars = ax3.bar(pivot.index, values, bottom=bottom, color=color_map[col], label=col)
            for i, val in enumerate(values):
                total = pivot.iloc[i].sum()
                pct = (val / total * 100) if total > 0 else 0
                # 값이 충분히 크고, 2000 이상일 때만 텍스트 표시
                if val > 2000:
                    ax3.text(
                        i, bottom[i] + val / 2,
                        f"{int(val):,}\n({pct:.1f}%)",
                        ha='center', va='center', fontsize=8, color='black'
                    )
            bottom += values

        ax3.set_ylabel("전력사용량(kWh)")   
        ax3.set_title(f"{selected_month}월 요일·작업유형별 전력사용량")
        ax3.set_xticklabels(pivot.index, rotation=0)

        # Legend에 전체비율 추가
        total = pivot.values.sum()
        labels_with_pct = []
        for col in load_order:
            col_sum = pivot[col].sum()
            pct = (col_sum / total) * 100 if total > 0 else 0
            labels_with_pct.append(f"{col} ({pct:.1f}%)")
        ax3.legend(labels_with_pct, loc='upper right', fontsize=9)

        fig2.tight_layout()
        plt.savefig(buf2, format='png', dpi=150)
        plt.close(fig2)
        buf2.seek(0)


        # 해설 자동 생성
        type_kor = {"Light_Load": "경부하", "Medium_Load": "중부하", "Maximum_Load": "최대부하"}
        most_type_per_day = pivot.idxmax(axis=1)
        most_type_kor = most_type_per_day.map(type_kor)

        # 1. 가장 흔한 패턴 찾기
        type_cnt = most_type_kor.value_counts()
        main_type = type_cnt.idxmax()
        main_days = [d for d, t in most_type_kor.items() if t == main_type]
        main_days_str = ", ".join(main_days)

        summary = [f"대부분 요일({main_days_str})은 '{main_type}'이 가장 높았습니다."]

        # 2. 예외(다른 부하가 높은 요일)
        exception_days = [d for d, t in most_type_kor.items() if t != main_type]
        if exception_days:
            exception_str = []
            for d in exception_days:
                kor = most_type_kor[d]
                exception_str.append(f"{d}요일은 '{kor}'이 가장 높음")
            summary.append(" / 예외: " + ", ".join(exception_str))

        # 3. 비정상적으로 치우친 요일(비율 60% 이상)
        threshold = 0.6
        insights = []
        for day in pivot.index:
            top_col = pivot.loc[day].idxmax()
            val = pivot.loc[day, top_col]
            total = pivot.loc[day].sum()
            ratio = val / total if total > 0 else 0
            if ratio >= threshold:
                kor = type_kor.get(top_col, top_col)
                insights.append(f"{day}요일은 '{kor}' 비중이 {ratio:.1%}로 매우 높음")
        if insights:
            summary.append(" / 특징: " + "; ".join(insights))

        explain_str = " ".join(summary)

        # 8. PDF 빌드
        out_buf = io.BytesIO()
        doc = SimpleDocTemplate(
            out_buf,
            leftMargin=30,   # 기본값은 72
            rightMargin=30,  # 기본값은 72
            topMargin=25,    # 기본값은 72
            bottomMargin=25  # 기본값은 72
        )                     
        elems = []
        styles = getSampleStyleSheet()
        styles['Title'].fontName = 'MalgunGothic'
        styles['BodyText'].fontName = 'MalgunGothic'

        # 제목 (짧게)
        elems.append(Paragraph(f"<b>2024년 {selected_month}월 청주공장 전기요금 분석 보고서</b>", styles["Title"]))
        elems.append(Spacer(1, 10))
        # 다단(좌: 표 제목, 우: 표)
        elems.append(multicol_table)
        elems.append(Spacer(1, 14))
        # 요일별 그래프
        elems.append(Paragraph("<b>■ 요일별 전력사용량 및 전기요금</b>", styles["BodyText"]))
        elems.append(Image(buf1, width=420, height=200))
        elems.append(Spacer(1, 12))
        # 월간 해설
        elems.append(Paragraph("<b>■ 월간 특징 및 해설</b>", styles["BodyText"]))
        for txt in 특징_문구:
            elems.append(Paragraph(f"- {txt}", styles["BodyText"]))
        elems.append(Spacer(1, 18))
        # 새 페이지
        elems.append(Paragraph("<b>■ 요일·작업유형별 전력사용량</b>", styles["BodyText"]))
        elems.append(Image(buf2, width=420, height=200))
        elems.append(Paragraph(f"<font size=9 color='gray'>{explain_str}</font>", styles["BodyText"]))
        doc.build(elems)
        out_buf.seek(0)
        return out_buf
            

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
            ax2.set_ylabel("전기요금 (만원)")
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/10000:.1f}만" if x else "0"))
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
            ax1.bar(grouped['단위'], grouped['전력사용량(kWh)'], color='skyblue', label='전력 사용량')
            ax2.plot(grouped['단위'], grouped['전기요금(원)'], color='red', marker='o', label='전기요금')
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

        #  고정 순서 및 색상 설정
        load_order = ["Light_Load", "Medium_Load", "Maximum_Load"]
        color_map = {
            "Light_Load": "#B3D7FF",
            "Medium_Load": "#FFEB99",
            "Maximum_Load": "#FF9999"
        }

        #  pivot 생성 및 순서 고정
        pivot = df_month.pivot_table(
            index='요일', columns='작업유형', values='전력사용량(kWh)', aggfunc='sum', fill_value=0
        ).reindex(list(dow_map.values())).fillna(0)
        pivot = pivot.reindex(columns=load_order, fill_value=0)

        #  시각화
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
    #
    # @reactive.Calc
    # def initial_estimated_total_cost():
    #     df = streamer.get().get_data()
    #     if df.empty:
    #         return "-"
    #     df["날짜"] = df["측정일시"].dt.date
    #     df_day = df.groupby("날짜")["예측_전기요금"].sum().reset_index(name="당일요금")
    #     df_day["누적요금"] = df_day["당일요금"].cumsum()

    #     # 초기 날짜 기준으로만 계산
    #     start_date = pd.to_datetime("2024-12-01").date()
    #     today = df_day["날짜"].max()
    #     days_elapsed = (today - start_date).days + 1
    #     if days_elapsed <= 0:
    #         return "-"
    #     current_total = df_day[df_day["날짜"] == today]["누적요금"].values[0]
    #     estimated_total = current_total * 31 / days_elapsed
    #     return f"{estimated_total:,.0f} 원"

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
                ui.div([
                    ui.tags.b("실시간 누적 요금"),
                    ui.br(),
                    ui.output_text("realtime_total_cost")
                ], style="margin-right: 30px; font-size: 18px;"),
                ui.div([
                    ui.tags.b("12월 총 예상 요금"),
                    ui.br(),
                    ui.output_text("estimated_total_cost")
                ], style="font-size: 18px;"),
            ),
            ui.hr(),
            ui.tags.div(
                ui.tags.b("12월 진행률"),
                ui.output_ui("december_progress_bar")
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
    
    # @output
    # @render.text
    # def estimated_total_cost():
    #     reactive.invalidate_later(3)
    #     df = streamer.get().get_data()
    #     if df.empty:
    #         return "-"
    #     df["날짜"] = df["측정일시"].dt.date
    #     df_day = df.groupby("날짜")["예측_전기요금"].sum().reset_index(name="당일요금")
    #     df_day["누적요금"] = df_day["당일요금"].cumsum()
    #     today = df_day["날짜"].max()
    #     start_date = pd.to_datetime("2024-12-01").date()
    #     days_elapsed = (today - start_date).days + 1
    #     current_total = df_day[df_day["날짜"] == today]["누적요금"].values[0]
    #     estimated_total = current_total * 31 / days_elapsed
    #     return f"{estimated_total:,.0f} 원"

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

    ################################
    # [B] 
    ################################













    
    ################################
    # [C] 실시간 전기요금 추이 그래프 출력
    ################################
    @output
    @render.plot
    def live_plot():
        reactive.invalidate_later(3)
        streaming_df = streamer.get().get_data()
        fig, ax = plt.subplots(figsize=(10, 3))  # 폭 10, 높이 4로 축소

        if streaming_df.empty:
            ax.text(0.5, 0.5, "시작 버튼을 눌러 데이터를 로드해주세요", ha="center", va="center", fontsize=14, color="gray")
            ax.axis("off")
            return fig

        time_unit = input.time_unit()
        streaming_df = transform_time(streaming_df, time_unit)
        grouped = streaming_df.groupby("단위")["예측_전기요금"].mean().reset_index()

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

        ax.plot(grouped["단위"], grouped["예측_전기요금"], marker="o", linestyle="-")
        ax.set_title("전기요금 실시간 추이")
        ax.set_xlabel("시간 단위")
        ax.set_ylabel("예측 전기요금(원)")
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="y", labelsize=10, pad=1.5)  # Y축 폰트 및 간격 조정

        fig.subplots_adjust(left=0.13, right=0.95, top=0.88, bottom=0.15)  # 여백 조절
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig


    # 최신 행 기준 작업유형과 예측요금을 카드 형태로 출력
    @output
    @render.ui
    def latest_info_cards():
        reactive.invalidate_later(3)
        streaming_df = streamer.get().get_data()
        if streaming_df.empty:
            return ui.div("데이터 없음", class_="text-muted", style="font-size: 14px;")

        latest = streaming_df.iloc[-1]
        작업유형 = latest.get("작업유형", "N/A")
        요금 = latest.get("예측_전기요금", "N/A")

        return ui.div(
            ui.card(
                ui.card_header("작업유형", style="font-size: 13px;"),
                ui.h4(str(작업유형), class_="fw-bold text-center", style="font-size: 20px;")
            ),
            ui.card(
                ui.card_header("전기요금", style="font-size: 13px;"),
                ui.h4(f"{요금:,.0f} 원" if pd.notna(요금) else "N/A", class_="fw-bold text-center", style="font-size: 20px;")
            ),
            style="display: flex; flex-direction: column; gap: 1rem;"
        )




##############
# 5. 앱 실행
##############
app = App(app_ui, server)


