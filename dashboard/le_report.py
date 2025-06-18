import io
import matplotlib.pyplot as plt
import matplotlib as mpl
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Image, Table, TableStyle)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.enums import TA_LEFT
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import os
from matplotlib import font_manager as mpl_font_manager
from pathlib import Path 
from reportlab.platypus import PageBreak


font_path = os.path.join(os.path.dirname(__file__), "www", "malgun.ttf")
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
def month_time_bin_plot(df, selected_month):
    

    df_month = df[df["월"] <= selected_month].copy()
    if df_month.empty:
        return None

    df_month["date"] = df_month["측정일시"].dt.floor("D")
    df_month["day"] = df_month["측정일시"].dt.day
    df_month["minutes"] = df_month["측정일시"].dt.hour * 60 + df_month["측정일시"].dt.minute

    # 시간대 구간
    bins = [0, 240, 480, 720, 960, 1200, 1440]
    labels = [
        "00:00–04:00","04:01–08:00","08:01–12:00",
        "12:01–16:00","16:01–20:00","20:01–24:00"
    ]
    df_month["time_bin"] = pd.cut(df_month["minutes"], bins=bins, labels=labels, right=True, include_lowest=True)
    grp = (
        df_month
        .groupby(["day","time_bin"], observed=True)["전력사용량(kWh)"]
        .mean()
        .reset_index()
    )
    pivot = (
        grp
        .pivot(index="day", columns="time_bin", values="전력사용량(kWh)")
        .reindex(columns=labels)
        .reindex(index=range(1,32))
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(8, 3.1))
    color_dict = {
        "00:00–04:00": "#B3D7FF",
        "04:01–08:00": "#FFEB99",
        "08:01–12:00": "#FF9999",
        "12:01–16:00": "#F9C0C0",
        "16:01–20:00": "#A1E3A1",
        "20:01–24:00": "#D1C4E9"
    }
    for lab in labels:
        ax.plot(pivot.index, pivot[lab], marker="o", label=lab, color=color_dict[lab])
    ax.set_xlabel("일자")
    ax.set_ylabel("평균 전력 사용량 (kWh)")
    ax.set_xticks(range(1, 32))
    ax.set_title(f"{selected_month}월 일자별 시간대(4시간)별 전력 사용량 추이")
    ax.legend(title="시간 구간", loc="best", fontsize=7)
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)

     # ==== 자동 해설문구 생성 ====
    timebin_mean = pivot.mean()
    peak_bin = timebin_mean.idxmax()
    peak_val = timebin_mean.max()

    # 누적에서 가장 많이 쓴 날
    pivot["합계"] = pivot.sum(axis=1)
    peak_day = pivot["합계"].idxmax()
    peak_day_val = pivot.loc[peak_day, "합계"]

    time_diff = timebin_mean.max() - timebin_mean.min()
    min_bin = timebin_mean.idxmin()
    min_val = timebin_mean.min()

    explain_lines = [
        f"• 누적 기준 <b>'{peak_bin}'</b> 시간대에 전력 사용량이 평균적으로 가장 많음. (평균 <b>{peak_val:,.1f}kWh</b>)",
        f"• 사용량이 가장 많은 날은 <b>{peak_day}일</b>이며, 총 <b>{peak_day_val:,.1f}kWh</b>가 사용됨.",
        f"• 시간대별 최대/최소 평균값 차이는 <b>{time_diff:,.1f}kWh</b> (<b>{peak_bin}</b>: {peak_val:,.1f}kWh, <b>{min_bin}</b>: {min_val:,.1f}kWh)",
        "• 특정 시간대의 집중 패턴, 일별 피크 발생일 등 설비 운영/에너지 관리 인사이트 도출에 활용할 수 있음."
    ]
    explain_str = "<br/>".join(explain_lines)

    return buf, explain_str

def le_report(train, selected_month, font_path=font_path):


    # 한글 폰트 등록
    pdfmetrics.registerFont(TTFont('MalgunGothic', font_path))
    mpl.rc('font', family='Malgun Gothic')
    mpl.rcParams['axes.unicode_minus'] = False

    # 3. 다단 레이아웃(좌: 표 제목, 우: 표)
    styles = getSampleStyleSheet()
    styles['Title'].fontName = 'MalgunGothic'
    styles['BodyText'].fontName = 'MalgunGothic'
    custom_left = ParagraphStyle(
        name='Left', parent=styles['BodyText'], alignment=TA_LEFT
    )

    # 1. 데이터 필터 및 요약값
    selected_month = int(selected_month)
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

# 1. 누적 요약 현황 표(좌: 제목, 우: 표)
    total_usage_cum = df_until_month["전력사용량(kWh)"].sum()
    total_cost_cum = df_until_month["전기요금(원)"].sum()
    days_cum = df_until_month['측정일시'].dt.date.nunique()
    avg_usage_cum = total_usage_cum / days_cum if days_cum > 0 else 0
    avg_cost_cum = total_cost_cum / days_cum if days_cum > 0 else 0
    peak_day = df_month.groupby(df_month['측정일시'].dt.day)["전기요금(원)"].sum().idxmax()

    # 2. 좌: 표 제목, 우: 누적 요약 표
    summary_title = f"■ 2024년 누적 전력소비 정보 현황 (1월~{selected_month}월)"
    summary_par = Paragraph(f"<b>{summary_title}</b>", styles['BodyText'])  # styles['Title']도 가능
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



    # ====== 월별 비교 표+그래프(좌우 2단, 한글화) ======
    prev_month = selected_month - 1
    df_prev = train[train['월'] == prev_month]
    usage_prev = df_prev["전력사용량(kWh)"].sum()
    cost_prev = df_prev["전기요금(원)"].sum()
    unit_prev = cost_prev / usage_prev if usage_prev > 0 else 0
    max_load_prev = (df_prev["작업유형"] == "Maximum_Load").sum() / len(df_prev) if len(df_prev) > 0 else 0

    usage_now = df_month["전력사용량(kWh)"].sum()
    cost_now = df_month["전기요금(원)"].sum()
    unit_now = cost_now / usage_now if usage_now > 0 else 0
    max_load_now = (df_month["작업유형"] == "Maximum_Load").sum() / len(df_month) if len(df_month) > 0 else 0

    cmp_table_data = [
        ["구분", f"{prev_month}월", f"{selected_month}월", "증감"],
        ["전력사용량(kWh)", f"{usage_prev:,.0f}", f"{usage_now:,.0f}", f"{usage_now-usage_prev:+,.0f}"],
        ["전기요금(원)", f"{cost_prev:,.0f}", f"{cost_now:,.0f}", f"{cost_now-cost_prev:+,.0f}"],
        ["단가(원/kWh)", f"{unit_prev:,.2f}", f"{unit_now:,.2f}", f"{unit_now-unit_prev:+.2f}"],
        ["과부하 비율(%)", f"{max_load_prev*100:.1f}", f"{max_load_now*100:.1f}", f"{(max_load_now-max_load_prev)*100:+.1f}"]
    ]
    cmp_table = Table(cmp_table_data, colWidths=[95, 60, 60, 65])
    cmp_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'MalgunGothic'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.3, colors.grey)
    ]))

# ---- 월별 비교 그래프 (우측, 내부 수치) ----
    buf_cmp = io.BytesIO()
    bar_colors = ['#C0C0C0', '#4472C4']
    fig_cmp, ax_cmp = plt.subplots(figsize=(4.5, 3.1))
    bars = ax_cmp.bar([f"{prev_month}월", f"{selected_month}월"], [usage_prev, usage_now], color=bar_colors, width=0.6)
    ax_cmp.set_ylabel("전력사용량 (kWh)")
    ax_cmp.set_title("월별 전력사용량 비교")
    for bar in bars:
        height = bar.get_height()
        # 내부 중앙에 표기
        ax_cmp.text(bar.get_x() + bar.get_width()/2, height*0.6, f"{int(height):,}", 
                    ha='center', va='center', fontsize=11, color='black', weight='bold')
    fig_cmp.tight_layout()
    plt.savefig(buf_cmp, format='png', dpi=150)
    plt.close(fig_cmp)
    buf_cmp.seek(0)
    # 좌: 표, 우: 그래프
    cmp_table_multicol = Table(
        [[cmp_table, Image(buf_cmp, width=200, height=140)]],
        colWidths=[325, 230]
    )
    cmp_table_multicol.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4)
    ]))


    # ====== 요일별 전력/요금 그래프 ======
    dow_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
    df_month['요일'] = df_month['측정일시'].dt.dayofweek.map(dow_map)
    by_dow = df_month.groupby('요일').agg({'전력사용량(kWh)': 'sum', '전기요금(원)': 'mean'}).reindex(list(dow_map.values()))
    weekday = ["월", "화", "수", "목", "금"]
    weekend = ["토", "일"]
    mean_weekday = by_dow.loc[weekday, "전력사용량(kWh)"].mean()
    mean_weekend = by_dow.loc[weekend, "전력사용량(kWh)"].mean()
    max_day = by_dow["전력사용량(kWh)"].idxmax()
    max_val = by_dow["전력사용량(kWh)"].max()
    min_day = by_dow["전력사용량(kWh)"].idxmin()
    min_val = by_dow["전력사용량(kWh)"].min()
    delta = max_val - min_val
    by_dow["단가"] = by_dow["전기요금(원)"] / by_dow["전력사용량(kWh)"]
    unit_day = by_dow["단가"].idxmax()
    unit_val = by_dow["단가"].max()
    dow_desc = [
        f"평일 평균 사용량은 {mean_weekday:,.0f}kWh, 주말 평균은 {mean_weekend:,.0f}kWh.",
        f"전력사용량이 가장 많은 요일은 {max_day}요일({max_val:,.0f}kWh), 가장 적은 요일은 {min_day}요일({min_val:,.0f}kWh).",
        f"요일별 최대/최소 사용량 차이는 {delta:,.0f}kWh.",
        f"가장 높은 단가의 요일은 {unit_day}요일({unit_val:,.0f}원/kWh)."
]

    buf1 = io.BytesIO()
    fig1, ax1 = plt.subplots(figsize=(7, 3.2))
    by_dow["전력사용량(kWh)"].plot(kind='bar', ax=ax1, color='skyblue', width=0.7, label="전력사용량(kWh)")
    ax2 = ax1.twinx()
    by_dow["전기요금(만원)"] = by_dow["전기요금(원)"] / 10000
    ax2.plot(by_dow.index, by_dow["전기요금(만원)"], color='red', marker='o', linewidth=2, label="전기요금(만원)")
    ax1.set_xlabel("요일")
    ax1.set_ylabel("전력사용량(kWh)")
    ax1.set_xticklabels(by_dow.index, rotation=0)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}만" if x else "0"))
    ax2.set_ylabel("전기요금(만원)")
    ax2.set_ylim(0, by_dow["전기요금(만원)"].max() * 1.5)
    ax1.set_title(f"{selected_month}월 요일별 전력사용량 및 전기요금")
    ax2.legend(['전기요금(만원)'], loc='upper right', bbox_to_anchor=(1, 1), fontsize=9)
    ax1.legend(['전력사용량(kWh)'], loc='upper left', bbox_to_anchor=(0, 1), fontsize=9)
    fig1.tight_layout()
    plt.savefig(buf1, format='png', dpi=150)
    plt.close(fig1)
    buf1.seek(0)


    # ====== 요일별 해설 ======

    # ====== 요일×작업유형별 전력사용량 (스택드바, 한글화) ======
    type_map = {"Light_Load": "경부하", "Medium_Load": "중부하", "Maximum_Load": "과부하"}
    pivot = df_month.pivot_table(
        index='요일', columns='작업유형', values='전력사용량(kWh)', aggfunc='sum', fill_value=0
    ).reindex(list(dow_map.values())).fillna(0)
    pivot.columns = [type_map.get(x, x) for x in pivot.columns]
    load_order = ["경부하", "중부하", "과부하"]
    color_map = {
        "경부하": "#B3D7FF",
        "중부하": "#FFEB99",
        "과부하": "#FF9999"
    }
    buf2 = io.BytesIO()
    fig2, ax3 = plt.subplots(figsize=(7, 3.1))
    bottom = np.zeros(len(pivot))
    for col in load_order:
        values = pivot[col].values
        bars = ax3.bar(pivot.index, values, bottom=bottom, color=color_map[col], label=col)
        for i, val in enumerate(values):
            total = pivot.iloc[i].sum()
            pct = (val / total * 100) if total > 0 else 0
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

    # ====== 해설 자동 생성 ======
    most_type_per_day = pivot.idxmax(axis=1)
    most_type_kor = most_type_per_day
    type_cnt = most_type_kor.value_counts()
    main_type = type_cnt.idxmax()
    main_days = [d for d, t in most_type_kor.items() if t == main_type]
    main_days_str = ", ".join(main_days)
    exception_days = [d for d, t in most_type_kor.items() if t != main_type]
    insights = []
    threshold = 0.6
    for day in pivot.index:
        top_col = pivot.loc[day].idxmax()
        val = pivot.loc[day, top_col]
        total = pivot.loc[day].sum()
        ratio = val / total if total > 0 else 0
        if ratio >= threshold:
            insights.append(f"{day}요일은 '{top_col}' 비중이 {ratio:.1%}로 매우 높았음.")

    lines = []
    lines.append(f"대부분 요일({main_days_str})은 '{main_type}'가 가장 높았음.")
    if exception_days:
        exception_str = []
        for d in exception_days:
            kor = most_type_kor[d]
            exception_str.append(f"{d}요일은 '{kor}'가 가장 높았음.")
        lines.append("예외: " + ", ".join(exception_str))
    if insights:
        lines.append("특징: " + "; ".join(insights))
    explain_str = "<br/>".join(lines)


# ====== PDF 빌드 ======
    out_buf = io.BytesIO()
    doc = SimpleDocTemplate(
        out_buf,
        leftMargin=15, rightMargin=15, topMargin=40, bottomMargin=40
    )
    elems = []
    elems.append(Paragraph(f"<b>2024년 {selected_month}월 청주공장 전기요금 분석 보고서</b>", styles["Title"]))
    elems.append(Spacer(1, 10))

    # 누적 현황 제목 + 표 (multicol_table 제거)
    elems.append(summary_par)
    elems.append(Spacer(1, 4))
    elems.append(table)
    elems.append(Spacer(1, 12))

    # 월별 비교 표+그래프 2단
    elems.append(Paragraph("<b>■ 월별 전력 사용량 및 전기요금</b>", styles["BodyText"]))
    elems.append(cmp_table_multicol)
    elems.append(Spacer(1, 18))

    # 요일별 그래프 → 해설문구
    elems.append(Paragraph("<b>■ 요일별 전력사용량 및 전기요금</b>", styles["BodyText"]))
    elems.append(Image(buf1, width=430, height=180))
    elems.append(Spacer(1, 6))
    for txt in dow_desc:
        elems.append(Paragraph(f"- {txt}", styles["BodyText"]))

        
    # 무조건 다음 페이지부터!
    elems.append(PageBreak())


    # 요일·작업유형별 그래프 → 해설문구
    elems.append(Spacer(1, 12))
    elems.append(Paragraph("<b>■ 요일·작업유형별 전력사용량</b>", styles["BodyText"]))
    elems.append(Image(buf2, width=430, height=180))
    
    elems.append(Paragraph(f"<font size=9 color='black'>{explain_str}</font>", styles["BodyText"]))

    #  여기 바로 아래에 추가!
    timebin_buf, timebin_explain = month_time_bin_plot(train, selected_month)
    if timebin_buf is not None:
        elems.append(Spacer(1, 14))
        elems.append(Paragraph(f"<b>■ 누적(1~{selected_month}월) 일자별 시간대(4시간)별 전력 사용량 추이</b>", styles["BodyText"]))
        elems.append(Image(timebin_buf, width=430, height=160))
        elems.append(Paragraph(timebin_explain, styles["BodyText"]))

        
    doc.build(elems)
    out_buf.seek(0)
    return out_buf