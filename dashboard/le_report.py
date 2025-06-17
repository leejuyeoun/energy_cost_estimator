import os
font_path = os.path.join(os.path.dirname(__file__), "www", "malgun.ttf")
def le_report(train, selected_month, font_path=font_path):
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

    # 한글 폰트 등록
    pdfmetrics.registerFont(TTFont('MalgunGothic', font_path))
    mpl.rc('font', family='Malgun Gothic')
    mpl.rcParams['axes.unicode_minus'] = False

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


