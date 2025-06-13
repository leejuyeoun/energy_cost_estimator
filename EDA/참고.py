# ================================
# 📦 1. Import
# ================================
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
from shared import RealTimeStreamer, StreamAccumulator
from shared import sensor_labels, static_df, streaming_df, spec_df_all, get_weather
import numpy as np
from datetime import datetime, timedelta
import matplotlib as mpl
import joblib
import warnings
from plotly.graph_objs import Figure, Scatter
import plotly.graph_objs as go
from shinywidgets import render_widget
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import pickle
import shap
from sklearn.impute import SimpleImputer
from collections import Counter
from pathlib import Path
import matplotlib.font_manager as fm
from sklearn.pipeline import Pipeline
import matplotlib.ticker as mticker
# 📍 server 구성 위쪽 (전역)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "www")
selected_log_index = reactive.Value(None)
app_dir = Path(__file__).parent

# # 모델 불러오기
# model_pipe = joblib.load(Path(__file__).parent / "www" / "model_pipe.pkl")
# model = model_pipe.named_steps["classifier"]
# shap_explainer = shap.TreeExplainer(model)


model_pipe = joblib.load(Path(__file__).parent / "www" / "model_pipe.pkl")  # ✅ pipeline 전체
model = model_pipe.named_steps["classifier"]  # classifier 추출
shap_explainer = shap.TreeExplainer(model)   # SHAP explainer 생성


# # model_pipe가 dict이면 내부 pipeline에서 classifier 꺼냄
# if isinstance(model_pipe, dict):
#     pipeline = model_pipe["pipeline"]
# else:
#     pipeline = model_pipe

# # classifier를 SHAP explainer에 전달

# model_pipeline = joblib.load("./www/model_pipeline.pkl")  # pipeline이 저장된 경로
# shap_explainer = shap.TreeExplainer(model_pipeline.named_steps["classifier"])

# if isinstance(model_pipe, dict):
#     print("📦 model_pipe 키 목록:", model_pipe.keys())


model = joblib.load(Path(__file__).parent / "www" / "model_xgb.pkl")

model_iso_path = Path(__file__).parent / "www" / "model_iso.pkl"
with open(model_iso_path, "rb") as f:
    model_iso = pickle.load(f)

# model = joblib.load(Path(__file__).parent / "www" / "model.pkl")
# 앱 디렉터리 설정

# 한글 폰트 설정: MaruBuri-Regular.ttf 직접 로드
font_path = app_dir / "MaruBuri-Regular.ttf"
font_prop = fm.FontProperties(fname=font_path)

warnings.filterwarnings('ignore')

mold_codes = ['ALL','8412', '8573', '8600', '8722', '8917']

plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

selected_cols = [
    'mold_code',
    'registration_time',
    'cast_pressure',         # 주조 압력
    'low_section_speed',     # 저속 구간 속도
    'biscuit_thickness',      # 비스킷 두께
    'molten_temp',           # 용탕 온도
    'high_section_speed',    # 고속 구간 속도
    'physical_strength',
    'facility_operation_cycleTime',
    'production_cycletime',
    'count',
    'Coolant_temperature',
    'sleeve_temperature',
    'molten_volume',
    'upper_mold_temp1',
    'EMS_operation_time',
]
df_selected = streaming_df[selected_cols].reset_index(drop=True)


cached_weather = {"time": None, "data": None}

def get_cached_weather(registration_time_str):
    global cached_weather
    try:
        # 문자열을 datetime으로 변환
        reg_time = pd.to_datetime(registration_time_str)

        # 캐시된 시간이 없거나 1시간 이상 차이 나면 업데이트
        if cached_weather["time"] is None or abs(reg_time - cached_weather["time"]) > timedelta(hours=1):
            new_weather = get_weather()  # 실제 날씨 API 호출
            cached_weather = {
                "time": reg_time,
                "data": new_weather
            }
        return cached_weather["data"]
    except Exception as e:
        print(f"[❌ get_cached_weather 오류] {e}")
        return "날씨 정보 없음"

# ================================
# 🖼️ 2. UI 정의
# ================================

app_ui = ui.page_fluid(
            ui.output_ui("dynamic_ui")  # 전체 UI는 서버에서 조건에 따라 출력
        )

# ================================
# ⚙️ 3. 서버 로직
# ================================
def server(input, output, session):
    # 초기 상태
    streamer = reactive.Value(RealTimeStreamer())
    accumulator = reactive.value(StreamAccumulator(static_df))
    current_data = reactive.Value(pd.DataFrame())
    is_streaming = reactive.Value(False)

    selected_log_time = reactive.Value(None)

    prediction_table_logs = reactive.Value([])  # TAB 3. [B] 로그 테이블용
    anomaly_detail_logs = reactive.Value([])
    # 로그인 상태 저장
    login_status = reactive.Value(False)
    
    alert_logs = reactive.Value([])  # 실시간 경고 누적
    anomaly_counter = reactive.Value(Counter())

    log_button_clicks = reactive.Value({})
    delete_clicks = reactive.Value({})
    # ================================
    # 스트리밍 제어
    # ================================
    @reactive.effect
    @reactive.event(input.start)
    def on_start():
        is_streaming.set(True)

    @reactive.effect
    @reactive.event(input.pause)
    def on_pause():
        is_streaming.set(False)

    @reactive.effect
    @reactive.event(input.reset)
    def on_reset():
        streamer.get().reset_stream()
        current_data.set(pd.DataFrame())
        is_streaming.set(False)

    @reactive.effect
    def stream_data():
        try:
            if not is_streaming.get():
                return
            reactive.invalidate_later(1
                                      )
            s = streamer.get()
            next_batch = s.get_next_batch(1)
            if next_batch is not None:
                current_data.set(s.get_current_data())

                # ✅ 누적기록 클래스도 업데이트 (전체 컬럼)
                accum = accumulator.get()
                accum.accumulate(next_batch)  # 내부 상태 갱신
            else:
                is_streaming.set(False)
        except Exception as e:
            print("⛔ 오류 발생:", e)
            is_streaming.set(False)


    # ================================
    # TAB 1: 실시간 공정 모니터링   Process Monitoring
    # ================================

    # ▶ 데이터 스트리밍 진행률을 퍼센트로 표시합니다.
    @output
    @render.ui
    def stream_status():
        try:
            status = "🟢 스트리밍 중" if is_streaming.get() else "🔴 정지됨"
            return status
        except Exception as e:
            return f"에러: {str(e)}"
        
    
    @output
    @render.ui
    def anomaly_alerts():
        try:
            df = current_data.get()
            if df.empty:
                return ui.div("데이터가 없습니다. 작업을 시작해주세요.", class_="text-muted")

            latest = df.iloc[-1].copy()

            # ================================
            # 🔹 모델 기반 예측 수행
            # ================================
            input_row = latest.drop(['passorfail', 'registration_time'], errors='ignore')
            required_features = model_iso.feature_names_in_

            for col in required_features:
                if col not in input_row:
                    input_row[col] = 0

            X_input = pd.DataFrame([input_row[required_features]])
            score = model_iso.decision_function(X_input)[0]

            # ✅ 판정 기준 설정
            score_thresholds = {
                "심각": -0.07342,
                "경도": -0.04480
            }

            # ✅ 이상 판단
            if score <= score_thresholds["심각"]:
                anomaly_score = "심각"
            elif score <= score_thresholds["경도"]:
                anomaly_score = "경도"
            else:
                anomaly_score = "정상"

            icon = "✅" if anomaly_score == "정상" else "❌"
            color_class = "alert alert-danger" if anomaly_score in ["경도", "심각"] else "alert alert-success"

            # 시각 정리
            reg_time = latest.get('registration_time')
            try:
                reg_time = pd.to_datetime(reg_time).strftime("%Y-%m-%d %H:%M:%S")
            except:
                reg_time = str(reg_time)

            return ui.div(
                ui.div(
                    ui.h6("실시간 공정 이상 탐지"),
                    ui.h4(f"{icon} {anomaly_score}", class_="fw-bold"),
                    ui.input_action_button("goto_2page", "이상탐지 확인하기", class_="btn btn-sm btn-outline-primary"),
                    class_=f"{color_class} p-3 rounded"
                )
            )

        except Exception as e:
            return ui.div(f"오류: {str(e)}", class_="text-danger")
        
        
    @output
    @render.ui
    def current_prediction2():
        try:
            df = current_data.get()
            if df.empty:
                return ui.div("데이터가 없습니다. 작업을 시작해주세요.", class_="text-muted")

            latest = df.iloc[-1]
            latest = pd.DataFrame([latest])  # 단일 행을 DataFrame으로 변환

            # ✅ registration_time 처리 및 파생 컬럼 생성
            latest["registration_time"] = pd.to_datetime(latest["registration_time"], errors="coerce")
            latest["time"] = latest["registration_time"].dt.strftime("%H:%M:%S")  # 시:분:초
            latest["date"] = latest["registration_time"].dt.strftime("%Y-%m-%d")  # 연-월-일
            latest["registration_time"] = latest["registration_time"].astype(str)

            # ✅ 모델에서 사용한 컬럼 정보 추출
            pipeline = model.best_estimator_
            preprocessor = pipeline.named_steps["preprocess"]
            numeric_features = preprocessor.transformers_[0][2]
            categorical_features = preprocessor.transformers_[1][2]
            model_features = numeric_features + categorical_features

            # ✅ 누락된 컬럼 보완
            for col in model_features:
                if col not in latest.columns:
                    latest[col] = 0.0 if col in numeric_features else "Unknown"
            print(f"✅ 누락된 컬럼 보완 완료")

            # ✅ 수치형 / 범주형 분리 (모델 기준으로)
            numeric_cols = numeric_features
            categorical_cols = categorical_features

            # ✅ NaN-only 수치형 컬럼 제외 후 결측치 처리
            valid_numeric_cols = [col for col in numeric_cols if not latest[col].isna().all()]
            print(f"📊 결측치 처리 대상 수치형 컬럼: {valid_numeric_cols}")

            latest[valid_numeric_cols] = pd.DataFrame(
                SimpleImputer(strategy="mean").fit_transform(latest[valid_numeric_cols]),
                columns=valid_numeric_cols,
                index=latest.index
            )
            print("✅ 수치형 결측치 처리 완료")

            # ✅ 범주형 결측치 처리
            latest[categorical_cols] = latest[categorical_cols].fillna("Unknown")
            print("✅ 범주형 결측치 처리 완료")

            # ✅ 모델 입력 형식 정렬
            X_live = latest[model_features]

            # ✅ 예측 수행
            prob = model.predict_proba(X_live)[0, 1]
            result = "불량" if prob >= 0.5 else "양품"
            icon = "❌" if result == "불량" else "✅"
            color_class = "alert alert-danger" if result == "불량" else "alert alert-success"


            # ✅ 결과 UI 출력
            return ui.div(
                ui.div(
                    ui.h6("실시간 품질 불량 판정"),
                    ui.h4(f"{icon} {result}", class_="fw-bold"),
                    class_="mb-2"
                ),
                ui.div(
                    ui.input_action_button("goto_3page", "불량탐지 확인하기", class_="btn btn-sm btn-outline-primary")
                ),
                class_=f"{color_class} p-3 rounded"
            )

        except Exception as e:
            print(f"⛔ current_prediction 오류 발생: {e}")
            return ui.div(f"오류: {str(e)}", class_="text-danger")
        




    @reactive.effect
    @reactive.event(input.goto_2page)
    def go_to_page_3():
        ui.update_navs("main_nav", "공정 이상 탐지   (Process Anomaly Detection)") 
    
    @reactive.effect
    @reactive.event(input.goto_3page)
    def go_to_page_3():
        ui.update_navs("main_nav", "품질 불량 판별   (Quality Defect Classification)") 


    @output
    @render.ui
    def current_weather():
        try:
            df = current_data.get()
            if df.empty:
                return ui.card(
                    ui.div("센서 데이터 없음 · 날씨 확인 불가", class_="p-1 bg-light shadow-sm rounded h-100")
                )

            # 최신 데이터의 시간 정보
            latest = df.iloc[-1]
            reg_time = latest.get("registration_time")
            if reg_time is None:
                return ui.card(
                    ui.div("📡 수집된 시간 정보 없음", class_="p-1 bg-light shadow-sm rounded h-100")
                )

            dt = pd.to_datetime(reg_time)
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M")

            # ✅ 날씨 문자열 반환 (예: "☁️ Seoul · 흐림 · 22℃ · 습도 40%")
            weather_info = get_cached_weather(reg_time)
            

            # ✅ 반드시 문자열 형태로 넣기
            return ui.card(
                ui.div([
                    ui.p(f"일자 {date_str} · 시간 {time_str}", class_="p-1 bg-light shadow-sm rounded h-100"),
                    ui.p(weather_info, class_="fw-bold fs-5")
                ], class_="p-3")
            )

        except Exception as e:
            return ui.card(
                ui.div(f"❌ 날씨 표시 오류: {str(e)}", class_="p-1 bg-light shadow-sm rounded h-100")
            )
                    
    # ================================
    # TAP 1 [A] - 스트리밍 표시
    # ================================
    for code in ["ALL"] + mold_codes:
            @output(id=f"stream_plot_{code}")
            @render.plot
            def _plot(code=code):  # ✅ 클로저 캡처
                try:
                    df = current_data.get()
                    if df.empty:
                        raise ValueError("데이터가 없습니다. 작업을 시작해주세요.")

                    df["registration_time"] = pd.to_datetime(df["registration_time"], errors="coerce")

                    # ✅ mold_code 필터링 (ALL이면 전체)
                    if code != "ALL":
                        df = df[df["mold_code"] == int(code)]

                    # ✅ 최근 30분 + tail(30)
                    t_latest = df["registration_time"].max()
                    df = df[df["registration_time"] >= t_latest - pd.Timedelta(minutes=30)]
                    df = df.tail(20)

                    # ✅ 사용자가 선택한 변수
                    selected_cols = input.selected_sensor_cols()
                    cols_to_plot = [col for col in selected_cols if col in df.columns]
                    if not cols_to_plot:
                        raise ValueError("선택된 센서 컬럼이 없습니다.")

                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                    fig, axs = plt.subplots(nrows=len(cols_to_plot), figsize=(10, 3.5 * len(cols_to_plot)), sharex=True)
                    if len(cols_to_plot) == 1:
                        axs = [axs]

                    for i, col in enumerate(cols_to_plot):
                        ax = axs[i]
                        ax.plot(df["registration_time"], df[col],
                                label=col,
                                color=colors[i % len(colors)],
                                linewidth=2,
                                marker='o', markersize=5)
                        

                        # ✅ 상한/하한선 표시 (단, code != "ALL"일 때만)
                        if code != "ALL" and int(code) not in [8573, 8600]:
                            spec_row = spec_df_all[
                                (spec_df_all["mold_code"] == int(code)) & (spec_df_all["variable"] == col)
                            ]
                            if not spec_row.empty:
                                upper = spec_row["upper"].values[0]
                                lower = spec_row["lower"].values[0]
                                ax.axhline(y=upper, color="red", linestyle="--", linewidth=1.2, label="상한")
                                ax.axhline(y=lower, color="blue", linestyle="--", linewidth=1.2, label="하한")

                        ax.legend(loc="upper left",prop=font_prop)
                        ax.grid(True)

                    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S')) 
                    fig.autofmt_xdate(rotation=0, ha='center')  # ✅ 짤리지 않게 정렬
                    fig.tight_layout()
                    fig.subplots_adjust(bottom=0.2) 
                    return fig

                except Exception as e:
                    print(f"⛔ stream_plot_{code} 오류:", e)
                    fig, ax = plt.subplots()
                    ax.text(0.5, 0.5, f"{str(e)}", ha="center", va="center", fontsize=12, color='red',fontproperties=font_prop)
                    ax.axis("off")
                    return fig

        # ✅ 알림 로그 (상단 요약용)
    @output
    @render.ui
    def current_prediction2():
        try:
            df = current_data.get()
            if df.empty:
                return ui.div("데이터가 없습니다. 작업을 시작해주세요.", class_="text-muted")

            latest = df.iloc[-1]
            latest = pd.DataFrame([latest])  # 단일 행을 DataFrame으로 변환

            # ✅ registration_time 처리 및 파생 컬럼 생성
            latest["registration_time"] = pd.to_datetime(latest["registration_time"], errors="coerce")
            latest["time"] = latest["registration_time"].dt.strftime("%H:%M:%S")  # 시:분:초
            latest["date"] = latest["registration_time"].dt.strftime("%Y-%m-%d")  # 연-월-일
            latest["registration_time"] = latest["registration_time"].astype(str)

            # ✅ 모델에서 사용한 컬럼 정보 추출
            pipeline = model.best_estimator_
            preprocessor = pipeline.named_steps["preprocess"]
            numeric_features = preprocessor.transformers_[0][2]
            categorical_features = preprocessor.transformers_[1][2]
            model_features = numeric_features + categorical_features

            # ✅ 누락된 컬럼 보완
            for col in model_features:
                if col not in latest.columns:
                    latest[col] = 0.0 if col in numeric_features else "Unknown"

            # ✅ 수치형 / 범주형 분리 (모델 기준으로)
            numeric_cols = numeric_features
            categorical_cols = categorical_features

            # ✅ NaN-only 수치형 컬럼 제외 후 결측치 처리
            valid_numeric_cols = [col for col in numeric_cols if not latest[col].isna().all()]

            latest[valid_numeric_cols] = pd.DataFrame(
                SimpleImputer(strategy="mean").fit_transform(latest[valid_numeric_cols]),
                columns=valid_numeric_cols,
                index=latest.index
            )

            # ✅ 범주형 결측치 처리
            latest[categorical_cols] = latest[categorical_cols].fillna("Unknown")
            # ✅ 모델 입력 형식 정렬
            X_live = latest[model_features]

            # ✅ 예측 수행
            prob = model.predict_proba(X_live)[0, 1]
            result = "불량" if prob >= 0.5 else "양품"
            icon = "❌" if result == "불량" else "✅"
            color_class = "alert alert-danger" if result == "불량" else "alert alert-success"

            # ✅ 시간 표시 처리
            try:
                reg_time = pd.to_datetime(latest["registration_time"].values[0]).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as time_err:
                print(f"⚠️ 시간 파싱 오류: {time_err}")
                reg_time = "시간 정보 없음"

            # ✅ 결과 UI 출력
            return ui.div(
                ui.div(
                    ui.h6("실시간 품질 불량 판별"),
                    ui.h4(f"{icon} {result}", class_="fw-bold"),
                    class_="mb-2"
                ),
                ui.div(
                    ui.input_action_button("goto_3page", "불량탐지 확인하기", class_="btn btn-sm btn-outline-primary")
                ),
                class_=f"{color_class} p-3 rounded"
            )

        except Exception as e:
            print(f"⛔ current_prediction 오류 발생: {e}")
            return ui.div(f"오류: {str(e)}", class_="text-danger")
    # ================================
    # TAP 1 [B] - 실시간 값 
    # ================================
    @output
    @render.ui
    def real_time_values():
        try:
            df = current_data.get()
            if df.empty:
                return ui.div("데이터 없음", class_="text-muted")

            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest

            sensor_colors = {
                "cast_pressure": "#ff7f0e",                 # 주조 압력
                "low_section_speed": "#d62728",             # 저속 구간 속도
                "biscuit_thickness": "#9467bd",             # 비스킷 두께
                "molten_temp": "#1f77b4",                   # 용탕 온도
                "high_section_speed": "#8c564b",            # 고속 구간 속도
                "physical_strength": "#e377c2",             # 물리적 강도
                "facility_operation_cycleTime": "#7f7f7f",  # 설비 작동 사이클
                "production_cycletime": "#bcbd22",          # 생산 사이클 타임
                "Coolant_temperature": "#17becf",           # 냉각수 온도
                "sleeve_temperature": "#aec7e8",            # 슬리브 온도
                "molten_volume": "#ffbb78",                 # 용탕 체적
                "EMS_operation_time": "#98df8a"             # EMS 작동 시간
            }
            sensor_korean_labels = {
                "cast_pressure": ("cast pressure", "(bar)"),
                "low_section_speed": ("low section speed", "(mm/s)"),
                "biscuit_thickness": ("biscuit thickness", "(mm)"),
                "molten_temp": ("molten_temp", "(℃)"),
                "high_section_speed": ("high_section_speed", "(mm/s)"),
                "physical_strength": ("physical_strength", "(MPa)"),
                "facility_operation_cycleTime": ("facility_operation_cycleTime", "(sec)"),
                "production_cycletime": ("production_cycletime", "(sec)"),
                "Coolant_temperature": ("Coolant_temperature", "(℃)"),
                "sleeve_temperature": ("sleeve_temperature", "(℃)"),
                "molten_volume": ("molten_volume", "(cc)"),
                "EMS_operation_time": ("EMS_operation_time", "(sec)"),
            }

            cards = []

            # ✅ mold_code 카드
            if 'mold_code' in df.columns:
                mold_code_val = latest['mold_code']
                cards.append(
                    ui.div(
                        ui.h6("Mold Code"),
                        ui.h4(str(mold_code_val), class_="fw-bold"),
                        class_="card p-3 mb-2 border border-info",
                        style="min-width: 200px;" 
                    )
                )

            # ✅ 사용자가 선택한 변수만 표시
            selected_cols = input.selected_sensor_cols()

            for col in selected_cols:
                if col in df.columns:
                    current_val = latest[col]
                    prev_val = prev[col]
                    diff = current_val - prev_val
                    percent_change = (diff / prev_val * 100) if prev_val != 0 else 0

                    arrow = "⬆️" if diff > 0 else "⬇️" if diff < 0 else "➡️"
                    color_class = "text-muted"

                    warning_class = ""
                    try:
                        mold_code_val = int(latest['mold_code'])
                        spec_row = spec_df_all[
                            (spec_df_all["mold_code"] == mold_code_val) &
                            (spec_df_all["variable"] == col)
                        ]
                        if not spec_row.empty:
                            lower_bound = spec_row["lower"].values[0]
                            upper_bound = spec_row["upper"].values[0]
                            if current_val < lower_bound or current_val > upper_bound:
                                warning_class = "border border-danger"
                    except Exception as e:
                        print(f"[스펙 확인 오류] {col}: {e}")

                    custom_color = sensor_colors.get(col, "#000000")
                    cards.append(
                        ui.div(
                            ui.h6(sensor_korean_labels.get(col, col)),
                            ui.h4(
                                f"{current_val:.1f}",
                                class_=color_class,
                                style=f"color: {custom_color}; font-weight: bold;"
                            ),
                            class_=f"card p-3 mb-2 {warning_class}"
                        )
                    )

            return ui.div(*cards, class_="d-flex flex-column gap-2")

        except Exception as e:
            return ui.div(f"오류: {str(e)}", class_="text-danger")

    # ================================
    # TAP 1 [C] - 실시간 로그
    # ================================
    @output
    @render.ui
    def recent_data_table():
        try:
            df = current_data.get()
            if df.empty:
                return ui.HTML("<p class='text-muted'>데이터 없음</p>")
            cols = [
                'mold_code',
                'registration_time',
                'molten_temp',
                'cast_pressure',
                'high_section_speed',
                'low_section_speed',
                'biscuit_thickness',
                'passorfail',
                'is_anomaly',
                'anomaly_level',
                'physical_strength',
                'heating_furnace',
                'tryshot_signal',
                'lower_mold_temp2',
                'facility_operation_cycleTime',
                'upper_mold_temp2',
                'production_cycletime',
                'count',
                'Coolant_temperature',
                'sleeve_temperature',
                'molten_volume',
                'upper_mold_temp1',
                'EMS_operation_time',
                'lower_mold_temp1', 
                'working'
            ]
        # 고속 구간 속도
    
            df = df[cols].round(2)  # 전체 데이터 출력
            df = df.iloc[::-1]       # 최근 데이터가 위로 오도록 역순 정렬

            rows = []

            # 헤더 행
            header_cells = [ui.tags.th(col) for col in df.columns]
            rows.append(ui.tags.tr(*header_cells))

            # 데이터 행
            for i, row in df.iterrows():
                is_latest = i == df.index[-1]
                style = "background-color: #fff7d1;" if is_latest else ""
                cells = [ui.tags.td(str(val)) for val in row]
                rows.append(ui.tags.tr(*cells, style=style))

            return ui.div(  # ✅ 스크롤 가능한 박스로 감싸기
                ui.tags.table(
                    {"class": "table table-sm table-striped table-bordered mb-0", "style": "font-size: 13px;"},
                    *rows
                ),
                style="max-height: 500px; overflow-y: auto;"  # ✅ 높이 제한 + 스크롤
            )

        except Exception as e:
            return ui.HTML(f"<p class='text-danger'>에러 발생: {str(e)}</p>")

    

    # ================================
    # TAP 1 [C] - 실시간 선택 다운로드 
    # ================================
    @output
    @render.ui
    def download_controls():
        return ui.div(
            ui.input_select("file_format", "다운로드 형식", {
                "csv": "CSV",
                "xlsx": "Excel",
                "pdf": "PDF"
            }, selected="csv"),
            ui.download_button("download_recent_data", "최근 로그 다운로드")
        )
    # ================================
    # TAP 1 [C] - 실시간 선택 다운로드 로직  
    # ================================
    @output
    @render.download(filename=lambda: f"recent_log.{input.file_format()}")
    def download_recent_data():
        def writer():
            df = current_data.get().tail(1000).round(2)
            file_format = input.file_format()

            if df.empty:
                return

            if file_format == "csv":
                yield df.to_csv(index=False).encode("utf-8")

            elif file_format == "xlsx":
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df.to_excel(writer, sheet_name="RecentLog", index=False)
                yield buffer.getvalue()

            elif file_format == "pdf":
                buffer = BytesIO()
                with PdfPages(buffer) as pdf:
                    fig, ax = plt.subplots(figsize=(8.5, 4))
                    ax.axis("off")
                    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1.2, 1.2)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                yield buffer.getvalue()
        return writer()
    # ================================
    # TAP 2 [D] - 이상 불량 알림 
    # ================================
    
    @reactive.Effect
    def update_selected_log_index():
        logs = list(reversed(prediction_table_logs.get()))
        for i in range(len(logs)):
            if input[f"log_{i}"]() > 0:  # 클릭된 버튼이 눌림
                selected_log_index.set(i)

    # ================================
    # TAB 2: [A] 이상 예측
    # ================================
    @output
    @render.plot
    def anomaly_variable_count():
        try:
            logs = anomaly_detail_logs.get()
            if not logs:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "이상 변수 없음", ha='center', va='center', fontproperties=font_prop)
                return fig

            # ✅ 카운터 초기화 후 새로 집계
            top_vars = []
            for row in logs:
                for key in ["top1", "top2", "top3"]:
                    var = row.get(key)
                    if pd.notna(var) and var != "":
                        top_vars.append(var)

            counts = Counter(top_vars)  # ← 이전 누적값 없이 새로 계산
            anomaly_counter.set(counts)  # 여전히 공유 저장소에는 저장함

            if not counts:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "이상 변수 없음", ha='center', va='center', fontproperties=font_prop)
                return fig

            # ✅ 정렬 후 시각화
            sorted_items = counts.most_common()
            vars_, values = zip(*sorted_items)

            fig, ax = plt.subplots(figsize=(10, max(4, len(vars_) * 0.4)))
            bars = ax.barh(vars_, values)
            ax.set_title("실시간 이상 변수 누적 카운트 (전체)", fontproperties=font_prop)
            ax.set_xlabel("횟수", fontproperties=font_prop)
            ax.set_ylabel("변수명", fontproperties=font_prop)

            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2,
                        f'{int(width)}', va='center', fontproperties=font_prop)

            plt.tight_layout()
            return fig

        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"오류: {str(e)}", ha='center', va='center', fontproperties=font_prop)
            return fig

    # ================================
    # TAB 2 [A]: 
    # ================================
    @reactive.effect
    @reactive.event(current_data)
    def update_alert_log():
        df = current_data.get()
        if df.empty:
            return

        score_thresholds = {
            "심각": -0.07342,
            "경도": -0.04480
        }

        latest = df.iloc[-1].copy()

        # 🔹 입력 벡터 구성
        input_row = latest.drop(['passorfail', 'registration_time'], errors='ignore')
        required_features = model_iso.feature_names_in_

        for col in required_features:
            if col not in input_row:
                input_row[col] = 0

        X_input = pd.DataFrame([input_row[required_features]])

        # 🔹 예측 및 점수 계산
        score = model_iso.decision_function(X_input)[0]
        pred = model_iso.predict(X_input)[0]

        # 🔹 SHAP top1~3 계산
        try:
            shap_explainer = shap.TreeExplainer(model_iso)
            shap_values = shap_explainer.shap_values(X_input)
            shap_row = shap_values[0]
            top_idx = np.argsort(np.abs(shap_row))[::-1][:3]
            top_names = [required_features[i] for i in top_idx]
            top_vals = [abs(shap_row[i]) for i in top_idx]
        except Exception:
            top_names = ["", "", ""]
            top_vals = [0.0, 0.0, 0.0]

        # 🔹 anomaly_level 판정
        if score <= score_thresholds["심각"]:
            level = "심각"
        elif score <= score_thresholds["경도"]:
            level = "경도"
        else:
            level = "정상"

        # 🔹 모든 예측 결과를 latest에 저장
        latest["anomaly_level"] = level
        latest["anomaly_score"] = score
        latest["is_anomaly"] = int(level in ["경도", "심각"])
        for i, col in enumerate(["top1", "top2", "top3"]):
            latest[col] = top_names[i]
        for i, col in enumerate(["top1_val", "top2_val", "top3_val"]):
            latest[col] = top_vals[i]
        latest["time"] = pd.to_datetime(latest["registration_time"]).strftime("%Y-%m-%d %H:%M:%S")

        # 🔹 알람 로그 저장 (심각/경도일 때만)
        if level in ["경도", "심각"]:
            logs = alert_logs.get() or []
            detail_logs = anomaly_detail_logs.get() or []

            logs.append({
                "time": latest["time"],
                "level": level.strip()
            })
            detail_logs.append(latest.to_dict())

            alert_logs.set(logs[:])
            anomaly_detail_logs.set(detail_logs[:])




    @reactive.effect
    @reactive.event(input.clear_alerts)
    def clear_alert_logs():
        alert_logs.set([])  # 또는 상태 변수 초기화
        anomaly_detail_logs.set([])
    
    
    # ================================
    # TAB 2 [C] 단위 시간 당 불량 관리도
    # ================================
    @output
    @render.plot
    def anomaly_p_chart():
        try:
            df = accumulator.get().get_data()

            # ✅ 필수 컬럼 존재 여부 확인
            if df.empty:
                raise ValueError("데이터가 비어 있습니다.")
            if 'registration_time' not in df.columns:
                raise ValueError("registration_time 컬럼이 존재하지 않습니다.")
            if 'is_anomaly' not in df.columns:
                raise ValueError("is_anomaly 컬럼이 존재하지 않습니다.")

            # ✅ datetime 파싱
            df['datetime'] = pd.to_datetime(df['registration_time'], errors='coerce')

            # ✅ 시간 단위 선택 (input ID: anomaly_chart_time_unit)
            unit = input.anomaly_chart_time_unit()
            if unit == "1시간":
                df['time_group'] = df['datetime'].dt.floor('H')
            elif unit == "3시간":
                df['time_group'] = df['datetime'].dt.floor('3H')
            elif unit == "일":
                df['time_group'] = df['datetime'].dt.date
            elif unit == "주":
                df['time_group'] = df['datetime'].dt.to_period('W')
            elif unit == "월":
                df['time_group'] = df['datetime'].dt.to_period('M')
            else:
                raise ValueError(f"선택된 시간 단위 '{unit}'를 처리할 수 없습니다.")

            # ✅ 그룹별 총 건수와 이상 건수 계산
            n_i = df.groupby('time_group').size()
            x_i = df[df['is_anomaly'] == -1].groupby('time_group').size()
            x_i = x_i.reindex(n_i.index, fill_value=0)

            # ✅ 불량률 및 중심선 계산
            p_i = x_i / n_i
            p_hat = x_i.sum() / n_i.sum()

            # ✅ 관리 한계선 계산
            std_err = np.sqrt(p_hat * (1 - p_hat) / n_i)
            ucl = p_hat + 3 * std_err
            lcl = (p_hat - 3 * std_err).clip(lower=0)

            # ✅ 최근 20개만 시각화
            last_n = 20
            df_plot = pd.DataFrame({
                "Group": n_i.index.astype(str),
                "DefectiveRate": p_i,
                "UCL": ucl,
                "LCL": lcl,
                "Center": p_hat
            }).sort_index().iloc[-last_n:].reset_index(drop=True)

            # ✅ 시각화
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_plot.index, df_plot["DefectiveRate"], marker="o", label="Defective Rate")
            ax.plot(df_plot.index, df_plot["UCL"], linestyle='--', color='red', label="UCL")
            ax.plot(df_plot.index, df_plot["LCL"], linestyle='--', color='red', label="LCL")
            ax.plot(df_plot.index, df_plot["Center"], linestyle=':', color='black', label="Center Line")
            ax.fill_between(df_plot.index, df_plot["LCL"], df_plot["UCL"], color='red', alpha=0.1)

            # ✅ 불량률 및 관리한계선에서 최소/최대 계산
            min_val = min(df_plot["DefectiveRate"].min(), df_plot["LCL"].min())
            max_val = max(df_plot["DefectiveRate"].max(), df_plot["UCL"].max())

            # ✅ 변화폭이 작을 경우, 확대 효과를 주기 위해 가중 마진
            range_val = max_val - min_val
            if range_val < 0.01:
                y_min = max(0, min_val - 0.005)
                y_max = min(1.0, max_val + 0.015)  # 아주 미세한 차이도 확대해서 보여줌
            else:
                y_margin = range_val * 0.3
                y_min = max(0, min_val - y_margin)
                y_max = min(1.0, max_val + y_margin)

            ax.set_ylim(y_min, y_max)

            # # ✅ x축 설정
            # ax.set_xticks(df_plot.index)
            # ax.set_xticklabels(df_plot["Group"], rotation=0, ha='right')
            
            # ✅ x축 설정
            if isinstance(df["time_group"].iloc[0], pd.Period):
            # 주/월 단위 등 Period → Timestamp → 월-일 포맷
                group_labels = df_plot["Group"].apply(lambda x: pd.Period(x).to_timestamp().strftime("%m-%d"))
            elif pd.api.types.is_datetime64_any_dtype(df["time_group"]):
                # 일/시 단위 등 datetime → 시:분:초 포맷
                group_labels = pd.to_datetime(df_plot["Group"], errors='coerce').dt.strftime("%H:%M:%S")
            else:
                # 기타 타입 (예외 상황) → 문자열 처리
                group_labels = df_plot["Group"].astype(str)

            # ✅ 라벨 출력 간격 조건 분기
            if unit in ["1시간", "3시간", "일"]:
                tick_interval = 3
                xticks = df_plot.index[::tick_interval]
                xticklabels = group_labels[::tick_interval]
            else:
                xticks = df_plot.index
                xticklabels = group_labels

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=0, ha='right', fontsize=9)
            
            ax.set_ylabel("공정 이상률",fontproperties=font_prop)
            ax.set_title(f"공정 이상률 관리도 (단위: {unit})",fontproperties=font_prop)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            fig.tight_layout(pad=2)
            fig.subplots_adjust(left=0.1)  # ✅ 왼쪽 여백 확보
            ax.margins(x=0)
            
            return fig

        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"오류 발생: {str(e)}", ha='center', va='center', color='red',fontproperties=font_prop)
            return fig
    
    
    # ========================================
    # TAB 2 - [D] 이상치 × 불량 Confusion 스타일
    # ========================================
    @output
    @render.ui
    def anomaly_fail_rate_ui():
        try:
            # ✅ 실시간 갱신 트리거 (1초마다 갱신)
            reactive.invalidate_later(1)

            df = accumulator.get().get_data()  # ✅ 실시간 누적 데이터 가져오기

            if df.empty:
                return ui.div("데이터 없습니다.작업을 시작해주세요.", class_="text-muted")

            # ✅ Confusion 영역별 필터링

            # total = len(df)
            count_a_f = len(df[(df["is_anomaly"] == -1) & (df["passorfail"] == 1)])  # 이상 + 불량
            count_a_p = len(df[(df["is_anomaly"] == -1) & (df["passorfail"] == 0)])  # 이상 + 정상
            count_n_f = len(df[(df["is_anomaly"] == 1) & (df["passorfail"] == 1)])  # 정상 + 불량
            count_n_p = len(df[(df["is_anomaly"] == 1) & (df["passorfail"] == 0)])  # 정상 + 정상
            total = count_a_f + count_a_p + count_n_f + count_n_p
            # ✅ 비율 계산
            def ratio(n): return f"{n:,}건 ({n/total:.2%})" if total > 0 else "0건 (0%)"

            return ui.div(
                [
                    ui.h5("이상 탐지 vs 불량 판정 매트릭스"),
                    ui.tags.table(
                        {"class": "table table-bordered text-center"},
                        ui.tags.thead(
                            ui.tags.tr(
                                ui.tags.th("구분"),
                                ui.tags.th("불량"),
                                ui.tags.th("정상")
                            )
                        ),
                        ui.tags.tbody([
                            ui.tags.tr([
                                ui.tags.th("이상치", {"class": "table-danger"}),
                                ui.tags.td(ratio(count_a_f)),
                                ui.tags.td(ratio(count_a_p))
                            ]),
                            ui.tags.tr([
                                ui.tags.th("정상치", {"class": "table-success"}),
                                ui.tags.td(ratio(count_n_f)),
                                ui.tags.td(ratio(count_n_p))
                            ])
                        ])
                    )
                ]
            )

        except Exception as e:
            return ui.div(f"⚠️ 오류 발생: {str(e)}", class_="text-danger")

    # ================================
    # TAB 2 - [D] 
    # ================================
    # @reactive.effect
    # @reactive.event(current_data)
    # def update_anomaly_details():
    #     df = current_data.get()
    #     if df.empty:
    #         return

    #     latest = df.iloc[-1]
    #     level = latest.get("anomaly_level", "정상")

    #     if level not in ["경도", "심각"]:
    #         return

    #     logs = anomaly_detail_logs.get() or []

    #     # 전체 컬럼 값 저장 (dict로 변환)
    #     row_data = latest.to_dict()
    #     row_data["level"] = level
    #     row_data["time"] = pd.to_datetime(latest["registration_time"]).strftime("%Y-%m-%d %H:%M:%S")

    #     logs.append(row_data)
    #     anomaly_detail_logs.set(logs)
        
    

    
    # @output
    # @render.ui
    # def anomaly_detail_table():
    #     try:
    #         logs = anomaly_detail_logs.get()
    #         if not logs:
    #             return ui.div("⚠️ 이상치 상세 로그 없음", class_="text-muted")

    #         rows = []

    #         for row in reversed(logs):
    #             level_value = row.get("anomaly_level", "없음")
    #             reg_time_raw = row.get("registration_time", "")
    #             try:
    #                 time_value = pd.to_datetime(reg_time_raw).strftime("%Y-%m-%d %H:%M:%S")
    #             except:
    #                 time_value = str(reg_time_raw)

    #             mold_code = row.get("mold_code", "미입력")

    #             top_features = []
    #             for i in range(1, 4):
    #                 var = row.get(f"top{i}", "-")
    #                 try:
    #                     val = float(row.get(var, "-")) if var != "-" else "-"
    #                 except:
    #                     val = "-"

    #                 # ▶️ IQR 상/하한 가져오기
    #                 try:
    #                     bounds_row = spec_df_all[
    #                         (spec_df_all["mold_code"] == int(mold_code)) & 
    #                         (spec_df_all["variable"] == var)
    #                     ]
    #                     lower = bounds_row["lower"].values[0]
    #                     upper = bounds_row["upper"].values[0]
    #                 except:
    #                     lower = "-"
    #                     upper = "-"

    #                 top_features.append((f"TOP {i}", var, val, lower, upper))

    #             level_color = "🔴" if level_value == "심각" else ("🟠" if level_value == "경도" else "✅")
    #             bg_color = "#fff5f5" if level_value == "심각" else ("#fffdf5" if level_value == "경도" else "#f5fff5")

    #             table_html = ui.tags.table(
    #                 {"class": "table table-bordered table-sm mb-1"},
    #                 ui.tags.thead(
    #                     ui.tags.tr(
    #                         ui.tags.th("순위"), ui.tags.th("변수명"),
    #                         ui.tags.th("수치"), ui.tags.th("하한"), ui.tags.th("상한")
    #                     )
    #                 ),
    #                 ui.tags.tbody(*[
    #                     ui.tags.tr(
    #                         ui.tags.td(rank),
    #                         ui.tags.td(var),
    #                         ui.tags.td(f"{val:.1f}" if isinstance(val, float) else val),
    #                         ui.tags.td(f"{lower:.1f}" if isinstance(lower, float) else lower),
    #                         ui.tags.td(f"{upper:.1f}" if isinstance(upper, float) else upper),
    #                     ) for rank, var, val, lower, upper in top_features
    #                 ])
    #             )

    #             rows.append(
    #                 ui.div(
    #                     ui.HTML(
    #                         f"{level_color} <b>{level_value}</b> | 🕒 {time_value} | mold_code: <b>{mold_code}</b><br>"
    #                     ),
    #                     table_html,
    #                     class_="border rounded p-2 mb-3",
    #                     style=f"background-color: {bg_color};"
    #                 )
    #             )

    #         return ui.div(*rows, class_="log-container", style="max-height: 450px; overflow-y: auto;")

    #     except Exception as e:
    #         return ui.div(f"❌ 로그 렌더링 오류: {str(e)}", class_="text-danger")

    


    # @reactive.effect
    # @reactive.event(input.clear_alerts2)
    # def clear_alert_logs():
    #     alert_logs.set([])               # 기존 경고/심각 로그 초기화
    #     anomaly_detail_logs.set([])      # ✅ SHAP 상세 로그도 함께 초기화

    
    # @output
    # @render.ui
    # def log_alert_for_defect():
    #     logs = alert_logs.get() or []  # logs가 None일 경우를 대비
    
    #     # level별 필터링 (없어도 0으로 반환되도록)
    #     mild_logs = [log for log in logs if log.get("level", "").strip() == "경도"]
    #     severe_logs = [log for log in logs if log.get("level", "").strip() == "심각"]
    #     count_badge = ui.div(
    #         ui.HTML(f"<span style='margin-right:10px;'>🟠 <b>경도</b>: {len(mild_logs)}</span> | "
    #                 f"<span style='margin-left:10px;'>🔴 <b>심각</b>: {len(severe_logs)}</span>"),
    #         class_="fw-bold mb-2"
    #     )
    #     return ui.div(count_badge, class_="log-container")

    
    
    @reactive.effect
    @reactive.event(current_data)
    def update_anomaly_details():
        df = current_data.get()
        if df.empty:
            return

        latest = df.iloc[-1]
        level = latest.get("anomaly_level", "정상")

        if level not in ["경도", "심각"]:
            return

        logs = anomaly_detail_logs.get() or []

        # 전체 컬럼 값 저장 (dict로 변환)
        row_data = latest.to_dict()
        row_data["level"] = level
        row_data["time"] = pd.to_datetime(latest["registration_time"]).strftime("%Y-%m-%d %H:%M:%S")
        
        # 고유 ID 추가 (현재 시간 + 로그 개수 기반)
        import time
        row_data["log_id"] = f"log_{int(time.time())}_{len(logs)}"

        logs.append(row_data)
        anomaly_detail_logs.set(logs)

    # 개별 삭제를 위한 reactive Value
    selected_for_deletion = reactive.Value("")

    # 삭제 버튼 클릭 처리
    @reactive.effect
    def handle_deletion():
        delete_id = selected_for_deletion.get()
        if delete_id:
            logs = anomaly_detail_logs.get() or []
            
            
            updated_logs = [log for log in logs if log.get("log_id") != delete_id]
            
            
            anomaly_detail_logs.set(updated_logs)
            selected_for_deletion.set("")  # 리셋

    @output
    @render.ui
    def anomaly_detail_table():
        try:
            logs = anomaly_detail_logs.get()
            if not logs:
                return ui.div("⚠️ 이상치 상세 로그 없음", class_="text-muted")

            rows = []

            # 원본 로그 순서 유지하면서 역순으로 표시
            reversed_logs = list(reversed(logs))
            for idx, row in enumerate(reversed_logs):
                log_id = row.get("log_id")
                if not log_id:
                    # fallback ID를 실제 로그에 할당
                    log_id = f"log_default_{len(logs) - idx - 1}"
                    row["log_id"] = log_id
                    # 원본 로그 업데이트
                    original_idx = len(logs) - idx - 1
                    logs[original_idx]["log_id"] = log_id
                    anomaly_detail_logs.set(logs)
                level_value = row.get("anomaly_level", "없음")
                reg_time_raw = row.get("registration_time", "")
                try:
                    time_value = pd.to_datetime(reg_time_raw).strftime("%Y-%m-%d %H:%M:%S")
                except:
                    time_value = str(reg_time_raw)

                mold_code = row.get("mold_code", "미입력")

                top_features = []
                for i in range(1, 4):
                    var = row.get(f"top{i}", "-")
                    try:
                        val = float(row.get(var, "-")) if var != "-" else "-"
                    except:
                        val = "-"

                    # ▶️ IQR 상/하한 가져오기
                    try:
                        bounds_row = spec_df_all[
                            (spec_df_all["mold_code"] == int(mold_code)) & 
                            (spec_df_all["variable"] == var)
                        ]
                        lower = bounds_row["lower"].values[0]
                        upper = bounds_row["upper"].values[0]
                    except:
                        lower = "-"
                        upper = "-"

                    top_features.append((f"TOP {i}", var, val, lower, upper))

                level_color = "🔴" if level_value == "심각" else ("🟠" if level_value == "경도" else "✅")
                bg_color = "#fff5f5" if level_value == "심각" else ("#fffdf5" if level_value == "경도" else "#f5fff5")

                table_html = ui.tags.table(
                    {"class": "table table-bordered table-sm mb-1"},
                    ui.tags.thead(
                        ui.tags.tr(
                            ui.tags.th("순위"), ui.tags.th("센서"),
                            ui.tags.th("수치"), ui.tags.th("하한"), ui.tags.th("상한")
                        )
                    ),
                    ui.tags.tbody(*[
                        ui.tags.tr(
                            ui.tags.td(rank),
                            ui.tags.td(var),
                            ui.tags.td(f"{val:.1f}" if isinstance(val, float) else val),
                            ui.tags.td(f"{lower:.1f}" if isinstance(lower, float) else lower),
                            ui.tags.td(f"{upper:.1f}" if isinstance(upper, float) else upper),
                        ) for rank, var, val, lower, upper in top_features
                    ])
                )

                # JavaScript를 사용한 삭제 버튼
                delete_js = f"""
                <button class="btn btn-sm btn-outline-danger" 
                        style="padding: 2px 8px; font-size: 12px; line-height: 1;" 
                        onclick="Shiny.setInputValue('delete_clicked', '{log_id}', {{priority: 'event'}});">
                    ✕
                </button>
                """

                # 헤더와 삭제 버튼이 포함된 div
                header_div = ui.div(
                    ui.div(
                        ui.HTML(f"{level_color} <b>{level_value}</b> |  {time_value} | mold_code: <b>{mold_code}</b>"),
                        style="flex: 1;"
                    ),
                    ui.HTML(delete_js),
                    style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;"
                )

                rows.append(
                    ui.div(
                        header_div,
                        table_html,
                        class_="border rounded p-2 mb-3",
                        style=f"background-color: {bg_color}; position: relative;"
                    )
                )

            return ui.div(*rows, class_="log-container", style="max-height: 450px; overflow-y: auto;")

        except Exception as e:
            return ui.div(f"❌ 로그 렌더링 오류: {str(e)}", class_="text-danger")

    # JavaScript 삭제 이벤트 처리
    @reactive.effect
    @reactive.event(input.delete_clicked)
    def handle_js_delete():
        delete_id = input.delete_clicked()
        if delete_id:
            print(f"JavaScript에서 삭제 요청: {delete_id}")  # 디버깅
            selected_for_deletion.set(delete_id)

    @reactive.effect
    @reactive.event(input.clear_alerts2)
    def clear_alert_logs():
        alert_logs.set([])               # 기존 경고/심각 로그 초기화
        anomaly_detail_logs.set([])      # ✅ SHAP 상세 로그도 함께 초기화
        selected_for_deletion.set("")    # 삭제 선택 상태 초기화

    @output
    @render.ui
    def log_alert_for_defect():
        logs = anomaly_detail_logs.get() or []  # anomaly_detail_logs를 참조하도록 수정

        # level별 필터링 (anomaly_level 또는 level 필드 확인)
        mild_logs = [log for log in logs if log.get("anomaly_level", log.get("level", "")).strip() == "경도"]
        severe_logs = [log for log in logs if log.get("anomaly_level", log.get("level", "")).strip() == "심각"]
        count_badge = ui.div(
            ui.HTML(f"<span style='margin-right:10px;'>🟠 <b>경도</b>: {len(mild_logs)}</span> | "
                    f"<span style='margin-left:10px;'>🔴 <b>심각</b>: {len(severe_logs)}</span>"),
            class_="fw-bold mb-2"
        )
        return ui.div(count_badge, class_="log-container")

    # ================================
    # TAB 3 - [A] : 품질 분석
    # ================================
    @output
    @render.plot
    def defect_rate_plot():
        try:
            unit = input.grouping_unit()  # "일", "주", "월"

            #df_vis = static_df.copy()
            df_vis = accumulator.get().get_data()

            # 문자열 날짜를 datetime으로 변환
            df_vis['datetime'] = pd.to_datetime(df_vis['registration_time'], errors="coerce")

            # 그룹핑 기준 추가
            if unit == "일":
                df_vis['group'] = df_vis['datetime'].dt.strftime('%Y-%m-%d')
            elif unit == "주":
                df_vis['group'] = df_vis['datetime'].dt.to_period('W').astype(str)
            elif unit == "월":
                df_vis['group'] = df_vis['datetime'].dt.to_period('M').astype(str)

            # 각 그룹별 불량률 계산
            group_result = df_vis.groupby(['group', 'passorfail']).size().unstack(fill_value=0)
    
            selected_group = input.selected_group()
            if selected_group not in group_result.index:
                raise ValueError("선택한 그룹에 대한 데이터가 없습니다.")
            counts = group_result.loc[selected_group]
    
            # 시각화
            fig, ax = plt.subplots()
            labels = ['양품', '불량']
            sizes = [counts.get(0, 0), counts.get(1, 0)]
            colors = ['#4CAF50', '#F44336']
    
            wedges, _, _ = ax.pie(
                sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90
            )
            ax.axis('equal')
            ax.set_title(f"{selected_group} ({unit} 기준) 불량률",fontproperties=font_prop)
            ax.legend(wedges, labels, title="예측 결과", loc="upper right", bbox_to_anchor=(1.1, 1))
    
            return fig
    
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"에러: {str(e)}", ha='center', va='center',fontproperties=font_prop)
            return fig
        
    @output
    @render.ui
    def group_choice():
        try:
            unit = input.grouping_unit()
            df_vis = accumulator.get().get_data()
            df_vis['datetime'] = pd.to_datetime(df_vis['registration_time'], errors="coerce")

            if unit == "일":
                df_vis['group'] = df_vis['datetime'].dt.strftime('%Y-%m-%d')
            elif unit == "주":
                df_vis['group'] = df_vis['datetime'].dt.to_period('W').astype(str)
            elif unit == "월":
                df_vis['group'] = df_vis['datetime'].dt.to_period('M').astype(str)

            unique_groups = sorted(df_vis['group'].dropna().unique())
            return ui.input_select("selected_group", "조회할 기간 선택", choices=unique_groups, selected=unique_groups[-1] if unique_groups else None)
        except:
            return ui.input_select("selected_group", "조회할 기간 선택", choices=["선택 불가"], selected=None)

    @output
    @render.plot
    def defect_rate_plot():
        try:
            # 기간 선택
            start_date, end_date = input.date_range()

            df_vis = accumulator.get().get_data()
            df_vis = df_vis.loc[:, ~df_vis.columns.duplicated()]  # 중복 열 제거
            df_vis['datetime'] = pd.to_datetime(df_vis['registration_time'], errors="coerce")

            # 날짜 필터링
            mask = (df_vis['datetime'].dt.date >= pd.to_datetime(start_date).date()) & \
                (df_vis['datetime'].dt.date <= pd.to_datetime(end_date).date())
            df_filtered = df_vis.loc[mask]

            if df_filtered.empty:
                raise ValueError("선택한 기간 내 데이터가 없습니다.")

            # ✅ 몰드코드 + 불량 여부별 카운트
            grouped = df_filtered.groupby(['mold_code', 'passorfail']).size().unstack(fill_value=0)
            grouped.columns = ['양품', '불량'] if 0 in grouped.columns else ['불량']
            grouped = grouped.reset_index()

            # ✅ 시각화 (stacked bar chart)
            import numpy as np
            mold_codes = grouped['mold_code']
            x = np.arange(len(mold_codes))
            width = 0.6

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x, grouped.get('양품', [0]*len(grouped)), width, label='양품', color='#4CAF50')
            ax.bar(x, grouped.get('불량', [0]*len(grouped)), width,
                bottom=grouped.get('양품', [0]*len(grouped)), label='불량', color='#F44336')

            ax.set_xlabel('몰드 코드',fontproperties=font_prop)
            ax.set_ylabel('개수',fontproperties=font_prop)
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
            ax.set_title(f"{start_date} ~ {end_date} 몰드코드별 누적 예측 결과",fontproperties=font_prop)
            ax.set_xticks(x)
            ax.set_xticklabels(mold_codes, rotation=0, ha='right')
            ax.legend(prop=font_prop)

            fig.tight_layout()
            fig.subplots_adjust(bottom=0.15)
            return fig

        except Exception as e:
            print(f"[defect_rate_plot] 에러: {e}")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"에러 발생: {str(e)}", ha='center', va='center',fontproperties=font_prop)
            return fig


    # ================================
    # TAP 3 [B]
    # ================================
    @output
    @render.ui
    def current_prediction():
        try:
            df = current_data.get()
            if df.empty:
                print("⚠️ current_data가 비어 있음")
                return ui.div("데이터 없음", class_="text-muted")

            # 최신 데이터 한 행
            latest = df.iloc[-1]

            if 'passorfail' not in latest:
                print("⚠️ 'passorfail' 컬럼이 존재하지 않음")
                return ui.div("예측값 없음", class_="text-muted")

            # 결합 확률은 이미 'passorfail' 컬럼에 예측값이 0~1로 들어온다고 가정
            prob = latest['passorfail']
            result = "불량" if prob >= 0.5 else "양품"
            icon = "❌" if result == "불량" else "✅"
            color_class = "alert alert-danger" if result == "불량" else "alert alert-success"

            reg_time = latest.get('registration_time')
            try:
                reg_time = pd.to_datetime(reg_time).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as time_err:
                print(f"⚠️ 시간 파싱 오류: {time_err}")
                reg_time = "시간 정보 없음"

            return ui.div(
                ui.div(
                    ui.h6("판정 결과"),
                    ui.h4(f"{icon} {result}", class_="fw-bold"),
                    class_="mb-2"
                ),
                ui.div(
                    ui.h6("판정 시간"),
                    ui.p(reg_time)
                ),
                class_=f"{color_class} p-3 rounded"
            )

        except Exception as e:
            print(f"⛔ current_prediction 오류 발생: {e}")
            return ui.div(f"오류: {str(e)}", class_="text-danger")

    @reactive.effect
    @reactive.event(current_data)
    def log_prediction_from_current_row():
        df = current_data.get()
        if df.empty or 'passorfail' not in df.columns:
            return

        row = df.iloc[-1]
        prob = row.get('passorfail', None)

        if pd.isna(prob):
            return

        result = "불량" if prob >= 0.5 else "양품"
        reg_time = row.get('registration_time')
        try:
            reg_time = pd.to_datetime(reg_time).strftime("%Y-%m-%d %H:%M:%S")
        except:
            reg_time = str(reg_time)

        logs = prediction_table_logs.get()
        logs.append({
            "판정 시간": reg_time,
            "결과": result
        })
        prediction_table_logs.set(logs[-20:])  # 최신 20개만 유지

    @output
    @render.ui
    def prediction_log_table():
        logs = prediction_table_logs.get()
        if not logs:
            return ui.div("예측 로그 없음", class_="text-muted")

        headers = ["판정 시간", "결과"]
        table_rows = [ui.tags.tr(*[ui.tags.th(h) for h in headers])]
        for i, log in enumerate(reversed(logs)):
            result = log["결과"]
            is_defect = result == "불량"
            row = ui.tags.tr(
                ui.tags.td(log["판정 시간"]),
                ui.tags.td(
                    ui.input_action_button(f"log_{i}", result, 
                        class_="btn btn-danger btn-sm" if is_defect else "btn btn-secondary btn-sm")
                )
            )
            table_rows.append(row)

        return ui.div(
            ui.tags.table(
                {"class": "table table-sm table-bordered table-striped mb-0"},
                *table_rows
            ),
            style="max-height: 250px; overflow-y: auto;"
        )
    
# ================================
    # TAP 3 [A] 단위 시간 당 불량 관리도
# ================================ 
    @output
    @render.plot
    def fail_rate_by_time():
        try:
            df = accumulator.get().get_data()
            if df.empty or 'passorfail' not in df.columns:
                raise ValueError("데이터 없음")

            if 'datetime' not in df.columns:
                df['datetime'] = pd.to_datetime(df['registration_time'], errors='coerce')

            unit = input.fail_time_unit()
            if unit == "1시간":
                df['time_group'] = df['datetime'].dt.floor('H')
            elif unit == "3시간":
                df['time_group'] = df['datetime'].dt.floor('3H')
            elif unit == "일":
                df['time_group'] = df['datetime'].dt.date
            elif unit == "주":
                df['time_group'] = df['datetime'].dt.to_period('W')
            elif unit == "월":
                df['time_group'] = df['datetime'].dt.to_period('M')

            # 그룹별 전체/불량 개수
            total_counts = df.groupby('time_group').size()
            fail_counts = df[df['passorfail'] == 1].groupby('time_group').size()
            rate = (fail_counts / total_counts).fillna(0)

            # 최근 20개
            rate = rate.sort_index().iloc[-20:]
            total_counts = total_counts.sort_index().loc[rate.index]

            # 평균 불량률
            p_bar = rate.mean()

            # 관리 상/하한선 계산
            ucl = []
            lcl = []
            for n in total_counts:
                std = (p_bar * (1 - p_bar) / n) ** 0.5
                ucl.append(min(1.0, p_bar + 3 * std))
                lcl.append(max(0.0, p_bar - 3 * std))

            labels = rate.index.astype(str)
            values = rate.values

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(labels, values, marker='o', label="불량률", color='blue')
            ax.plot(labels, [p_bar] * len(labels), linestyle='--', label="평균", color='gray')
            ax.plot(labels, ucl, linestyle='--', label="UCL", color='red')
            ax.plot(labels, lcl, linestyle='--', label="LCL", color='red')
            ax.fill_between(labels, lcl, ucl, color='red', alpha=0.1)


            ax.set_title(f"관리도 기반 불량률 분석 ({unit}) - 최근 20개",fontproperties=font_prop)
            ax.set_xlabel("시간 단위",fontproperties=font_prop)
            ax.set_ylabel("불량률",fontproperties=font_prop)
            # ✅ 시각화를 위한 y축 범위 계산
            min_val = min(min(values), min(lcl))
            max_val = max(max(values), max(ucl))
            range_val = max_val - min_val

            # ✅ 극소 불량률 보정
            if max_val < 0.01:
                y_min, y_max = -0.005, 0.03  # 완전 플랫 방지용 확대
            elif range_val < 0.01:
                y_min = max(0, min_val - 0.005)
                y_max = min(1.0, max_val + 0.02)
            else:
                y_margin = range_val * 0.3
                y_min = max(0, min_val - y_margin)
                y_max = min(1.0, max_val + y_margin)

            ax.set_ylim(y_min, y_max)
            ax.legend(prop=font_prop)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(0, len(labels), 3))  # ✅ 3칸마다 하나만 보여줌
            ax.set_xticklabels(labels[::3], fontproperties=font_prop, rotation=0)
            plt.tight_layout()
            fig.subplots_adjust(left=0.08,bottom=0.15)
            ax.margins(x=0)
            return fig

        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"에러 발생: {str(e)}", ha='center', va='center',fontproperties=font_prop)
            return fig

# ================================
# TAP 3 [B]
# ================================
    
    @reactive.Effect
    def handle_log_click():
        logs = list(reversed(prediction_table_logs.get()))
        prev_clicks = log_button_clicks.get()

        for i, log in enumerate(logs):
            btn_id = f"log_{i}"
            current_click = input[btn_id]()

            if prev_clicks.get(btn_id, -1) != current_click:
                selected_log_time.set(log["판정 시간"])
                prev_clicks[btn_id] = current_click  # 클릭 수 갱신

        log_button_clicks.set(prev_clicks)
    @output
    @render.plot
    def shap_explanation_plot():
        try:
            reg_time = selected_log_time.get()
    
            if reg_time is None:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "불량 로그를 선택하세요", ha='center', fontproperties=font_prop)
                return fig
    
            # 판정 시간 일치하는 row 찾기
            df = current_data.get()
            df['registration_time'] = df['registration_time'].astype(str)
            row_match = df[df['registration_time'] == str(reg_time)]
    
            if row_match.empty:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "해당 시간의 입력값을 찾을 수 없습니다", ha='center', fontproperties=font_prop)
                return fig
    
            logs = list(reversed(prediction_table_logs.get()))
            log = next((l for l in logs if l["판정 시간"] == reg_time), None)
            if log is None:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "해당 로그를 찾을 수 없습니다", ha='center', fontproperties=font_prop)
                return fig
    
            if log["결과"] != "불량":
                fig, ax = plt.subplots()
                ax.axis("off")
                ax.text(0.5, 0.5, "양품입니다\nSHAP 해석은 불량에만 제공됩니다", ha='center', va='center', color='gray', fontproperties=font_prop)
                return fig
    
            # ============================
            # SHAP 계산 로직
            # ============================
            input_row = row_match.iloc[0].drop(['passorfail', 'registration_time'], errors='ignore')
            required_features = model_pipe.feature_names_in_.tolist()
    
            # 범주형 변수 없을 경우 처리
            try:
                ct = model_pipe.named_steps["preprocess"]
                cat_cols = ct.transformers_[1][2]  # 없으면 오류 → except로 처리
            except Exception:
                cat_cols = []
    
            # 누락된 컬럼 보완
            for col in required_features:
                if col not in input_row:
                    input_row[col] = "0" if col in cat_cols else 0
            input_row = input_row[required_features]
    
            # 데이터프레임 구성 및 형 변환
            input_df = pd.DataFrame([input_row])
            for col in cat_cols:
                if col in input_df.columns:
                    input_df[col] = input_df[col].astype(str)
    
            # 전처리 및 SHAP 계산
            X_transformed = model_pipe.named_steps["preprocess"].transform(input_df)
            shap_raw = shap_explainer.shap_values(X_transformed)
    
            # ✅ SHAP 값 안전하게 가져오기
            if isinstance(shap_raw, list):
                if len(shap_raw) == 1:
                    shap_val = shap_raw[0][0]
                else:
                    shap_val = shap_raw[1][0]  # 일반적으로 1은 "불량" 클래스
            else:
                shap_val = shap_raw[0]
    
            # 변수 이름 정리 및 그래프
            feature_names = model_pipe.named_steps["preprocess"].get_feature_names_out()
            shap_series = pd.Series(shap_val, index=feature_names).abs().sort_values(ascending=False).head(5)
            shap_series.index = shap_series.index.str.replace(r'^(num__|cat__)', '', regex=True)
    
            fig, ax = plt.subplots()
            shap_series.plot(kind='barh', ax=ax)
            ax.invert_yaxis()
            ax.set_title("SHAP 기여도 상위 변수", fontproperties=font_prop)
            ax.set_xlabel("기여도 크기 (절댓값 기준)", fontproperties=font_prop)
            return fig
    
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"오류 발생: {str(e)}", ha='center', color='red', fontproperties=font_prop)
            return fig
    
    



# ================================
    # TAP 0  - 로그인 기능
# ================================ 
    # 로그인 버튼 처리
    @reactive.effect
    @reactive.event(input.login_button)
    def login():
        if input.username() == "admin" and input.password() == "1234":
            login_status.set(True)
        else:
            ui.notification_show("❌ 로그인 실패", duration=3)

    # 로그아웃 버튼 처리
    @reactive.effect
    @reactive.event(input.logout_button)
    def logout():
        login_status.set(False)

    # 전체 UI 렌더링
    @output
    @render.ui
    def dynamic_ui():
        if not login_status.get():
            # 로그인 화면 반환
            return ui.div(  # ✅ 전체 로그인 UI를 감싸서 가운데 정렬 + 너비 제한
            ui.card(
                ui.div(
                    ui.tags.img(
                        src="./logo2.png",
                        style="max-width: 300px; margin-bottom: 20px;"
                    ),
                    class_="text-center"
                ),
                ui.card_header("LS 기가 팩토리 로그인"),
                ui.input_text("username", "아이디"),
                ui.input_password("password", "비밀번호"),
                ui.input_action_button("login_button", "로그인", class_="btn btn-primary w-100"),
                ui.p("ID: admin / PW: 1234", class_="text-muted")
            ),
            style="max-width: 400px; margin: 0 auto; padding-top: 30px;"  # ✅ 핵심 스타일
        )
        else:
            return ui.page_fluid(
                        ui.tags.head(
                            ui.tags.link(rel="stylesheet", href="./style.css")
                        ),
                        
                        ui.page_navbar(
                            # ================================
                            # TAB 1: 실시간 공정 모니터링   Process Monitoring
                            # ================================
                            ui.nav_panel("실시간 공정 모니터링   (Process Monitoring)",

        # ▶ 좌측: 시작/정지/리셋 버튼 + 상태
        ui.column(3,
            ui.div(
                ui.input_action_button("start", "▶ 시작", class_="btn-success me-2"),
                ui.input_action_button("pause", "⏸ 일시정지", class_="btn-warning me-2"),
                ui.input_action_button("reset", "🔄 리셋", class_="btn-secondary me-2"),
                ui.output_ui("stream_status"),
            )
        ),
        ui.div(
            ui.div(ui.output_ui("anomaly_alerts"), class_="col-2"),
            ui.div(ui.output_ui("current_prediction2"), class_="col-2"),
            ui.div(ui.output_ui("current_weather"), class_="col-8"),
            class_="row g-3 align-items-stretch",
        ),
                                ui.layout_columns(
                                    # [A] 실시간 그래프
                                    ui.card(
                                    ui.card_header("실시간 센서 스트리밍"),
                                        ui.div(
                                            # 왼쪽: 탭 그래프
                                            ui.div(
                                                ui.input_checkbox_group(
                                                    id="selected_sensor_cols",
                                                    label="시각화할 센서 선택",
                                                    choices=list(sensor_labels.keys()),  # ✅ 튜플 대신 문자열 리스트
                                                    selected=[list(sensor_labels.keys())[0],list(sensor_labels.keys())[1],list(sensor_labels.keys())[2]],  # ✅ 기본 선택도 문자열 리스트
                                                    inline=True
                                                ),
                                                ui.navset_tab(
                                                    *[
                                                        ui.nav_panel(
                                                            f"몰드코드 {code}",
                                                            ui.output_plot(f"stream_plot_{code}", height="400px")
                                                        )
                                                        for code in mold_codes
                                                    ]
                                                ),
                                                class_="flex-fill me-3"  # 오른쪽 여백
                                            ),
                                            # 오른쪽: 실시간 값
                                            ui.div(
                                                ui.output_ui("real_time_values"),
                                                class_="flex-fill"
                                            ),
                                            class_="d-flex align-items-start"  # 가로 정렬
                                        ),
                                        class_="p-3"
                                    ),
                                    
                                ),
                                # [C] 실시간 로그
                                ui.card(
                                    ui.card_header("실시간 로그"),
                                    ui.div(
                                        ui.h5("실시간 로그"),
                                        ui.output_table("recent_data_table"),
                                        ui.output_ui("download_controls")  # 형식 선택 + 다운로드 버튼
                                    )
                                ),  
                            ),
                            
                            # ================================
                            # TAB 2: 이상 예측
                            # ================================
                            ui.nav_panel("공정 이상 탐지   (Process Anomaly Detection)",
                                ui.layout_columns(
                                    #TAB 2 [C] 시간에 따른 이상 분석
                                    ui.card(
                                        ui.card_header("이상 탐지 알림"),
                                        ui.output_ui("log_alert_for_defect"),
                                        ui.output_ui("anomaly_detail_table"),
                                        ui.input_action_button("clear_alerts", "알림 확인", class_="btn btn-sm btn-secondary")
                                    ),
                                    # TAB 2 [B] 이상 탐지 알림
                                    
                                    ui.card(
                                        ui.card_header("주요 변수의 이상 발생 횟수"),
                                        ui.output_plot("anomaly_variable_count", height="300px")
                                    ),
                                    col_widths=[6, 6]
                                ),
                                ui.layout_columns(
                                    ui.card(
                                        ui.card_header("시간에 따른 이상 분석"),
                                        ui.div(
                                            ui.input_select(
                                                "anomaly_chart_time_unit", 
                                                "시간 단위 선택", 
                                                choices=["1시간", "3시간", "일", "주", "월"], 
                                                selected="일"
                                            ),
                                            class_="mb-3"
                                        ),
                                        ui.output_plot("anomaly_p_chart", height="300px")
                                    ),
                                    
                # [D] [D] 이상치 내 불량률
                                    ui.card(
                                        ui.card_header("이상치 내 불량률"),
                                        ui.output_ui("anomaly_fail_rate_ui")
                                        
                                    ),
                                    col_widths=[6, 6]
                                )
                            ),
                            # ================================
                            # TAB 3: 품질
                            # ================================
                    
                                ui.nav_panel("품질 불량 판별   (Quality Defect Classification)",
                                    # TAB 3 [A] 
                                    ui.layout_columns(
                                        ui.card(
                                            ui.card_header("품질 불량 판별"),
                                            ui.output_ui("current_prediction"),
                                            ui.output_ui("prediction_log_table")
                                        ),
                                        # TAB 3 [B]
                                        ui.card(# TAB 3 [D]# TAB 3 [D]# TAB 3 [D]# TAB 3 [D]
                                            ui.card_header("품질 불량 판별 주요 센서"),
                                            ui.output_plot("shap_explanation_plot")
                                            
                                        )
                                        
                                    ),
                                    # TAB 3 [C]
                                    ui.layout_columns(
                                        ui.card(
                                            ui.card_header("단위 시간 당 불량 관리도"),
                                            ui.input_select(
                                                "fail_time_unit", 
                                                "시간 단위 선택", 
                                                choices=["1시간", "3시간", "일", "주", "월"], 
                                                selected="일"
                                            ),
                                            ui.output_plot("fail_rate_by_time", height="350px"),
                                        ),
                                        ui.card(
                                            ui.card_header("몰드 코드별 품질 불량 횟수"),
                                            ui.input_date_range(
                                                "date_range", 
                                                "기간 선택", 
                                                start="2019-02-21",  # 데이터 시작일
                                                end="2019-03-12",    # 데이터 종료일 # 기본값
                                            ),
                                            ui.output_plot("defect_rate_plot", height="300px")
                                        )
                                    )
                                ),
                                ui.nav_panel("부록 (Annexes)",
                                    ui.page_fluid(
                                    
                                        # 1단계
                                        ui.card(
                                            ui.card_header(ui.h3("프로젝트 개요 및 데이터 준비")),
                                            ui.HTML("""
                                            <h5>1. 대시보드 간단 소개</h5>
                                            <b>사용자:</b> 생산 라인 책임자 및 주요 관리자<br>
                                            <b>목적:</b> 공정 이상 탐지 및 불량 예측을 통한 자원·인력 낭비 방지, 신제품 공정 데이터 확보<br>
                                            <b>기능:</b> 실시간 데이터 스트리밍, 이상 탐지, 주요 원인 파악, 불량 예측
                                            <hr>
                                            <h5>2. 데이터 이슈 및 전처리 과정</h5>
                                            <b>데이터 이슈</b>
                                            이상치/결측치가 실제로 이싱치인지 결측치인지 판단이 어려움<br>
                                            일부 변수의 오기입 여부 불명확(1449 등 이상 데이터 다수 존재)<br><br>
                                            <b>주요 전처리</b>
                                            불필요 칼럼 ('id', 'line', 'name', 'mold_name', 'emergency_stop') 삭제<br>
                                            molten_temp 결측 → 최근값으로 대체 (온도 급락 불가능성 가정)<br>
                                            결측치만 있는 행 1개 삭제<br>
                                            production_cycletime = 0 → 전체 제거 (실생산 아님)<br>
                                            low_section_speed 이상치 제거<br>
                                            molten_volume 결측 → 최근값 대체 (변동 시에만 기록되는 특성 고려)<br>
                                            cast_pressure 200 이하 양품 25개 행 → boxplot 기반으로 삭제<br>
                                            1449번 등 명확한 이상행, upper3/lower3 변수 전체 제거<br>
                                            EMS_operation_time = 0인 행 삭제<br>
                                            heating_furnace, tryshot_signal 결측치 → 'unknown'으로 대체<br>
                                            불균형 데이터 (정상:불량 비율 고려) → XGBoost scale_pos_weight = 정상/불량 (예: 70,333/3,279 = 21.45)로 조정

                                            <hr>
                                            <h5>전처리 결과 시각화 보기</h5>
                                            <details style="margin-top: 10px;">
                                              <summary style="font-size: 16px; cursor: pointer;">수치형 변수 Boxplot</summary>
                                              <img src="수치형변수별_boxplot.png" style="width: 100%; margin: 10px 0;">
                                            </details>

                                            <details style="margin-top: 15px;">
                                              <summary style="font-size: 16px; cursor: pointer;">mold_code별 Boxplot</summary>
                                              <img src="mold_code별_boxplot.png" style="width: 100%; margin: 10px 0;">
                                            </details>
                                            """)
                                        ),


                                        # 2단계
                                        ui.card(
                                            ui.card_header(ui.h3("모델 구성 및 설정")),
                                            ui.HTML("""
                                            <h5>3. 사용 모델 및 간단 원리</h5>

                                            <b>Isolation Forest (이상 탐지):</b>
                                            mold_code별로 개별 모델 학습 및 예측<br>
                                            수치형 변수만 추출해 결측값은 평균으로 대체<br>
                                            contamination=0.1, random_state=42 설정<br>
                                            예측 결과로 is_anomaly (-1: 이상치, 1: 정상) 생성<br>
                                            decision_function 기반 anomaly_score 계산<br>
                                            anomaly_score 분위수 기준으로 anomaly_level(정상/경도/심각) 분류<br><br>

                                            <b>SHAP (이상 탐지 주요 변수 해석):</b>
                                            각 mold_code별 IsolationForest 모델에 TreeExplainer 적용<br>
                                            is_anomaly = -1인 이상치 샘플에 대해 SHAP 값 계산<br>
                                            SHAP 절댓값 기준 상위 3개 변수 추출<br><br>
                                                    
                                            <b>XGBoost (불량 예측)</b>
                                            수치형 변수는 StandardScaler, 범주형 변수는 OneHotEncoder 적용<br>
                                            ColumnTransformer로 수치형/범주형 전처리 후 Pipeline 구성<br>
                                            use_label_encoder=False, eval_metric='logloss' 설정<br>
                                            불균형 보정: scale_pos_weight = 21.45 적용<br>
                                            주요 하이퍼파라미터 튜닝 대상: learning_rate (0.1~0.35), max_depth (3~5), n_estimators (42)<br>
                                            GridSearchCV로 f1_macro, recall 기준 각각 교차검증 수행<br>
                                            최종 선택 모델은 f1_score 또는 recall 기준으로 평가<br><br>
                                            
                                            <b>SHAP (불량 예측 해석)</b>
                                            XGBoost 예측 결과 중 logit score 기준으로 SHAP 값 계산<br>
                                            TreeExplainer로 SHAP 값 도출: 각 변수의 로짓값 기여도 분석<br>
                                            절댓값 기준 SHAP 값이 큰 상위 변수 3개 추출<br>
                                            해당 변수들이 불량 확률을 높이는데 얼마나 기여했는지 정량 해석<br><br>
                                                    
                                            <b>Feature Importance (불량 주요 변수 해석)</b>
                                            XGBoost 최적 모델(best_estimator_)에서 전처리 후 피처 이름 추출<br>
                                            ColumnTransformer 기반 피처 이름: get_feature_names_out() 사용<br>
                                            모델에서 학습된 feature_importances_ 값을 함께 DataFrame으로 정리<br>
                                            중요도 기준 상위 변수 10개 추출 및 시각화 가능<br>
                                            주요 변수 기준으로 불량에 영향을 미치는 요인을 해석에 활용<br><br>
                                                    
                                            <hr>
                                            <h5>4. 모델 선정 이유</h5>
                                            <div style="display: flex; gap: 30px;">

                                              <!-- Isolation Forest 테이블 -->
                                              <div style="flex: 1;">
                                                <b>Isolation Forest (이상 탐지)</b><br>
                                                <table class="table table-bordered table-sm" style="font-size: 14px;">
                                                  <tr><th>항목</th><th>이유</th></tr>
                                                  <tr><td>학습 방식</td><td>비지도 학습 (라벨 없음)</td></tr>
                                                  <tr><td>전제 조건</td><td>정상 데이터 다수, 이상치 소수</td></tr>
                                                  <tr><td>불균형 영향</td><td>없음 (라벨 사용 안 함)</td></tr>
                                                  <tr><td>적합한 상황</td><td>이상치 여부만 판단하고자 할 때</td></tr>
                                                  <tr><td>파라미터로 이상 비율 설정</td><td>contamination으로 직접 제어 가능</td></tr>
                                                  <tr><td>사용 목적</td><td>공정에서 벗어난 이상 패턴 조기 탐지</td></tr>
                                                </table>
                                              </div>

                                              <!-- XGBoost 테이블 -->
                                              <div style="flex: 1;">
                                                <b>XGBoost (불량 판별)</b><br>
                                                <table class="table table-bordered table-sm" style="font-size: 14px;">
                                                  <tr><th>항목</th><th>이유</th></tr>
                                                  <tr><td>학습 방식</td><td>지도 학습 (정답 라벨 사용)</td></tr>
                                                  <tr><td>전제 조건</td><td>불량 데이터가 희소한 불균형 분류</td></tr>
                                                  <tr><td>불균형 보정 기능</td><td>scale_pos_weight로 가중치 조절</td></tr>
                                                  <tr><td>적합한 상황</td><td>불량/정상 구분 필요할 때</td></tr>
                                                  <tr><td>성능</td><td>높은 정확도 + 해석력 (Feature Importance 제공)</td></tr>
                                                  <tr><td>사용 목적</td><td>품질 불량 판별</td></tr>
                                                </table>
                                              </div>

                                            </div>
                                            """)
                                        ),

                                        # 3단계
                                        ui.card(
                                            ui.card_header(ui.h3("모델 성능(불량 판단 모델)")),
                                            ui.HTML("""
                                            <h5>5. 모델 성능 평가 결과</h5>

                                            <div style="display: flex; gap: 30px;">
                                              <div style="flex: 1;">
                                                <b>XGBoost Confusion Matrix (F1 기준)</b><br>
                                                <img src="XGBoost Confusion Matrix (f1 기준).png" style="width: 100%; margin-top: 10px;">
                                              </div>

                                              <div style="flex: 1;">
                                                <b>XGBoost Confusion Matrix (Recall 기준)</b><br>
                                                <img src="XGBoost Confusion Matrix (Recall 기준).png" style="width: 100%; margin-top: 10px;">
                                              </div>
                                            </div>

                                            <br><br>
                                            <b>간단 해석:</b>
                                            - F1 기준 모델은 정밀도(Precision)와 재현율(Recall) 사이의 균형을 중시, 전체적인 예측 안정성이 우수함 (Recall: 0.984, F1: 0.966)<br>
                                            - Recall 기준 모델은 불량을 놓치지 않는 것에 집중하여 민감도(Recall)가 높은 반면, Precision은 소폭 낮아짐 (Recall: 0.981, Precision: 0.944)<br>
                                            - 상황에 따라 정확한 예측(F1)과 불량 최소 누락(Recall) 중 업무 목적에 맞게 선택 가능
                                            """)
                                        )

                                    )
                                ),
                                ui.nav_spacer(),  # 선택
                            ui.nav_panel("🔓 로그아웃",  # ✅ 여기 추가!
                                ui.layout_column_wrap(
                                    ui.h4("로그아웃 하시겠습니까?"),
                                    ui.input_action_button("logout_button", "로그아웃", class_="btn btn-danger")
                                )
                            ),
                                id="main_nav",
                                title = "LS 기가 팩토리"
                            )
                        )
            
        
            
            
# ================================
# 🚀 4. 앱 실행
# ================================
app = App(app_ui, server, static_assets=STATIC_DIR)