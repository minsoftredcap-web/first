
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st


# 맷플롯립 폰트 설정 (나눔고딕 지정)
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

# 폰트 캐시 강제 리프레시 (리눅스 서버 환경 대응)
try:
  fm._load_fontmanager(try_read_cache=False)
except Exception:
  pass
    
st.set_page_config(
    page_title="수능 예측 분석",
    page_icon="📊",
    layout="wide"
)

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SUBJECTS = ["국어", "수학", "영어", "탐구1", "탐구2"]
EXAMS = ["고2 3월", "고2 6월", "고2 9월", "고2 10월"]

# 등급별 대표 백분위 추정값
# 실제 백분위가 아니라 계산을 위한 근사값
GRADE_PERCENTILE = {
    1: 98.0,
    2: 92.5,
    3: 83.0,
    4: 68.5,
    5: 50.0,
    6: 31.5,
    7: 17.0,
    8: 7.5,
    9: 2.0
}

# 가정에 따른 시험별 가중치
EXAM_WEIGHTS = {
    "고2 3월": 0.10,
    "고2 6월": 0.20,
    "고2 9월": 0.30,
    "고2 10월": 0.40
}


def percentile_to_grade(p):
    if p >= 96:
        return 1
    elif p >= 89:
        return 2
    elif p >= 77:
        return 3
    elif p >= 60:
        return 4
    elif p >= 40:
        return 5
    elif p >= 23:
        return 6
    elif p >= 11:
        return 7
    elif p >= 4:
        return 8
    return 9


def calculate_base_percentile(grades):
    """응시한 시험의 등급을 가중 평균하여 기준값 계산"""
    valid = [
        (exam, grades[exam])
        for exam in EXAMS
        if grades.get(exam) is not None
    ]

    if not valid:
        return None

    total_weight = sum(
        EXAM_WEIGHTS[exam] for exam, _ in valid
    )

    return sum(
        GRADE_PERCENTILE[grade] * EXAM_WEIGHTS[exam]
        for exam, grade in valid
    ) / total_weight


def predict_percentile(base, grad_share, grad_advantage,
                       growth, subject):
    """
    단순 가정 기반 시나리오.
    grad_advantage: 재수생 성적 우위를 백분위 포인트로 가정.
    growth: 고2 이후 본인의 실력 변화를 백분위 포인트로 가정.
    """

    if subject == "영어":
        # 영어는 절대평가이므로 재수생 비율로 직접 등급을 깎지 않음
        return None

    # 상위권일수록 경쟁 압력이 커진다는 가정
    if base >= 90:
        factor = 1.2
    elif base >= 80:
        factor = 1.0
    else:
        factor = 0.7

    competition_penalty = (
        grad_share * grad_advantage * factor
    )

    predicted = base + growth - competition_penalty

    return max(0.0, min(100.0, predicted))


st.title("📊 고2 성적으로 수능 예상하기")

st.write(
    "고2 모의고사 성적과 재수생 유입 가정을 이용해 "
    "고3 수능 성적을 시나리오별로 계산합니다."
)

st.warning(
    "이 프로그램은 실제 수능 결과로 검증된 예측 모델이 아닙니다. "
    "등급별 백분위 환산, 가중치, 재수생 경쟁 압력 및 "
    "향후 실력 변화는 사용자가 설정한 가정입니다."
)

# --------------------------------------------------
# 1. 성적 입력
# --------------------------------------------------

st.header("1. 고2 모의고사 성적 입력")

with st.form("score_form"):
    scores = {}

    for exam in EXAMS:
        st.subheader(exam)
        cols = st.columns(5)
        scores[exam] = {}

        for i, subject in enumerate(SUBJECTS):
            with cols[i]:
                grade = st.selectbox(
                    subject,
                    ["미응시"] + list(range(1, 10)),
                    key=f"{exam}_{subject}"
                )

                scores[exam][subject] = (
                    None if grade == "미응시" else grade
                )

    st.header("2. 재수생 유입 및 학습 변화 가정")

    grad_share = st.slider(
        "졸업생·검정고시 등 응시자 비율 (%)",
        min_value=0,
        max_value=50,
        value=33,
        step=1,
        help="2026학년도 실제 비율을 참고한 기본값입니다. "
             "미래 수능 비율을 의미하지는 않습니다."
    ) / 100

    grad_advantage = st.slider(
        "재수생 성적 우위 가정 (백분위 포인트)",
        min_value=0,
        max_value=20,
        value=10,
        step=1,
        help="재수생이 현재 학생보다 상대적으로 강하다는 "
             "가정을 수치화한 값입니다. 공식 통계 추정치는 아닙니다."
    )

    growth_mid = st.slider(
        "고2 이후 본인의 실력 변화 가정 (백분위 포인트)",
        min_value=-10,
        max_value=10,
        value=2,
        step=1,
        help="공부량, 약점 보완 및 시험 적응에 따른 "
             "가상의 변화를 설정합니다."
    )

    submitted = st.form_submit_button(
        "수능 예상 성적 계산",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# 2. 예측
# --------------------------------------------------

if submitted:
    scenario_settings = {
        "보수적": {
            "advantage_multiplier": 1.4,
            "growth_delta": -3
        },
        "중간": {
            "advantage_multiplier": 1.0,
            "growth_delta": 0
        },
        "낙관적": {
            "advantage_multiplier": 0.6,
            "growth_delta": 2
        }
    }

    results = []
    base_values = {}

    for subject in SUBJECTS:
        subject_grades = {
            exam: scores[exam][subject]
            for exam in EXAMS
        }

        if subject == "영어":
            # 절대평가: 가장 최근 응시 등급 표시
            valid = [
                scores[exam][subject]
                for exam in EXAMS
                if scores[exam][subject] is not None
            ]

            if not valid:
                continue

            latest_grade = valid[-1]

            row = {
                "과목": subject,
                "기준값": f"최근 {latest_grade}등급",
                "보수적": f"{latest_grade}등급",
                "중간": f"{latest_grade}등급",
                "낙관적": f"{latest_grade}등급"
            }

            results.append(row)
            continue

        base = calculate_base_percentile(subject_grades)

        if base is None:
            continue

        base_values[subject] = base

        row = {
            "과목": subject,
            "기준 백분위(추정)": round(base, 1)
        }

        for scenario, setting in scenario_settings.items():
            predicted_p = predict_percentile(
                base=base,
                grad_share=grad_share,
                grad_advantage=(
                    grad_advantage
                    * setting["advantage_multiplier"]
                ),
                growth=growth_mid + setting["growth_delta"],
                subject=subject
            )

            predicted_grade = percentile_to_grade(predicted_p)

            row[scenario] = f"{predicted_grade}등급"
            row[f"{scenario} 백분위"] = round(predicted_p, 1)

        results.append(row)

    if not results:
        st.error("성적을 하나 이상 입력해 주세요.")
    else:
        df = pd.DataFrame(results)

        st.header("3. 예상 수능 성적")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "보수적·중간·낙관적 결과는 통계적 신뢰구간이 아닙니다. "
            "서로 다른 가정을 적용한 계산 결과입니다."
        )

        # ------------------------------------------
        # 3. 시나리오별 그래프
        # ------------------------------------------

        st.header("4. 시나리오별 예상 등급 비교")

        graph_subjects = list(base_values.keys())

        if graph_subjects:
            fig, ax = plt.subplots(figsize=(11, 5))

            x = np.arange(len(graph_subjects))
            width = 0.24

            colors = {
                "보수적": "#B0C4DE",
                "중간": "#4682B4",
                "낙관적": "#1F4E79"
            }

            for i, scenario in enumerate(
                ["보수적", "중간", "낙관적"]
            ):
                values = []

                for subject in graph_subjects:
                    base = base_values[subject]
                    setting = scenario_settings[scenario]

                    p = predict_percentile(
                        base=base,
                        grad_share=grad_share,
                        grad_advantage=(
                            grad_advantage
                            * setting["advantage_multiplier"]
                        ),
                        growth=growth_mid + setting["growth_delta"],
                        subject=subject
                    )

                    values.append(percentile_to_grade(p))

                ax.bar(
                    x + (i - 1) * width,
                    values,
                    width,
                    label=scenario,
                    color=colors[scenario]
                )

            ax.set_xticks(x)
            ax.set_xticklabels(graph_subjects)
            ax.set_yticks(range(1, 10))
            ax.set_ylim(9.5, 0.5)
            ax.set_ylabel("예상 등급 (위쪽이 1등급)")
            ax.set_title("재수생 경쟁 및 학습 변화 가정별 비교")
            ax.legend()
            ax.grid(axis="y", alpha=0.25)

            st.pyplot(fig)
            plt.close(fig)

        # ------------------------------------------
        # 4. 입력 성적 확인 및 다운로드
        # ------------------------------------------

        st.header("5. 입력한 모의고사 성적")

        input_rows = []

        for subject in SUBJECTS:
            row = {"과목": subject}

            for exam in EXAMS:
                grade = scores[exam][subject]
                row[exam] = (
                    f"{grade}등급"
                    if grade is not None
                    else "미응시"
                )

            input_rows.append(row)

        input_df = pd.DataFrame(input_rows)
        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "예측 결과 CSV 다운로드",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="수능_예측결과.csv",
            mime="text/csv"
        )

        st.info(
            "실제 수능 등급을 더 정확하게 추정하려면 "
            "고3 6월·9월 모의평가의 표준점수와 백분위를 추가하고, "
            "과거 모의고사-수능 대응 자료로 모델을 검증해야 합니다."
        )
