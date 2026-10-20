import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 리눅스 서버(Streamlit Cloud) 한글 폰트 설정 (최상단 고정)
# --------------------------------------------------
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

try:
  font_name = fm.FontProperties(fname=font_path).get_name()
  plt.rcParams["font.family"] = font_name
except Exception:
  plt.rcParams["font.family"] = "sans-serif"

plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="수능 예측 분석", page_icon="📊", layout="wide"
)

SUBJECTS = ["국어", "수학", "영어", "탐구1", "탐구2"]
EXAMS = ["고2 3월", "고2 6월", "고2 9월", "고2 10월"]

# --------------------------------------------------
# 과목별/등급별 비율(%) 설정
# --------------------------------------------------
GRADE_RATIOS = {
    "국어": {
        1: 2.9,
        2: 5.0,
        3: 9.5,
        4: 15.2,
        5: 19.1,
        6: 19.0,
        7: 15.2,
        8: 9.6,
        9: 4.5,
    },
    "수학": {
        1: 2.2,
        2: 5.0,
        3: 10.5,
        4: 16.3,
        5: 18.2,
        6: 20.8,
        7: 15.6,
        8: 7.2,
        9: 4.2,
    },
    "영어": {
        1: 4.7,
        2: 13.6,
        3: 19.4,
        4: 21.3,
        5: 16.6,
        6: 10.9,
        7: 7.0,
        8: 5.1,
        9: 1.3,
    },
}

# 탐구1, 탐구2는 국어와 영어 비율의 중간치(산술평균)로 세팅
GRADE_RATIOS["탐구1"] = {
    g: round((GRADE_RATIOS["국어"][g] + GRADE_RATIOS["영어"][g]) / 2, 2)
    for g in range(1, 10)
}
GRADE_RATIOS["탐구2"] = GRADE_RATIOS["탐구1"].copy()


def get_grade_percentile_map(subject):
  """등급별 비율을 기반으로 해당 등급의 대표 백분위(100 - 중간 누적상위%) 계산"""
  ratios = GRADE_RATIOS[subject]
  percentile_map = {}
  cum_sum = 0.0

  for grade in range(1, 10):
    r = ratios[grade]
    mid_cum = cum_sum + (r / 2.0)  # 해당 등급 구간의 중간 누적 상위 %
    percentile_map[grade] = max(0.0, min(100.0, 100.0 - mid_cum))
    cum_sum += r

  return percentile_map


# 시험별 가중치 (3모 10%, 6모 20%, 9모 30%, 10모 40%)
EXAM_WEIGHTS = {
    "고2 3월": 0.10,
    "고2 6월": 0.20,
    "고2 9월": 0.30,
    "고2 10월": 0.40,
}


def percentile_to_grade(p, subject="국어"):
  """백분율(백분위)을 해당 과목의 등급 비율 매핑에 맞춰 환산"""
  top_p = 100.0 - p  # 상위 %
  ratios = GRADE_RATIOS.get(subject, GRADE_RATIOS["국어"])

  cum_sum = 0.0
  for grade in range(1, 10):
    cum_sum += ratios[grade]
    if top_p <= cum_sum:
      return grade
  return 9


def calculate_base_percentile(grades, subject):
  """응시한 시험의 등급을 입력받아 과목별 비율 기반 백분위 가중평균 산출"""
  valid = [
      (exam, grades[exam]) for exam in EXAMS if grades.get(exam) is not None
  ]

  if not valid:
    return None

  p_map = get_grade_percentile_map(subject)
  total_weight = sum(EXAM_WEIGHTS[exam] for exam, _ in valid)

  return (
      sum(p_map[grade] * EXAM_WEIGHTS[exam] for exam, grade in valid)
      / total_weight
  )


def predict_percentile(base, grad_share, grad_advantage, growth):
  """백분위 예측 로직"""
  if base >= 90:
    factor = 1.2
  elif base >= 80:
    factor = 1.0
  else:
    factor = 0.7

  competition_penalty = grad_share * grad_advantage * factor
  predicted = base + growth - competition_penalty

  return max(0.0, min(100.0, predicted))


# --------------------------------------------------
# 메인 UI
# --------------------------------------------------

st.title("📊 고2 성적으로 수능 예상하기")

st.write(
    "과목별 등급 비율(국/수/영/탐) 데이터를 적용하여 고3 수능"
    " 성적을 시나리오별로 예측합니다."
)

st.warning(
    "이 프로그램은 입력하신 과목별 등급 비율 분포 및 가정을 바탕으로"
    " 추정하는 모델입니다."
)

# --------------------------------------------------
# 1. 성적 입력
# --------------------------------------------------

st.header("1. 고2 모의고사 성적 입력")
st.caption(
    "※ 각 시험별로 취득한 **등급(1~9등급)**을 선택하세요. 미응시 시"
    " '미응시'를 선택합니다."
)

with st.form("score_form"):
  scores = {}

  for exam in EXAMS:
    st.subheader(exam)
    cols = st.columns(5)
    scores[exam] = {}

    for i, subject in enumerate(SUBJECTS):
      with cols[i]:
        grade = st.selectbox(
            f"{subject}",
            ["미응시"] + list(range(1, 10)),
            key=f"{exam}_{subject}",
        )
        scores[exam][subject] = None if grade == "미응시" else grade

  st.header("2. 재수생 유입 및 학습 변화 가정")

  grad_share = (
      st.slider(
          "졸업생·검정고시 등 응시자 비율 (%)",
          min_value=0,
          max_value=50,
          value=33,
          step=1,
          help="2026학년도 실제 비율을 참고한 기본값입니다.",
      )
      / 100
  )

  grad_advantage = st.slider(
      "재수생 성적 우위 가정 (백분위 포인트)",
      min_value=0,
      max_value=20,
      value=10,
      step=1,
      help="재수생이 현재 학생보다 상대적으로 강하다는 가정입니다.",
  )

  growth_mid = st.slider(
      "고2 이후 본인의 실력 변화 가정 (백분위 포인트)",
      min_value=-10,
      max_value=10,
      value=2,
      step=1,
      help=(
          "공부량, 약점 보완 및 시험 적응에 따른 가상의 변화를"
          " 설정합니다."
      ),
  )

  submitted = st.form_submit_button(
      "수능 예상 성적 계산", type="primary", use_container_width=True
  )


# --------------------------------------------------
# 2. 예측 계산
# --------------------------------------------------

if submitted:
  scenario_settings = {
      "보수적": {"advantage_multiplier": 1.4, "growth_delta": -3},
      "중간": {"advantage_multiplier": 1.0, "growth_delta": 0},
      "낙관적": {"advantage_multiplier": 0.6, "growth_delta": 2},
  }

  results = []
  base_values = {}

  for subject in SUBJECTS:
    subject_grades = {exam: scores[exam][subject] for exam in EXAMS}

    base_p = calculate_base_percentile(subject_grades, subject)

    if base_p is None:
      continue

    base_values[subject] = base_p

    row = {
        "과목": subject,
        "기준 백분위(추정)": f"{round(base_p, 1)}%",
        "기준 등급(추정)": f"{percentile_to_grade(base_p, subject)}등급",
    }

    for scenario, setting in scenario_settings.items():
      predicted_p = predict_percentile(
          base=base_p,
          grad_share=grad_share,
          grad_advantage=(
              grad_advantage * setting["advantage_multiplier"]
          ),
          growth=growth_mid + setting["growth_delta"],
      )

      predicted_grade = percentile_to_grade(predicted_p, subject)
      row[scenario] = (
          f"{predicted_grade}등급 (백분위 {round(predicted_p, 1)}%)"
      )

    results.append(row)

  if not results:
    st.error("성적을 하나 이상 입력해 주세요.")
  else:
    df = pd.DataFrame(results)

    st.header("3. 예상 수능 성적")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(
        "보수적·중간·낙관적 결과는 설정하신 과목별 등급 비율 분포 및"
        " 시나리오 가정을 적용한 계산 결과입니다."
    )

    # ------------------------------------------
    # 3. 시나리오별 백분위 비교 그래프
    # ------------------------------------------

    st.header("4. 시나리오별 예상 백분위 비교")

    graph_subjects = list(base_values.keys())

    if graph_subjects:
      fig, ax = plt.subplots(figsize=(11, 5))

      x = np.arange(len(graph_subjects))
      width = 0.24

      colors = {"보수적": "#B0C4DE", "중간": "#4682B4", "낙관적": "#1F4E79"}

      for i, scenario in enumerate(["보수적", "중간", "낙관적"]):
        values = []

        for subject in graph_subjects:
          setting = scenario_settings[scenario]
          base = base_values[subject]
          p = predict_percentile(
              base=base,
              grad_share=grad_share,
              grad_advantage=(
                  grad_advantage * setting["advantage_multiplier"]
              ),
              growth=growth_mid + setting["growth_delta"],
          )
          values.append(p)

        rects = ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=scenario,
            color=colors[scenario],
        )

        # 막대 위에 수치 표시
        for rect in rects:
          height = rect.get_height()
          ax.annotate(
              f"{height:.1f}%",
              xy=(rect.get_x() + rect.get_width() / 2, height),
              xytext=(0, 3),
              textcoords="offset points",
              ha="center",
              va="bottom",
              fontsize=8,
          )

      ax.set_xticks(x)
      ax.set_xticklabels(graph_subjects)
      ax.set_ylim(0, 110)
      ax.set_ylabel("예상 백분위 (%)")
      ax.set_title(
          "재수생 경쟁 및 학습 변화 가정별 비교 (가중치: 3모 10%, 6모 20%,"
          " 9모 30%, 10모 40%)"
      )
      ax.legend(loc="upper left")
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
        val = scores[exam][subject]
        row[exam] = f"{val}등급" if val is not None else "미응시"

      input_rows.append(row)

    input_df = pd.DataFrame(input_rows)
    st.dataframe(input_df, use_container_width=True, hide_index=True)

    st.download_button(
        "예측 결과 CSV 다운로드",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="수능_예측결과.csv",
        mime="text/csv",
    )
