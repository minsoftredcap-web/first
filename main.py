 
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

# 영어 등급별 가상 점수 매핑 (가중 평균 계산용)
ENGLISH_GRADE_SCORE = {
    1: 95.0,
    2: 85.0,
    3: 75.0,
    4: 65.0,
    5: 55.0,
    6: 45.0,
    7: 35.0,
    8: 25.0,
    9: 15.0,
}

# 시험별 가중치 (3모 10%, 6모 20%, 9모 30%, 10모 40%)
EXAM_WEIGHTS = {
    "고2 3월": 0.10,
    "고2 6월": 0.20,
    "고2 9월": 0.30,
    "고2 10월": 0.40,
}


def percentile_to_grade(p):
  """백분위를 수능 등급으로 환산"""
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


def score_to_english_grade(score):
  """영어 점수를 절대평가 등급으로 환산"""
  if score >= 90:
    return 1
  elif score >= 80:
    return 2
  elif score >= 70:
    return 3
  elif score >= 60:
    return 4
  elif score >= 50:
    return 5
  elif score >= 40:
    return 6
  elif score >= 30:
    return 7
  elif score >= 20:
    return 8
  return 9


def calculate_base_percentile(percentiles):
  """응시한 시험의 백분율(백분위)을 가중 평균하여 기준값 계산"""
  valid = [
      (exam, percentiles[exam])
      for exam in EXAMS
      if percentiles.get(exam) is not None
  ]

  if not valid:
    return None

  total_weight = sum(EXAM_WEIGHTS[exam] for exam, _ in valid)

  return (
      sum(p * EXAM_WEIGHTS[exam] for exam, p in valid) / total_weight
  )


def calculate_base_english(grades):
  """영어 가중 평균 점수 계산"""
  valid = [
      (exam, grades[exam]) for exam in EXAMS if grades.get(exam) is not None
  ]

  if not valid:
    return None

  total_weight = sum(EXAM_WEIGHTS[exam] for exam, _ in valid)

  return (
      sum(ENGLISH_GRADE_SCORE[grade] * EXAM_WEIGHTS[exam] for exam, grade in valid)
      / total_weight
  )


def predict_percentile(base, grad_share, grad_advantage, growth, subject):
  """국어, 수학, 탐구 백분위 예측"""
  if base >= 90:
    factor = 1.2
  elif base >= 80:
    factor = 1.0
  else:
    factor = 0.7

  competition_penalty = grad_share * grad_advantage * factor
  predicted = base + growth - competition_penalty

  return max(0.0, min(100.0, predicted))


def predict_english(base_score, growth_english):
  """영어 점수 예측 (절대평가 고려)"""
  predicted_score = base_score + (growth_english * 1.5)
  return max(0.0, min(100.0, predicted_score))


# --------------------------------------------------
# 메인UI 시작
# --------------------------------------------------

st.title("📊 고2 성적으로 수능 예상하기")

st.write(
    "고2 모의고사 성적(국/수/탐은 백분위, 영어는 등급)과 재수생 유입"
    " 가정을 이용해 고3 수능 성적을 시나리오별로 계산합니다."
)

st.warning(
    "이 프로그램은 실제 수능 결과로 검증된 예측 모델이 아닙니다. "
    "가중치, 재수생 경쟁 압력 및 향후 실력 변화는 사용자가 설정한"
    " 가정입니다."
)

# --------------------------------------------------
# 1. 성적 입력
# --------------------------------------------------

st.header("1. 고2 모의고사 성적 입력")
st.caption(
    "※ 영어는 **등급(1~9)**을 선택하고, 나머지 과목은 **백분위(0~100)**를"
    " 입력하세요. 미응시 시 빈칸으로 두거나 0을 유지하세요."
)

with st.form("score_form"):
  scores = {}

  for exam in EXAMS:
    st.subheader(exam)
    cols = st.columns(5)
    scores[exam] = {}

    for i, subject in enumerate(SUBJECTS):
      with cols[i]:
        if subject == "영어":
          # 영어는 등급 선택 Box
          grade = st.selectbox(
              f"{subject} (등급)",
              ["미응시"] + list(range(1, 10)),
              key=f"{exam}_{subject}",
          )
          scores[exam][subject] = None if grade == "미응시" else grade
        else:
          # 국/수/탐은 백분위 Number Input
          val = st.number_input(
              f"{subject} (백분위 %)",
              min_value=0.0,
              max_value=100.0,
              value=None,
              step=0.1,
              format="%.1f",
              key=f"{exam}_{subject}",
              help="미응시한 경우 비워두세요.",
          )
          scores[exam][subject] = val

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
      "고2 이후 본인의 실력 변화 가정 (백분위/점수 포인트)",
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
    subject_scores = {exam: scores[exam][subject] for exam in EXAMS}

    # 영어 과목 예측 로직 (등급 기반 처리)
    if subject == "영어":
      base_score = calculate_base_english(subject_scores)
      if base_score is None:
        continue

      base_values[subject] = base_score
      row = {"과목": subject, "기준 성적(추정)": f"{round(base_score, 1)}점"}

      for scenario, setting in scenario_settings.items():
        pred_score = predict_english(
            base_score=base_score,
            growth_english=growth_mid + setting["growth_delta"],
        )
        pred_grade = score_to_english_grade(pred_score)
        row[scenario] = f"{pred_grade}등급 ({round(pred_score, 1)}점)"

      results.append(row)
      continue

    # 국어, 수학, 탐구1, 탐구2 예측 로직 (백분위 기반 처리)
    base_p = calculate_base_percentile(subject_scores)

    if base_p is None:
      continue

    base_values[subject] = base_p

    row = {"과목": subject, "기준 성적(추정)": f"백분위 {round(base_p, 1)}%"}

    for scenario, setting in scenario_settings.items():
      predicted_p = predict_percentile(
          base=base_p,
          grad_share=grad_share,
          grad_advantage=(
              grad_advantage * setting["advantage_multiplier"]
          ),
          growth=growth_mid + setting["growth_delta"],
          subject=subject,
      )

      predicted_grade = percentile_to_grade(predicted_p)
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
        "보수적·중간·낙관적 결과는 통계적 신뢰구간이 아닙니다. "
        "서로 다른 가정을 적용한 계산 결과입니다."
    )

    # ------------------------------------------
    # 3. 시나리오별 백분위/등급 그래프
    # ------------------------------------------

    st.header("4. 시나리오별 예상 백분위 비교 (영어는 예상 점수)")

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

          if subject == "영어":
            base_score = base_values[subject]
            p_score = predict_english(
                base_score=base_score,
                growth_english=growth_mid + setting["growth_delta"],
            )
            values.append(p_score)  # 영어는 예상 점수로 표시
          else:
            base = base_values[subject]
            p = predict_percentile(
                base=base,
                grad_share=grad_share,
                grad_advantage=(
                    grad_advantage * setting["advantage_multiplier"]
                ),
                growth=growth_mid + setting["growth_delta"],
                subject=subject,
            )
            values.append(p)  # 국/수/탐은 백분위 수치로 표시

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
              f"{height:.1f}",
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
      ax.set_ylabel("예상 백분위 / 점수 (%)")
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
        if val is None:
          row[exam] = "미응시"
        else:
          row[exam] = f"{val}등급" if subject == "영어" else f"{val}%"

      input_rows.append(row)

    input_df = pd.DataFrame(input_rows)
    st.dataframe(input_df, use_container_width=True, hide_index=True)

    st.download_button(
        "예측 결과 CSV 다운로드",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="수능_예측결과.csv",
        mime="text/csv",
    )

    st.info(
        "실제 수능 등급을 더 정확하게 추정하려면 고3 6월·9월 모의평가의"
        " 표준점수와 백분위를 추가하고, 과거 모의고사-수능 대응 자료로"
        " 모델을 검증해야 합니다."
    )
