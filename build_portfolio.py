import csv
import json
import math
import os
import random
import statistics
import zipfile
from collections import Counter, defaultdict


ROOT = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(ROOT, "diabetes_uci.zip")
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "outputs")


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_data():
    csv_path = os.path.join(DATA_DIR, "diabetic_data.csv")
    mapping_path = os.path.join(DATA_DIR, "IDS_mapping.csv")
    if os.path.exists(csv_path) and os.path.exists(mapping_path):
        return csv_path, mapping_path
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    return csv_path, mapping_path


def is_missing(value):
    return value is None or value.strip() in {"", "?", "Unknown/Invalid"}


def to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def age_midpoint(age_band):
    # UCI stores age as strings like [50-60)
    cleaned = age_band.replace("[", "").replace(")", "")
    low, high = cleaned.split("-")
    return (int(low) + int(high)) // 2


def bucket_los(days):
    if days <= 2:
        return "1-2 days"
    if days <= 5:
        return "3-5 days"
    if days <= 8:
        return "6-8 days"
    return "9-14 days"


def read_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def pct(part, whole):
    return round(100 * part / whole, 1) if whole else 0.0


def top_counter(counter, n=10):
    total = sum(counter.values())
    return [
        {"name": name, "count": count, "pct": pct(count, total)}
        for name, count in counter.most_common(n)
    ]


def grouped_rates(rows, field, limit=None):
    groups = defaultdict(lambda: {"encounters": 0, "readmit_lt30": 0})
    for row in rows:
        key = row.get(field, "Unknown")
        if is_missing(key):
            key = "Missing"
        groups[key]["encounters"] += 1
        if row["readmitted"] == "<30":
            groups[key]["readmit_lt30"] += 1
    items = []
    for key, value in groups.items():
        items.append(
            {
                "name": key,
                "encounters": value["encounters"],
                "readmit_lt30": value["readmit_lt30"],
                "rate": pct(value["readmit_lt30"], value["encounters"]),
            }
        )
    items.sort(key=lambda x: (-x["rate"], -x["encounters"]))
    return items[:limit] if limit else items


def build_dashboard_sample(rows):
    sample_path = os.path.join(OUTPUT_DIR, "dashboard_sample.csv")
    fields = [
        "encounter_id",
        "patient_nbr",
        "race",
        "gender",
        "age",
        "time_in_hospital",
        "num_lab_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "A1Cresult",
        "insulin",
        "change",
        "diabetesMed",
        "readmitted",
        "readmit_lt30",
        "los_bucket",
        "age_midpoint",
    ]
    with open(sample_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows[:20000]:
            out = {key: row.get(key, "") for key in fields if key in row}
            out["readmit_lt30"] = 1 if row["readmitted"] == "<30" else 0
            out["los_bucket"] = bucket_los(to_int(row["time_in_hospital"]))
            out["age_midpoint"] = age_midpoint(row["age"])
            writer.writerow(out)
    return sample_path


def sigmoid(x):
    if x < -35:
        return 0.0
    if x > 35:
        return 1.0
    return 1 / (1 + math.exp(-x))


def auc_score(y_true, y_score):
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.0
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2
        positives_in_tie = sum(label for _, label in pairs[i : j + 1])
        rank_sum += positives_in_tie * avg_rank
        i = j + 1
    return round((rank_sum - pos * (pos + 1) / 2) / (pos * neg), 3)


def encode_features(row):
    features = {
        "bias": 1.0,
        "time_in_hospital": to_int(row["time_in_hospital"]) / 14,
        "num_lab_procedures": to_int(row["num_lab_procedures"]) / 130,
        "num_procedures": to_int(row["num_procedures"]) / 6,
        "num_medications": to_int(row["num_medications"]) / 80,
        "number_outpatient": min(to_int(row["number_outpatient"]), 10) / 10,
        "number_emergency": min(to_int(row["number_emergency"]), 10) / 10,
        "number_inpatient": min(to_int(row["number_inpatient"]), 10) / 10,
        "number_diagnoses": to_int(row["number_diagnoses"]) / 16,
        "age_midpoint": age_midpoint(row["age"]) / 100,
        "diabetesMed_yes": 1.0 if row["diabetesMed"] == "Yes" else 0.0,
        "med_change_yes": 1.0 if row["change"] == "Ch" else 0.0,
    }
    for value in ["No", "Steady", "Up", "Down"]:
        features[f"insulin_{value}"] = 1.0 if row["insulin"] == value else 0.0
    for value in ["None", "Norm", ">7", ">8"]:
        features[f"A1C_{value}"] = 1.0 if row["A1Cresult"] == value else 0.0
    return features


def train_lightweight_model(rows):
    random.seed(42)
    model_rows = rows[:]
    random.shuffle(model_rows)
    model_rows = model_rows[:25000]
    split = int(len(model_rows) * 0.8)
    train = model_rows[:split]
    test = model_rows[split:]
    feature_names = sorted(encode_features(train[0]).keys())
    weights = {name: 0.0 for name in feature_names}
    lr = 0.08
    l2 = 0.0005

    for _ in range(24):
        random.shuffle(train)
        for row in train:
            y = 1.0 if row["readmitted"] == "<30" else 0.0
            x = encode_features(row)
            score = sum(weights[name] * x[name] for name in feature_names)
            error = sigmoid(score) - y
            for name in feature_names:
                if name == "bias":
                    weights[name] -= lr * error * x[name]
                else:
                    weights[name] -= lr * (error * x[name] + l2 * weights[name])

    y_true = []
    y_prob = []
    for row in test:
        x = encode_features(row)
        prob = sigmoid(sum(weights[name] * x[name] for name in feature_names))
        y_prob.append(prob)
        y_true.append(1 if row["readmitted"] == "<30" else 0)

    threshold = 0.16
    preds = [1 if p >= threshold else 0 for p in y_prob]
    tp = sum(1 for y, p in zip(y_true, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, preds) if y == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    top_weights = sorted(
        [{"feature": k, "weight": round(v, 3)} for k, v in weights.items() if k != "bias"],
        key=lambda x: abs(x["weight"]),
        reverse=True,
    )[:10]
    return {
        "model_type": "Lightweight logistic regression baseline",
        "target": "Readmission within 30 days",
        "training_rows": len(train),
        "test_rows": len(test),
        "threshold": threshold,
        "accuracy": round((tp + tn) / len(test), 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "auc": auc_score(y_true, y_prob),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "top_model_signals": top_weights,
    }


def analyze(csv_path):
    rows = []
    missing_counts = Counter()
    numeric_fields = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]
    numeric_values = {field: [] for field in numeric_fields}

    for row in read_rows(csv_path):
        rows.append(row)
        for key, value in row.items():
            if is_missing(value):
                missing_counts[key] += 1
        for field in numeric_fields:
            numeric_values[field].append(to_int(row[field]))

    total = len(rows)
    unique_patients = len({row["patient_nbr"] for row in rows})
    readmit_lt30 = sum(1 for row in rows if row["readmitted"] == "<30")
    readmit_any = sum(1 for row in rows if row["readmitted"] != "NO")

    age_counter = Counter(row["age"] for row in rows)
    race_counter = Counter("Missing" if is_missing(row["race"]) else row["race"] for row in rows)
    readmission_counter = Counter(row["readmitted"] for row in rows)
    a1c_counter = Counter(row["A1Cresult"] for row in rows)
    insulin_counter = Counter(row["insulin"] for row in rows)
    los_counter = Counter(bucket_los(to_int(row["time_in_hospital"])) for row in rows)

    metrics = {
        "project_title": "Diabetes Readmission Analytics Dashboard",
        "data_source": "UCI Machine Learning Repository - Diabetes 130-US Hospitals for Years 1999-2008",
        "source_url": "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "kpis": {
            "encounters": total,
            "unique_patients": unique_patients,
            "features": len(rows[0]) if rows else 0,
            "readmitted_within_30_days": readmit_lt30,
            "readmitted_within_30_days_rate": pct(readmit_lt30, total),
            "any_readmission_rate": pct(readmit_any, total),
            "avg_time_in_hospital": round(statistics.mean(numeric_values["time_in_hospital"]), 2),
            "avg_lab_procedures": round(statistics.mean(numeric_values["num_lab_procedures"]), 2),
            "avg_medications": round(statistics.mean(numeric_values["num_medications"]), 2),
        },
        "distributions": {
            "readmission": top_counter(readmission_counter, 5),
            "age": top_counter(age_counter, 10),
            "race": top_counter(race_counter, 10),
            "A1Cresult": top_counter(a1c_counter, 10),
            "insulin": top_counter(insulin_counter, 10),
            "length_of_stay_bucket": top_counter(los_counter, 10),
        },
        "grouped_readmission_rates": {
            "age": grouped_rates(rows, "age"),
            "race": grouped_rates(rows, "race"),
            "A1Cresult": grouped_rates(rows, "A1Cresult"),
            "insulin": grouped_rates(rows, "insulin"),
            "medical_specialty_top": grouped_rates(rows, "medical_specialty", 12),
        },
        "data_quality": [
            {
                "field": field,
                "missing_count": count,
                "missing_rate": pct(count, total),
            }
            for field, count in missing_counts.most_common(12)
        ],
    }
    sample_path = build_dashboard_sample(rows)
    metrics["model"] = train_lightweight_model(rows)
    metrics_path = os.path.join(OUTPUT_DIR, "summary_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return rows, metrics, metrics_path, sample_path


def bar_list(items, value_key="rate", label_suffix="%"):
    max_value = max((item[value_key] for item in items), default=1)
    html = []
    for item in items:
        value = item[value_key]
        width = max(4, value / max_value * 100)
        label = item["name"]
        sub = f'{item.get("encounters", item.get("count", "")):,} encounters' if item.get("encounters") else f'{item.get("count", ""):,} records'
        html.append(
            f"""
            <div class="bar-row">
              <div class="bar-label"><strong>{label}</strong><span>{sub}</span></div>
              <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>
              <div class="bar-value">{value}{label_suffix}</div>
            </div>
            """
        )
    return "\n".join(html)


def model_signal_list(signals):
    rows = []
    for signal in signals:
        direction = "Higher risk signal" if signal["weight"] > 0 else "Lower risk signal"
        rows.append(
            f"<tr><td>{signal['feature']}</td><td>{signal['weight']}</td><td>{direction}</td></tr>"
        )
    return "\n".join(rows)


def generate_dashboard(metrics):
    k = metrics["kpis"]
    model = metrics["model"]
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{metrics["project_title"]}</title>
  <style>
    :root {{
      --ink: #1f2933;
      --muted: #627083;
      --line: #d7dde5;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --blue: #2563eb;
      --teal: #0f766e;
      --amber: #b45309;
      --red: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.45;
    }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 28px 36px 24px;
    }}
    header h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    header p {{
      margin: 0;
      max-width: 980px;
      color: var(--muted);
      font-size: 15px;
    }}
    main {{
      padding: 24px 36px 40px;
      max-width: 1280px;
      margin: 0 auto;
    }}
    section {{
      margin-bottom: 24px;
    }}
    h2 {{
      font-size: 18px;
      margin: 0 0 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}
    .two {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .panel, .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric .value {{
      font-size: 27px;
      font-weight: 700;
      margin-top: 8px;
    }}
    .metric .note {{
      color: var(--muted);
      margin-top: 5px;
      font-size: 13px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 180px 1fr 62px;
      align-items: center;
      gap: 12px;
      padding: 7px 0;
      border-bottom: 1px solid #edf1f5;
    }}
    .bar-row:last-child {{ border-bottom: 0; }}
    .bar-label strong {{
      display: block;
      font-size: 13px;
    }}
    .bar-label span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .bar-track {{
      height: 12px;
      background: #e9eef5;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 12px;
      background: linear-gradient(90deg, var(--teal), var(--blue));
    }}
    .bar-value {{
      text-align: right;
      font-weight: 700;
      font-size: 13px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid #edf1f5;
      padding: 8px 6px;
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .callout {{
      border-left: 4px solid var(--blue);
      background: #eef5ff;
      padding: 14px 16px;
      border-radius: 8px;
      color: #22324a;
    }}
    .model-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin: 12px 0 16px;
    }}
    .model-box {{
      background: #f8fafc;
      border: 1px solid #e4e9f0;
      border-radius: 8px;
      padding: 12px;
    }}
    .model-box span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .model-box strong {{
      display: block;
      margin-top: 4px;
      font-size: 21px;
    }}
    footer {{
      color: var(--muted);
      font-size: 12px;
      padding-top: 14px;
      border-top: 1px solid var(--line);
    }}
    @media (max-width: 920px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      .grid, .two, .model-grid {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 130px 1fr 54px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{metrics["project_title"]}</h1>
    <p>Portfolio project using real clinical encounter data from 130 U.S. hospitals. The dashboard summarizes readmission patterns, data quality, care utilization indicators, and a baseline predictive model for 30-day readmission risk.</p>
  </header>
  <main>
    <section class="grid">
      <div class="metric"><div class="label">Hospital Encounters</div><div class="value">{k["encounters"]:,}</div><div class="note">1999-2008 diabetes inpatient records</div></div>
      <div class="metric"><div class="label">Unique Patients</div><div class="value">{k["unique_patients"]:,}</div><div class="note">Longitudinal patient-level records</div></div>
      <div class="metric"><div class="label">30-Day Readmission</div><div class="value">{k["readmitted_within_30_days_rate"]}%</div><div class="note">{k["readmitted_within_30_days"]:,} encounters readmitted within 30 days</div></div>
      <div class="metric"><div class="label">Average Length of Stay</div><div class="value">{k["avg_time_in_hospital"]}</div><div class="note">days per encounter</div></div>
    </section>

    <section class="two">
      <div class="panel">
        <h2>Readmission Rate by Age Group</h2>
        {bar_list(metrics["grouped_readmission_rates"]["age"])}
      </div>
      <div class="panel">
        <h2>Readmission Rate by A1C Result</h2>
        {bar_list(metrics["grouped_readmission_rates"]["A1Cresult"])}
      </div>
    </section>

    <section class="two">
      <div class="panel">
        <h2>Insulin Medication Segment</h2>
        {bar_list(metrics["grouped_readmission_rates"]["insulin"])}
      </div>
      <div class="panel">
        <h2>Length of Stay Distribution</h2>
        {bar_list(metrics["distributions"]["length_of_stay_bucket"], "pct", "%")}
      </div>
    </section>

    <section class="panel">
      <h2>Predictive Analytics Baseline</h2>
      <div class="callout">Target: predict whether a diabetes inpatient encounter is followed by readmission within 30 days. This is a baseline logistic model built for portfolio demonstration and model interpretation, not clinical deployment.</div>
      <div class="model-grid">
        <div class="model-box"><span>AUC</span><strong>{model["auc"]}</strong></div>
        <div class="model-box"><span>Accuracy</span><strong>{model["accuracy"]}</strong></div>
        <div class="model-box"><span>Precision</span><strong>{model["precision"]}</strong></div>
        <div class="model-box"><span>Recall</span><strong>{model["recall"]}</strong></div>
        <div class="model-box"><span>F1</span><strong>{model["f1"]}</strong></div>
      </div>
      <table>
        <thead><tr><th>Model Signal</th><th>Weight</th><th>Interpretation</th></tr></thead>
        <tbody>{model_signal_list(model["top_model_signals"])}</tbody>
      </table>
    </section>

    <section class="two">
      <div class="panel">
        <h2>Data Quality Checks</h2>
        <table>
          <thead><tr><th>Field</th><th>Missing Records</th><th>Missing Rate</th></tr></thead>
          <tbody>
            {"".join(f"<tr><td>{x['field']}</td><td>{x['missing_count']:,}</td><td>{x['missing_rate']}%</td></tr>" for x in metrics["data_quality"])}
          </tbody>
        </table>
      </div>
      <div class="panel">
        <h2>Project Value</h2>
        <p>This project demonstrates the full healthcare analytics workflow: data profiling, KPI design, cohort segmentation, dashboard reporting, data quality review, and a predictive modeling baseline.</p>
        <p>It is designed for Healthcare Data Analyst, Clinical BI Analyst, Quality Reporting Analyst, and entry-level Population Health Analytics applications.</p>
      </div>
    </section>

    <footer>
      Data source: {metrics["data_source"]}. Dataset page: {metrics["source_url"]}. License: {metrics["license"]}.
    </footer>
  </main>
</body>
</html>
"""
    path = os.path.join(ROOT, "diabetes_readmission_dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def generate_readme(metrics):
    readme = f"""# Diabetes Readmission Analytics Portfolio

## Project Summary

This portfolio project uses the UCI Diabetes 130-US Hospitals dataset to analyze inpatient diabetes encounters, monitor 30-day readmission patterns, evaluate data quality, and build a baseline predictive model for readmission risk.

The project is designed for Healthcare Data Analyst, Clinical BI Analyst, Quality Reporting Analyst, Population Health Analytics, and Healthcare AI transition roles.

## Data Source

- Dataset: Diabetes 130-US Hospitals for Years 1999-2008
- Source: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- License: Creative Commons Attribution 4.0 International
- Scope: {metrics["kpis"]["encounters"]:,} inpatient diabetes encounters from 130 U.S. hospitals and integrated delivery networks

## Business / Clinical Questions

1. What share of diabetes inpatient encounters result in readmission within 30 days?
2. Which patient and care-utilization segments show higher readmission rates?
3. What data quality issues should be reviewed before reporting or modeling?
4. Can a baseline predictive model identify patients at higher readmission risk?

## Dashboard Pages / Sections

- Executive KPIs: encounters, unique patients, 30-day readmission rate, average length of stay
- Segment Analysis: readmission rate by age, A1C result, insulin status, and length of stay bucket
- Predictive Analytics Baseline: logistic regression model metrics and top model signals
- Data Quality Checks: missing-value profile for high-impact fields

## Key Results

- Total encounters: {metrics["kpis"]["encounters"]:,}
- Unique patients: {metrics["kpis"]["unique_patients"]:,}
- 30-day readmission rate: {metrics["kpis"]["readmitted_within_30_days_rate"]}%
- Any readmission rate: {metrics["kpis"]["any_readmission_rate"]}%
- Baseline model AUC: {metrics["model"]["auc"]}

## Tools Demonstrated

- SQL-ready relational thinking
- Python data cleaning and feature engineering
- Healthcare KPI design
- Clinical cohort segmentation
- Data quality profiling
- BI dashboard design
- Baseline predictive analytics

## Resume Bullets

- Built a healthcare analytics portfolio project using 101K+ diabetes inpatient encounters from 130 U.S. hospitals to analyze 30-day readmission patterns, patient segments, care utilization, and data quality.
- Designed a BI-style dashboard tracking readmission KPIs, age and A1C cohorts, insulin medication segments, length-of-stay patterns, and missing-data checks for clinical reporting use cases.
- Developed a baseline predictive model for 30-day readmission risk using Python feature engineering and logistic regression, evaluating performance with AUC, precision, recall, and F1 score.

## Interview Pitch

I built a healthcare analytics project using real inpatient diabetes encounter data from U.S. hospitals. I cleaned and profiled the data, created readmission KPIs, segmented patients by clinical and utilization factors, checked data quality issues, and built a baseline predictive model for 30-day readmission risk. The project demonstrates the kind of workflow I want to apply in healthcare data analyst and clinical BI roles: turning patient-level data into reliable reporting, quality insights, and predictive analytics.
"""
    path = os.path.join(ROOT, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(readme)
    return path


def generate_project_brief(metrics):
    brief = f"""Healthcare Analytics Portfolio Brief

Project: Diabetes Readmission Analytics Dashboard

Target role fit:
Healthcare Data Analyst | Clinical BI Analyst | Quality Reporting Analyst | Population Health Analytics Associate

Dataset:
UCI Diabetes 130-US Hospitals for Years 1999-2008
{metrics["source_url"]}

Portfolio narrative:
This project analyzes real inpatient diabetes encounter data from 130 U.S. hospitals to identify patterns associated with 30-day readmission. It demonstrates a practical healthcare analytics workflow: data profiling, KPI development, cohort analysis, dashboard reporting, data quality checks, and a baseline predictive model.

Why this fits my background:
- RBT experience: clinical documentation, client-level progress tracking, behavioral health data accuracy
- Dialysis-X: healthcare market analytics, patient population analysis, KPI dashboard thinking
- Miaomi: patient-related data cleaning, feature engineering, predictive modeling
- SQL/eBay project: structured analysis, segmentation, A/B testing, business recommendations

Main metrics:
- Encounters: {metrics["kpis"]["encounters"]:,}
- Unique patients: {metrics["kpis"]["unique_patients"]:,}
- 30-day readmission rate: {metrics["kpis"]["readmitted_within_30_days_rate"]}%
- Average length of stay: {metrics["kpis"]["avg_time_in_hospital"]} days
- Baseline model AUC: {metrics["model"]["auc"]}

Suggested resume project entry:

Diabetes Readmission Analytics Dashboard | Healthcare BI & Predictive Analytics Project
- Built a healthcare analytics dashboard using 101K+ diabetes inpatient encounters from 130 U.S. hospitals to analyze 30-day readmission patterns, patient cohorts, care utilization, and data quality.
- Created KPI views for readmission rate, length of stay, A1C result, insulin medication status, and missing-data checks to support clinical reporting and quality analytics.
- Developed a baseline Python predictive model for 30-day readmission risk, using feature engineering and logistic regression to evaluate model performance and identify high-risk signals.

Short interview answer:
I chose a diabetes readmission project because readmission is a common healthcare quality and population health use case. I used real hospital encounter data, profiled data quality, built readmission KPIs, segmented patient cohorts, and added a baseline predictive model. It helped me connect my clinical documentation experience as an RBT with healthcare analytics skills like SQL-style data modeling, BI reporting, and predictive analytics.
"""
    path = os.path.join(ROOT, "portfolio_brief.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(brief)
    return path


def main():
    ensure_dirs()
    csv_path, _ = extract_data()
    _, metrics, metrics_path, sample_path = analyze(csv_path)
    dashboard_path = generate_dashboard(metrics)
    readme_path = generate_readme(metrics)
    brief_path = generate_project_brief(metrics)
    print(json.dumps({
        "dashboard": dashboard_path,
        "readme": readme_path,
        "brief": brief_path,
        "metrics": metrics_path,
        "sample": sample_path,
    }, indent=2))


if __name__ == "__main__":
    main()
