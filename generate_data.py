"""
=============================================================
  STEP 1: GENERATE SYNTHETIC LEAD DATA
=============================================================
  This script creates a fake (synthetic) dataset of 5,000
  property buyers in Malaysia. We inject realistic patterns
  so the ML model has meaningful rules to learn from.

  Output: leads_dataset.csv
=============================================================
"""

import numpy as np
import pandas as pd

# ── Set a random seed so results are the same every run ──
np.random.seed(42)
N = 5000  # number of fake leads to generate


# =============================================================
# HELPER: randomly pick from a list with weighted probability
# =============================================================
def choice(options, weights, size):
    """Like rolling a weighted dice over a list of options."""
    return np.random.choice(options, size=size, p=weights)


# =============================================================
# 1. DEMOGRAPHIC & FINANCIAL FEATURES
# =============================================================

# Employment type — government jobs are most common in Malaysia
employment_type = choice(
    ["Government", "Private", "Self-Employed", "Unemployed"],
    [0.30, 0.45, 0.20, 0.05],
    N
)

# Monthly income in Ringgit Malaysia (RM)
# We generate income differently per employment type
monthly_income = np.where(
    employment_type == "Government",
    np.random.normal(5500, 1200, N).clip(2000, 15000),
    np.where(
        employment_type == "Private",
        np.random.normal(6000, 2000, N).clip(1800, 25000),
        np.where(
            employment_type == "Self-Employed",
            np.random.normal(7000, 3000, N).clip(1500, 40000),
            np.random.normal(1500, 500, N).clip(500, 3000)  # Unemployed
        )
    )
).round(0)

# Existing monthly loan commitments (car, personal loan, etc.)
existing_loan_commitment = (monthly_income * np.random.uniform(0.0, 0.65, N)).round(0)

# Debt-to-income ratio — banks want this BELOW 0.7
debt_to_income_ratio = (existing_loan_commitment / monthly_income).round(3)

# Has CCRIS / CTOS issues? (bad credit history)
# Unemployed people have higher chance of bad credit
ccris_issue = np.where(
    employment_type == "Unemployed",
    np.random.choice([1, 0], size=N, p=[0.70, 0.30]),
    np.random.choice([1, 0], size=N, p=[0.12, 0.88])
)


# =============================================================
# 2. PROJECT FIT FEATURES
# =============================================================

# Property price the lead is interested in (RM)
property_price = choice(
    [350000, 450000, 550000, 650000, 800000, 1000000],
    [0.20,   0.25,   0.25,   0.15,   0.10,   0.05],
    N
)

# Budget the lead actually stated
# We add some noise — some people over-reach their budget
budget_stated = (property_price * np.random.uniform(0.70, 1.20, N)).round(-3)

# Budget-to-price fit: 1.0 means perfect match
budget_fit_ratio = (budget_stated / property_price).round(3)

# Location match: does the lead want THIS project's location?
location_match = choice(
    ["Perfect", "Acceptable", "Not Ideal"],
    [0.40, 0.35, 0.25],
    N
)

# Number of bedrooms needed vs available
bedroom_match = choice(
    ["Match", "Partial", "No Match"],
    [0.55, 0.30, 0.15],
    N
)


# =============================================================
# 3. ENGAGEMENT / BEHAVIOUR FEATURES
# =============================================================

# How did the lead find out about this project?
lead_source = choice(
    ["Facebook Ad", "Walk-in", "Agent Referral", "Website", "Billboard"],
    [0.35, 0.15, 0.25, 0.15, 0.10],
    N
)

# How many times has the lead replied to messages? (0–20)
response_count = np.random.poisson(lam=4, size=N).clip(0, 20)

# Days since first contact (0 = today, 90 = three months ago)
days_since_contact = np.random.randint(0, 91, N)

# Has the lead visited the showroom?
showroom_visit = np.random.choice([1, 0], size=N, p=[0.30, 0.70])


# =============================================================
# 4. INTENT / URGENCY FEATURES
# =============================================================

# Purpose of buying
buy_purpose = choice(
    ["Own Stay", "Investment", "Both"],
    [0.50, 0.35, 0.15],
    N
)

# Timeline: how soon do they want to buy?
buy_timeline_months = choice(
    [1, 3, 6, 12, 24],
    [0.10, 0.20, 0.30, 0.25, 0.15],
    N
)


# =============================================================
# 5. LABEL THE LEADS  ← this is what the model learns to predict
# =============================================================
# Rules (based on domain knowledge):
#   GREEN (2) = Ready to buy, financially strong, engaged
#   YELLOW (1) = Possible buyer, but has one or two blockers
#   RED (0)   = Poor financial fit, bad credit, or unresponsive

def assign_label(i):
    """
    For each lead (row i), apply business rules to assign
    Green / Yellow / Red.
    """

    green_score = 0  # we'll tally up green signals
    red_flags   = 0  # and red flags separately

    # ── Financial health ──
    if monthly_income[i] >= 5000:
        green_score += 2
    elif monthly_income[i] < 2500:
        red_flags += 2

    if debt_to_income_ratio[i] < 0.40:
        green_score += 2
    elif debt_to_income_ratio[i] > 0.65:
        red_flags += 2

    if ccris_issue[i] == 1:
        red_flags += 3  # very serious — banks won't approve loan

    # ── Budget & project fit ──
    if budget_fit_ratio[i] >= 0.90:
        green_score += 2
    elif budget_fit_ratio[i] < 0.70:
        red_flags += 2

    if location_match[i] == "Perfect":
        green_score += 1
    elif location_match[i] == "Not Ideal":
        red_flags += 1

    if bedroom_match[i] == "Match":
        green_score += 1
    elif bedroom_match[i] == "No Match":
        red_flags += 1

    # ── Engagement ──
    if response_count[i] >= 5:
        green_score += 2
    elif response_count[i] == 0:
        red_flags += 2

    if showroom_visit[i] == 1:
        green_score += 2

    if days_since_contact[i] <= 7:
        green_score += 1
    elif days_since_contact[i] > 60:
        red_flags += 1

    # ── Urgency ──
    if buy_timeline_months[i] <= 3:
        green_score += 2
    elif buy_timeline_months[i] >= 12:
        red_flags += 1

    if employment_type[i] == "Unemployed":
        red_flags += 3

    # ── Final decision ──
    if red_flags >= 4:
        return 0  # RED
    elif green_score >= 9:
        return 2  # GREEN
    else:
        return 1  # YELLOW


# Apply the labelling function to every row
labels = np.array([assign_label(i) for i in range(N)])


# =============================================================
# 6. ASSEMBLE INTO A DATAFRAME AND SAVE
# =============================================================

df = pd.DataFrame({
    # Financial
    "monthly_income_rm":       monthly_income,
    "employment_type":         employment_type,
    "existing_loan_rm":        existing_loan_commitment,
    "debt_to_income_ratio":    debt_to_income_ratio,
    "ccris_issue":             ccris_issue,

    # Project fit
    "property_price_rm":       property_price,
    "budget_stated_rm":        budget_stated,
    "budget_fit_ratio":        budget_fit_ratio,
    "location_match":          location_match,
    "bedroom_match":           bedroom_match,

    # Engagement
    "lead_source":             lead_source,
    "response_count":          response_count,
    "days_since_contact":      days_since_contact,
    "showroom_visit":          showroom_visit,

    # Intent
    "buy_purpose":             buy_purpose,
    "buy_timeline_months":     buy_timeline_months,

    # Target label
    "lead_label":              labels   # 0=Red, 1=Yellow, 2=Green
})

# Save to CSV
df.to_csv("leads_dataset.csv", index=False)

# ── Quick summary so you can see what was generated ──
print("✅ Dataset created: leads_dataset.csv")
print(f"   Total rows : {len(df):,}")
print("\n📊 Label distribution:")
label_map = {0: "🔴 Red", 1: "🟡 Yellow", 2: "🟢 Green"}
for code, name in label_map.items():
    count = (df["lead_label"] == code).sum()
    pct   = count / N * 100
    print(f"   {name} : {count:,} leads ({pct:.1f}%)")

print("\n📋 First 3 rows:")
print(df.head(3).to_string())