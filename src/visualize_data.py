# ============================================================
# STUDENT PLACEMENT / MNC MATCHING PROJECT
# PRESENTATION-READY VISUALIZATION
# Generates ONLY the most important graphs for ONE PPT SLIDE
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. LOAD DATASETS
# ------------------------------------------------------------
student_df = pd.read_excel("Outputs/final_student_dataset.xlsx")
skill_gap_df = pd.read_excel("Outputs/cleaned_skill_gaps.xlsx")

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 2. CREATE ONE PRESENTATION PAGE (2 x 3 GRAPHS)
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    "Student Placement Analysis – Key Insights",
    fontsize=22,
    fontweight="bold",
    y=0.98
)

# ============================================================
# GRAPH 1 — OVERALL PLACEMENT STATUS
# ============================================================
ax = axes[0, 0]

placement_counts = student_df["placement_status"].value_counts()

sns.barplot(
    x=placement_counts.index,
    y=placement_counts.values,
    ax=ax
)

ax.set_title("Overall Placement Status", fontweight="bold")
ax.set_xlabel("Placement Status")
ax.set_ylabel("Students")

for i, value in enumerate(placement_counts.values):
    ax.text(i, value, str(value), ha="center", va="bottom", fontweight="bold")

# ============================================================
# GRAPH 2 — PLACEMENT RATE BY BRANCH
# ============================================================
ax = axes[0, 1]

placement_rate_branch = (
    student_df.assign(
        placed=student_df["placement_status"].eq("Placed")
    )
    .groupby("branch")["placed"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

sns.barplot(
    x=placement_rate_branch.index,
    y=placement_rate_branch.values,
    ax=ax
)

ax.set_title("Placement Rate by Branch", fontweight="bold")
ax.set_xlabel("Branch")
ax.set_ylabel("Placement Rate (%)")
ax.tick_params(axis="x", rotation=35)

for i, value in enumerate(placement_rate_branch.values):
    ax.text(i, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)

# ============================================================
# GRAPH 3 — CGPA VS PLACEMENT
# ============================================================
ax = axes[0, 2]

sns.boxplot(
    data=student_df,
    x="placement_status",
    y="cgpa",
    ax=ax
)

ax.set_title("CGPA vs Placement Status", fontweight="bold")
ax.set_xlabel("Placement Status")
ax.set_ylabel("CGPA")

# ============================================================
# GRAPH 4 — TECHNICAL SKILLS VS PLACEMENT
# ============================================================
ax = axes[1, 0]

technical_skills = [
    "coding",
    "dsa",
    "cs_fundamentals",
    "sql",
    "excel",
    "power_bi",
    "statistics",
    "data_visualization"
]

technical_skills = [
    col for col in technical_skills
    if col in student_df.columns
]

technical_placement = (
    student_df.groupby("placement_status")[technical_skills]
    .mean()
    .T
)

technical_placement.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Technical Skills by Placement", fontweight="bold")
ax.set_xlabel("Technical Skill")
ax.set_ylabel("Average Score")
ax.tick_params(axis="x", rotation=45)
ax.legend(title="Status", fontsize=7)

# ============================================================
# GRAPH 5 — TOP SKILL GAPS
# ============================================================
ax = axes[1, 1]

if not skill_gap_df.empty and "skill" in skill_gap_df.columns and "gap" in skill_gap_df.columns:

    skill_gap_summary = (
        skill_gap_df.groupby("skill")["gap"]
        .mean()
        .sort_values(ascending=False)
        .head(8)
        .sort_values()
    )

    sns.barplot(
        x=skill_gap_summary.values,
        y=skill_gap_summary.index,
        ax=ax
    )

    ax.set_title("Top Skill Gaps", fontweight="bold")
    ax.set_xlabel("Average Skill Gap")
    ax.set_ylabel("Skill")

else:
    ax.text(
        0.5, 0.5,
        "Skill-gap data unavailable",
        ha="center",
        va="center"
    )
    ax.set_title("Top Skill Gaps", fontweight="bold")
    ax.axis("off")

# ============================================================
# GRAPH 6 — PLACEMENT READINESS VS PLACEMENT
# ============================================================
ax = axes[1, 2]

if "placement_readiness_band" in student_df.columns:

    readiness_placement = pd.crosstab(
        student_df["placement_readiness_band"],
        student_df["placement_status"],
        normalize="index"
    ) * 100

    readiness_placement.plot(
        kind="bar",
        stacked=True,
        ax=ax
    )

    ax.set_title("Readiness vs Placement", fontweight="bold")
    ax.set_xlabel("Readiness Band")
    ax.set_ylabel("Percentage (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Status", fontsize=7)

else:
    ax.text(
        0.5, 0.5,
        "Readiness data unavailable",
        ha="center",
        va="center"
    )
    ax.set_title("Readiness vs Placement", fontweight="bold")
    ax.axis("off")

# ------------------------------------------------------------
# 3. FINAL PRESENTATION FORMATTING
# ------------------------------------------------------------
for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save as high-resolution image for PowerPoint
plt.savefig(
    "Outputs/placement_analysis_one_page.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nPresentation graph generated successfully.")
print("Saved as: Outputs/placement_analysis_one_page.png")