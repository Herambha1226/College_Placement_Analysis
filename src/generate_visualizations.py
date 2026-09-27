
# STUDENT PLACEMENT / MNC MATCHING PROJECT
# INDIVIDUAL VISUALIZATIONS (saved to images/ folder)
#
# Unlike visualize_data.py (one combined dashboard PNG), this
# script generates each chart as its own high-res PNG file,
# saved under images/, so each can be used independently
# (README, reports, slides, etc.)


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 0. SETUP

IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100
PALETTE = "viridis"


def save_fig(fig, filename, dpi=300):
    """Save a figure to the images/ folder and close it."""
    path = os.path.join(IMAGES_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")



# 1. LOAD DATA

student_df = pd.read_excel("Outputs/final_student_dataset.xlsx")
skill_gap_df = pd.read_excel("Outputs/cleaned_skill_gaps.xlsx")

placed_df = student_df[student_df["placement_status"] == "Placed"]


# 2. OVERALL PLACEMENT STATUS (countplot)

fig, ax = plt.subplots(figsize=(6, 5))
order = student_df["placement_status"].value_counts().index
sns.countplot(data=student_df, x="placement_status", hue="placement_status", order=order, palette=PALETTE, legend=False, ax=ax)
ax.set_title("Overall Placement Status", fontsize=14, fontweight="bold")
ax.set_xlabel("Placement Status")
ax.set_ylabel("Number of Students")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontweight="bold")
save_fig(fig, "01_overall_placement_status.png")


# 3. PLACEMENT RATE BY BRANCH

fig, ax = plt.subplots(figsize=(8, 5))
rate_by_branch = (
    student_df.assign(placed=student_df["placement_status"].eq("Placed").astype(int))
    .groupby("branch")["placed"].mean().sort_values(ascending=False) * 100
)
sns.barplot(x=rate_by_branch.index, y=rate_by_branch.values, hue=rate_by_branch.index, palette=PALETTE, legend=False, ax=ax)
ax.set_title("Placement Rate by Branch", fontsize=14, fontweight="bold")
ax.set_xlabel("Branch")
ax.set_ylabel("Placement Rate (%)")
ax.tick_params(axis="x", rotation=30)
for i, v in enumerate(rate_by_branch.values):
    ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
save_fig(fig, "02_placement_rate_by_branch.png")


# 4. CGPA DISTRIBUTION BY PLACEMENT STATUS (boxplot)

fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(data=student_df, x="placement_status", y="cgpa", hue="placement_status", palette=PALETTE, legend=False, ax=ax)
ax.set_title("CGPA Distribution vs Placement Status", fontsize=14, fontweight="bold")
ax.set_xlabel("Placement Status")
ax.set_ylabel("CGPA")
save_fig(fig, "03_cgpa_vs_placement.png")


# 5. CGPA DISTRIBUTION (histogram + KDE)

fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(data=student_df, x="cgpa", hue="placement_status", kde=True,
             element="step", palette=PALETTE, ax=ax)
ax.set_title("CGPA Distribution by Placement Status", fontsize=14, fontweight="bold")
ax.set_xlabel("CGPA")
ax.set_ylabel("Count")
save_fig(fig, "04_cgpa_distribution.png")


# 6. TECHNICAL SKILLS: PLACED vs NOT PLACED (grouped bar)

technical_skills = [
    "coding", "dsa", "cs_fundamentals", "sql",
    "excel", "power_bi", "statistics", "data_visualization"
]
technical_skills = [c for c in technical_skills if c in student_df.columns]

tech_by_status = student_df.groupby("placement_status")[technical_skills].mean().T

fig, ax = plt.subplots(figsize=(9, 5))
tech_by_status.plot(kind="bar", ax=ax, colormap=PALETTE)
ax.set_title("Average Technical Skill Scores by Placement", fontsize=14, fontweight="bold")
ax.set_xlabel("Technical Skill")
ax.set_ylabel("Average Score")
ax.tick_params(axis="x", rotation=40)
ax.legend(title="Placement Status")
save_fig(fig, "05_technical_skills_by_placement.png")

# 7. TOP SKILL GAPS (from skill_gap_df)

if not skill_gap_df.empty and {"skill", "gap"}.issubset(skill_gap_df.columns):
    top_gaps = (
        skill_gap_df.groupby("skill")["gap"].mean()
        .sort_values(ascending=False).head(10).sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top_gaps.values, y=top_gaps.index, hue=top_gaps.index, palette=PALETTE, legend=False, ax=ax)
    ax.set_title("Top 10 Average Skill Gaps", fontsize=14, fontweight="bold")
    ax.set_xlabel("Average Skill Gap")
    ax.set_ylabel("Skill")
    save_fig(fig, "06_top_skill_gaps.png")


# 8. PLACEMENT READINESS BAND vs PLACEMENT (stacked %)

if "placement_readiness_band" in student_df.columns:
    band_order = ["Low", "Developing", "Ready", "Highly Ready"]
    existing_order = [b for b in band_order if b in student_df["placement_readiness_band"].unique()]
    readiness_ct = pd.crosstab(
        student_df["placement_readiness_band"],
        student_df["placement_status"],
        normalize="index"
    ).reindex(existing_order) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    readiness_ct.plot(kind="bar", stacked=True, ax=ax, colormap=PALETTE)
    ax.set_title("Placement Readiness Band vs Placement Outcome", fontsize=14, fontweight="bold")
    ax.set_xlabel("Readiness Band")
    ax.set_ylabel("Percentage (%)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Status")
    save_fig(fig, "07_readiness_band_vs_placement.png")


# 9. PREFERRED CAREER DISTRIBUTION (top 10)

if "preferred_career" in student_df.columns:
    career_counts = student_df["preferred_career"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=career_counts.values, y=career_counts.index, hue=career_counts.index, palette=PALETTE, legend=False, ax=ax)
    ax.set_title("Top 10 Preferred Career Choices", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Students")
    ax.set_ylabel("Preferred Career")
    save_fig(fig, "08_preferred_career_distribution.png")

# 10. HIGHEST PACKAGE (LPA) DISTRIBUTION FOR PLACED STUDENTS

if "highest_package_lpa" in student_df.columns and not placed_df.empty:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(data=placed_df, x="highest_package_lpa", kde=True,
                 color=sns.color_palette(PALETTE, 1)[0], ax=ax)
    ax.set_title("Package (LPA) Distribution — Placed Students", fontsize=14, fontweight="bold")
    ax.set_xlabel("Highest Package (LPA)")
    ax.set_ylabel("Number of Students")
    save_fig(fig, "09_package_distribution.png")


# 11. BRANCH vs AVERAGE PACKAGE (placed students only)

if "highest_package_lpa" in student_df.columns and not placed_df.empty:
    avg_package_branch = (
        placed_df.groupby("branch")["highest_package_lpa"]
        .mean().sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=avg_package_branch.index, y=avg_package_branch.values, hue=avg_package_branch.index, palette=PALETTE, legend=False, ax=ax)
    ax.set_title("Average Package by Branch (Placed Students)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Branch")
    ax.set_ylabel("Average Package (LPA)")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(avg_package_branch.values):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    save_fig(fig, "10_avg_package_by_branch.png")


# 12. BACKLOGS vs PLACEMENT STATUS
if "backlogs" in student_df.columns:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=student_df, x="backlogs", hue="placement_status", palette=PALETTE, ax=ax)
    ax.set_title("Active Backlogs vs Placement Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Backlogs")
    ax.set_ylabel("Number of Students")
    ax.legend(title="Placement Status")
    save_fig(fig, "11_backlogs_vs_placement.png")


# 13. CORRELATION HEATMAP OF KEY NUMERIC FEATURES

corr_cols = [
    "cgpa", "aptitude_score", "coding", "dsa", "communication",
    "projects_count", "internship_months", "certifications_count",
    "hackathons_count", "placement_readiness_score", "interview_score"
]
corr_cols = [c for c in corr_cols if c in student_df.columns]

fig, ax = plt.subplots(figsize=(9, 7))
corr_matrix = student_df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=PALETTE, square=True,
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Correlation Heatmap — Key Academic & Readiness Features", fontsize=14, fontweight="bold")
save_fig(fig, "12_correlation_heatmap.png")


# 14. PLACEMENT READINESS SCORE vs INTERVIEW SCORE (scatter)

if {"placement_readiness_score", "interview_score"}.issubset(student_df.columns):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=student_df, x="placement_readiness_score", y="interview_score",
        hue="placement_status", alpha=0.6, palette=PALETTE, ax=ax
    )
    ax.set_title("Placement Readiness Score vs Interview Score", fontsize=14, fontweight="bold")
    ax.set_xlabel("Placement Readiness Score")
    ax.set_ylabel("Interview Score")
    ax.legend(title="Placement Status")
    save_fig(fig, "13_readiness_vs_interview_score.png")


# 15. INTERNSHIP MONTHS vs PLACEMENT STATUS

if "internship_months" in student_df.columns:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.violinplot(data=student_df, x="placement_status", y="internship_months",
                    hue="placement_status", palette=PALETTE, legend=False, ax=ax)
    ax.set_title("Internship Experience vs Placement Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Placement Status")
    ax.set_ylabel("Internship Months")
    save_fig(fig, "14_internship_months_vs_placement.png")

# 16. COMPANY MATCH STATUS DISTRIBUTION

if "company_match_status" in student_df.columns:
    match_counts = student_df["company_match_status"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = sns.color_palette(PALETTE, len(match_counts))
    ax.pie(match_counts.values, labels=match_counts.index, autopct="%1.1f%%",
           colors=colors, startangle=90, wedgeprops={"edgecolor": "white"})
    ax.set_title("Company Match Status Distribution", fontsize=14, fontweight="bold")
    save_fig(fig, "15_company_match_status.png")

print("\nAll visualizations generated successfully.")
print(f"Saved {len(os.listdir(IMAGES_DIR))} images to the '{IMAGES_DIR}/' folder.")
