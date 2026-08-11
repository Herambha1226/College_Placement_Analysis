# ============================================================
# STUDENT PLACEMENT / MNC MATCHING PROJECT
# DATA VISUALIZATION USING MATPLOTLIB + SEABORN
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. LOAD DATASETS
# ------------------------------------------------------------

student_df = pd.read_excel("/mnt/data/student records.xlsx")
placement_df = pd.read_excel("/mnt/data/SRC_Student_Placement.xlsx")
skill_gap_df = pd.read_excel("/mnt/data/SRC_Skill_Gaps.xlsx")
mnc_df = pd.read_excel("/mnt/data/SRC_MNC_Criteria.xlsx")
role_df = pd.read_excel("/mnt/data/SRC_Role_Requirements.xlsx")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ------------------------------------------------------------
# 2. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("Student Records Shape:", student_df.shape)
print("Placement Dataset Shape:", placement_df.shape)
print("Skill Gap Dataset Shape:", skill_gap_df.shape)
print("MNC Criteria Shape:", mnc_df.shape)
print("Role Requirements Shape:", role_df.shape)

print("\nStudent Records Columns:")
print(student_df.columns.tolist())

# ============================================================
# 3. PLACEMENT STATUS DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

sns.countplot(
    data=student_df,
    x="placement_status",
    order=student_df["placement_status"].value_counts().index
)

plt.title("Student Placement Status Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()


# ============================================================
# 4. PLACEMENT STATUS - PIE CHART
# ============================================================

placement_counts = student_df["placement_status"].value_counts()

plt.figure(figsize=(7, 7))

plt.pie(
    placement_counts.values,
    labels=placement_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Percentage of Students by Placement Status")
plt.tight_layout()
plt.show()


# ============================================================
# 5. BRANCH-WISE STUDENT DISTRIBUTION
# ============================================================

plt.figure(figsize=(12, 6))

branch_order = student_df["branch"].value_counts().index

sns.countplot(
    data=student_df,
    x="branch",
    order=branch_order
)

plt.title("Student Distribution by Branch")
plt.xlabel("Branch")
plt.ylabel("Number of Students")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# 6. BRANCH-WISE PLACEMENT STATUS
# ============================================================

branch_placement = pd.crosstab(
    student_df["branch"],
    student_df["placement_status"]
)

branch_placement.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Branch-wise Placement Status")
plt.xlabel("Branch")
plt.ylabel("Number of Students")
plt.xticks(rotation=45)
plt.legend(title="Placement Status")
plt.tight_layout()
plt.show()


# ============================================================
# 7. PLACEMENT RATE BY BRANCH
# ============================================================

placement_rate_branch = (
    student_df.assign(
        placed=student_df["placement_status"].eq("Placed")
    )
    .groupby("branch")["placed"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=placement_rate_branch.index,
    y=placement_rate_branch.values
)

plt.title("Placement Rate by Branch")
plt.xlabel("Branch")
plt.ylabel("Placement Rate (%)")
plt.xticks(rotation=45)

for i, value in enumerate(placement_rate_branch.values):
    plt.text(i, value + 1, f"{value:.1f}%", ha="center")

plt.tight_layout()
plt.show()


# ============================================================
# 8. CGPA DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=student_df,
    x="cgpa",
    bins=20,
    kde=True
)

plt.title("CGPA Distribution of Students")
plt.xlabel("CGPA")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()


# ============================================================
# 9. CGPA VS PLACEMENT STATUS
# ============================================================

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=student_df,
    x="placement_status",
    y="cgpa"
)

plt.title("CGPA Distribution by Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("CGPA")
plt.tight_layout()
plt.show()


# ============================================================
# 10. 10TH AND 12TH PERCENTAGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    student_df["tenth_pct"],
    bins=20,
    kde=True,
    label="10th Percentage",
    alpha=0.5
)

sns.histplot(
    student_df["twelfth_pct"],
    bins=20,
    kde=True,
    label="12th Percentage",
    alpha=0.5
)

plt.title("10th vs 12th Percentage Distribution")
plt.xlabel("Percentage")
plt.ylabel("Number of Students")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 11. BACKLOG DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 6))

sns.countplot(
    data=student_df,
    x="backlogs"
)

plt.title("Distribution of Student Backlogs")
plt.xlabel("Number of Backlogs")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()


# ============================================================
# 12. BACKLOGS VS PLACEMENT STATUS
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=student_df,
    x="placement_status",
    y="backlogs"
)

plt.title("Backlogs vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Number of Backlogs")
plt.tight_layout()
plt.show()


# ============================================================
# 13. OFFERS COUNT DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 6))

sns.countplot(
    data=student_df,
    x="offers_count"
)

plt.title("Distribution of Number of Offers")
plt.xlabel("Number of Offers")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()


# ============================================================
# 14. HIGHEST PACKAGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=student_df,
    x="highest_package_lpa",
    bins=25,
    kde=True
)

plt.title("Highest Package Distribution")
plt.xlabel("Highest Package (LPA)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()


# ============================================================
# 15. CGPA VS HIGHEST PACKAGE
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=student_df,
    x="cgpa",
    y="highest_package_lpa",
    hue="placement_status",
    alpha=0.7
)

plt.title("CGPA vs Highest Package")
plt.xlabel("CGPA")
plt.ylabel("Highest Package (LPA)")
plt.tight_layout()
plt.show()


# ============================================================
# 16. INTERNSHIP MONTHS VS PLACEMENT
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=student_df,
    x="placement_status",
    y="internship_months"
)

plt.title("Internship Experience vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Internship Duration (Months)")
plt.tight_layout()
plt.show()


# ============================================================
# 17. CERTIFICATIONS VS PLACEMENT
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=student_df,
    x="placement_status",
    y="certifications_count"
)

plt.title("Certifications vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Number of Certifications")
plt.tight_layout()
plt.show()


# ============================================================
# 18. PROJECTS VS PLACEMENT
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=student_df,
    x="placement_status",
    y="projects_count"
)

plt.title("Projects vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Number of Projects")
plt.tight_layout()
plt.show()


# ============================================================
# 19. HACKATHONS VS PLACEMENT
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=student_df,
    x="placement_status",
    y="hackathons_count"
)

plt.title("Hackathons vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Number of Hackathons")
plt.tight_layout()
plt.show()


# ============================================================
# 20. SKILL DISTRIBUTION
# ============================================================

skill_columns = [
    "coding",
    "dsa",
    "cs_fundamentals",
    "sql",
    "excel",
    "power_bi",
    "statistics",
    "data_visualization",
    "communication",
    "teamwork",
    "problem_solving",
    "analytical_thinking",
    "presentation",
    "business_knowledge",
    "cloud",
    "linux",
    "networking",
    "troubleshooting",
    "security",
    "scripting",
    "monitoring",
    "docker",
    "devops",
    "cybersecurity",
    "siem",
    "risk_compliance",
    "documentation",
    "testing",
    "api_testing",
    "git",
    "requirements"
]

skill_columns = [
    col for col in skill_columns
    if col in student_df.columns
]

skill_means = student_df[skill_columns].mean().sort_values(ascending=False)

plt.figure(figsize=(14, 8))

sns.barplot(
    x=skill_means.values,
    y=skill_means.index
)

plt.title("Average Student Skill Scores")
plt.xlabel("Average Skill Score")
plt.ylabel("Skill")
plt.tight_layout()
plt.show()


# ============================================================
# 21. TOP 10 SKILLS
# ============================================================

top_skills = skill_means.head(10)

plt.figure(figsize=(12, 7))

sns.barplot(
    x=top_skills.values,
    y=top_skills.index
)

plt.title("Top 10 Skills by Average Score")
plt.xlabel("Average Score")
plt.ylabel("Skill")
plt.tight_layout()
plt.show()


# ============================================================
# 22. LOWEST 10 SKILLS
# ============================================================

lowest_skills = skill_means.tail(10).sort_values()

plt.figure(figsize=(12, 7))

sns.barplot(
    x=lowest_skills.values,
    y=lowest_skills.index
)

plt.title("10 Skills with Lowest Average Scores")
plt.xlabel("Average Score")
plt.ylabel("Skill")
plt.tight_layout()
plt.show()


# ============================================================
# 23. SKILL SCORE DISTRIBUTION
# ============================================================

selected_skills = [
    col for col in [
        "coding",
        "dsa",
        "sql",
        "excel",
        "power_bi",
        "statistics",
        "data_visualization"
    ]
    if col in student_df.columns
]

plt.figure(figsize=(12, 7))

sns.boxplot(
    data=student_df[selected_skills]
)

plt.title("Distribution of Major Technical Skills")
plt.xlabel("Skill")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# 24. TECHNICAL SKILLS VS PLACEMENT
# ============================================================

technical_skills = [
    col for col in [
        "coding",
        "dsa",
        "cs_fundamentals",
        "sql",
        "excel",
        "power_bi",
        "statistics",
        "data_visualization"
    ]
    if col in student_df.columns
]

technical_placement = (
    student_df.groupby("placement_status")[technical_skills]
    .mean()
    .T
)

technical_placement.plot(
    kind="bar",
    figsize=(14, 7)
)

plt.title("Average Technical Skill Scores by Placement Status")
plt.xlabel("Skill")
plt.ylabel("Average Score")
plt.xticks(rotation=45)
plt.legend(title="Placement Status")
plt.tight_layout()
plt.show()


# ============================================================
# 25. SOFT SKILLS VS PLACEMENT
# ============================================================

soft_skills = [
    col for col in [
        "communication",
        "teamwork",
        "problem_solving",
        "analytical_thinking",
        "presentation"
    ]
    if col in student_df.columns
]

soft_placement = (
    student_df.groupby("placement_status")[soft_skills]
    .mean()
    .T
)

soft_placement.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title("Average Soft Skill Scores by Placement Status")
plt.xlabel("Skill")
plt.ylabel("Average Score")
plt.xticks(rotation=45)
plt.legend(title="Placement Status")
plt.tight_layout()
plt.show()


# ============================================================
# 26. SKILL CORRELATION HEATMAP
# ============================================================

correlation_columns = [
    col for col in [
        "cgpa",
        "tenth_pct",
        "twelfth_pct",
        "backlogs",
        "academic_consistency",
        "quantitative_aptitude",
        "logical_reasoning",
        "verbal_aptitude",
        "communication",
        "coding",
        "dsa",
        "cs_fundamentals",
        "sql",
        "excel",
        "power_bi",
        "statistics",
        "data_visualization",
        "problem_solving",
        "analytical_thinking",
        "projects_count",
        "internship_months",
        "certifications_count",
        "hackathons_count",
        "offers_count",
        "highest_package_lpa"
    ]
    if col in student_df.columns
]

plt.figure(figsize=(18, 14))

correlation_matrix = student_df[correlation_columns].corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("Correlation Heatmap of Student Placement Features")
plt.tight_layout()
plt.show()


# ============================================================
# 27. CGPA VS CODING
# ============================================================

if "coding" in student_df.columns:

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=student_df,
        x="cgpa",
        y="coding",
        hue="placement_status",
        alpha=0.7
    )

    plt.title("CGPA vs Coding Skill")
    plt.xlabel("CGPA")
    plt.ylabel("Coding Score")
    plt.tight_layout()
    plt.show()


# ============================================================
# 28. CODING VS DSA
# ============================================================

if "coding" in student_df.columns and "dsa" in student_df.columns:

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=student_df,
        x="coding",
        y="dsa",
        hue="placement_status",
        alpha=0.7
    )

    plt.title("Coding Skill vs DSA Skill")
    plt.xlabel("Coding Score")
    plt.ylabel("DSA Score")
    plt.tight_layout()
    plt.show()


# ============================================================
# 29. ROLE DISTRIBUTION
# ============================================================

role_column = None

for col in [
    "recommended_role_from_profile",
    "placed_role",
    "company_role"
]:
    if col in student_df.columns:
        role_column = col
        break

if role_column:

    role_counts = student_df[role_column].value_counts().head(15)

    plt.figure(figsize=(14, 8))

    sns.barplot(
        x=role_counts.values,
        y=role_counts.index
    )

    plt.title(f"Top Roles Based on {role_column}")
    plt.xlabel("Number of Students")
    plt.ylabel("Role")
    plt.tight_layout()
    plt.show()


# ============================================================
# 30. CAREER FAMILY DISTRIBUTION
# ============================================================

if "career_family" in student_df.columns:

    career_counts = student_df["career_family"].value_counts()

    plt.figure(figsize=(12, 7))

    sns.barplot(
        x=career_counts.values,
        y=career_counts.index
    )

    plt.title("Student Distribution by Career Family")
    plt.xlabel("Number of Students")
    plt.ylabel("Career Family")
    plt.tight_layout()
    plt.show()


# ============================================================
# 31. PLACEMENT READINESS BAND
# ============================================================

if "placement_readiness_band" in student_df.columns:

    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=student_df,
        x="placement_readiness_band",
        order=student_df["placement_readiness_band"].value_counts().index
    )

    plt.title("Placement Readiness Band Distribution")
    plt.xlabel("Placement Readiness Band")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


# ============================================================
# 32. PLACEMENT READINESS VS PLACEMENT STATUS
# ============================================================

if "placement_readiness_band" in student_df.columns:

    readiness_placement = pd.crosstab(
        student_df["placement_readiness_band"],
        student_df["placement_status"],
        normalize="index"
    ) * 100

    readiness_placement.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 7)
    )

    plt.title("Placement Status Percentage by Readiness Band")
    plt.xlabel("Placement Readiness Band")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=30)
    plt.legend(title="Placement Status")
    plt.tight_layout()
    plt.show()


# ============================================================
# 33. SKILL GAP ANALYSIS
# ============================================================

if not skill_gap_df.empty:

    skill_gap_summary = (
        skill_gap_df.groupby("skill")["gap"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )

    plt.figure(figsize=(14, 8))

    sns.barplot(
        x=skill_gap_summary.values,
        y=skill_gap_summary.index
    )

    plt.title("Top 15 Skills with Highest Average Skill Gap")
    plt.xlabel("Average Skill Gap")
    plt.ylabel("Skill")
    plt.tight_layout()
    plt.show()


# ============================================================
# 34. SKILL GAP PRIORITY DISTRIBUTION
# ============================================================

if "priority" in skill_gap_df.columns:

    plt.figure(figsize=(8, 6))

    priority_order = ["High", "Medium", "Low"]

    sns.countplot(
        data=skill_gap_df,
        x="priority",
        order=[
            p for p in priority_order
            if p in skill_gap_df["priority"].unique()
        ]
    )

    plt.title("Skill Gap Priority Distribution")
    plt.xlabel("Priority")
    plt.ylabel("Number of Skill Gaps")
    plt.tight_layout()
    plt.show()


# ============================================================
# 35. SKILL GAP DISTRIBUTION
# ============================================================

if "gap" in skill_gap_df.columns:

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=skill_gap_df,
        x="gap",
        bins=25,
        kde=True
    )

    plt.title("Distribution of Skill Gaps")
    plt.xlabel("Skill Gap")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# ============================================================
# 36. CURRENT SCORE VS REQUIRED SCORE
# ============================================================

if "current_score" in skill_gap_df.columns and "required_score" in skill_gap_df.columns:

    sample_gap = skill_gap_df.sample(
        min(3000, len(skill_gap_df)),
        random_state=42
    )

    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        data=sample_gap,
        x="current_score",
        y="required_score",
        hue="priority",
        alpha=0.5
    )

    plt.title("Current Skill Score vs Required Skill Score")
    plt.xlabel("Current Score")
    plt.ylabel("Required Score")
    plt.tight_layout()
    plt.show()


# ============================================================
# 37. TOTAL SKILL GAP VS PLACEMENT
# ============================================================

if "total_skill_gap" in student_df.columns:

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=student_df,
        x="placement_status",
        y="total_skill_gap"
    )

    plt.title("Total Skill Gap vs Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("Total Skill Gap")
    plt.tight_layout()
    plt.show()


# ============================================================
# 38. HIGH PRIORITY SKILL GAPS VS PLACEMENT
# ============================================================

if "high_priority_gap_count" in student_df.columns:

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=student_df,
        x="placement_status",
        y="high_priority_gap_count"
    )

    plt.title("High Priority Skill Gaps vs Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("High Priority Gap Count")
    plt.tight_layout()
    plt.show()


# ============================================================
# 39. COMPANY MATCH STATUS
# ============================================================

if "company_match_status" in student_df.columns:

    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=student_df,
        x="company_match_status",
        order=student_df["company_match_status"].value_counts().index
    )

    plt.title("Company Match Status Distribution")
    plt.xlabel("Company Match Status")
    plt.ylabel("Number of Students")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


# ============================================================
# 40. COMPANY SKILL FIT SCORE
# ============================================================

if "company_skill_fit_score" in student_df.columns:

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=student_df,
        x="company_skill_fit_score",
        bins=25,
        kde=True
    )

    plt.title("Distribution of Company Skill Fit Score")
    plt.xlabel("Company Skill Fit Score")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.show()


# ============================================================
# 41. COMPANY SKILL FIT VS PLACEMENT
# ============================================================

if "company_skill_fit_score" in student_df.columns:

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=student_df,
        x="placement_status",
        y="company_skill_fit_score"
    )

    plt.title("Company Skill Fit Score vs Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("Company Skill Fit Score")
    plt.tight_layout()
    plt.show()


# ============================================================
# 42. COMPANY SELECTIVITY DISTRIBUTION
# ============================================================

if "company_selectivity" in student_df.columns:

    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=student_df,
        x="company_selectivity",
        order=student_df["company_selectivity"].value_counts().index
    )

    plt.title("Company Selectivity Distribution")
    plt.xlabel("Company Selectivity")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


# ============================================================
# 43. TOP COMPANIES BY RECORD COUNT
# ============================================================

if "company_company_id" in student_df.columns:

    company_counts = (
        student_df["company_company_id"]
        .value_counts()
        .head(15)
    )

    plt.figure(figsize=(12, 7))

    sns.barplot(
        x=company_counts.values,
        y=company_counts.index
    )

    plt.title("Top 15 Companies by Student Records")
    plt.xlabel("Number of Records")
    plt.ylabel("Company")
    plt.tight_layout()
    plt.show()


# ============================================================
# 44. COMPANY ACADEMIC ELIGIBILITY
# ============================================================

if "company_academic_eligible" in student_df.columns:

    plt.figure(figsize=(8, 6))

    sns.countplot(
        data=student_df,
        x="company_academic_eligible"
    )

    plt.title("Company Academic Eligibility")
    plt.xlabel("Academic Eligibility")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.show()


# ============================================================
# 45. ACADEMIC ELIGIBILITY VS PLACEMENT
# ============================================================

if "company_academic_eligible" in student_df.columns:

    eligibility_placement = pd.crosstab(
        student_df["company_academic_eligible"],
        student_df["placement_status"],
        normalize="index"
    ) * 100

    eligibility_placement.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Placement Status by Company Academic Eligibility")
    plt.xlabel("Company Academic Eligibility")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=0)
    plt.legend(title="Placement Status")
    plt.tight_layout()
    plt.show()


# ============================================================
# 46. NUMERICAL FEATURE DISTRIBUTIONS
# ============================================================

numerical_columns = [
    col for col in [
        "cgpa",
        "tenth_pct",
        "twelfth_pct",
        "backlogs",
        "academic_consistency",
        "quantitative_aptitude",
        "logical_reasoning",
        "verbal_aptitude",
        "aptitude_score",
        "coding",
        "dsa",
        "cs_fundamentals",
        "sql",
        "excel",
        "power_bi",
        "statistics",
        "data_visualization",
        "communication",
        "teamwork",
        "problem_solving",
        "analytical_thinking",
        "presentation",
        "projects_count",
        "internship_months",
        "certifications_count",
        "hackathons_count",
        "offers_count",
        "highest_package_lpa",
        "skill_gap_count",
        "total_skill_gap",
        "maximum_skill_gap",
        "high_priority_gap_count",
        "company_skill_fit_score"
    ]
    if col in student_df.columns
]

for col in numerical_columns:

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=student_df,
        x=col,
        bins=20,
        kde=True
    )

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# ============================================================
# 47. NUMERICAL FEATURES VS PLACEMENT
# ============================================================

important_numeric = [
    col for col in [
        "cgpa",
        "tenth_pct",
        "twelfth_pct",
        "academic_consistency",
        "quantitative_aptitude",
        "coding",
        "dsa",
        "sql",
        "communication",
        "analytical_thinking",
        "projects_count",
        "internship_months",
        "certifications_count",
        "hackathons_count",
        "skill_gap_count",
        "total_skill_gap",
        "company_skill_fit_score"
    ]
    if col in student_df.columns
]

for col in important_numeric:

    plt.figure(figsize=(9, 6))

    sns.boxplot(
        data=student_df,
        x="placement_status",
        y=col
    )

    plt.title(f"{col} vs Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


# ============================================================
# 48. PAIRPLOT OF IMPORTANT FEATURES
# ============================================================

pairplot_columns = [
    col for col in [
        "cgpa",
        "coding",
        "dsa",
        "sql",
        "communication",
        "projects_count",
        "internship_months",
        "highest_package_lpa"
    ]
    if col in student_df.columns
]

if len(pairplot_columns) >= 2:

    pair_data = student_df[
        pairplot_columns + ["placement_status"]
    ].sample(
        min(1000, len(student_df)),
        random_state=42
    )

    sns.pairplot(
        pair_data,
        hue="placement_status",
        diag_kind="hist"
    )

    plt.show()


# ============================================================
# 49. OVERALL NUMERICAL CORRELATION WITH PLACEMENT
# ============================================================

if "placement_status" in student_df.columns:

    analysis_df = student_df.copy()

    analysis_df["placed_binary"] = (
        analysis_df["placement_status"] == "Placed"
    ).astype(int)

    numeric_df = analysis_df.select_dtypes(
        include=np.number
    )

    placement_corr = (
        numeric_df.corr()["placed_binary"]
        .drop("placed_binary")
        .sort_values()
    )

    plt.figure(figsize=(12, 10))

    sns.barplot(
        x=placement_corr.values,
        y=placement_corr.index
    )

    plt.title("Correlation of Numerical Features with Placement")
    plt.xlabel("Correlation with Placement")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


# ============================================================
# 50. FINAL TOP FEATURES FOR PLACEMENT
# ============================================================

top_positive_features = placement_corr.sort_values(
    ascending=False
).head(15)

plt.figure(figsize=(12, 8))

sns.barplot(
    x=top_positive_features.values,
    y=top_positive_features.index
)

plt.title("Top 15 Features Positively Associated with Placement")
plt.xlabel("Correlation with Placement")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# ============================================================
# END OF VISUALIZATION CODE
# ============================================================