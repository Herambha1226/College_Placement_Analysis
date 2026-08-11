"""
preprocess.py
=================================================================
Cleaning pipeline for the Student Placement / MNC-Matching data.

Datasets handled
-----------------
  1. SRC_MNC_Criteria.xlsx        - per-company eligibility & skill minimums
  2. SRC_Role_Requirements.xlsx   - per-role skill minimums
  3. SRC_Skill_Gaps.xlsx          - long-format (student x skill) gap records
  4. SRC_Student_Placement.xlsx   - 1 row per REAL student (1,200 students)
  5. student_records.xlsx         - 1 row per (student, synthetic-variant)
                                     already merged with role/company/skill-gap
                                     features (5,000 rows) -> used as the base
                                     for the final ML-ready dataset.

Output (written to output_dir, default /mnt/user-data/outputs)
-----------------------------------------------------------------
  cleaned_mnc_criteria.xlsx
  cleaned_role_requirements.xlsx
  cleaned_skill_gaps.xlsx
  cleaned_student_placement.xlsx      <- 1 row per real student (master lookup)
  final_student_dataset.xlsx          <- 1 row per record, ML-ready (THE deliverable)
  cleaning_report.txt                 <- summary of what was found/fixed

Usage
-----
    pipeline = Preprocess(data_dir="DataSets", output_dir="Outputs")
    pipeline.run_all()
"""

import os
import numpy as np
import pandas as pd


class Preprocess:

    # Columns that are genuinely allowed to be 0 as a real value
    # (0 = "skill/requirement not applicable to this role", not missing data)
    ZERO_IS_VALID_PREFIXES = ("min_", "role_req_min_", "company_min_")

    def __init__(self, data_dir="DataSets", output_dir="Outputs"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.paths = {
            "mnc_criteria": os.path.join(data_dir, "SRC_MNC_Criteria.xlsx"),
            "role_requirements": os.path.join(data_dir, "SRC_Role_Requirements.xlsx"),
            "skill_gaps": os.path.join(data_dir, "SRC_Skill_Gaps.xlsx"),
            "student_placement": os.path.join(data_dir, "SRC_Student_Placement.xlsx"),
            "student_records": os.path.join(data_dir, "student_records.xlsx"),
        }

        # Raw frames (populated by read_all_datasets)
        self.raw = {}

        # Cleaned frames (populated by the clean_* methods)
        self.clean = {}

        # Human-readable log of every fix applied, per dataset
        self.report_lines = []

    # -----------------------------------------------------------------
    # Step 1 & 2: load + quick look (kept close to your original design)
    # -----------------------------------------------------------------
    def read_all_datasets(self):
        for key, path in self.paths.items():
            self.raw[key] = pd.read_excel(path)
        return self.raw

    def check_all_data_info(self):
        for name, df in self.raw.items():
            print(f"\n{'=' * 70}\n{name}  shape={df.shape}\n{'=' * 70}")
            df.info()
        return "All dataset info printed."

    def get_top_five_rows(self):
        for name, df in self.raw.items():
            print(f"\n--- {name} (top 5) ---")
            print(df.head())

    # -----------------------------------------------------------------
    # Generic, reusable cleaning helpers
    # -----------------------------------------------------------------
    def _log(self, dataset, msg):
        self.report_lines.append(f"[{dataset}] {msg}")

    def _strip_and_standardize_strings(self, df, dataset):
        """Trim whitespace, collapse internal double-spaces, keep original case
        (categories in this data are already consistently cased, but this
        keeps the pipeline safe against future dirty input)."""
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            before = df[col].copy()
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
            changed = (before.astype("string") != df[col]).sum()
            if changed:
                self._log(dataset, f"Trimmed/normalized whitespace in '{col}' ({changed} cells)")
        return df

    def _drop_exact_duplicate_rows(self, df, dataset):
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        if n_removed:
            self._log(dataset, f"Dropped {n_removed} fully duplicated rows")
        return df

    def _drop_duplicate_merge_columns(self, df, dataset):
        """student_records.xlsx carries columns like 'company_min_x' and
        'company_min_x.1' left over from an earlier merge. Where the pair is
        byte-identical, keep one copy and drop the '.1' suffix version."""
        dupe_suffixed = [c for c in df.columns if c.endswith(".1")]
        for col in dupe_suffixed:
            base = col[:-2]
            if base in df.columns:
                identical = df[col].fillna("__NA__").astype(str).equals(
                    df[base].fillna("__NA__").astype(str)
                )
                if identical:
                    df = df.drop(columns=[col])
                    self._log(dataset, f"Dropped duplicate column '{col}' (identical to '{base}')")
                else:
                    # keep both but rename meaningfully instead of silently losing data
                    df = df.rename(columns={col: f"{base}_alt"})
                    self._log(dataset, f"'{col}' differed from '{base}' -> renamed to '{base}_alt'")
        return df

    def _fix_id_dtype(self, df, dataset, id_cols):
        for col in id_cols:
            if col in df.columns:
                df[col] = df[col].astype("string").str.strip().str.upper()
        return df

    def _clip_percentage_columns(self, df, dataset, cols):
        """Percentages / 0-100 scored fields must lie within [0, 100]."""
        for col in cols:
            if col not in df.columns:
                continue
            n_bad = ((df[col] < 0) | (df[col] > 100)).sum()
            if n_bad:
                self._log(dataset, f"Clipped {n_bad} out-of-range values in '{col}' to [0, 100]")
                df[col] = df[col].clip(lower=0, upper=100)
        return df

    def _clip_cgpa(self, df, dataset, col="cgpa"):
        if col in df.columns:
            n_bad = ((df[col] < 0) | (df[col] > 10)).sum()
            if n_bad:
                self._log(dataset, f"Clipped {n_bad} out-of-range '{col}' values to [0, 10]")
                df[col] = df[col].clip(lower=0, upper=10)
        return df

    def _fill_missing(self, df, dataset):
        """Numeric skill/requirement columns: missing == 'not required' -> 0.
        Everything else numeric: median impute. Categorical: 'Unknown'."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            n_missing = df[col].isna().sum()
            if n_missing == 0:
                continue
            if col.startswith(self.ZERO_IS_VALID_PREFIXES):
                df[col] = df[col].fillna(0)
                self._log(dataset, f"Filled {n_missing} missing '{col}' with 0 (not-required)")
            else:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                self._log(dataset, f"Filled {n_missing} missing '{col}' with median ({median_val})")

        cat_cols = df.select_dtypes(include=["string", "object"]).columns
        for col in cat_cols:
            n_missing = df[col].isna().sum()
            if n_missing:
                df[col] = df[col].fillna("Unknown")
                self._log(dataset, f"Filled {n_missing} missing '{col}' with 'Unknown'")
        return df

    def _downcast_numeric(self, df):
        """Shrink memory footprint without losing precision that matters here."""
        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")
        return df

    # -----------------------------------------------------------------
    # Step 3: per-dataset cleaning
    # -----------------------------------------------------------------
    def clean_mnc_criteria(self):
        df = self.raw["mnc_criteria"].copy()
        df = self._strip_and_standardize_strings(df, "mnc_criteria")
        df = self._drop_exact_duplicate_rows(df, "mnc_criteria")
        df = self._fix_id_dtype(df, "mnc_criteria", ["company_id"])
        df = self._clip_cgpa(df, "mnc_criteria", "min_cgpa")
        df = self._clip_percentage_columns(
            df, "mnc_criteria",
            [c for c in df.columns if c.startswith("min_") or c in
             ("min_10th_pct", "min_12th_pct")]
        )
        df = self._fill_missing(df, "mnc_criteria")
        assert df["company_id"].is_unique, "company_id must be unique in MNC_Criteria"
        self.clean["mnc_criteria"] = self._downcast_numeric(df)
        return self.clean["mnc_criteria"]

    def clean_role_requirements(self):
        df = self.raw["role_requirements"].copy()
        df = self._strip_and_standardize_strings(df, "role_requirements")
        df = self._drop_exact_duplicate_rows(df, "role_requirements")
        df = self._clip_percentage_columns(
            df, "role_requirements", [c for c in df.columns if c.startswith("min_")]
        )
        df = self._fill_missing(df, "role_requirements")
        assert df["role"].is_unique, "role must be unique in Role_Requirements"
        self.clean["role_requirements"] = self._downcast_numeric(df)
        return self.clean["role_requirements"]

    def clean_skill_gaps(self):
        df = self.raw["skill_gaps"].copy()
        df = self._strip_and_standardize_strings(df, "skill_gaps")
        df = self._drop_exact_duplicate_rows(df, "skill_gaps")
        df = self._fix_id_dtype(df, "skill_gaps", ["student_id"])
        df = self._clip_percentage_columns(df, "skill_gaps", ["current_score", "required_score"])

        # gap must equal max(required - current, 0); recompute defensively
        recomputed_gap = (df["required_score"] - df["current_score"]).clip(lower=0)
        mismatches = (~np.isclose(df["gap"], recomputed_gap)).sum()
        if mismatches:
            self._log("skill_gaps", f"Recomputed {mismatches} inconsistent 'gap' values")
        df["gap"] = recomputed_gap

        valid_priorities = {"Low", "Medium", "High"}
        bad_priority = ~df["priority"].isin(valid_priorities)
        if bad_priority.sum():
            self._log("skill_gaps", f"Fixed {bad_priority.sum()} invalid 'priority' labels -> 'Low'")
            df.loc[bad_priority, "priority"] = "Low"

        df = self._fill_missing(df, "skill_gaps")
        df["priority"] = df["priority"].astype("category")
        self.clean["skill_gaps"] = self._downcast_numeric(df)
        return self.clean["skill_gaps"]

    def clean_student_placement(self):
        df = self.raw["student_placement"].copy()
        df = self._strip_and_standardize_strings(df, "student_placement")
        df = self._drop_exact_duplicate_rows(df, "student_placement")
        df = self._fix_id_dtype(df, "student_placement", ["student_id"])
        df = self._clip_cgpa(df, "student_placement")

        pct_like = [c for c in df.columns if c not in (
            "student_id", "batch_year", "branch", "cgpa", "backlogs",
            "preferred_career", "recommended_role_from_profile", "career_family",
            "placement_status", "placed_role", "offers_count",
            "highest_package_lpa", "projects_count", "internship_months",
            "certifications_count", "hackathons_count", "relevant_internship",
        )]
        df = self._clip_percentage_columns(df, "student_placement", pct_like)

        df["backlogs"] = df["backlogs"].clip(lower=0)
        df["offers_count"] = df["offers_count"].clip(lower=0)
        df["highest_package_lpa"] = df["highest_package_lpa"].clip(lower=0)

        # referential/logical consistency: Not Placed <=> no role/offers/package
        not_placed = df["placement_status"] == "Not Placed"
        bad = not_placed & ((df["placed_role"] != "Not Placed") | (df["offers_count"] > 0)
                             | (df["highest_package_lpa"] > 0))
        if bad.sum():
            self._log("student_placement", f"Fixed {bad.sum()} 'Not Placed' rows with placement leftovers")
            df.loc[bad, ["placed_role"]] = "Not Placed"
            df.loc[bad, ["offers_count", "highest_package_lpa"]] = 0

        df = self._fill_missing(df, "student_placement")
        assert df["student_id"].is_unique, "student_id must be unique in Student_Placement"

        for col in ("branch", "placement_status", "placed_role", "career_family",
                    "recommended_role_from_profile", "preferred_career"):
            df[col] = df[col].astype("category")

        self.clean["student_placement"] = self._downcast_numeric(df)
        return self.clean["student_placement"]

    def clean_student_records(self):
        df = self.raw["student_records"].copy()
        df = self._strip_and_standardize_strings(df, "student_records")
        df = self._drop_exact_duplicate_rows(df, "student_records")
        df = self._drop_duplicate_merge_columns(df, "student_records")
        df = self._fix_id_dtype(df, "student_records", ["student_id", "source_student_id"])
        df = self._clip_cgpa(df, "student_records")
        df = self._clip_cgpa(df, "student_records", "company_min_cgpa")

        skip = {
            "student_id", "source_student_id", "synthetic_record", "batch_year",
            "branch", "cgpa", "backlogs", "preferred_career",
            "recommended_role_from_profile", "career_family", "placement_status",
            "placed_role", "offers_count", "highest_package_lpa",
            "projects_count", "internship_months", "certifications_count",
            "hackathons_count", "relevant_internship", "placement_readiness_band",
            "company_company_id", "company_role", "company_career_family",
            "company_selectivity", "company_note", "company_location_preference",
            "company_experience_required", "company_expected_ctc_lpa",
            "company_eligible_graduation_batches", "company_allowed_branches",
            "company_max_active_backlogs", "company_minimum_internship_months",
            "company_minimum_certifications", "company_match_status",
            "skill_gap_count", "total_skill_gap", "maximum_skill_gap",
            "high_priority_gap_count", "company_academic_eligible",
            "company_skill_fit_score",  # unbounded fit score, NOT a 0-100 pct
        }
        pct_like = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]
        df = self._clip_percentage_columns(df, "student_records", pct_like)

        # referential integrity vs. the role/company lookup tables
        valid_roles = set(self.clean["role_requirements"]["role"])
        bad_role = ~df["recommended_role_from_profile"].isin(valid_roles)
        if bad_role.sum():
            self._log("student_records", f"{bad_role.sum()} rows reference an unknown role")

        valid_companies = set(self.clean["mnc_criteria"]["company_id"])
        bad_company = ~df["company_company_id"].isin(valid_companies)
        if bad_company.sum():
            self._log("student_records", f"{bad_company.sum()} rows reference an unknown company_id")

        df = self._fill_missing(df, "student_records")

        for col in ("branch", "placement_status", "placed_role", "career_family",
                    "recommended_role_from_profile", "placement_readiness_band",
                    "company_company_id", "company_role", "company_selectivity",
                    "company_match_status"):
            if col in df.columns:
                df[col] = df[col].astype("category")

        assert df["student_id"].is_unique, "student_id must be unique in student_records"
        self.clean["student_records"] = self._downcast_numeric(df)
        return self.clean["student_records"]

    # -----------------------------------------------------------------
    # Step 4: engineered features that don't already exist as columns
    # -----------------------------------------------------------------
    def build_skill_gap_features(self):
        """Per-student aggregate features recomputed straight from the
        cleaned long-format skill_gaps table (used to sanity-check the
        pre-computed aggregate columns already sitting in student_records)."""
        sg = self.clean["skill_gaps"]
        agg = sg.groupby("student_id").agg(
            recomputed_skill_gap_count=("gap", lambda s: (s > 0).sum()),
            recomputed_total_skill_gap=("gap", "sum"),
            recomputed_max_skill_gap=("gap", "max"),
            recomputed_avg_skill_gap=("gap", "mean"),
            recomputed_high_priority_gap_count=("priority", lambda s: (s == "High").sum()),
        ).reset_index()
        return agg

    # -----------------------------------------------------------------
    # Step 5: build the single, finalized, ML-ready dataset
    # -----------------------------------------------------------------
    def build_final_dataset(self):
        """
        student_records.xlsx is already the fully-merged table (student
        profile + role requirements + company/MNC criteria + skill-gap
        aggregates + placement outcome), augmented to 5,000 rows for ML use.
        After cleaning, it IS the finalized dataset - we just validate it
        against a fresh recompute of the skill-gap aggregates for safety.
        """
        final = self.clean["student_records"].copy()

        check = self.build_skill_gap_features()
        # student_records is keyed by source_student_id back to the real student
        merged_check = final.merge(
            check, left_on="source_student_id", right_on="student_id",
            how="left", suffixes=("", "_check")
        )
        mismatch = (~np.isclose(
            merged_check["total_skill_gap"],
            merged_check["recomputed_total_skill_gap"],
            equal_nan=True,
        )).sum()
        self._log("final_dataset",
                   f"Validated total_skill_gap against fresh recompute: "
                   f"{mismatch} mismatches out of {len(final)} rows")

        # final column order: id/profile first, then everything else as-is
        id_cols = ["student_id", "source_student_id", "batch_year", "branch"]
        other_cols = [c for c in final.columns if c not in id_cols]
        final = final[id_cols + other_cols]

        self.clean["final_dataset"] = final
        return final

    # -----------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------
    def run_all(self):
        self.read_all_datasets()

        self.clean_mnc_criteria()
        self.clean_role_requirements()
        self.clean_skill_gaps()
        self.clean_student_placement()
        self.clean_student_records()
        self.build_final_dataset()

        self.save_all()
        self.write_report()
        return self.clean["final_dataset"]

    def save_all(self):
        name_map = {
            "mnc_criteria": "cleaned_mnc_criteria.xlsx",
            "role_requirements": "cleaned_role_requirements.xlsx",
            "skill_gaps": "cleaned_skill_gaps.xlsx",
            "student_placement": "cleaned_student_placement.xlsx",
            "final_dataset": "final_student_dataset.xlsx",
        }
        for key, filename in name_map.items():
            out_path = os.path.join(self.output_dir, filename)
            self.clean[key].to_excel(out_path, index=False)
            print(f"Saved {out_path}  shape={self.clean[key].shape}")

    def write_report(self):
        out_path = os.path.join(self.output_dir, "cleaning_report.txt")
        with open(out_path, "w") as f:
            f.write("DATA CLEANING REPORT\n")
            f.write("=" * 70 + "\n\n")
            if not self.report_lines:
                f.write("No issues found - all source files were already consistent.\n")
            for line in self.report_lines:
                f.write(line + "\n")
            f.write("\n" + "=" * 70 + "\n")
            f.write("FINAL DATASET SUMMARY\n")
            fd = self.clean["final_dataset"]
            f.write(f"final_student_dataset.xlsx: {fd.shape[0]} rows x {fd.shape[1]} columns\n")
            f.write(f"Unique students (student_id): {fd['student_id'].nunique()}\n")
            f.write(f"Unique real students (source_student_id): {fd['source_student_id'].nunique()}\n")
            f.write(f"Placement rate: {(fd['placement_status'] == 'Placed').mean():.1%}\n")
        print(f"Saved {out_path}")


if __name__ == "__main__":
    pipeline = Preprocess(
        data_dir="/mnt/user-data/uploads",
        output_dir="/mnt/user-data/outputs",
    )
    final_df = pipeline.run_all()
    print("\nFinal dataset preview:")
    print(final_df.head())