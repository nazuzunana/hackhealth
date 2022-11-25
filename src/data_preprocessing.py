import pandas as pd
import re

pd.options.display.max_colwidth = 200

def diagnoses_preprocess(filename_in = "../data/raw/Dg from report.xlsx", filename_out = "../data/preprocessed/diagnoses.pkl"):
    # Load
    df = pd.read_excel(filename_in)
    # Split Codes and Descriptions whenever possible
    df["mainDgCode"] = df["mainDgCode"].str.strip()
    df["OtherDgCode"] = df["OtherDgCode"].str.strip()
    df[["main_code", "main_description"]] = df["mainDgCode"].str.split(":", n = 1, expand = True)
    condition = df["OtherDgCode"].str.match("[A-Z]\d", na = False) # Code starts with number and letter if present
    df.loc[condition,  ["other_code", "other_description"]] = df.loc[condition, "OtherDgCode"].str.split(":", n = 1, expand = True).values
    df.loc[~condition, "other_description"] = df.loc[~condition, "OtherDgCode"] 
    # Prepare target variable
    df["is_ckd"] = df["main_code"].str.startswith("N18", na = False) \
                    | df["main_description"].str.lower().str.contains("chronické onemocnění ledvin", na = False) \
                    | df["other_code"].str.startswith("N18", na = False) \
                    | df["other_description"].str.lower().str.contains("chronické onemocnění ledvin", na = False)
    df["is_ckd"] = df["is_ckd"].astype(int)
    df["is_dia"] = df["main_description"].str.lower().str.contains("diabetes mellitus", na = False) \
                    | df["other_description"].str.lower().str.contains("diabetes mellitus", na = False)
    df["is_dia"] = df["is_dia"].astype(int)
    # Add data about stadium of ckd from main and other diagnosis
    df["ckd_stadium"] = df.loc[df["is_ckd"] == 1, "mainDgCode"].str.extract(r'chronické onemocnění ledvin, stadium (\d)', re.IGNORECASE, expand=True)
    df.loc[df["ckd_stadium"].isna(), "ckd_stadium"] = df.loc[df["ckd_stadium"].isna(), "OtherDgCode"].str.extract(r'chronické onemocnění ledvin, stadium (\d)', re.IGNORECASE, expand=True).values
    # Add relevant grouped metrics
    df_grouped = df.groupby(["Patient"]).agg(min_date = pd.NamedAgg("Date", "min"),
                           max_date = pd.NamedAgg("Date", "max"),
                           cnt_rows = pd.NamedAgg("Date", "count"),
                           cnt_visits = pd.NamedAgg("Date", "nunique"),
                           is_ckd_patient = pd.NamedAgg("is_ckd", "max"),
                           is_dia_patient = pd.NamedAgg("is_dia", "max")).sort_values("cnt_rows")
    df_grouped_ckd = df.loc[df["is_ckd"] == 1, :].groupby("Patient").agg(min_ckd_date = pd.NamedAgg("Date", "min")).reset_index() # only ckd patients
    df_grouped_dia = df.loc[df["is_dia"] == 1, :].groupby("Patient").agg(min_dia_date = pd.NamedAgg("Date", "min")).reset_index() # only dia patients
    df_grouped["date_range"] = df_grouped["max_date"] - df_grouped["min_date"]
    df_grouped = df_grouped.reset_index()\
        .merge(df_grouped_ckd, on = "Patient", how = "left")\
        .merge(df_grouped_dia, on ="Patient", how = "left")
    df_joined = df.merge(df_grouped, on = "Patient", how = "inner") # merge the datasets together
    # add cummulative ckd/dia, assume patient is ckd/dia from the moment it first appears #TODO: check this with healtcare personnel
    df_joined["is_ckd_cum"] = (df_joined["min_ckd_date"] <= df_joined["Date"]).astype(int)
    df_joined["is_dia_cum"] = (df_joined["min_ckd_date"] <= df_joined["Date"]).astype(int)
    df_joined["is_ckd_cum"].fillna(0, inplace=True)
    df_joined["is_dia_cum"].fillna(0, inplace=True)
    # df_joined.loc[df_joined["Patient"] == 30877, :].sort_values("Date") # Example for checking the result
    # Final cosmetic modifications and write the output
    df_joined.rename(columns = {"Patient": "patient_id", "Date": "date"}, inplace=True)
    for col in ["main_code", "main_description", "other_code", "other_description"]:
        df_joined[col] = df_joined[col].str.strip()
    df_joined.drop(columns = ["mainDgCode","OtherDgCode"]).to_pickle(filename_out)

if __name__ == "__main__":
    print("Processing diagnoses...")
    diagnoses_preprocess()
    print("Processing diagnoses done.")