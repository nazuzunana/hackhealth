
import pandas as pd
import re
pd.options.display.max_colwidth = 200

def diagnoses_clean(filename_out = "../data/clean/diagnoses.pkl"):
    df = pd.read_pickle("../data/preprocessed/diagnoses.pkl")
    diag_unique_patients = df[["patient_id"]].drop_duplicates()
    print(f"N. unique patients in diagnoses: {len(diag_unique_patients)}")
    print(f"N. rows diag: {len(df)}")
    df_age_sex = pd.read_excel("../data/raw/hackath 112022 - Age SEX CKD.xlsx")
    df_bmi = pd.read_excel("../data/raw/hackath 112022 - BMI weight height CKD 1.xlsx")
    # validate NAs and how the dataset match on top of each other
    print("### NAs age sex")
    print(df_age_sex.isna().sum())
    print("### NAs bmi")
    print(df_bmi.isna().sum())

    age_unique_patients = df_age_sex[["Patient"]].drop_duplicates()
    bmi_unique_patients = df_bmi[["Patient"]].drop_duplicates()
    print("### Datasets joinability")
    print(f'{len(diag_unique_patients.merge(age_unique_patients, how = "inner", left_on = ["patient_id"], right_on = "Patient"))/len(diag_unique_patients)*100} % have age sex info')
    print(f'{len(diag_unique_patients.merge(bmi_unique_patients, how = "inner", left_on = ["patient_id"], right_on = "Patient"))/len(diag_unique_patients)*100} % have bmi info')
    # Prepare and join age
    df_age_sex.rename(columns = {"Patient": "patient_id", "Sex": "sex", "Age": "age_2022"}, inplace = True)
    df = df.merge(df_age_sex, on = "patient_id", how = "left")
    df["age_date"] = pd.to_datetime(pd.Timestamp.now())
    df["age"] = df["age_2022"] -  pd.to_timedelta(df["age_date"] - df["date"]).dt.days/365
    # Prepare and join BMI
    df_bmi["bmi_fom"] = df_bmi['date'].dt.normalize() - pd.offsets.MonthBegin(1)
    df_bmi.rename(columns = {"BMI": "bmi", "Patient": "patient_id"}, inplace = True)
    # Prepare skeleton, interpolate and join bmi data
    date_range = pd.DataFrame({"fom": pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="MS",inclusive="both")})
    diag_unique_patients["key"] = 0
    date_range['key'] = 0
    skeleton = diag_unique_patients.merge(date_range, on = "key", how = "inner")
    skeleton_bmi = skeleton.merge(df_bmi, left_on = ["patient_id", "fom"], right_on = ["patient_id", "bmi_fom"], how = "left")
    skeleton_bmi = skeleton_bmi.drop_duplicates(subset = ["patient_id", "fom"])
    skeleton_bmi["bmi"] = skeleton_bmi.groupby('patient_id')["bmi"].apply(lambda group: group.interpolate(method='linear', limit_area = "inside"))
    skeleton_bmi["bmi"] = skeleton_bmi.groupby('patient_id')['bmi'].apply(lambda x: x.bfill().ffill())
    skeleton_bmi.drop(columns = ["date"], inplace = True)
    df["date_fom"] = df['date'].dt.normalize() - pd.offsets.MonthBegin(1)
    df = df.merge(skeleton_bmi, left_on = ["patient_id", "date_fom"], right_on = ["patient_id", "bmi_fom"], how = "left")
    # Final cosmetic modifications and store the data
    df.drop(columns = ["key", "Weight (kg)", "Height (cm)", "age_date", "bmi_fom"], inplace = True, errors="ignore")
    print(f"N. rows diag after joining: {len(df)}")
    df.to_pickle(filename_out)

if __name__ == "__main__":
    print("Diganoses clean start...")
    diagnoses_clean()
    print("Diagnoses clean finished.")