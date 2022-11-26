import pandas as pd
import numpy as np

import datetime
from typing import Tuple
import re

PREPROC_PTH = "../data/preprocessed"
CLEAN_PTH = "../data/clean"
GRF_TEST_IDS = ["17341.0", "3086.0"]

class LabDataProcessor:
    """
    Class for raw lab data processing
    """

    def __init__(self) -> None:
        pass

    def load_data(self, fname: str, state: str = "raw", sep: str = ";") -> pd.DataFrame:
        data = pd.read_csv(f"../data/{state}/{fname}", sep=sep)
        return data

    def write_data(
        self, data: pd.DataFrame, fname: str, state: str = PREPROC_PTH
    ) -> None:
        data.to_pickle(f"{state}/{fname}.pkl")

    def parse_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        data["EntryDate"] = pd.to_datetime(
            data["EntryDate"], infer_datetime_format=True
        )
        data["EntryTime"] = pd.to_datetime(
            data["EntryTime"], infer_datetime_format=True
        )
        data["POD"] = data["EntryTime"].apply(lambda x: self.part_of_day(x))
        data.drop(columns=["EntryTime"], inplace=True)
        return data

    def parse_num_helper(self, x):
        x = str(x).replace("-", "").strip()
        if x and len(x) > 0:
            return float(x)
        else:
            return None

    def parse_numeric(self, data: pd.DataFrame) -> pd.DataFrame:
        num_cols = ["ValueNumber", "RefHigh", "RefLow"]
        for col in num_cols:
            data[col] = data[col].apply(lambda x: self.parse_num_helper(x))
        return data

    def part_of_day(self, x: datetime.datetime) -> int:
        if x.hour > 0 and x.hour <= 4:
            return 0
        elif x.hour > 4 and x.hour <= 8:
            return 1
        elif x.hour > 8 and x.hour <= 12:
            return 2
        elif x.hour > 12 and x.hour <= 16:
            return 3
        elif x.hour > 16 and x.hour <= 20:
            return 4
        else:
            return 5

    def add_unified_nclp(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds NCLP_E column to have same NCLP codes based on Analyte column.
        :param data:
        :return:
        """
        # extract coding dict in form {53.0: ['s_a1 antitrypsin'],
        #  54.0: ['s_a1 antitrypsin'],
        #  80.0: ['Elfo Alfa 1'],
        #  82.0: ['Elfo Alfa 1'],
        #  88.0: ['Alfa-1-globulin'],
        #  116.0: ['Elfo Alfa 2'],
        #  118.0: ['Elfo Alfa 2'],
        #  124.0: ['Alfa-2-globulin'],
        #  149.0: ['ADV'],
        #  159.0: ['s_Borrelia IgM'],
        #  175.0: ['Brucella abortus protilátky - (agl)'], ....
        # key -> NCLP code
        # value -> all Analyte forms found in data
        coding = (
            data.groupby(["NCLP"])["Analyte"]
            .apply(lambda grp: list(grp.value_counts().index))
            .to_dict()
        )
        # revert codes so we have Analyte as key and NCLP as single code
        codes = {}
        for k in coding.keys():
            for n in coding[k]:
                codes[n] = k

        data["NCLP_E"] = data.Analyte.apply(lambda cell: codes.get(cell, np.nan))

        return data

    def test_separator(self, data: pd.DataFrame) -> Tuple[(pd.DataFrame,) * 2]:
        nclp = data[~data["NCLP_E"].isna()]
        code = data[~data["Code"].isna()]
        return nclp, code

    def nclp_result(self, data: pd.DataFrame) -> pd.DataFrame:
        data["result"] = np.where(
            np.logical_or(
                data["ValueNumber"] < data["RefLow"],
                data["ValueNumber"] > data["RefHigh"],
            ),
            1,
            0,
        )
        return data

    def run(self):
        data = self.load_data(fname="LabsALL 2015-2022.csv")
        data = self.parse_dates(data=data)
        data = self.parse_numeric(data=data)
        data = self.add_unified_nclp(data=data)
        nclp_data, code_data = self.test_separator(data=data)
        nclp_data = self.nclp_result(data=nclp_data)
        self.write_data(data=nclp_data, fname="nclp")


class DiagDataProcessor:
    """
    Class for diagnosis data processing
    """

    def __init__(self) -> None:
        pass

    def diagnoses_preprocess(
        self,
        filename_in="../data/raw/Dg from report.xlsx",
        filename_out="../data/preprocessed/diagnoses.pkl",
    ):
        # Load
        df = pd.read_excel(filename_in)
        # Split Codes and Descriptions whenever possible
        df["mainDgCode"] = df["mainDgCode"].str.strip()
        df["OtherDgCode"] = df["OtherDgCode"].str.strip()
        df[["main_code", "main_description"]] = df["mainDgCode"].str.split(
            ":", n=1, expand=True
        )
        condition = df["OtherDgCode"].str.match(
            "[A-Z]\d", na=False
        )  # Code starts with number and letter if present
        df.loc[condition, ["other_code", "other_description"]] = (
            df.loc[condition, "OtherDgCode"].str.split(":", n=1, expand=True).values
        )
        df.loc[~condition, "other_description"] = df.loc[~condition, "OtherDgCode"]
        # Prepare target variable
        df["is_ckd"] = (
            df["main_code"].str.startswith("N18", na=False)
            | df["main_description"]
            .str.lower()
            .str.contains("chronické onemocnění ledvin", na=False)
            | df["other_code"].str.startswith("N18", na=False)
            | df["other_description"]
            .str.lower()
            .str.contains("chronické onemocnění ledvin", na=False)
        )
        df["is_ckd"] = df["is_ckd"].astype(int)
        df["is_dia"] = df["main_description"].str.lower().str.contains(
            "diabetes mellitus", na=False
        ) | df["other_description"].str.lower().str.contains(
            "diabetes mellitus", na=False
        )
        df["is_dia"] = df["is_dia"].astype(int)
        # Add data about stadium of ckd from main and other diagnosis
        df["ckd_stadium"] = df.loc[df["is_ckd"] == 1, "mainDgCode"].str.extract(
            r"chronické onemocnění ledvin, stadium (\d)", re.IGNORECASE, expand=True
        )
        df.loc[df["ckd_stadium"].isna(), "ckd_stadium"] = (
            df.loc[df["ckd_stadium"].isna(), "OtherDgCode"]
            .str.extract(
                r"chronické onemocnění ledvin, stadium (\d)", re.IGNORECASE, expand=True
            )
            .values
        )
        # Add relevant grouped metrics
        df_grouped = (
            df.groupby(["Patient"])
            .agg(
                min_date=pd.NamedAgg("Date", "min"),
                max_date=pd.NamedAgg("Date", "max"),
                cnt_rows=pd.NamedAgg("Date", "count"),
                cnt_visits=pd.NamedAgg("Date", "nunique"),
                is_ckd_patient=pd.NamedAgg("is_ckd", "max"),
                is_dia_patient=pd.NamedAgg("is_dia", "max"),
            )
            .sort_values("cnt_rows")
        )
        df_grouped_ckd = (
            df.loc[df["is_ckd"] == 1, :]
            .groupby("Patient")
            .agg(min_ckd_date=pd.NamedAgg("Date", "min"))
            .reset_index()
        )  # only ckd patients
        df_grouped_dia = (
            df.loc[df["is_dia"] == 1, :]
            .groupby("Patient")
            .agg(min_dia_date=pd.NamedAgg("Date", "min"))
            .reset_index()
        )  # only dia patients
        df_grouped["date_range"] = df_grouped["max_date"] - df_grouped["min_date"]
        df_grouped = (
            df_grouped.reset_index()
            .merge(df_grouped_ckd, on="Patient", how="left")
            .merge(df_grouped_dia, on="Patient", how="left")
        )
        df_joined = df.merge(
            df_grouped, on="Patient", how="inner"
        )  # merge the datasets together
        # add cummulative ckd/dia, assume patient is ckd/dia from the moment it first appears #TODO: check this with healtcare personnel
        df_joined["is_ckd_cum"] = (
            df_joined["min_ckd_date"] <= df_joined["Date"]
        ).astype(int)
        df_joined["is_dia_cum"] = (
            df_joined["min_ckd_date"] <= df_joined["Date"]
        ).astype(int)
        df_joined["is_ckd_cum"].fillna(0, inplace=True)
        df_joined["is_dia_cum"].fillna(0, inplace=True)
        # df_joined.loc[df_joined["Patient"] == 30877, :].sort_values("Date") # Example for checking the result
        # Final cosmetic modifications and write the output
        df_joined.rename(
            columns={"Patient": "patient_id", "Date": "date"}, inplace=True
        )
        for col in ["main_code", "main_description", "other_code", "other_description"]:
            df_joined[col] = df_joined[col].str.strip()
        df_joined.drop(columns=["mainDgCode", "OtherDgCode"]).to_pickle(filename_out)

    def diagnoses_clean(self, filename_out="../data/clean/diagnoses.pkl"):
        df = pd.read_pickle("../data/preprocessed/diagnoses.pkl")
        diag_unique_patients = df[["patient_id"]].drop_duplicates()
        print(f"N. unique patients in diagnoses: {len(diag_unique_patients)}")
        print(f"N. rows diag: {len(df)}")
        df_age_sex = pd.read_excel("../data/raw/hackath 112022 - Age SEX CKD.xlsx")
        df_bmi = pd.read_excel(
            "../data/raw/hackath 112022 - BMI weight height CKD 1.xlsx"
        )
        # validate NAs and how the dataset match on top of each other
        print("### NAs age sex")
        print(df_age_sex.isna().sum())
        print("### NAs bmi")
        print(df_bmi.isna().sum())

        age_unique_patients = df_age_sex[["Patient"]].drop_duplicates()
        bmi_unique_patients = df_bmi[["Patient"]].drop_duplicates()
        print("### Datasets joinability")
        print(
            f'{len(diag_unique_patients.merge(age_unique_patients, how = "inner", left_on = ["patient_id"], right_on = "Patient"))/len(diag_unique_patients)*100} % have age sex info'
        )
        print(
            f'{len(diag_unique_patients.merge(bmi_unique_patients, how = "inner", left_on = ["patient_id"], right_on = "Patient"))/len(diag_unique_patients)*100} % have bmi info'
        )
        # Prepare and join age
        df_age_sex.rename(
            columns={"Patient": "patient_id", "Sex": "sex", "Age": "age_2022"},
            inplace=True,
        )
        df = df.merge(df_age_sex, on="patient_id", how="left")
        df["age_date"] = pd.to_datetime(pd.Timestamp.now())
        df["age"] = (
            df["age_2022"] - pd.to_timedelta(df["age_date"] - df["date"]).dt.days / 365
        )
        # Prepare and join BMI
        df_bmi["bmi_fom"] = df_bmi["date"].dt.normalize() - pd.offsets.MonthBegin(1)
        df_bmi.rename(columns={"BMI": "bmi", "Patient": "patient_id"}, inplace=True)
        # Prepare skeleton, interpolate and join bmi data
        date_range = pd.DataFrame(
            {
                "fom": pd.date_range(
                    start=df["date"].min(),
                    end=df["date"].max(),
                    freq="MS",
                    inclusive="both",
                )
            }
        )
        diag_unique_patients["key"] = 0
        date_range["key"] = 0
        skeleton = diag_unique_patients.merge(date_range, on="key", how="inner")
        skeleton_bmi = skeleton.merge(
            df_bmi,
            left_on=["patient_id", "fom"],
            right_on=["patient_id", "bmi_fom"],
            how="left",
        )
        skeleton_bmi = skeleton_bmi.drop_duplicates(subset=["patient_id", "fom"])
        skeleton_bmi["bmi"] = skeleton_bmi.groupby("patient_id")["bmi"].apply(
            lambda group: group.interpolate(method="linear", limit_area="inside")
        )
        skeleton_bmi["bmi"] = skeleton_bmi.groupby("patient_id")["bmi"].apply(
            lambda x: x.bfill().ffill()
        )
        skeleton_bmi.drop(columns=["date"], inplace=True)
        df["date_fom"] = df["date"].dt.normalize() - pd.offsets.MonthBegin(1)
        df = df.merge(
            skeleton_bmi,
            left_on=["patient_id", "date_fom"],
            right_on=["patient_id", "bmi_fom"],
            how="left",
        )
        # Final cosmetic modifications and store the data
        df.drop(
            columns=["key", "Weight (kg)", "Height (cm)", "age_date", "bmi_fom"],
            inplace=True,
            errors="ignore",
        )
        print(f"N. rows diag after joining: {len(df)}")
        df.to_pickle(filename_out)

    def run(self) -> None:
        print("Processing diagnoses...")
        self.diagnoses_preprocess()
        print("Processing diagnoses done.")

        print("Diganoses clean start...")
        self.diagnoses_clean()
        print("Diagnoses clean finished.")


class DataFinalizer:
    """
    Join and finalize data
    """

    def __init__(
        self, process_lab_data: bool = False, process_diag_data: bool = False
    ) -> None:
        self.process_lab_data = process_lab_data
        self.process_diag_data = process_diag_data

    def load_data(self) -> Tuple[(pd.DataFrame,) * 2]:
        if self.process_diag_data:
            dgp = DiagDataProcessor()
            dgp.run()
        if self.process_lab_data:
            dlp = LabDataProcessor()
            dlp.run()

        lab_data = pd.read_pickle(f"{PREPROC_PTH}/nclp.pkl")
        diag_data = pd.read_pickle(f"{CLEAN_PTH}/diagnoses.pkl")

        return lab_data, diag_data

    def flatten_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data["year"] = data["date"].apply(lambda x: x.year)
        agg_last = ["age_2022", "sex", "is_dia", "is_ckd"]
        agg_median = ["bmi"]
        data = data.groupby("patient_id", as_index=False).agg(
            {
                **{col: "last" for col in agg_last},
                **{col: "median" for col in agg_median},
            }
        )
        return data

    def reduce_lab_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data.drop(
            columns=[
                "Report",
                "ID",
                "EntryDate",
                "Code",
                "Analyte",
                "ValueText",
                "RefHigh",
                "RefLow",
                "Unit",
            ],
            inplace=True,
        )
        return data

    def merge_data(
        self, lab_data: pd.DataFrame, diag_data: pd.DataFrame
    ) -> pd.DataFrame:
        if "Patient" in lab_data.columns.values:
            lab_data.rename(columns={"Patient": "patient_id"}, inplace=True)
        elif "patient_id" in lab_data.columns.values:
            pass
        else:
            raise "Mismatch in patient_id column name"

        data = pd.merge(left=lab_data, right=diag_data, how="left", on="patient_id")
        return data

    def get_relevant_tests(self, data:pd.DataFrame, num_of_tests:int=10)->list:
        pat = data.patient_id.unique().tolist()
        test_order = data.groupby(['NCLP_E']).apply(lambda grp: len(grp['patient_id'].unique())/len(pat)*100).sort_values(ascending=False)
        tmp = test_order.index.values
        tmp = [str(x) for x in tmp]
        relevant_tests = tmp[:num_of_tests]
        relevant_tests = relevant_tests + GRF_TEST_IDS
        return relevant_tests

    def one_hot_enc(self, data:pd.DataFrame) -> pd.DataFrame:
        relevant_tests = self.get_relevant_tests(data=data)
        data["NCLP_E"] = data["NCLP_E"].astype(str)
        sub_data = data[data['NCLP_E'].isin(relevant_tests)]
        # Get one hot encoding of column NCLP_E
        one_hot = pd.get_dummies(sub_data['NCLP_E'])
        # Drop column B as it is now encoded
        df = sub_data.drop('NCLP_E',axis = 1)
        # Join the encoded df
        df = df.join(one_hot)
        return df

    def run(self):
        lab_data, diag_data = self.load_data()
        diag_data = self.flatten_data(data=diag_data)
        lab_data = self.reduce_lab_data(data=lab_data)
        data = self.merge_data(lab_data=lab_data, diag_data=diag_data)
        data = self.one_hot_enc(data=data)
        data.to_pickle(f"{CLEAN_PTH}/full_data.pkl")


if __name__ == "__main__":
    df = DataFinalizer()
    df.run()
