import pandas as pd
import numpy as np

import datetime
from typing import Tuple

PREPROC_PTH = "../data/preprocessed"
CLEAN_PTH = "../data/clean"

class DataProcessor:
    """
    Class for raw data processing
    """

    def __init__(self) -> None:
        pass

    def load_data(self, fname:str, state:str='raw', sep:str = ';') -> pd.DataFrame:
        data = pd.read_csv(f'../data/{state}/{fname}', sep=sep)
        return data

    def write_data(self, data:pd.DataFrame, fname:str, state:str=PREPROC_PTH) -> None:
        data.to_pickle(f'{state}/{fname}.pkl')

    def parse_dates(self, data:pd.DataFrame) -> pd.DataFrame:
        data['EntryDate'] = pd.to_datetime(data['EntryDate'], infer_datetime_format=True)
        data['EntryTime'] = pd.to_datetime(data['EntryTime'], infer_datetime_format=True)
        data['POD'] = data['EntryTime'].apply(lambda x: self.part_of_day(x))
        data.drop(columns=['EntryTime'], inplace=True)
        return data

    def parse_num_helper(self,x):
        x = str(x).replace('-','').strip()
        if x and len(x)>0:
            return float(x)
        else:
            return None

    def parse_numeric(self, data:pd.DataFrame) -> pd.DataFrame:
        num_cols = ['ValueNumber', 'RefHigh', 'RefLow']
        for col in num_cols:
            data[col] = data[col].apply(lambda x: self.parse_num_helper(x))
        return data

    def part_of_day(self, x:datetime.datetime) -> int:
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

    def test_separator(self, data:pd.DataFrame) -> Tuple[(pd.DataFrame,)*2]:
        nclp = data[~data['NCLP'].isna()]
        code = data[~data['Code'].isna()]
        return nclp, code

    def nclp_result(self, data:pd.DataFrame) -> pd.DataFrame:
        data["result"] = np.where(np.logical_or(data['ValueNumber'] < data['RefLow'], data['ValueNumber'] > data['RefHigh']), 1, 0)
        return data

    def run(self):
        data = self.load_data(fname='LabsALL 2015-2022.csv')
        data = self.parse_dates(data=data)
        data = self.parse_numeric(data=data)
        nclp_data, code_data = self.test_separator(data=data)
        nclp_data = self.nclp_result(data=nclp_data)
        self.write_data(data=nclp_data, fname="nclp")

if __name__ == "__main__":
    dp = DataProcessor()
    dp.run()