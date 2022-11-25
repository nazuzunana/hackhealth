# hackhealth

## Data Dictionary 

### Preprocessed/diagnoses.csv
* patient_id
* date
* main_code
* main_description
* other_code
* other_description
* is_ckd - 1 if one of the diagnosis is CKD
* is_dia - 1 if one of the diagnosis is Diabetes
* ckd_stadium	- stadium of CKD (not filled for all entries) of the diagnoses
* min_date	- patient first diagnosis
* max_date	- patient max diagnosis
* cnt_rows	- number of rows of the patient
* cnt_visits	- number of visits (unique dates of the patient)
* is_ckd_patient	- is the patient ckd (whenever in time)
* is_dia_patient	- is the patient diabetic (whenever in time)
* date_range	- diff between first and last visit in days
* min_ckd_date	- first date of ckd diagnosis of the patient
* min_dia_date	- first date of diabetes diagnoses of the patient
* is_ckd_cum	- cummulative CKD (once patient is diagnosed, it is flagged as CKD from that point on)
* is_dia_cum - cummulative diabetes (once patient is diagnosed, it is flagged as Dia from that point on)
