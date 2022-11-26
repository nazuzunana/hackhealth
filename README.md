# EHH Ettioled Challenge 6
This is code repository for the European Healthcare Hackathon from 26.11.2022 for challeng #6

## Challenge description
Chronical Kidney Disease (CKD) in late stages has huge human and financial cost, including regular dialysis, 
costly medications and in late stages transplantation (in better case) or death. The main challenges with
CKD are:

- Chronical Kidney Disease (CKD) involves gradual loss of kidney function. It can progress to end-stage kidney failure, which is fatal without artificial filtering (dialysis) or a kidney transplant and is linked to other health complications. As CKD progresses, it affects the quality of life for the patients as the late stages of the disease requires dialysis.
- Each patient that requires dialysis costs  approx. 1 million CZK, inducing costs of 6 billions CZK per year with more than 6,000 dialysis patients in 2021. 
- Research made CDC (Center for Disease Control and Prevention) claims the disease affects more than 15% of the population. As the disease probability is linked to age, the number of people suffering from it will only increase.
- There is little prevention and education w.r.t. CKD, which results to the disease being identified in later stages, where it already is life threatening and requires costly treatments.
- If identified early, the medications available can significantly slow down the progression. 


## Proposed goals and solution 

Using data, we aim to identify people prone to having CKD to detect the disease as early as possible.

 - Employ different models and incentives based on the probability:
   - Group 1 (age < 40) : Early Detection - educate and use available data without forcing additional tests (e.g. BMI, 
   - Group 2 (age 40 - 65) : Riskier - Identify “available” tests that have predictive power w.r.t. to CKD, incentivize people to test and report the results
   - Group 3 (age >65) : High alert - Incentivize people to undergo specialized tests that determine the disease.

## Data 

### Preparation and exploration


### Methodology



### Findings

### Recommendations

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
