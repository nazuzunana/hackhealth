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

Using data, we aim to identify people prone to having CKD to detect the disease as early as possible using 
most prevalent test and baseline information like age or bmi to enable simple indentification at early stage
for more descriptive tests.

## Data 

For this challenge we got anonymized data from IKEM containing test results and list of diagnoses including CKD. 


### Preparation and exploration

For data exploration we have used pandas_explain module which provides comprehensive reports about data.
One of our goals in the data preparation was to aggregate tests into single NCLP code as in essence they 
measure the same metrics. 

**Laboratory dataset**

Full data exploration results for laboratory test can be found it this dashboard 
[Laboratory data report](./documentation/labs_data_report.html) where you can see that we have evenly distributed
data across time. As a main data feature from this code we have selected NCLP code which corresponds to state code
for specific test. From our exploratory dashboard we can see that the NCLP code is present at nearly 90% of entries. 
The missing values were suplemented from the Analyte column based on creating mapping between different names to NCLP code.

After analysing Code feature in the dateset we identified that it contained only test requiered for transplantation procedure.
Thus we concluded that patients with these samples were either in very late stages of CKD or completely healthy,
which are considered as outliers in our usecase thus we opted for removing those entries.


**Diagnoses dataset**

Full data exploration results for diagnosis can be found in this dashboard 
[Diagnosis data report](./documentation/diag_clean.html). In this dataset we can see that the age of participants
is highly concentrated in older population with mean age of 67 years. This can lead to bias to age as a predictor
as in real world with increasing age we get higher tendency for chronical disease. Second finding in this dataset is that 
it may look like we have diagnosis just from 2 years, 2015 and 2022. After consultation with data owner that this could
be caused by the fact that they have selected patients which had entry check (first diagnosis) in 2015 and then add subsequent
cohort of CKD patients from 2022. 

**Synthetic dataset**

This was used to ˚



### Methodology

We have used IKEM data to build predictive model that would predict whether the patient has CKD based on laboratory
results.  We have employed Machine Learning and Data Science to come up with a solution that helps determine risky 
patients together with a strategy which would target and incentivize groups at risk. 
Our predictive model can accurately determine a CKD patient by identifying core subgroup of 20% in our data
which contained 99%

### Further suggestions

- Employ different models and incentives based on the probability:
   - Group 1 (age < 40) : Early Detection - educate and use available data without forcing additional tests (e.g. BMI, 
   - Group 2 (age 40 - 65) : Riskier - Identify “available” tests that have predictive power w.r.t. to CKD, incentivize people to test and report the results
   - Group 3 (age >65) : High alert - Incentivize people to undergo specialized tests that determine the disease.
   
## Authors

Pavel Milicka - pmilicka (at) deloittece.com
Kajetan Poliak - kpoliak (at) deloittece.com
Matej Marcisin - mmarcisin (at) deloittece.com
