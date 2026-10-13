# Evaluation: GPO-Anual-report.pdf

- Config: `all-MiniLM-L6-v2 | 800/150 | k=5 | th=0.2 | rerank ms-marco-MiniLM-L-6-v2@20`
- Answer/judge model: `auto`

## Retrieval

{'hit_rate': 1.0, 'mrr': 0.944, 'avg_chunks': 5.0}

## Answers

{'accuracy': 1.0, 'citation_accuracy': 0.833}

| id | correct | cites page | judge reason |
|---|---|---|---|
| sample-1 | ✅ | True | The system answer accurately provides the correct number of depository libraries visited in FY 2018 as stated in the reference answer. |
| sample-2 | ✅ | True | The system answer accurately identifies govinfo as the replacement system and December 2018 as the date of full deployment and retirement of FDsys. |
| sample-3 | ✅ | True | The system answer accurately provides the procurement amount of more than $375 million (and specifically $375.7 million) for FY 2018, matching the reference answer. |
| copies-distributed | ✅ | True | The system answer accurately provides the exact number of copies and titles distributed according to the reference answer. |
| passports | ✅ | True | The system answer accurately provides the exact number of passports produced in FY 2018 as stated in the reference answer. |
| fdlp-webinars | ✅ | False | The system answer accurately captures both the number of webinars and participants mentioned in the reference answer, adding specific details without contradiction. |
| revenues | ✅ | True | The system answer accurately provides GPO's total revenues for FY 2018 in both exact figures and rounded format, matching the reference answer. |
| staff | ✅ | False | The system answer accurately provides the correct number of staff (1,737) for FY 2018 as stated in the reference answer. |
| libraries-nationwide | ✅ | True | The system answer accurately provides the correct number of depository libraries nationwide as stated in the reference answer. |
| nara-renovation | ✅ | True | The system answer correctly identifies the renovated space as approximately 25,000 square feet, which aligns with the reference answer's mention of approximately 25,000 square feet. |
| preservation-stewards | ✅ | True | The system answer accurately states that GPO signed 20 additional Preservation Steward agreements in FY 2018, matching the reference answer. |
| govinfo-beta | ✅ | True | The system answer correctly identifies December 2017 as the time the govinfo website moved out of beta. |
| trap-not-in-report | ✅ | None | The system correctly stated that the document does not contain information about GPO's budget for fiscal year 2025, which aligns with the reference answer. |
