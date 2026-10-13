# Evaluation: GPO-Anual-report.pdf

- Config: `all-MiniLM-L6-v2 | 800/150 | k=5 | th=0.2 | rerank ms-marco-MiniLM-L-6-v2@20`
- Answer/judge model: `auto`

## Retrieval

{'hit_rate': 1.0, 'mrr': 0.944, 'avg_chunks': 5.0}

## Answers

{'accuracy': 1.0, 'citation_accuracy': 1.0}

| id | correct | cites page | judge reason |
|---|---|---|---|
| sample-1 | ✅ | True | The system answer accurately provides the correct number of depository libraries visited by GPO staff in FY 2018, matching the reference answer. |
| sample-2 | ✅ | True | The system correctly identified the replacement system as govinfo and the full deployment/retirement date as December 2018. |
| sample-3 | ✅ | True | The system answer accurately provides the key figure of $375.7 million for printing and information products procured by GPO in FY 2018, matching the reference answer. |
| copies-distributed | ✅ | True | The system answer accurately provides the exact number of copies and titles mentioned in the reference answer. |
| passports | ✅ | True | The system answer accurately provides the exact number of passports produced in FY 2018 as stated in the reference answer. |
| fdlp-webinars | ✅ | True | The system answer accurately provides the number of webinars and participants for FY 2018, matching the key facts in the reference answer with specific details from the document. |
| revenues | ✅ | True | The system answer accurately states GPO's total revenues for FY 2018, matching both the exact figure and the rounded value provided in the reference answer. |
| staff | ✅ | True | The system answer accurately provides the correct number of staff (1,737) for FY 2018 as stated in the reference answer. |
| libraries-nationwide | ✅ | True | The system answer correctly states the exact number of depository libraries nationwide as provided in the reference answer. |
| nara-renovation | ✅ | True | The system answer correctly states the renovated space as approximately 25,000 square feet, which aligns with the reference answer. |
| preservation-stewards | ✅ | True | The system answer accurately provides the correct number of agreements signed in FY 2018 as stated in the reference answer. |
| govinfo-beta | ✅ | True | The system answer correctly identifies December 2017 as the date the govinfo website moved out of beta. |
| trap-not-in-report | ✅ | None | The system correctly identified that the information regarding GPO's budget for fiscal year 2025 was not found in the documents. |
