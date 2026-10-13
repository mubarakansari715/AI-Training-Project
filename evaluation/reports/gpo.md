# Evaluation: GPO-Anual-report.pdf

- Config: `all-MiniLM-L6-v2 | 800/150 | k=5 | th=0.2 | rerank ms-marco-MiniLM-L-6-v2@20`
- Answer/judge model: `gemini-3.5-flash-lite`

## Retrieval

{'hit_rate': 1.0, 'mrr': 0.944, 'avg_chunks': 5.0}

## Answers

{'accuracy': 1.0, 'citation_accuracy': 1.0}

| id | correct | cites page | judge reason |
|---|---|---|---|
| sample-1 | ✅ | True | The system answer correctly identifies the number of depository libraries visited in FY 2018 as 166, which matches the reference answer. |
| sample-2 | ✅ | True | The system accurately identified govinfo as the replacement system and provided the correct deployment timeline matching the reference answer. |
| sample-3 | ✅ | True | The system answer accurately captures both the approximate figure ('more than $375 million') and the specific amount ('approximately $375.7 million') mentioned in the reference answer. |
| copies-distributed | ✅ | True | The system answer accurately provides both the number of copies and titles distributed in FY 2018 as stated in the reference answer. |
| passports | ✅ | True | The system answer includes both the exact figure and the rounded figure mentioned in the reference answer. |
| fdlp-webinars | ✅ | True | The system answer accurately captures both key facts from the reference answer (over 85 webinars and more than 10,000 participants) and includes additional matching details without contradiction. |
| revenues | ✅ | True | The system answer accurately provides the total revenues for FY 2018 in both exact and rounded formats, matching the reference answer. |
| staff | ✅ | True | The system answer accurately provides the correct number of staff (1,737) for FY 2018 as stated in the reference answer. |
| libraries-nationwide | ✅ | True | The system answer accurately provides the correct number of depository libraries nationwide, matching the reference answer. |
| nara-renovation | ✅ | True | The system answer accurately captures the figures (25,000 and over 24,000 square feet) mentioned in the reference answer. |
| preservation-stewards | ✅ | True | The system answer accurately provides the correct number of Preservation Steward agreements signed in FY 2018 as stated in the reference answer. |
| govinfo-beta | ✅ | True | The system answer accurately states that the govinfo website moved out of beta in December 2017, matching the reference answer. |
| trap-not-in-report | ✅ | None | The system correctly stated that the information was not found in the documents, matching the reference answer. |
