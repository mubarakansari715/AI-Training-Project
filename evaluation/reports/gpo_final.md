# Evaluation: GPO-Anual-report.pdf

- Config: `all-MiniLM-L6-v2 | 800/150 | k=5 | th=0.2 | rerank ms-marco-MiniLM-L-6-v2@20 | sections`
- Answer/judge model: `openai/gpt-oss-120b`

## Retrieval

{'hit_rate': 1.0, 'mrr': 0.944, 'avg_chunks': 5.0}

## Answers

{'accuracy': 1.0, 'citation_accuracy': 0.917}

| id | correct | cites page | judge reason |
|---|---|---|---|
| sample-1 | ✅ | True | The system answer correctly states the number of depository libraries (166) and the number of states (25) as specified in the reference answer. |
| sample-2 | ✅ | True | The system answer correctly identifies govinfo as the replacement for FDsys and accurately states the timeline of deployment, including the move out of beta in December 2017 and the full retirement of FDsys in December 2018, which aligns with the reference answer. |
| sample-3 | ✅ | True | The system answer correctly states the amount as $375.7 million (or more than $375 million), which matches the key facts in the reference answer. |
| copies-distributed | ✅ | True | The system answer accurately states the number of copies (802,834) and titles (4,211) as specified in the reference answer. |
| passports | ✅ | True | The system answer provides the exact figure of 15,901,600 passports and the rounded figure of 16 million, which aligns perfectly with the reference answer. |
| fdlp-webinars | ✅ | True | The system answer provides specific numbers (85 webinars, 10,214 participants) that align with the reference answer's 'over 85' and 'more than 10,000', and correctly identifies the key facts without contradiction. |
| revenues | ✅ | True | The system answer provides the exact figure ($874,466,000) and the rounded figure ($874.5 million) matching the reference answer. |
| staff | ✅ | True | The system answer correctly states the number of staff (1,737) as specified in the reference answer. |
| libraries-nationwide | ✅ | True | The system answer correctly states the number 1,133, which matches the reference answer. |
| nara-renovation | ✅ | False | The system answer correctly identifies both figures mentioned in the reference (over 24,000 and approximately 25,000 square feet) and accurately reflects the nuance that different sections of the document provide slightly different numbers. |
| preservation-stewards | ✅ | True | The system answer correctly states that GPO signed 20 additional Preservation Steward agreements in FY 2018, matching the reference answer. |
| govinfo-beta | ✅ | True | The system answer correctly identifies December 2017 as the date the website moved out of beta, matching the reference answer. |
| trap-not-in-report | ✅ | None | The system correctly states that the information is not found in the documents, aligning with the reference answer which indicates the document does not contain this information. |
