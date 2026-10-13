# Evaluation: GPO-Anual-report.pdf

- Config: `all-MiniLM-L6-v2 | 800/150 | k=5 | th=0.2 | rerank ms-marco-MiniLM-L-6-v2@20`
- Answer/judge model: `gemini-3.5-flash-lite`

## Retrieval

{'hit_rate': 1.0, 'mrr': 1.0, 'avg_chunks': 5.0}

## Answers

{'accuracy': 0.96, 'citation_accuracy': 0.96}

| id | correct | cites page | judge reason |
|---|---|---|---|
| gen-01 | ✅ | True | The system answer accurately provides the correct page count of 1,242 pages as stated in the reference answer. |
| gen-02 | ✅ | True | The system answer correctly identifies the number of additional Preservation Steward agreements as 20 for FY 2018. |
| gen-03 | ✅ | True | The system answer correctly identifies Title 44 as the relevant title from the U.S. Code, matching the reference answer. |
| gen-04 | ✅ | True | The system answer correctly identifies 'Article I, section 5' as the required section from the Constitution, while adding harmless extra detail and source information. |
| gen-05 | ✅ | True | The system answer successfully identifies the three key words required by the reference answer. |
| gen-06 | ✅ | True | The system answer correctly identifies the appropriation amount for Congressional Publishing in FY 1980 as $91.6 million, and the extra detail provided does not contradict the reference. |
| gen-07 | ✅ | True | The system answer accurately provides the requested funding amount of $79.5 million for fiscal year 2018. |
| gen-08 | ✅ | True | The system answer correctly identifies that the copies are available in PDF format, matching the reference answer. |
| gen-09 | ✅ | True | The system answer accurately provides the key fact from the reference answer ('more than 14,587') along with appropriate context. |
| gen-10 | ✅ | True | The system answer correctly identifies fiscal year 2018 as the time when the space was expanded, matching the reference answer. |
| gen-11 | ✅ | True | The system answer correctly identifies the number of libraries as 1,133, matching the reference answer while providing additional context from the document. |
| gen-12 | ✅ | True | The system answer correctly provides the exact number of titles (34,367) found in the reference answer. |
| gen-13 | ✅ | True | The system answer correctly identifies the fiscal year 2014 as the launch year of the FDLP Academy, matching the reference answer. |
| gen-14 | ✅ | True | The system answer correctly identifies Gregg Harper, Committee on House Administration Chairman, as the person who introduced the act. |
| gen-15 | ✅ | True | The system answer correctly identifies the number of titles available online through the GPO in fiscal year 2018 as 2,500,000, matching the reference answer. |
| gen-16 | ✅ | True | The system answer accurately identifies the Business Operations Revolving Fund as the checking account for the Government Publishing Office with the U.S. Treasury, matching the reference answer. |
| gen-17 | ✅ | True | The system correctly identified the Business Operations Revolving Fund as the answer. |
| gen-18 | ❌ | False | The system stated that the information was not in the document, but the ground truth indicates the answer is present. |
| gen-19 | ✅ | True | The system accurately provided the net income of $20.4 million for FY 2018 as stated in the reference answer. |
| gen-20 | ✅ | True | The system answer accurately captures all the key sources of reimbursement to the Revolving Fund as stated in the reference answer. |
| gen-21 | ✅ | True | The system answer correctly identifies 'five years' as the time period while providing helpful additional context from the document. |
| gen-22 | ✅ | True | The system answer correctly identifies the smart cards as HSPD-12 PIV smart cards, matching the key facts of the reference answer. |
| gen-23 | ✅ | True | The system answer accurately states the exact figure from the reference answer along with relevant context. |
| gen-24 | ✅ | True | The system answer accurately provides the net accounts receivable as of September 30, 2018, matching the reference answer while including helpful additional context. |
| gen-25 | ✅ | True | The system answer correctly identifies the fiscal year as 2019, matching the reference answer. |
