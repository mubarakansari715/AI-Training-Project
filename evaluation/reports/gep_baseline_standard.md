# Evaluation: GEP-Jun-2026.pdf

- Config: `all-MiniLM-L6-v2 | 800/150 | k=5 | th=0.2 | rerank ms-marco-MiniLM-L-6-v2@20`
- Answer/judge model: `auto`

## Retrieval

{'hit_rate': 0.929, 'mrr': 0.929, 'avg_chunks': 5.7}

## Answers

{'accuracy': 0.8, 'citation_accuracy': 0.714}

| id | correct | cites page | judge reason |
|---|---|---|---|
| sample-south-asia | ❌ | False | The system answer provides incorrect historical and projected growth figures that do not match the reference answer (e.g., 6.6% in 2026 versus the reference's 6.3% softening). |
| sample-ch3-debt | ✅ | True | The system answer accurately captures the key points of the reference answer regarding sovereign debt in developing economies without introducing any contradictions. |
| sample-exec-risks | ✅ | False | The system answer successfully captures the main downside risks to the global outlook outlined in the reference answer, such as conflict-related energy supply disruptions and higher prices, without contradicting the source. |
| eap-growth | ✅ | True | The system answer accurately states the growth forecast for East Asia and Pacific in 2026 as 4.2 percent, matching the reference answer. |
| ssa-growth | ✅ | True | The system answer accurately captures the 2026 growth forecast for Sub-Saharan Africa and the 0.3 percentage point downward revision mentioned in the reference answer. |
| lac-growth | ✅ | False | The system answer accurately states the 2026 growth forecast of 2.2 percent for Latin America and the Caribbean, matching the reference answer. |
| eca-growth | ❌ | False | The system stated the information was not provided, whereas the reference answer clearly gives the 2026 growth forecast as 2.1 percent. |
| mna-growth | ✅ | True | The system accurately identifies the 1.6 percent growth forecast for the region in 2026, aligning with the reference answer. |
| sar-vs-ssa | ❌ | True | The system answer misses the aggregate regional growth forecasts of 6.3 percent for South Asia and 4 percent for Sub-Saharan Africa, and instead provides individual country figures and incorrect comparisons. |
| global-growth | ✅ | True | The system answer accurately captures the global growth forecast of 2.5 percent for 2026 as stated in the reference. |
| box-1-1-ai | ✅ | True | The system answer accurately captures the key facts from the reference answer regarding AI's near-term investment boost, medium-term productivity growth, and the wide range of optimistic and pessimistic estimates, while providing additional relevant details from the document without contradiction. |
| box-2-1-lics | ✅ | True | The system answer accurately states the expected growth rate of 5.4 percent for low-income countries in 2026, matching the reference answer. |
| box-3-1-hidden-debt | ✅ | True | The system answer accurately captures the key facts from the reference answer regarding the association between hidden debt revelations and increased sovereign borrowing costs, while providing additional relevant context without contradiction. |
| ch4-commodity | ✅ | True | The system answer accurately captures the key facts from the reference answer regarding the impact of commodity price increases on revenues and spending, including the 0.4 percent increase after five years and the gradual spending of windfalls. |
| trap-japan-unemployment | ✅ | None | The system correctly identified that the report does not contain Japan's unemployment rate for 2025, matching the reference answer. |
