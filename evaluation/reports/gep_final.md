# Evaluation: GEP-Jun-2026.pdf

- Config: `all-MiniLM-L6-v2 | 800/150 | k=5 | th=0.2 | rerank ms-marco-MiniLM-L-6-v2@20 | sections`
- Answer/judge model: `openai/gpt-oss-120b`

## Retrieval

{'hit_rate': 1.0, 'mrr': 0.964, 'avg_chunks': 12.9}

## Answers

{'accuracy': 0.933, 'citation_accuracy': 1.0}

| id | correct | cites page | judge reason |
|---|---|---|---|
| sample-south-asia | ✅ | True | The system answer correctly states the key growth figures for South Asia (6.3% in 2026, ~7% in 2027-28) and India (6.6% in FY 2026/27), along with the primary drivers mentioned in the reference, without contradicting the ground truth. |
| sample-ch3-debt | ✅ | True | The system answer accurately includes all key points from the reference answer, such as the rise in debt from under 40% to over 70% of GDP, the non-linear relationship with higher spreads and yields (110 and 30 basis points), the specific vulnerabilities of certain countries, and the recommended policy measures, without contradicting the reference. |
| sample-exec-risks | ❌ | True | The system answer omits the upside risk mentioned in the reference answer, which states that broader AI investment and adoption could lift activity. |
| eap-growth | ✅ | True | The system answer correctly identifies the 2026 growth forecast as 4.2 percent and includes the context of moderation from 5 percent in 2025, which aligns with the key facts in the reference answer. |
| ssa-growth | ✅ | True | The system answer correctly identifies the 4 percent growth forecast for 2026, which is the primary fact requested, and does not contradict the reference. |
| lac-growth | ✅ | True | The system answer correctly identifies the 2.2 percent growth forecast for 2026, which is the primary fact requested in the question and present in the reference answer. |
| eca-growth | ✅ | True | The system answer correctly states the 2.1 percent growth forecast for 2026, which is the key fact in the reference answer, and the additional detail about excluding Russia and Türkiye does not contradict the reference. |
| mna-growth | ✅ | True | The system answer correctly states the 1.6 percent growth forecast for 2026, the 4 percent growth in 2025, and the 2.7-percentage-point downgrade, matching all key facts in the reference answer. |
| sar-vs-ssa | ✅ | True | The system answer correctly identifies the 2026 growth forecasts for South Asia (6.3%) and Sub-Saharan Africa (4%) and accurately states that South Asia is forecast to grow faster, matching all key facts in the reference answer. |
| global-growth | ✅ | True | The system answer correctly states the 2.5 percent growth forecast for 2026, which matches the key fact in the reference answer. |
| box-1-1-ai | ✅ | True | The system answer accurately captures all key points from the reference answer, including the near-term investment boost, medium-term productivity gains, the wide range of estimates, and the specific optimistic scenario where the 2030s become the strongest decade since the 1970s. |
| box-2-1-lics | ✅ | True | The system answer correctly states the expected growth rate of 5.4% for 2026 and the 0.3 percentage point decrease from previous forecasts, which aligns with the key facts in the reference answer. |
| box-3-1-hidden-debt | ✅ | True | The system answer correctly identifies that hidden debt revelations are associated with large and significant increases in sovereign spreads and highlights the need for improved debt transparency, which aligns with the key facts in the reference answer. |
| ch4-commodity | ✅ | True | The system answer correctly states that a 1 percent increase in commodity prices raises revenues and primary spending by about 0.4 percent after five years, and notes that windfalls are gradually spent rather than saved, which aligns with the reference answer. |
| trap-japan-unemployment | ✅ | None | The system correctly states that the report does not provide the requested information, aligning with the reference answer. |
