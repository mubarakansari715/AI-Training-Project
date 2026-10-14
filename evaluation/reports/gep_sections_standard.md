# Evaluation: GEP-Jun-2026.pdf

- Config: `all-MiniLM-L6-v2 | 800/150 | k=5 | th=0.2 | rerank ms-marco-MiniLM-L-6-v2@20 | sections`
- Answer/judge model: `auto`

## Retrieval

{'hit_rate': 1.0, 'mrr': 0.964, 'avg_chunks': 6.1}

## Answers

{'accuracy': 0.8, 'citation_accuracy': 0.929}

| id | correct | cites page | judge reason |
|---|---|---|---|
| sample-south-asia | ❌ | True | The system answer omits the key fact that India is projected to grow 6.6% in fiscal year 2026/27. |
| sample-ch3-debt | ❌ | True | The system answer omits the specific debt‑to‑GDP rise (under 40% to over 70%) and the basis‑point increases (110 and 30 bps) and does not mention the larger effects for countries with default history, low ratings, frontier status, short‑term debt reliance or weak governance, which are key facts in the reference. |
| sample-exec-risks | ✅ | True | The system answer lists all the main downside risks mentioned in the reference, matching the key facts without contradiction. |
| eap-growth | ✅ | True | The system answer correctly states the 2026 growth forecast of 4.2 percent for East Asia and Pacific, matching the key fact in the reference. |
| ssa-growth | ✅ | True | It correctly states the 2026 growth forecast of 4 percent. |
| lac-growth | ✅ | True | The answer correctly states the 2026 growth forecast of 2.2 percent, matching the reference. |
| eca-growth | ❌ | True | The system answer gives 2.8% instead of the reference’s 2.1% and omits the weakening in about 70% of economies, contradicting the key facts. |
| mna-growth | ✅ | True | The system answer correctly states the 2026 growth forecast of 1.6 percent for the MNA region, matching the key fact in the reference. |
| sar-vs-ssa | ✅ | True | The answer correctly states the 2026 growth forecasts for South Asia (6.3%) and Sub‑Saharan Africa (4%) and notes that South Asia is forecast to grow faster, matching the reference facts. |
| global-growth | ✅ | True | The system answer correctly states the 2026 global growth forecast of 2.5 percent, matching the key fact in the reference. |
| box-1-1-ai | ✅ | True | The answer includes all key facts from the reference—near‑term investment boost, medium‑term productivity growth, wide estimate range, optimistic 2030s growth exceeding 2000s average, and smaller gains under less optimistic assumptions—without contradiction. |
| box-2-1-lics | ✅ | False | The system answer correctly states the 5.4% growth forecast for 2026 and the 0.3‑percentage‑point drop from earlier forecasts, matching the key facts of the reference. |
| box-3-1-hidden-debt | ✅ | True | The system answer accurately states the key facts from the reference: hidden debt revelations lead to large, significant increases in sovereign spreads (higher borrowing costs) and highlight the need for better debt transparency. |
| ch4-commodity | ✅ | True | The system answer correctly states that a 1% rise in commodity prices raises both revenues and primary spending by about 0.4% after five years and that the windfall is gradually spent, matching the reference facts while adding permissible extra detail. |
| trap-japan-unemployment | ✅ | None | The system answer correctly states that the report does not provide Japan’s unemployment rate for 2025, matching the reference answer. |
