# Evaluation: GEP-Jun-2026.pdf

- Config: `all-MiniLM-L6-v2 | 800/150 | k=5 | th=0.2 | rerank ms-marco-MiniLM-L-6-v2@20 margin 5 | sections | hybrid BM25 | decompose`
- Answer/judge model: `auto`

## Retrieval

{'hit_rate': 0.833, 'mrr': 0.667, 'avg_chunks': 5.0}

## Answers

{'accuracy': 0.167, 'citation_accuracy': 0.667, 'avg_tokens': 2622, 'total_cost_usd': 0.002281, 'avg_cost_usd': 0.00038}

| id | correct | cites page | judge reason |
|---|---|---|---|
| chart-moderate-risk | ❌ | True | The system stated that the information was unavailable, whereas the reference answer successfully extracted the data points and calculated the percentage-point difference from the chart. |
| chart-debt-service | ✅ | False | The system answer accurately provides the 2010 and 2025 debt-service cost percentages (6% vs ~7% within acceptable rounding margin, and ~11% for 2025), capturing the key facts from the reference answer. |
| chart-energy-revenue | ❌ | True | The system stated that the information was unavailable, but the reference answer clearly provides the figure from the document (about 38 percent). |
| chart-fiscal-rule-swf | ❌ | False | The system answer states that the information is unavailable, whereas the reference answer provides the specific percentage from the document. |
| chart-importers-high-risk | ❌ | True | The reference answer states the share was about 60 percent, but the system incorrectly claimed the information was not provided. |
| chart-picture-advanced | ❌ | True | The system stated the information was not provided, whereas the reference answer actually contains the required data point (roughly 95 percent). |
