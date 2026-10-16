# Evaluation: GEP-Jun-2026.pdf

- Config: `all-MiniLM-L6-v2 | 800/150 | k=5 | th=0.2 | rerank ms-marco-MiniLM-L-6-v2@20 margin 5 | sections | hybrid BM25 | decompose`
- Answer/judge model: `auto`

## Retrieval

{'hit_rate': 1.0, 'mrr': 1.0, 'avg_chunks': 4.8}

## Answers

{'accuracy': 1.0, 'citation_accuracy': 1.0, 'avg_tokens': 6904, 'total_cost_usd': 0.014475, 'avg_cost_usd': 0.002413}

| id | correct | cites page | judge reason |
|---|---|---|---|
| chart-moderate-risk | ✅ | True | The system answer accurately identifies the moderate risk percentages for 2015 and 2026 and calculates the correct percentage point difference, aligning closely with the reference answer while providing additional relevant breakdowns. |
| chart-debt-service | ✅ | True | The system accurately provided the debt-service costs for both 2010 (~7%) and 2025 (~11%), matching the reference answer. |
| chart-energy-revenue | ✅ | True | The system answer accurately identifies the share of commodity nontax revenue for energy exporters as approximately 38% (or 38.2%), which matches the reference answer's figure of about 38 percent. |
| chart-fiscal-rule-swf | ✅ | True | The system answer correctly identifies the key percentage (approximately 40%) of energy exporters with both a fiscal rule and a sovereign wealth fund, matching the reference answer. |
| chart-importers-high-risk | ✅ | True | The system answer accurately identifies that about 60 percent of commodity importers were at high risk of debt distress in 2020, matching the reference answer. |
| chart-picture-advanced | ✅ | True | The system's estimated value of approximately 100 percent falls within the acceptable range of 90 to 100 percent specified in the reference answer. |
