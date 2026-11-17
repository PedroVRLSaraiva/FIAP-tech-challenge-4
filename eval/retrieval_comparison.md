# Comparação de recuperação (k=10, 21 perguntas, sem LLM de geração)

## A. Técnicas de seleção (sem filtros)

| Configuração | Textos distintos | Redundância ↓ | Cobertura de temas | Cosseno médio | ms/pergunta |
|---|---|---|---|---|---|
| baseline (vetorial pura) | 9.14 / 10 | 0.668 | 0.73 | 0.611 | 13 |
| + agrupar idênticos | 10.00 / 10 | 0.653 | 0.76 | 0.609 | 10 |
| + MMR | 9.38 / 10 | 0.654 | 0.76 | 0.611 | 9 |
| + agrupar idênticos + MMR | 10.00 / 10 | 0.643 | 0.79 | 0.609 | 10 |
| híbrida (BM25+RRF) | 9.00 / 10 | 0.651 | 0.65 | 0.592 | 37 |
| híbrida + agrupar + MMR | 10.00 / 10 | 0.590 | 0.71 | 0.576 | 31 |
| rerank | 9.33 / 10 | 0.657 | 0.70 | 0.587 | 335 |
| rerank + agrupar | 10.00 / 10 | 0.642 | 0.73 | 0.588 | 148 |
| rerank + agrupar + MMR | 10.00 / 10 | 0.620 | 0.78 | 0.586 | 104 |
| tudo (híbrida+rerank+agrupar+MMR) | 10.00 / 10 | 0.584 | 0.79 | 0.567 | 165 |

## Varredura de λ (agrupar idênticos + MMR)

| λ | Textos distintos | Redundância ↓ | Cobertura de temas | Cosseno médio |
|---|---|---|---|---|
| 0.4 | 10.00 | 0.632 | 0.83 | 0.608 |
| 0.5 | 10.00 | 0.643 | 0.79 | 0.609 |
| 0.6 | 10.00 | 0.647 | 0.78 | 0.609 |
| 0.7 | 10.00 | 0.648 | 0.77 | 0.609 |
| 0.8 | 10.00 | 0.651 | 0.76 | 0.609 |
| 1.0 | 10.00 | 0.653 | 0.76 | 0.609 |

## B. Filtros (perguntas com filtro esperado)

| Configuração | Filtro | Conformidade com o filtro | Textos distintos | Cosseno médio |
|---|---|---|---|---|
| baseline (vetorial pura) | sem filtro | 0.51 | 9.33 | 0.561 |
| baseline (vetorial pura) | com filtro | 1.00 | 9.67 | 0.541 |
| agrupar + MMR | sem filtro | 0.50 | 10.00 | 0.557 |
| agrupar + MMR | com filtro | 1.00 | 10.00 | 0.538 |
