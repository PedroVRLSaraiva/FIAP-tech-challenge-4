# Comparativo: antes e depois da seleção de evidências

Registro dos resultados **antes** (busca vetorial pura, a versão que a equipe revisou) e **depois** (seleção de evidências com agrupamento de textos idênticos, MMR, reranking, filtros por metadados e modo temas). Todos os números vêm de execuções reais; os arquivos de origem estão em `eval/`.

## 1. Resumo

- **A revisão da equipe estava certa:** as evidências eram relevantes (precisão@10 de 0,95), mas **repetidas**. A métrica de relevância não enxergava isso. Passamos a medir diversidade.
- **Diversidade:** textos distintos nas 10 evidências foram de **9,08 para 10,00** (perguntas 2 e 5: de 7 para 10). Frases comuns viram uma só evidência com a contagem real na base (ex.: "Produto de qualidade" em 28 avaliações).
- **Filtros:** sem filtro, só 0,51 das evidências satisfaz uma restrição como "nota baixa"; com filtro, 1,00. A extração automática acertou 26 de 28.
- **Perguntas agregadas:** o modo temas troca 10 reviews parecidas por temas com tamanho ("55 de 244 avaliações recuperadas, 22%").
- **O que não melhorou:** relevância e fidelidade nas perguntas que já iam bem ficaram estáveis ou um pouco abaixo (fidelidade 0,96 → 0,92, por dois detalhes sem suporte). A latência subiu de 2,9 para 3,6 s. A busca híbrida não ajudou e ficou desligada.

## 2. Como medimos

| Métrica | Definição | Tipo |
|---|---|---|
| Textos distintos | Quantos textos diferentes (ignorando caixa, acento e pontuação) entre as 10 evidências | Exata |
| Redundância | Similaridade (cosseno) média entre pares de evidências escolhidas; menor é melhor | Proxy |
| Cobertura de temas | Fração dos 6 temas de um KMeans fixo (100 vizinhos) que as evidências alcançam | Proxy |
| Cosseno médio | Similaridade média das evidências com a pergunta | Proxy de relevância (penaliza o reranker por construção) |
| Conformidade com o filtro | Fração das evidências que satisfaz o filtro esperado | Exata |
| Precisão@10 | Fração das 10 evidências que tratam do assunto da pergunta | **Pré-avaliação**, por leitura |
| Fidelidade | 1 se toda afirmação é sustentada pelas evidências citadas; 0,5 se parcial | **Pré-avaliação**, por leitura |

As notas de precisão@10 e fidelidade foram atribuídas lendo cada resposta contra as evidências (e, nas dúvidas, o texto completo da review). **Não foram revisadas pela equipe** e a mesma rubrica foi usada nas duas rodadas.

## 3. Antes (linha de base)

Configuração: busca vetorial pura, top-10 por similaridade, sem filtros, sem informação de frequência. Arquivo: `eval/results_baseline.csv`. 20 perguntas (5 do enunciado, 8 de tema, 4 fora de escopo, 3 sem evidência).

- 20 de 20 comportamentos esperados (13 respondidas, 7 recusadas).
- Precisão@10 média de 0,95 e fidelidade de 0,96, sobre as 13 respondidas.
- Textos distintos nas 10 evidências: **9,08** em média, com mínimo de 7 nas perguntas 2 e 5.
- Latência média de 2,9 s nas perguntas respondidas.
- Repetições observadas: "Muita demora na entrega" ocupava 4 dos 10 slots na pergunta 5 (e 9 nos 50 candidatos da pergunta de prazo); "Produto de qualidade" aparece 23 vezes no top-50 da pergunta de qualidade.
- Sem categoria, situação da entrega nem frequência nas evidências; sem como restringir a nota, o período ou a entrega.

## 4. Depois

Configuração recomendada: agrupar idênticos + MMR (λ = 0,5) + reranking + extração automática de filtros; híbrida e temas desligados. Arquivo: `eval/results.csv` (28 perguntas, incluindo 8 com filtros).

- 28 de 28 comportamentos esperados (21 respondidas, 7 recusadas); 0 citações inválidas.
- Textos distintos: **10,00** em todas as perguntas.
- Precisão@10 / fidelidade nas 13 comparáveis: 0,94 / 0,92. Nas 8 com filtro: 0,91 / 1,00. Nas 21: 0,93 / 0,95.
- Latência média de 3,6 s nas 13 comparáveis.

### Por pergunta (antes → depois)

| # | Pergunta | Textos distintos | Precisão@10 | Fidelidade |
|---|---|---|---|---|
| 1 | Quais são os principais problemas relatados pelos clientes em… | 9 → 10 | 1,0 → 0,9 | 1,0 → 1,0 |
| 2 | O que os clientes relatam sobre problemas relacionados à entr… | 7 → 10 | 1,0 → 0,9 | 1,0 → 1,0 |
| 3 | Quais aspectos da experiência de compra são mais elogiados pe… | 10 → 10 | 1,0 → 1,0 | 1,0 → 1,0 |
| 4 | Quais padrões podem ser identificados nas avaliações de clien… | 10 → 10 | 0,8 → 1,0 | 1,0 → 1,0 |
| 5 | Quais avaliações sustentam a conclusão de que existem problem… | 7 → 10 | 1,0 → 0,8 | 1,0 → 0,5 |
| 6 | Há reclamações sobre produtos que chegaram quebrados ou danif… | 9 → 10 | 1,0 → 1,0 | 1,0 → 0,5 |
| 7 | O que os clientes dizem sobre a qualidade dos produtos? | 8 → 10 | 1,0 → 1,0 | 0,5 → 1,0 |
| 8 | Existem reclamações sobre o atendimento ou a falta de respost… | 10 → 10 | 0,8 → 0,8 | 1,0 → 1,0 |
| 9 | Clientes reclamam de receber um produto diferente do anunciad… | 10 → 10 | 1,0 → 1,0 | 1,0 → 1,0 |
| 10 | O que os clientes dizem sobre a embalagem dos produtos? | 10 → 10 | 1,0 → 1,0 | 1,0 → 1,0 |
| 11 | Os clientes elogiam a rapidez da entrega? | 9 → 10 | 1,0 → 1,0 | 1,0 → 1,0 |
| 12 | Quais problemas aparecem quando o produto nunca chegou? | 9 → 10 | 1,0 → 1,0 | 1,0 → 1,0 |
| 13 | Os clientes reclamam de falta de itens no pedido? | 10 → 10 | 0,8 → 0,8 | 1,0 → 1,0 |

Leitura:
- **Perguntas 2 e 5** (as mais repetidas) foram de 7 para 10 textos distintos. Na 2 a resposta passou a cobrir status desatualizado, Correios e desencontro na entrega; na 5, o modelo usou o dado de entrega para apontar que uma das avaliações vem de um pedido que chegou antes do prazo.
- **Pergunta 4** ganhou o filtro automático `nota ≤ 2`: as evidências são todas de clientes insatisfeitos (antes, duas eram ruído com nota 4 e 5), e a precisão@10 foi de 0,8 para 1,0.
- **Pergunta 5** perdeu relevância (1,0 → 0,8): o reranking e o MMR afastaram as evidências do "prazo" exato e trouxeram problemas de entrega em geral. É o custo da diversidade.
- **Pergunta 6** perdeu fidelidade: a resposta cita "itens de casa", que nenhuma evidência citada sustenta.

## 5. Comparação das técnicas (sem LLM de geração)

`python -m eval.compare_retrieval`: 21 perguntas respondíveis, k = 10, sem filtros.

| Configuração | Textos distintos (de 10) | Redundância ↓ | Cobertura de temas ↑ | Cosseno médio | ms/pergunta |
|---|---|---|---|---|---|
| baseline (vetorial pura) | 9,14 | 0,668 | 0,73 | 0,611 | 13 |
| + agrupar idênticos | 10,00 | 0,653 | 0,76 | 0,609 | 10 |
| + MMR | 9,38 | 0,654 | 0,76 | 0,611 | 9 |
| + agrupar idênticos + MMR | 10,00 | 0,643 | 0,79 | 0,609 | 10 |
| híbrida (BM25+RRF) | 9,00 | 0,651 | 0,65 | 0,592 | 37 |
| híbrida + agrupar + MMR | 10,00 | 0,590 | 0,71 | 0,576 | 31 |
| rerank | 9,33 | 0,657 | 0,70 | 0,587 | 335 |
| rerank + agrupar | 10,00 | 0,642 | 0,73 | 0,588 | 148 |
| rerank + agrupar + MMR | 10,00 | 0,620 | 0,78 | 0,586 | 104 |
| tudo (híbrida+rerank+agrupar+MMR) | 10,00 | 0,584 | 0,79 | 0,567 | 165 |

- **Agrupar idênticos** é o ganho mais claro: 9,14 → 10,00 textos distintos, sem custo de relevância.
- **MMR** soma redundância menor e cobertura de temas maior (0,76 → 0,79 sobre o agrupamento).
- **Híbrida** piora o cosseno e a cobertura de temas neste conjunto: as perguntas são semânticas e as reviews curtas. Ficou desligada.
- **Reranking** tem cosseno menor porque deixa de ordenar por cosseno. A leitura qualitativa mostrou ganho de aderência ao tema (pergunta de atendimento: trocou avaliações sobre entrega por avaliações sobre atendimento). Ficou ligado, com custo de 0,1 a 0,3 s.

Varredura de λ (agrupar idênticos + MMR):

| λ | Textos distintos | Redundância ↓ | Cobertura de temas ↑ | Cosseno médio |
|---|---|---|---|---|
| 0,4 | 10,00 | 0,632 | 0,83 | 0,608 |
| 0,5 | 10,00 | 0,643 | 0,79 | 0,609 |
| 0,6 | 10,00 | 0,647 | 0,78 | 0,609 |
| 0,7 | 10,00 | 0,648 | 0,77 | 0,609 |
| 0,8 | 10,00 | 0,651 | 0,76 | 0,609 |
| 1,0 | 10,00 | 0,653 | 0,76 | 0,609 |

Valores menores de λ cobrem mais temas sem custo visível no cosseno; adotamos 0,5, no limite inferior da faixa usual de 0,5 a 0,7.

## 6. Filtros

| Configuração | Filtro | Conformidade com o filtro | Textos distintos | Cosseno médio |
|---|---|---|---|---|
| baseline (vetorial pura) | sem filtro | 0,51 | 9,33 | 0,561 |
| baseline (vetorial pura) | com filtro | 1,00 | 9,67 | 0,541 |
| agrupar + MMR | sem filtro | 0,50 | 10,00 | 0,557 |
| agrupar + MMR | com filtro | 1,00 | 10,00 | 0,538 |

Perguntas 21 a 28 na rodada nova (filtros extraídos automaticamente da pergunta, todos corretos):

| # | Pergunta | Recorte aplicado |
|---|---|---|
| 21 | Principais problemas em avaliações com nota baixa | nota ≤ 2 |
| 22 | O que clientes elogiam em avaliações com nota alta | nota ≥ 4 |
| 23 | Problemas em pedidos entregues com atraso | entregue com atraso |
| 24 | O que dizem sobre produtos de telefonia | categoria: telephony |
| 25 | Reclamações nas compras de 2017 | compra de jan/2017 a dez/2017 |
| 26 | Problemas de móveis e decoração em avaliações negativas | nota ≤ 2; categoria: furniture decor |
| 27 | Eletrônicos entregues com atraso | categoria: electronics; entregue com atraso |
| 28 | Elogios nas compras do primeiro semestre de 2018 | compra de jan/2018 a jun/2018 |

A extração errou em 2 das 28 perguntas: "Como funciona um motor de carro?" e a pergunta da Tesla receberam `categoria: auto` só pela palavra "carro". Ambas foram recusadas do mesmo jeito. O critério de aceite do plano ("principais problemas em avaliações com nota baixa" devolve só reviews com nota ≤ 2) foi atendido na pergunta 21: as 10 evidências têm nota 1 ou 2.

## 7. Temas (KMeans)

Rodada `eval/results_temas.csv` (perguntas 1, 3, 4 e 21), com conjunto de até 600 vizinhos e `k = avaliações // 40`:

| Pergunta | Temas | Exemplo de tema (tamanho no conjunto recuperado) |
|---|---|---|
| 1. Principais problemas | 6 | Demora na entrega, 55 de 244 (22%); Produto diferente ou com defeito, 50 de 244 (20%); Não entrega e falta de atendimento, 48 de 244 (20%) |
| 3. Aspectos elogiados | 6 | Entrega no prazo e compra conforme o combinado, 159 de 600 (26%); Entrega correta e bom atendimento, 112 de 600 (19%) |
| 4. Padrões em insatisfeitos (nota ≤ 2) | 3 | Não recebimento do produto, 16 de 39 (41%); Falta de consideração e compromisso, 12 de 39 (31%) |
| 21. Problemas com nota baixa (nota ≤ 2) | 3 | Atrasos e falta de comunicação, 10 de 26 (38%); Produto não recebido, 9 de 26 (35%) |

Achados: (1) com 150 vizinhos o KMeans não tinha variedade para separar temas, e com 600 separa produto errado, falta de respeito, Correios, "não consegui avaliar porque não chegou", qualidade e atendimento; (2) a silhueta é ≈ 0,04 para qualquer k, então ela não serve para escolher k; (3) na pergunta 1 aparece um tema só de elogios, porque "problemas" atrai reviews como "nunca tive problemas", e o modelo avisou isso na resposta.

## 8. Limitações do comparativo

- A pré-avaliação é de uma só pessoa e não foi revisada; a rubrica é a mesma nas duas rodadas, mas a amostra é pequena (13 perguntas comparáveis).
- As perguntas 21 a 28 só existem na rodada nova; não há "antes" para elas além da conformidade com o filtro.
- Textos distintos só reconhece textos idênticos, não paráfrases.
- A latência depende do reranking (CPU) e da chamada de extração de filtros.

## 9. Como reproduzir

```bash
python -m eval.compare_retrieval                                   # seção 5, sem LLM de geração
python -m eval.run_eval --preset baseline --out eval/results_baseline.csv   # "antes"
python -m eval.run_eval                                            # "depois" -> eval/results.csv
python -m eval.run_eval --preset temas --ids 1,3,4,21 --out eval/results_temas.csv
```

`results_baseline.csv` guardado nesta pasta foi gerado pela versão anterior do código, com 20 perguntas; para reproduzi-lo com o código atual use `--preset baseline` e `--ids 1,...,20`.
