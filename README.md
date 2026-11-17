# Voice of Customer Intelligence com RAG

Site de busca em linguagem natural sobre as avaliações de clientes da base *Brazilian E-Commerce Public Dataset by Olist*. O usuário faz uma pergunta e recebe uma resposta **restrita às avaliações recuperadas**, com citações `[n]` e as evidências (texto, nota, data, categoria, situação da entrega, score). Quando não há evidência suficiente ou a pergunta está fora do escopo, o sistema recusa em vez de inventar.

> Projeto do Tech Challenge, Fase 4 (FIAP). Os números deste documento vêm de execuções reais sobre a base completa. O comparativo completo do antes e depois está em [`Documentação/Comparativo antes e depois.md`](Documentação/Comparativo%20antes%20e%20depois.md).

## Arquitetura

```
                 ┌────────────────────── OFFLINE (build_index.py) ──────────────────────┐
 Olist reviews ► │ load ─► clean/filter ─► build docs ─► ⋈ pedidos/itens/produtos        │
 + pedidos       │   ─► embed (batch, cache) ─► ChromaDB (vetor + metadados)             │
                 └───────────────────────────────────────────────────────────────────────┘

                 ┌────────────────────────── ONLINE (app.py) ───────────────────────────┐
 Pergunta ─────► │ normalize ─► filtros (manuais + extraídos pelo LLM, só se houver pista)│
                 │   ─► embed query ─► busca vetorial COM pré-filtro de metadados         │
                 │   ─► score gate ─► [BM25 + RRF] ─► agrupar idênticos ─► [rerank] ─► MMR │
                 │        └─ ou: KMeans em temas                                          │
                 │   ─► contexto [1..k] ─► LLM (prompt restritivo) ─► validador de citações│
                 │   ─► UI (evidências, temas, filtros aplicados, exportação CSV)         │
                 └───────────────────────────────────────────────────────────────────────┘
```

## Tecnologias

| Camada | Escolha |
|---|---|
| Linguagem | Python 3.12 |
| Dados | pandas |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dim, vetores normalizados) |
| Banco vetorial | ChromaDB (persistente, similaridade cosseno, filtros por metadados) |
| Busca lexical | `rank_bm25` (BM25) |
| Reranking | `sentence-transformers`, cross-encoder multilíngue `mmarco-mMiniLMv2-L12-H384-v1` |
| Agrupamento por tema | `scikit-learn` (KMeans, TF-IDF) |
| LLM | OpenAI, modelo `gpt-6-luna` (SDK `openai`) |
| Interface | Streamlit |
| Testes | pytest (69 testes) |

Versões fixadas em `requirements.txt`.

## Instalação e execução

```bash
git clone https://github.com/PedroVRLSaraiva/FIAP-tech-challenge-4.git && cd FIAP-tech-challenge-4
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Dataset: baixa os CSVs da Olist do Kaggle (kagglehub) para data/
#    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
python download_data.py

# 2. Chave da OpenAI (usada para embeddings e geração)
cp .env.example .env        # edite e preencha OPENAI_API_KEY

# 3. Construir o índice (gera data/processed/ e chroma_db/; idempotente)
python build_index.py            # --reindex recria o índice reaproveitando o cache de embeddings (sem custo de API)
                                 # --rebuild recria tudo, inclusive os embeddings (gera custo de API)

# 4. Subir o site
streamlit run app.py
```

O reranking baixa o modelo (~470 MB) na primeira vez que é usado.

Passo a passo detalhado, com saídas esperadas, custos, testes e problemas comuns: seção [Reprodutibilidade](#reprodutibilidade).

Outros comandos:

```bash
pytest                                     # testes automatizados
python notebooks/eda.py                    # exploratória da base (números para este README)
python -m src.calibrate                    # calibração dos limiares do gate (só usa embeddings; ver abaixo)
python -m eval.compare_retrieval           # compara as técnicas de seleção, sem LLM de geração
python -m eval.run_eval                    # avalia eval/perguntas.csv na configuração recomendada -> eval/results.csv
python -m eval.run_eval --preset baseline --out eval/results_baseline.csv   # busca vetorial pura (comportamento anterior)
python -m eval.run_eval --preset temas --ids 1,3,4,21 --out eval/results_temas.csv
```

## Preparação da base

Fonte: `olist_order_reviews_dataset.csv`, enriquecida com os dados do pedido (ver abaixo). Cada review válida vira **um documento**.

| Etapa | Decisão | Justificativa |
|---|---|---|
| Seleção | Apenas reviews com `review_comment_message` não vazia | Sem texto não há o que recuperar; a nota isolada não carrega o motivo da avaliação |
| Ausentes | Reviews sem texto são descartadas; título (quando existe) é concatenado à mensagem | O título costuma resumir a opinião |
| Limpeza | Espaços e quebras normalizados, URLs trocadas por `[url]`, caracteres de controle removidos. **Sem** remover acentos, stopwords ou alterar caixa | Embeddings de transformers dependem do texto natural |
| Textos curtos | Descartados abaixo de 15 caracteres ou 3 palavras (ex.: "ok", "Muito bom") | Trazem **poucos insights sobre a real opinião do cliente**: dizem que gostou ou não, mas não o motivo. Como o objetivo é entender o porquê, esses textos só adicionariam ruído ao índice e ao top-k |
| Duplicatas | Exatas por `(order_id, texto)` removidas; `review_id` repetido mantém a primeira | O dataset tem `review_id` repetido |
| Documentos | Um documento por review, **sem chunking** | Reviews são curtas; o p95 de tamanho fica muito abaixo do limite do modelo: p95 de 192 caracteres (~33 palavras), máximo de 208 caracteres |
| Metadados | `review_score`, `data` e os campos derivados do pedido (tabela abaixo) | Alimentam os cartões de evidência, os filtros e o contexto enviado ao LLM |

Funil de dados (saída de `python notebooks/eda.py`):

| Etapa | Reviews |
|---|---|
| Total no CSV | 99.224 (58,7% sem mensagem) |
| Com mensagem | 40.950 |
| Descartadas por tamanho (< 15 caracteres ou < 3 palavras) | 4.976 |
| Descartadas por duplicata `(order_id, texto)` | 6 |
| Descartadas por `review_id` repetido | 266 |
| **Documentos finais** | **35.702** |

Metadados derivados do pedido (`olist_orders`, `olist_order_items`, `olist_products` e a tradução das categorias), juntados por `order_id`:

| Campo | Como é calculado |
|---|---|
| `order_status` | Status do pedido |
| `categoria` | **Uma por pedido**: a categoria mais frequente entre os itens; em empate, a do primeiro item. Nome em inglês (tradução oficial); as 2 categorias sem tradução (`pc_gamer` e uma de cozinha) mantêm o nome original. Só 0,7% dos pedidos têm mais de uma categoria |
| `ano_mes` (AAAAMM) e `data_compra_ts` (epoch) | Data da compra. O Chroma só filtra intervalos sobre números |
| `atraso_dias` | Data de entrega − data estimada, em dias (negativo = antes do prazo) |
| `entregue_atrasado` | `atraso_dias > 0` |

Valores ausentes (pedido não entregue, produto sem categoria) são **omitidos** dos metadados, porque o Chroma não aceita `None`; um documento sem o campo nunca satisfaz um filtro sobre ele. Cobertura: todos os 35.702 documentos têm pedido, 34.709 têm categoria e 33.939 têm entrega registrada. Entre os pedidos entregues, **11,7% chegaram atrasados**, e a nota média cai de 3,75 para 2,04 nesses pedidos.

## Estratégia do pipeline RAG

1. **Processamento da pergunta** (`src/query.py`): normalização (espaços e limite de tamanho). A pergunta segue direto para o embedding, **sem reescrita pelo LLM** (query expansion fora do escopo).
2. **Filtros** (`src/filters.py`, `src/filter_extraction.py`): filtros manuais da barra lateral e filtros extraídos da pergunta (ver seção própria).
3. **Embedding** da pergunta com o mesmo modelo e a mesma normalização do índice.
4. **Recuperação** (`src/retrieval.py`): top-50 no Chroma **com o filtro dentro da consulta** (pré-filtro, e não depois do top-k); score = `1 − distância cosseno`.
5. **Gate de evidência** (`src/guardrails.py`), sobre os candidatos vetoriais e antes de chamar o LLM de geração.
6. **Seleção das evidências** (`src/selection.py`): corte por `SCORE_MIN`, agrupar textos idênticos, reranking e MMR, ou então agrupamento em temas. Padrão: `top_k` = 10, ajustável no site.
7. **Contexto** (`src/context.py`): reviews numeradas `[n] (nota | data | categoria | entrega | texto idêntico em N avaliações) "texto"`, delimitadas por `<reviews>…</reviews>`.
8. **Geração** (`src/prompts.py`, `src/llm.py`): prompt que restringe a resposta ao contexto, exige citações `[n]`, trata o conteúdo das reviews como dado (mitiga prompt injection), deixa claro que se trata de uma amostra e só permite citar números que estejam escritos no contexto.
9. **Validação de citações** (`src/answer.py`): índices fora de `1..k` são removidos e registrados, intervalos como `[1]–[4]` são expandidos, e frases sem citação geram alerta na UI.

## Seleção das evidências (diversidade e utilidade)

Antes, o top-k eram as 10 reviews **mais parecidas com a pergunta** e nada mais. Isso gera repetição: uma pergunta como "prazo de entrega" trazia a mesma frase várias vezes, e uma frase tão comum quanto "Produto de qualidade" aparece em 28 avaliações da base. Cada técnica abaixo pode ser ligada ou desligada na barra lateral:

| Técnica | O que faz | Padrão |
|---|---|---|
| **Agrupar textos idênticos** (`src/diversify.py`) | Colapsa cópias do mesmo texto (ignora caixa, acento e pontuação) em uma só e informa em quantas avaliações da **base inteira** ele aparece e com que notas. Libera slots e transforma repetição em dado de frequência | Ligado |
| **MMR** (λ = 0,5) | A cada passo escolhe a review que maximiza `λ·relevância − (1−λ)·similaridade com as já escolhidas` | Ligado |
| **Reranking** (`src/rerank.py`) | Um cross-encoder lê pergunta e review juntas e reordena por utilidade, não por similaridade de vetores | Ligado |
| **Busca híbrida** (`src/lexical.py`) | BM25 sobre os mesmos textos, fundido com a busca vetorial por Reciprocal Rank Fusion (`1/(60+posição)`) | Desligado |
| **Temas** (`src/themes.py`) | Agrupa as avaliações recuperadas com KMeans, nomeia cada tema por palavras frequentes e informa o tamanho; o LLM organiza a resposta por tema | Desligado |

**Por que esses padrões.** Medimos cada técnica sem chamar o LLM de geração (`python -m eval.compare_retrieval`, 21 perguntas, k = 10):

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

- **Agrupar idênticos** leva os textos distintos de 9,14 para 10,00 sem perder relevância (cosseno 0,611 → 0,609).
- **MMR** sobre isso reduz a redundância e aumenta a cobertura de temas (0,76 → 0,79). A varredura de λ mostra que valores menores cobrem mais temas sem custo no cosseno; adotamos 0,5, o limite inferior da faixa usual de 0,5 a 0,7.
- **Híbrida** ficou desligada: ela **reduz** o cosseno médio (0,611 → 0,592) e a cobertura de temas (0,73 → 0,65) neste conjunto, em que as perguntas são semânticas e as reviews curtas. O BM25 traz reviews que repetem as palavras da pergunta, mas pouco informativas.
- **Reranking** ficou ligado. O cosseno médio cai (0,609 → 0,586) **porque o reranker deixa de ordenar por cosseno**, então esse proxy o penaliza por construção. Na leitura das evidências ele troca itens sobre entrega, que o baseline trazia na pergunta de atendimento, por avaliações que falam de atendimento. O custo é de 0,1 a 0,3 s por pergunta.

Varredura de λ (agrupar idênticos + MMR):

| λ | Textos distintos | Redundância ↓ | Cobertura de temas ↑ | Cosseno médio |
|---|---|---|---|---|
| 0,4 | 10,00 | 0,632 | 0,83 | 0,608 |
| 0,5 | 10,00 | 0,643 | 0,79 | 0,609 |
| 0,6 | 10,00 | 0,647 | 0,78 | 0,609 |
| 0,7 | 10,00 | 0,648 | 0,77 | 0,609 |
| 0,8 | 10,00 | 0,651 | 0,76 | 0,609 |
| 1,0 | 10,00 | 0,653 | 0,76 | 0,609 |

As métricas são proxies: **redundância** é a similaridade média entre as evidências escolhidas, **cobertura de temas** é a fração dos 6 temas de um KMeans fixo sobre os 100 vizinhos que as evidências alcançam, e **cosseno médio** é a similaridade com a pergunta.

## Filtros por metadados

Nota, categoria, período da compra e entrega com atraso, aplicados **na consulta ao vetor** (`where` do Chroma) e, na busca híbrida, pela mesma máscara em pandas (testes garantem que os dois dão o mesmo resultado).

- **Manuais:** controles da barra lateral.
- **Automáticos** (`src/filter_extraction.py`): o LLM devolve um JSON (`nota_min`, `nota_max`, `categorias`, `periodo_inicio`, `periodo_fim`, `entregue_atrasado`) que é **validado contra o esquema**: nota de 1 a 5, categorias só da lista real, período AAAAMM válido, intervalos coerentes; campo inválido é descartado. Exemplos: "nota baixa" → `nota ≤ 2`; "clientes insatisfeitos" → `nota ≤ 2`; "em 2017" → jan a dez/2017; "entregues com atraso" → `entregue_atrasado`.
- **Custo controlado:** o LLM só é consultado quando a pergunta contém uma pista de filtro (nota, período, categoria, atraso). As demais perguntas não gastam requisição. O resultado fica em cache em disco.
- **Prioridade:** o filtro manual vence o automático no mesmo campo. Os filtros aplicados aparecem como etiquetas acima da resposta e entram no prompt ("recorte aplicado à busca"), para a resposta dizer sobre qual recorte fala.
- **Guardrail:** se os filtros zerarem os candidatos, o sistema informa que não há avaliações naquele recorte, sem chamar o LLM.
- **Contexto enriquecido:** cada evidência leva categoria e situação da entrega ("entregue 12 dias após o prazo"). Isso permitiu respostas como a da pergunta 27, em que o modelo notou que uma avaliação que diz "entregue no prazo" não combina com o recorte de pedidos atrasados.

Efeito medido (8 perguntas com filtro esperado, k = 10):

| Configuração | Filtro | Conformidade com o filtro | Textos distintos | Cosseno médio |
|---|---|---|---|---|
| baseline (vetorial pura) | sem filtro | 0,51 | 9,33 | 0,561 |
| baseline (vetorial pura) | com filtro | 1,00 | 9,67 | 0,541 |
| agrupar + MMR | sem filtro | 0,50 | 10,00 | 0,557 |
| agrupar + MMR | com filtro | 1,00 | 10,00 | 0,538 |

Sem filtro, só metade das evidências (0,51) satisfaz a restrição pedida; com filtro, todas. O cosseno cai um pouco porque o recorte tem menos candidatos próximos da pergunta.

## Temas (KMeans)

Para perguntas agregadas ("principais problemas"), o modo temas recupera até 600 vizinhos (só entram os com `score ≥ SCORE_MIN`), agrupa-os em 3 a 6 temas (`k = avaliações // 40`), nomeia cada um por palavras-chave TF-IDF e escolhe exemplos distintos e centrais de cada tema. A resposta passa a vir organizada por tema, com o tamanho de cada um **sobre as avaliações recuperadas**.

Dois achados da medição: a silhueta do KMeans é praticamente constante (≈ 0,04) para qualquer k em perguntas genéricas, então k é função do tamanho do conjunto; e com 150 vizinhos não havia variedade, por isso o conjunto é de até 600.

## Interface

Pergunta e exemplos clicáveis; barra lateral com `top_k`, opções de seleção, filtros e histórico; resposta com citações `[n]` que levam ao cartão da evidência; cartões com nota, data, categoria, situação da entrega, score e selo de "texto idêntico em N avaliações"; temas com barras de tamanho; etiquetas do recorte aplicado; **exportação das evidências em CSV**; painel "Como a resposta foi gerada" com filtros, gate, scores, seleção, latência por etapa e o prompt completo.

## Tratamento de falta de evidência

| Caso | Mecanismo |
|---|---|
| Nenhum resultado relevante | `max_score < T_MAX` → recusa sem chamar o LLM de geração |
| Evidência insuficiente | Menos de `MIN_DOCS` reviews com `score ≥ SCORE_MIN` → recusa |
| Fora de escopo / sem suporte no contexto | O prompt manda responder `FORA_DE_ESCOPO` ou `SEM_EVIDENCIA`; a saída é convertida no status correspondente |

Nas recusas, a UI mostra as avaliações mais próximas encontradas, rotuladas como **não conclusivas**.

**Calibração dos limiares.** O cosseno entre embeddings varia de modelo para modelo, então `T_MAX`, `SCORE_MIN` e `MIN_DOCS` em `src/config.py` foram calibrados sobre a base real com `python -m src.calibrate` (15 perguntas dentro e 15 fora do escopo; só usa a API de embeddings) e conferidos nas 20 perguntas de `eval/perguntas.csv`.

| Conjunto | `max_score` |
|---|---|
| Dentro do escopo | 0,500 a 0,843. As mais baixas são perguntas abstratas, como "padrões em clientes insatisfeitos" (0,50 a 0,54) |
| Fora do escopo | 0,256 a 0,486, exceto "melhor time de futebol" (0,617, por causa de reviews de camisas de time) |
| Sem evidência (assunto de e-commerce, sem reviews sobre o tema) | 0,483 a 0,525 |

**Os conjuntos se sobrepõem:** uma pergunta abstrata legítima pontua no mesmo patamar de um assunto que a base não cobre. Nenhum limiar único separa os dois. Por isso o gate é uma **primeira barreira grossa**, calibrada para **não recusar perguntas legítimas** (`T_MAX = 0,48`, `SCORE_MIN = 0,45`, `MIN_DOCS = 3`), e descarta só o que é claramente alheio à base (capital da França, receita de bolo). O julgamento semântico fica com a **segunda barreira**, os tokens `SEM_EVIDENCIA` e `FORA_DE_ESCOPO` do LLM. Recusar uma pergunta válida é pior do que gastar uma chamada de geração num caso limítrofe. A margem da pergunta abstrata mais baixa (0,500 contra `T_MAX` 0,48) é pequena, e o custo é que casos sem evidência próximos do limiar dependem do LLM para recusar (3 de 3 recusados na avaliação).

## Decisões técnicas

- **Embeddings da OpenAI (`text-embedding-3-small`):** multilíngue, sem infraestrutura local e uma única chave para embeddings e geração. O custo para indexar ~36 mil reviews curtas é de poucos centavos; os vetores ficam em cache em `.cache/`, e `--reindex` reaproveita o cache.
- **ChromaDB:** roda local, persiste em disco, filtra por metadados dentro da consulta e é simples de reproduzir.
- **Sem chunking:** cada review já é uma unidade semântica curta.
- **Python puro, sem framework de orquestração:** cada etapa fica explícita e explicável.
- **Gate determinístico antes do LLM:** reduz custo e alucinação; o token `SEM_EVIDENCIA` é a segunda barreira.
- **`LLMClient` isolando o provedor:** o pipeline não depende do SDK da OpenAI.
- **Sem reescrita da query:** a pergunta vai direto para o embedding. Reduz uma chamada de LLM por pergunta (custo, latência e exposição ao limite de requisições do modelo) e deixa a busca determinística. O preço são perguntas abstratas com score mais baixo, tratado na calibração do gate.
- **Textos curtos descartados:** "ok" e "Muito bom" dizem pouco sobre a real opinião do cliente (ver tabela de preparação).
- **Seleção por técnicas opcionais, com padrões escolhidos por medição:** nada foi ligado por padrão sem número ou leitura que o justificasse (ver "Seleção das evidências").
- **No máximo 2 requisições de LLM por pergunta** (extração de filtros, só quando há pista, e geração), o que importa porque o modelo tem limite diário de requisições.

## Exemplos de perguntas e respostas

Respostas reais do pipeline na configuração recomendada (`python -m eval.run_eval`), com o texto como o site o entrega.

**1. Frequência em vez de repetição.** `O que os clientes dizem sobre a qualidade dos produtos?`

> - Nas avaliações recuperadas, vários clientes elogiam a qualidade, descrevendo os produtos como bons, ótimos ou excelentes [1][3][6][8][9][10].
> - Há também comentários mais moderados: um cliente diz que a qualidade correspondeu ao esperado [7], enquanto outro afirma que ela deixa a desejar [2].

Evidências citadas (3 de 8):

- `[6]` nota 5/5, score 0.63 (computers accessories; entregue 12 dias antes do prazo; texto idêntico em 28 avaliações da base): "Produto de qualidade"
- `[1]` nota 5/5, score 0.62 (computers accessories; entregue 19 dias antes do prazo): "Produto de qualidade! Compromisso com o cliente e qualidade no produto resumem essa compra."
- `[2]` nota 4/5, score 0.63 (bed bath table; entregue 18 dias antes do prazo): "Qualidade do produto deixa a desejar"

**2. Pergunta abstrata do enunciado, com filtro automático.** `Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?`  
Recorte aplicado automaticamente: **nota ≤ 2**

> Nas avaliações recuperadas, os principais padrões são:
>
> - **Problemas de entrega e conclusão do serviço:** há relatos de produto não entregue, serviço incompleto e mercadoria atrasada [2][3][6].
> - **Falta de comunicação e de compromisso com o cliente:** clientes mencionam ausência de satisfação sobre o atraso e falta de compromisso ou consideração [3][6][7][10].
> - **Produto diferente do esperado:** aparecem queixas sobre cor enviada incorretamente e material de má qualidade [1][8].
> - **Insatisfação intensa:** alguns clientes dizem não recomendar a compra ou não querer comprar novamente [5][9].
>
> Esses padrões refletem apenas esta amostra de avaliações com nota ≤ 2.

Evidências citadas (3 de 9):

- `[1]` nota 1/5, score 0.47 (bed bath table; entregue 15 dias antes do prazo): "Regular. Péssimo material , falta de respeito com consumidor, mandam as cores q bem entendem"
- `[2]` nota 1/5, score 0.46 (bed bath table; entregue 6 dias após o prazo): "Impossível avaliar um serviço que não foi completado"
- `[3]` nota 1/5, score 0.48: "Produto não entregue. Produto não entregue e sem dar satisfação ao cliente."

**3. Filtros combinados (nota e categoria).** `Quais problemas de móveis e decoração aparecem nas avaliações negativas?`  
Recorte aplicado automaticamente: **nota ≤ 2; categoria: furniture decor**

> Nas avaliações negativas recuperadas, aparecem estes problemas:
>
> - **Defeitos de acabamento e fabricação:** pé trocado, solda aparente, acabamento ruim e quadro que não encosta na parede [1][2][8].
> - **Produtos danificados ou com riscos:** mesa chegou danificada e espelhos e produto vieram arranhados [3][4][6].
> - **Material de baixa qualidade ou frágil:** críticas à qualidade dos espelhos, da malha da cortina e do material de objetos [4][7][10].
> - **Instabilidade ou problemas de apoio:** mesa bamba e apoio do tampo que escorrega [3].
> - **Embalagem sem proteção adequada:** estofados e mesa chegaram sem proteção suficiente [1][3].
> - **Divergência ou falta de itens:** produto diferente da descrição, cortina diferente da imagem e uma cortina ausente [6][9].
> - **Atraso na entrega:** também houve reclamação de atraso sem explicação [5].

Evidências citadas (3 de 10):

- `[1]` nota 1/5, score 0.56 (furniture decor; entregue 6 dias antes do prazo): "A cadeira decorativa veio c/ pé trocado, c/ solda aparente, acabamento mal feito. estofados vieram s/ nenhuma proteçã..."
- `[2]` nota 2/5, score 0.49 (furniture decor; entregue 9 dias antes do prazo): "Verifiquei que o acabamento deixa muito a desejar."
- `[3]` nota 1/5, score 0.48 (furniture decor; entregue 3 dias após o prazo): "péssimo. além de ter sido muito mal embalado sem nenhuma proteção e por isso chegou completamente danificado, a mesa ..."

**4. Pergunta agregada em modo temas.** `Quais são os principais problemas relatados pelos clientes em suas avaliações?`

> Nas avaliações recuperadas, os principais problemas relatados foram:
>
> - **Demora na entrega — 55 de 244 avaliações recuperadas (22%)**: clientes mencionam atrasos e falta de satisfação sobre a entrega [1][2].
> - **Produto diferente, com defeito ou troca não resolvida — 50 de 244 (20%)**: há relatos de produto divergente do pedido, avaria e demora para atender pedidos de troca ou devolução [3][4].
> - **Não entrega e falta de atendimento — 48 de 244 (20%)**: clientes dizem não receber o produto nem esclarecimentos ou atenção às solicitações [5][6].
> - **Falta de respeito ou atenção — 37 de 244 (15%)**: aparecem queixas de falta de respeito e atenção ao cliente [7][8].
> - **Produto não recebido, impossibilitando a avaliação — 37 de 244 (15%)**: clientes afirmam que o pedido ainda não chegou e, por isso, não podem avaliar o produto [9][10].
>
> O Tema F, com **17 de 244 avaliações recuperadas (7%)**, traz exemplos positivos de qualidade, agilidade e atenção ao cliente, não de problemas [11][12].

**5. Recusa pelo gate (fora de escopo).** `Qual a capital da França?`

> Não encontrei avaliações suficientes para responder a essa pergunta.

Status `sem_evidencia`; `max_score` = 0.28. O gate barrou antes de chamar o LLM de geração.

**6. Recusa pelo LLM (assunto plausível, sem reviews sobre o tema).** `O que os clientes dizem sobre o serviço de assinatura de streaming?`

> Não encontrei avaliações suficientes para responder a essa pergunta.

Status `sem_evidencia`; `max_score` = 0.52. O gate deixou passar e o LLM recusou (`SEM_EVIDENCIA`).

## Avaliação

`eval/perguntas.csv` tem 28 perguntas: 5 do enunciado, 8 de temas específicos, 8 que exigem **filtros**, 4 fora de escopo e 3 sem evidência. Para cada uma o `eval/run_eval.py` grava status, evidências, scores, citações, filtros aplicados, latência e a pré-avaliação.

**Antes e depois** (perguntas 1 a 13, as mesmas nas duas rodadas; antes = busca vetorial pura, `eval/results_baseline.csv`; depois = agrupar + MMR + rerank + filtros automáticos, `eval/results.csv`):

| Métrica | Antes | Depois |
|---|---|---|
| Textos distintos entre as 10 evidências (média) | 9,08 (mínimo 7, nas perguntas 2 e 5) | **10,00** (10 em todas) |
| Precisão@10 (relevância, pré-avaliação) | 0,95 | 0,94 |
| Fidelidade (pré-avaliação) | 0,96 | 0,92 |
| Evidências citadas na resposta (média de 10) | 7,4 | 7,2 |
| Latência média | 2,9 s | 3,6 s |

Novas capacidades, medidas só na rodada nova:

| Métrica | Resultado |
|---|---|
| Comportamento esperado (responder ou recusar), 28 perguntas | **28 de 28** |
| Perguntas com filtro (21 a 28): precisão@10 / fidelidade | 0,91 / 1,00 |
| Extração automática de filtros correta | 26 de 28 |
| Evidências que satisfazem o filtro pedido | 0,51 sem filtro → 1,00 com filtro |
| Citações inválidas / frases sem citação | 0 / 1 |

Por pergunta (antes → depois):

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

**O que a leitura mostra.** A precisão@10 do baseline (0,95) media apenas **relevância**, e por isso escondia a repetição que a equipe apontou na revisão: as evidências eram relevantes, mas muitas vezes cópias umas das outras. A diversidade passou a ser medida (textos distintos) e subiu de 9,08 para 10,00; nas perguntas 2 e 5, que eram as piores, de 7 para 10. **A relevância e a fidelidade não melhoraram nas perguntas que já iam bem, e a fidelidade caiu um pouco (0,96 → 0,92):** na pergunta 6 a resposta cita "itens de casa", que nenhuma evidência citada sustenta, e na 5 duas citações dão apoio fraco ao "prazo". Os ganhos estão na cobertura, na frequência informada e nas perguntas que dependem de recorte, não na precisão das perguntas simples. A latência subiu de 2,9 para 3,6 s por causa do reranking e da extração de filtros.

> As notas de precisão@10, fidelidade e acerto da recusa são uma **pré-avaliação** feita lendo cada resposta contra as evidências (as colunas finais de `eval/results.csv` trazem a observação de cada caso). A equipe deve conferir antes da entrega. **Precisão@10** é a fração das 10 evidências que tratam do assunto da pergunta; **fidelidade** é 1 quando toda afirmação é sustentada pelas evidências citadas (0,5 para parcial).

## Limitações conhecidas

- Perguntas agregadas ("principais problemas") são respondidas sobre uma **amostra** das reviews recuperadas, não sobre a base inteira. No modo temas, os tamanhos valem para o conjunto recuperado, e não para a base.
- Agrupar idênticos só reconhece textos iguais, não paráfrases ("Falta de respeito com o cliente" e "Falta de consideração com o cliente" continuam separados). O MMR atenua, mas não cria variedade que a base não tem.
- Em perguntas genéricas o conjunto recuperado inclui elogios (a palavra "problemas" atrai "nunca tive problemas"); no modo temas isso vira um tema à parte, que o modelo sinaliza, e o filtro de nota evita o efeito.
- A extração automática de filtros errou 2 de 28 vezes: aplicou `categoria: auto` em "Como funciona um motor de carro?" e na pergunta da Tesla, só pela palavra "carro". Em ambas a recusa se manteve.
- Os limiares do gate dependem de calibração empírica. Perguntas abstratas pontuam perto do limiar e casos sem evidência próximos dele dependem do LLM para recusar; o rótulo da recusa (`sem_evidencia` ou `fora_de_escopo`) pode variar entre execuções.
- Sem reescrita da query, perguntas com vocabulário distante do das reviews têm score mais baixo.
- O reranker é um modelo pequeno (mMARCO multilíngue); os proxies automáticos não medem se ele ajuda, só a leitura das evidências.
- A busca híbrida não ajudou neste conjunto de perguntas e está desligada por padrão.
- Quantificadores como "a maioria" valem apenas para a amostra recuperada, não para a base inteira.
- Apenas parte das reviews tem comentário em texto, e textos muito curtos são descartados na preparação, então elogios genéricos como "Muito bom" não aparecem nas respostas. 993 documentos não têm categoria.

## Reprodutibilidade

Esta seção leva do zero até o site e a avaliação rodando na sua máquina. O caminho foi **validado num clone limpo** (ambiente virtual novo, sem `.env`, sem `data/` e sem `chroma_db/`): instalação do `requirements.txt`, os 69 testes sem chave da OpenAI, download dos CSVs, EDA, construção do índice (com o cache de embeddings, para não gastar API) e abertura do app (renderização completa pelo `AppTest` do Streamlit, sem exceções). A geração dos embeddings pela API, as chamadas ao LLM e o uso do site no navegador foram executados antes no ambiente principal, não no clone.

### 1. Pré-requisitos

| Item | Detalhe |
|---|---|
| Python | 3.12 (testado com 3.12.10) |
| Sistema | macOS (testado). No Windows, ative o venv com `.venv\Scripts\activate` |
| Disco | ≈ 2,4 GB: ambiente virtual ≈ 1,3 GB (o `torch` pesa), `chroma_db/` ≈ 330 MB num índice recém-construído, `.cache/` ≈ 210 MB, `data/` ≈ 53 MB e o modelo do reranker ≈ 470 MB (cache do Hugging Face). Reconstruir o índice várias vezes na mesma pasta deixa o `chroma_db/` maior |
| Conta OpenAI | Chave com créditos e acesso a `text-embedding-3-small` e ao modelo de geração (`gpt-6-luna` por padrão) |
| Rede | Kaggle (dataset), OpenAI e, só na primeira vez que o reranking é usado, Hugging Face (modelo) |

### 2. Passo a passo

**Passo 1. Código e ambiente.**

```bash
git clone https://github.com/PedroVRLSaraiva/FIAP-tech-challenge-4.git && cd FIAP-tech-challenge-4
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Todas as versões estão fixadas. Na primeira instalação o `torch` torna o download demorado.

**Passo 2. Testes** (ainda sem dados nem chave; ver seção "Testes" abaixo).

```bash
pytest          # esperado: 69 passed
```

**Passo 3. Dataset.** Baixa 5 CSVs da Olist via `kagglehub` para `data/`:

```bash
python download_data.py
```

| Arquivo | Tamanho | Para quê |
|---|---|---|
| `olist_order_reviews_dataset.csv` | 14 MB | Base de conhecimento (as avaliações) |
| `olist_orders_dataset.csv` | 18 MB | Status e datas do pedido (período e atraso) |
| `olist_order_items_dataset.csv` | 15 MB | Itens do pedido (para a categoria) |
| `olist_products_dataset.csv` | 2,4 MB | Categoria de cada produto |
| `product_category_name_translation.csv` | 3 KB | Tradução das categorias para o inglês |

O `kagglehub` guarda os arquivos em `~/.cache/kagglehub`, então repetir o comando é rápido. Se ele pedir autenticação no Kaggle, siga a documentação do `kagglehub`.

**Passo 4. Chave e modelos.**

```bash
cp .env.example .env     # abra o .env e preencha OPENAI_API_KEY
```

| Variável | Obrigatória | Padrão | Uso |
|---|---|---|---|
| `OPENAI_API_KEY` | sim | | Embeddings e geração. Nunca versione o `.env` (já está no `.gitignore`) |
| `LLM_MODEL` | não | `gpt-6-luna` | Modelo de geração (e de extração de filtros) |
| `EMBEDDING_MODEL` | não | `text-embedding-3-small` | Modelo de embeddings. Se trocar, rode `build_index.py --reindex`: cada modelo tem o seu arquivo de cache |
| `RERANK_MODEL` | não | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Cross-encoder do reranking |

**Passo 5. Exploração da base (opcional).** Confere os números de preparação usados neste README:

```bash
python notebooks/eda.py
```

Saída esperada (início): `Linhas: 99224`, `review_id duplicados: 814`, `order_id com >1 review: 547` e a distribuição de notas (1: 11,5%, 2: 3,2%, 3: 8,2%, 4: 19,3%, 5: 57,8%). Imprime também o comprimento dos textos e 20 exemplos descartados por tamanho.

**Passo 6. Construir o índice.**

```bash
python build_index.py
```

O que acontece: limpa e filtra as avaliações, junta com os dados do pedido, gera os embeddings pela API em lotes de 256 (≈ 140 requisições; em torno de um milhão de tokens, poucos centavos de dólar), grava `data/processed/reviews_docs.parquet` e monta o `chroma_db/`.

Saída esperada (as linhas principais):

```
total                              99224
com_mensagem                       40950
descartadas_curtas                 4976
descartadas_duplicadas_pedido      6
descartadas_review_id_repetido     266
documentos_finais                  35702
com pedido: 35702 | com categoria: 34709 | com entrega registrada: 33939
Índice pronto: 35702 documentos em ...s
```

- A primeira execução depende da velocidade da API (alguns minutos). Com o cache de embeddings já presente, leva cerca de 1 minuto (62 s no clone limpo).
- O cache é salvo a cada 2.048 embeddings: se a execução cair, rodar de novo **retoma** de onde parou.
- É idempotente (`upsert`): rodar duas vezes não duplica. A contagem final é conferida contra o número de documentos.
- `--reindex` recria o índice reaproveitando o cache (sem custo de API); `--rebuild` recria tudo, inclusive os embeddings (gera custo de API).

**Passo 7. Conferir o índice.**

```bash
python - <<'EOF'
from src import vector_store
col = vector_store.get_collection()
print(col.count())                                   # esperado: 35702
print(col.get(limit=1, include=["metadatas"])["metadatas"][0])
EOF
```

A saída traz os metadados de um documento: `review_score`, `data`, `order_status`, `categoria`, `ano_mes`, `data_compra_ts`, `atraso_dias` e `entregue_atrasado`.

**Passo 8. Subir o site.**

```bash
streamlit run app.py        # http://localhost:8501  (outra porta: --server.port 8502)
```

Na primeira carga o app monta o índice BM25 (alguns segundos) e, na primeira busca com reranking ligado, baixa o modelo do cross-encoder. Três verificações rápidas:

1. Clique no exemplo "Quais problemas aparecem nos pedidos entregues com atraso?": a resposta traz a etiqueta `entregue com atraso` e as evidências mostram "entregue N dias após o prazo".
2. Pergunte "Quais os principais problemas relatados nas avaliações com nota baixa?": o recorte `nota ≤ 2` aparece como etiqueta.
3. Clique em "Qual a capital da França?": o site recusa e mostra avaliações "não conclusivas".

### 3. Reproduzir a avaliação e as comparações

| Comando | O que faz | Uso de API | Saída |
|---|---|---|---|
| `python -m src.calibrate` | Calibra `T_MAX` e `SCORE_MIN` com 15 perguntas dentro e 15 fora do escopo | Só embeddings (30 consultas) | Tabelas no terminal |
| `python -m eval.compare_retrieval` | Compara as técnicas de seleção em 21 perguntas, sem LLM de geração (alguns minutos; carrega o reranker) | Só embeddings (21 consultas) | `eval/retrieval_comparison.csv` e `.md`, `eval/themes_demo.json` |
| `python -m eval.run_eval` | Roda as 28 perguntas na configuração recomendada | LLM: até ≈ 24 gerações + até 9 extrações de filtro | `eval/results.csv` |
| `python -m eval.run_eval --preset baseline --ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 --out eval/results_baseline.csv` | Reproduz o "antes" (busca vetorial pura, 20 perguntas) | LLM: ≈ 16 requisições | `eval/results_baseline.csv` |
| `python -m eval.run_eval --preset temas --ids 1,3,4,21 --out eval/results_temas.csv` | Modo temas nas perguntas agregadas | LLM: 4 gerações | `eval/results_temas.csv` |

Atenção:

- **`run_eval` sobrescreve o arquivo de saída**, inclusive as colunas de avaliação manual (`precisao_at_k`, `fidelidade`, `acerto_recusa`, `observacoes`). Copie o `results.csv` antes de rodar de novo, ou use `--out`.
- O `gpt-6-luna` tinha, na conta usada no projeto, limite de **50 requisições por dia**. Confira o limite da sua conta. O código falha na hora, com mensagem clara, quando o limite diário ou os créditos acabam.
- Rode sempre da **raiz do projeto**, com `python -m` (por exemplo `python -m eval.run_eval`), para o pacote `src` ser encontrado.

### 4. O que muda de uma execução para outra

| Determinístico (deve bater com o README) | Pode variar |
|---|---|
| Limpeza, filtros, funil de dados e ids dos documentos | Texto das respostas do LLM (o `gpt-6-luna` não aceita `temperature`, então a redação varia) |
| Embeddings (ficam em cache; a OpenAI pode atualizar o modelo, mas o cache protege) | O rótulo da recusa (`sem_evidencia` ou `fora_de_escopo`) em casos limítrofes |
| Sementes: `SEED = 42` para `random`, `numpy` e KMeans | Latências |
| Métricas de recuperação do `compare_retrieval` (textos distintos, redundância, cobertura de temas, conformidade com o filtro): o mesmo índice e os mesmos vetores dão os mesmos números, com diferenças de arredondamento | A ordem de candidatos com score empatado, porque o HNSW do Chroma é uma busca aproximada |
| Extração de filtros de uma pergunta já vista (cache em `.cache/filters.json`) | A extração de filtros de uma pergunta nova (depende do LLM) |

Para comparar com o README, use primeiro as métricas que não dependem do LLM. Precisão@10 e fidelidade vêm de leitura manual (`eval/results.csv` traz a pré-avaliação e a observação de cada caso).

### 5. Testes

```bash
pytest                              # tudo: 69 testes, em poucos segundos
pytest tests/test_filters.py -q     # um arquivo
pytest -k mmr -q                    # por nome
```

Os testes **não usam chave da OpenAI, rede nem `data/` e `chroma_db/`**: o LLM, o Chroma e o reranker são substituídos por dublês (o Chroma em memória só aparece no teste que compara filtros). Rodam num clone limpo, sem `.env`. Os avisos de depreciação do `chromadb`/`pydantic` na saída são inofensivos.

| Arquivo | Testes | O que cobre |
|---|---|---|
| `tests/test_data_prep.py` | 5 | Limpeza (acentos e caixa preservados; espaços, quebras e URLs normalizados), corte de textos curtos, funil de filtros com contagens, `review_id` único |
| `tests/test_filters.py` | 25 | Filtros: `where` do Chroma (condição única, `$and`, booleano `False`), prioridade do filtro manual, descrição legível, validação do JSON do LLM (campo inválido descartado); **equivalência entre o `where` do Chroma e a máscara em pandas** em 7 casos; pista barata de filtro; extração com LLM falso (JSON, cache, erros) |
| `tests/test_selection.py` | 17 | Normalização e tokenização, BM25 com máscara, contagem de textos idênticos, agrupar idênticos, MMR, seleção (baseline = top-k, agrupar, rerank), fusão RRF, temas (KMeans), enriquecimento com pedidos (categoria por pedido, atraso) e metadados primitivos |
| `tests/test_answer_flow.py` | 22 | Validador de citações (índice fora do intervalo, intervalos `[1]–[3]`, citação após o ponto, linha de abertura), gate de evidência, tokens `SEM_EVIDENCIA`/`FORA_DE_ESCOPO`, seleção e contexto (contagem de idênticos, temas) e o fluxo completo com dublês: resposta com citações, gate que impede a chamada ao LLM, recusas, pergunta vazia, filtros manuais e automáticos, recorte vazio, MMR e agrupar |

### 6. Estrutura do repositório

```
.
├── README.md                      este documento
├── requirements.txt               dependências com versões fixas
├── .env.example                   modelo do .env (chave e modelos)
├── .gitignore                     ignora .env, .venv, data/, chroma_db/ e .cache/
├── app.py                         site Streamlit: busca, filtros, temas, evidências, CSV, painel de debug
├── download_data.py               baixa os 5 CSVs da Olist (kagglehub) para data/
├── build_index.py                 pipeline offline: CSVs → documentos → embeddings → ChromaDB
│
├── src/                           o pipeline RAG
│   ├── config.py                  todos os parâmetros: caminhos, modelos, limiares do gate, opções de seleção
│   ├── data_prep.py               limpeza, textos curtos, deduplicação, junção com pedidos, metadados
│   ├── embeddings.py              embeddings da OpenAI: lotes, retry, normalização, cache em disco
│   ├── vector_store.py            ChromaDB: coleção, upsert em lotes, consulta com filtro, busca por id
│   ├── query.py                   normalização da pergunta
│   ├── filters.py                 objeto Filters: where do Chroma, máscara pandas, descrição, validação
│   ├── filter_extraction.py       extração de filtros pelo LLM (JSON validado, pista barata, cache)
│   ├── retrieval.py               busca vetorial com pré-filtro, fusão híbrida (RRF) e seleção simples
│   ├── lexical.py                 índice BM25 e contagem de textos idênticos na base
│   ├── diversify.py               agrupar textos idênticos e MMR
│   ├── rerank.py                  cross-encoder (reranking)
│   ├── themes.py                  agrupamento por tema (KMeans) e palavras-chave
│   ├── selection.py               escolhe as evidências (agrupar, rerank, MMR) ou os temas
│   ├── options.py                 SearchOptions: liga e desliga cada técnica
│   ├── context.py                 monta o contexto numerado [n] enviado ao LLM
│   ├── prompts.py                 prompts do LLM (restritivo, anti-injeção, regras de temas)
│   ├── llm.py                     cliente do LLM com retry e mensagens de erro seguras
│   ├── guardrails.py              gate de evidência e leitura dos tokens de recusa
│   ├── answer.py                  orquestra o fluxo completo e valida as citações
│   └── calibrate.py               calibração dos limiares do gate
│
├── notebooks/
│   └── eda.py                     exploratória da base (números do funil de dados)
│
├── eval/                          avaliação
│   ├── perguntas.csv              28 perguntas (id, pergunta, tipo, esperado, filtro_esperado)
│   ├── run_eval.py                roda as perguntas no pipeline e grava o CSV de resultados
│   ├── compare_retrieval.py       compara as técnicas de seleção sem LLM de geração
│   ├── results.csv                "depois": configuração recomendada, 28 perguntas, com pré-avaliação
│   ├── results_baseline.csv       "antes": busca vetorial pura, 20 perguntas (versão anterior do código)
│   ├── results_temas.csv          modo temas em 4 perguntas agregadas
│   ├── retrieval_comparison.csv   métricas por configuração e pergunta
│   ├── retrieval_comparison.md    tabelas resumidas da comparação
│   └── themes_demo.json           temas, tamanhos e palavras-chave de 3 perguntas
│
├── tests/                         69 testes (ver seção Testes)
│
├── Documentação/
│   ├── Plano de Execução.md       plano por fases, com checklist e status
│   └── Comparativo antes e depois.md   resultados antes e depois da seleção de evidências
│
│   ── gerados na sua máquina, fora do git ──
├── .venv/                         ambiente virtual
├── data/                          CSVs da Olist e data/processed/reviews_docs.parquet (documentos + metadados)
├── chroma_db/                     índice vetorial persistente do ChromaDB
└── .cache/                        embeddings_<modelo>.npy + _ids.json (um arquivo por modelo) e filters.json
```

### 7. Problemas comuns

| Sintoma | Causa e solução |
|---|---|
| `Índice não encontrado. Rode python build_index.py` | O índice ainda não foi construído (Passo 6) |
| `OPENAI_API_KEY não configurada` | Faltou preencher o `.env` (Passo 4) |
| `A conta da OpenAI está sem créditos (insufficient_quota)` | Adicione créditos na OpenAI |
| `Limite diário de requisições do modelo atingido (RPD)` | Aguarde a janela liberar (a mensagem traz a sugestão da API) ou use um modelo/conta com limite maior. Cada pergunta usa no máximo 2 requisições |
| `Modelo '...' não encontrado para esta chave` | Ajuste `LLM_MODEL` no `.env` para um modelo que a sua chave acesse |
| Log `Parâmetro 'temperature' não suportado pelo modelo; removendo.` | Esperado com o `gpt-6-luna`: o cliente retira o parâmetro e segue |
| `ModuleNotFoundError: No module named 'src'` | Rode da raiz do projeto e com `python -m` (ex.: `python -m eval.run_eval`) |
| `CSVs de pedidos não encontrados: índice sem metadados de pedido` | Rode `python download_data.py` e depois `python build_index.py --reindex`. Sem isso o site funciona, mas sem filtros de categoria, período e entrega |
| Primeira busca lenta com reranking | O app baixa o modelo (~470 MB) uma vez. Sem rede, desligue "Reordenar com cross-encoder" na barra lateral |
| `capture() takes 1 positional argument` no terminal | Ruído da telemetria do Chroma, inofensivo |
| Porta 8501 ocupada | `streamlit run app.py --server.port 8502` |

### 8. Recomeçar do zero

```bash
rm -rf chroma_db data/processed     # índice e documentos processados (recriados por build_index.py)
rm -rf .cache                       # embeddings (regerar custa API) e cache de filtros
python build_index.py               # recria tudo; sem o cache, gera os embeddings de novo
```

Para refazer só o índice sem custo de API, mantenha `.cache/` e use `python build_index.py --reindex`.
