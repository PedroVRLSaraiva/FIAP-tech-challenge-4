# Plano de execução: Voice of Customer Intelligence com RAG

## Status do plano

Legenda: `- [x]` feito e verificado; `- [ ]` pendente. Quando um item ficou **parcial** ou diferente do previsto, a nota está na própria linha. Requisitos do site e fases 0 a 8: **110 de 120 itens concluídos**. Checklist final de entrega (seção 9): **6 de 8**. Seção 6 (segundo momento): **25 de 29 itens aplicados**.

| Fase | Situação | O que falta |
|---|---|---|
| 0. Setup | Concluída | |
| 1. Base | Concluída | |
| 2. Base vetorial | Concluída | |
| 3. Pipeline RAG | Concluída | |
| 4. Evidências e rastreabilidade | Concluída | |
| 5. Falta de evidência | Concluída com ressalvas | Gráfico da calibração; testar ponta a ponta as 2 perguntas de calibração que passam pelo gate (clima e futebol) |
| 6. Site | Implementado e testado no navegador (inclui filtros, temas e exportação CSV) | Clone limpo; as 5 perguntas do enunciado no navegador; teste com um colega |
| 7. Avaliação | Rodada antes e depois (28 perguntas), com pré-avaliação | A equipe conferir precisão@k, fidelidade e acerto da recusa; listar os ajustes feitos |
| 8. Entregáveis | Em andamento | Parágrafo de problema de negócio no README; vídeo |
| Seção 6 (opcional) | **Aplicada**, exceto 6.A reescrita da query e 6.F Databricks (a equipe decidiu não fazer) | Itens parciais: precisão@k manual das perguntas com filtro, sem filtro (6.B); `top_k` padrão (6.E) |

**Mudanças de escopo registradas:** reescrita da query pelo LLM retirada do obrigatório (seção 6.A); textos curtos descartados por trazerem poucos insights sobre a opinião real (Fase 1.4); gate de score calibrado para ser permissivo (Fase 5.3); seção 6 aplicada, exceto a reescrita da query e a trilha Databricks. Resultados antes e depois em `Documentação/Comparativo antes e depois.md`.

## 1. Entregável

Um **site simples de busca em linguagem natural** sobre as avaliações de clientes da Olist, para uso da equipe. O usuário digita uma pergunta e o site devolve:

1. uma resposta gerada pelo LLM, **restrita ao contexto recuperado** e com citações `[n]`;
2. a lista de **evidências** (reviews recuperadas, com nota, data e score de similaridade);
3. um aviso explícito quando **não há evidência suficiente** ou a pergunta está **fora de escopo**.

Além do site, o repositório público precisa de `README.md` e o vídeo de até 7 minutos exigidos pelo desafio.

**Escopo atual:** apenas os itens obrigatórios do desafio. Os itens opcionais (busca híbrida, MMR, reranking, metadados e filtros, exportação, clusterização) estão reunidos na seção 6 e ficam para um segundo momento, junto com as melhorias de qualidade da seleção das evidências (6.E).

### Requisitos funcionais do site

- [x] Campo de pergunta em linguagem natural e botão de busca.
- [x] Perguntas de exemplo clicáveis (as do enunciado).
- [x] Resposta com citações `[n]` que apontam para as evidências listadas abaixo dela.
- [x] Lista de evidências: texto, nota, data e score.
- [x] Estado visual distinto para "sem evidência suficiente" e "fora de escopo".
- [x] Painel lateral com o parâmetro `top_k`.
- [x] Painel "Como a resposta foi gerada" (modo debug): pergunta usada na busca, scores, gate, prompt enviado. Serve para mostrar o raciocínio técnico no vídeo.

## 2. Arquitetura

```
                 ┌──────────────────── OFFLINE (build_index.py) ────────────────────┐
 Olist reviews ► │ load ─► clean/filter ─► build docs ─► embed (batch) ─► ChromaDB   │
                 └──────────────────────────────────────────────────────────────────┘

                 ┌───────────────────────── ONLINE (app.py) ─────────────────────────┐
 Pergunta ─────► │ normalize ─► embed query ─► retrieve top-N ─► score gate            │
                 │   ─► build context [1..k] ─► LLM (prompt restritivo)               │
                 │   ─► citation validator ─► UI                                      │
                 └───────────────────────────────────────────────────────────────────┘
```

## 3. Decisões de stack

| Camada | Decisão | Alternativa / observação |
|---|---|---|
| Linguagem | Python 3.12 | |
| Dados | pandas + `olist_order_reviews_dataset.csv` | Os demais CSVs só entram no segundo momento (metadados) |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dim), via API | Mesma chave do LLM. Vetores normalizados e cacheados em disco, com um arquivo de cache por modelo |
| Banco vetorial | ChromaDB `PersistentClient`, espaço `cosine` | FAISS se o Chroma der problema |
| LLM | **OpenAI** (SDK `openai`, `chat.completions`), chave em `OPENAI_API_KEY`, modelo `gpt-6-luna` em `LLM_MODEL`, atrás de uma interface `LLMClient.generate(prompt) -> str` | A interface permite trocar o provedor sem mexer no pipeline |
| Orquestração | Python puro (módulos em `src/`) | Deixa cada etapa visível no vídeo |
| Interface | Streamlit | Sessão e cache nativos, sem front separado |
| Testes | pytest | |
| Config | `.env` + `src/config.py` | Nada de chave no código |

Todos os parâmetros ajustáveis (modelo, `top_k`, limiares, `MIN_DOCS`, tamanho mínimo de texto) ficam em `src/config.py`.

## 4. Estrutura do repositório

```
.
├── README.md
├── requirements.txt          # versões fixadas
├── .env.example              # OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL
├── .gitignore                # data/, .env, chroma_db/, cache de embeddings
├── app.py                    # site Streamlit
├── build_index.py            # pipeline offline: CSV → índice
├── src/
│   ├── config.py
│   ├── data_prep.py          # load, limpeza, documentos
│   ├── embeddings.py         # modelo, lote, cache em disco
│   ├── vector_store.py       # Chroma: upsert, query
│   ├── retrieval.py          # top-N, seleção, score gate
│   ├── query.py              # normalização da pergunta
│   ├── context.py            # montagem do contexto [1..k]
│   ├── prompts.py            # system prompt e template
│   ├── llm.py                # LLMClient (OpenAI)
│   ├── answer.py             # orquestra o fluxo e valida citações
│   └── guardrails.py         # limiares, MIN_DOCS, sentinela SEM_EVIDENCIA
├── notebooks/                # EDA e calibração de limiares
├── eval/                     # perguntas.csv e results.csv
├── tests/
└── data/                     # ignorado pelo git
```

## 5. Fases (escopo obrigatório)

### Fase 0. Setup do projeto

**Objetivo:** ambiente reprodutível e dados no disco.

- [x] Criar o repositório público no GitHub com a estrutura da seção 4. Publicado em https://github.com/PedroVRLSaraiva/FIAP-tech-challenge-4 (público, branch `main`, 46 arquivos).
- [x] Criar o venv e o `requirements.txt` com versões fixadas: `pandas`, `kagglehub`, `chromadb`, `streamlit`, `openai`, `python-dotenv`, `pytest`.
- [x] Preencher o `.gitignore` (`data/`, `.env`, `chroma_db/`, `.cache/`) e criar o `.env.example` com `OPENAI_API_KEY`, `LLM_MODEL` e `EMBEDDING_MODEL`.
- [x] Colocar a chave da OpenAI no `.env` local (nunca commitar).
- [x] Baixar o dataset do Kaggle (`olistbr/brazilian-ecommerce`) para `data/` e documentar o comando de download no README.
- [x] Gravar `LLM_MODEL=gpt-6-luna` no `.env`.
- [x] Validar que a chave e o modelo funcionam com uma chamada mínima ao `chat.completions`. Conferir também quais parâmetros o modelo aceita (`temperature`, `max_tokens` ou `max_completion_tokens`) e ajustar o `LLMClient`. (o `gpt-6-luna` recusa `temperature`; o `LLMClient` remove o parâmetro sozinho e usa `max_completion_tokens`)
- [x] Fixar seeds (`random`, `numpy`) em `config.py`.

**Saída:** `pip install -r requirements.txt` funciona do zero, os CSVs estão em `data/` e a chamada à OpenAI responde. **Parcial:** instalação, dados e chamada à OpenAI funcionam; falta validar num clone limpo.

### Fase 1. Exploração e preparação da base

**Objetivo:** transformar `olist_order_reviews_dataset.csv` em uma tabela de documentos limpos.

**1.1 Exploratória (notebook)**

- [x] Carregar `olist_order_reviews_dataset.csv` (colunas: `review_id`, `order_id`, `review_score`, `review_comment_title`, `review_comment_message`, `review_creation_date`, `review_answer_timestamp`).
- [x] Medir: total de linhas, `review_id` duplicados, `order_id` com mais de uma review, distribuição de `review_score`, % de nulos em título e mensagem.
- [x] Medir o comprimento do texto (caracteres e tokens): média, mediana, p95, máximo. (feito em caracteres e palavras; tokens não foram medidos, mas o p95 de 192 caracteres fica muito abaixo do limite de 8191 tokens)
- [x] Registrar os números para o README (cada descarte precisa de justificativa quantificada).

**1.2 Seleção e valores ausentes**

- [x] Manter somente linhas com `review_comment_message` não nulo e não vazio após `strip()`.
- [x] Decidir sobre `review_comment_title`: se existir, concatenar como `"{titulo}. {mensagem}"`. Registrar a decisão.
- [x] Descartar reviews sem texto, com justificativa: sem texto não há o que recuperar, e a nota isolada não carrega o motivo.

**1.3 Limpeza do texto**

- [x] Normalizar `\r\n`, tabs e espaços múltiplos.
- [x] Substituir URLs por um token (`[url]`) ou remover.
- [x] Remover caracteres de controle e lixo de encoding.
- [x] **Não** remover acentos, stopwords nem fazer stemming (embeddings de transformers dependem do texto natural).
- [x] Não alterar caixa. O modelo é multilíngue e case-aware.

**1.4 Deduplicação e textos curtos**

- [x] Remover duplicatas exatas por `(order_id, texto_normalizado)`.
- [x] Avaliar duplicatas do mesmo texto em pedidos diferentes. Manter se forem pedidos distintos, e registrar a decisão. (decisão: manter; só a duplicata do mesmo pedido é removida. Registrada no README)
- [x] Definir `MIN_CHARS` (ex.: 15 caracteres ou 3 palavras) a partir da distribuição da EDA. Listar 20 exemplos descartados para validar o corte. (`notebooks/eda.py` imprime 20 exemplos descartados)
- [x] **Justificativa do corte (decisão tomada):** textos muito curtos (ex.: "ok", "Muito bom") trazem poucos insights sobre a real opinião do cliente: dizem que gostou ou não, mas não dizem o motivo. Como o objetivo é entender o porquê, esses textos só adicionariam ruído ao índice e ao top-k. O corte é mantido e registrado no README, junto com o percentual descartado.

**1.5 Montagem dos documentos**

- [x] Um documento por review (reviews são curtas, então **sem chunking**). Justificar com o p95 de tamanho medido na EDA, que deve ficar abaixo do limite de tokens do modelo (8191 tokens no `text-embedding-3-small`).
- [x] Schema do documento: `id` (= `review_id`), `text` e `metadata` com `review_score` e `data` (derivada de `review_creation_date`, como string `YYYY-MM-DD`). Esses campos alimentam os cartões de evidência.
- [x] Salvar em `data/processed/reviews_docs.parquet`.

**Critério de aceite:** parquet gerado, sem nulos em `text`, com contagem antes e depois de cada filtro registrada em log. **Atendido** (0 nulos em `text`; funil impresso por `notebooks/eda.py` e registrado no README).

### Fase 2. Base vetorial

**Objetivo:** índice persistente, gerado por um único comando.

- [x] Configurar o cliente da OpenAI para embeddings (`EMBEDDING_MODEL=text-embedding-3-small`), com retry e falha imediata quando a conta está sem créditos.
- [x] Gerar embeddings dos documentos em lotes (256 textos por requisição) e normalizar os vetores.
- [x] Cachear os embeddings em disco (`.cache/embeddings.npy` + ids), com retomada se a execução cair.
- [x] Criar a collection do Chroma com `metadata={"hnsw:space": "cosine"}`.
- [x] Fazer `upsert` em lotes (≤ 5000), com `ids`, `documents`, `embeddings` e `metadatas`. Os metadados só aceitam tipos primitivos.
- [x] Salvar em `chroma_db/` e garantir idempotência (rodar duas vezes não duplica).
- [x] Expor o comando `python build_index.py [--rebuild]`.
- [x] Teste de sanidade: `collection.count()` igual ao número de documentos, e uma query de teste devolve reviews coerentes.

**Critério de aceite:** `python build_index.py` constrói o índice do zero e imprime contagem e tempo total. **Atendido** (35.702 documentos, contagem conferida pelo `build_index.py`).

### Fase 3. Pipeline RAG

**Objetivo:** função `answer(question, params) -> AnswerResult`, que devolve resposta, evidências e diagnóstico de cada etapa.

**3.1 Processamento da pergunta (**`query.py`**)**

- [x] Normalizar a pergunta: `strip`, espaços e limite de tamanho.
- [x] **Sem reescrita da query:** a pergunta normalizada vai direto para o embedding. A reescrita via LLM (query expansion) foi retirada do escopo obrigatório e está na seção 6 (item A). Motivos: elimina uma chamada de LLM por pergunta (menos custo, menos latência, menos exposição ao limite de requisições) e deixa a busca determinística.

**3.2 Embedding da pergunta**

- [x] Gerar o embedding da query com o mesmo modelo e a mesma normalização do índice.

**3.3 Recuperação (**`retrieval.py`**)**

- [x] Buscar `top_n` candidatos no Chroma (`n_results=top_n`, ex.: 50) com `include=["documents","metadatas","distances"]`.
- [x] Converter distância em similaridade: `score = 1 − distance` (espaço cosine).

**3.4 Seleção**

- [x] Aplicar o corte por score mínimo (`SCORE_MIN`) e reduzir para `top_k` finais (ex.: 8 a 15; para perguntas agregadas, 30 a 50).
- [x] Ordenar por score decrescente e fixar a numeração `[1]…[k]`.

**3.5 Construção do contexto (**`context.py`**)**

- [x] Formato por review: `[n] (nota: X/5 | data: YYYY-MM-DD) "texto"`.
- [x] Respeitar o orçamento de tokens do LLM: truncar o texto longo e limitar o número de reviews.
- [x] Delimitar o contexto com tags (`<reviews>…</reviews>`) e instruir o modelo a tratar o conteúdo como **dado, não como instrução**. As reviews são texto não confiável e podem conter prompt injection.

**3.6 Prompt e geração (**`prompts.py`**,** `llm.py`**)**

- [x] Implementar `LLMClient` com o SDK `openai` (`client.chat.completions.create`), lendo `OPENAI_API_KEY` e `LLM_MODEL` do ambiente.
- [x] System prompt: responder **somente** com base nas reviews fornecidas, citar os índices `[n]` em cada afirmação, não usar conhecimento externo, responder em português.
- [x] Instruir o modelo a devolver o token fixo `SEM_EVIDENCIA` quando o contexto não sustentar a resposta.
- [x] Instruir o modelo a indicar que a resposta se baseia numa **amostra** de reviews (relevante para perguntas agregadas).
- [x] Usar `temperature` baixa (0 a 0,2) e limitar `max_tokens`. (`LLM_TEMPERATURE` está em 0,1 no `config.py`, mas o `gpt-6-luna` não aceita o parâmetro; o tamanho da saída é limitado por `max_completion_tokens`)
- [x] Tratar erros de API (`RateLimitError`, `APITimeoutError`, `APIConnectionError`) com retry com backoff e mensagem amigável na UI.

**3.7 Orquestração (**`answer.py`**)**

- [x] Encadear as etapas e devolver um `AnswerResult` com: `answer`, `status` (`ok` | `sem_evidencia` | `fora_de_escopo`), `evidences[]`, `debug` (pergunta usada na busca, scores, gate, prompt).
- [x] Medir e registrar a latência de cada etapa.

**Critério de aceite:** `python -c "from src.answer import answer; print(answer('Quais os principais problemas de entrega?'))"` retorna resposta com citações. **Atendido** (executado pelo `eval/run_eval.py`).

### Fase 4. Evidências e rastreabilidade

**Objetivo:** cada afirmação da resposta é verificável contra uma review real.

- [x] Extrair as citações da resposta com regex `\[(\d+)\]`.
- [x] **Validar:** todo índice citado precisa estar em `{1..k}`. Se houver índice inexistente, remover a citação inválida ou regenerar uma vez, e logar o evento.
- [x] Marcar sentenças afirmativas sem nenhuma citação (alerta na UI ou no log).
- [x] No `AnswerResult`, mapear `[n]` → `review_id`, texto, nota, data e score.
- [x] Na UI, tornar cada `[n]` da resposta um link âncora para o cartão da evidência correspondente.
- [x] Mostrar também as evidências recuperadas que **não** foram citadas (diferencia o recuperado do usado).
- [x] Calcular a taxa de citações válidas por pergunta (alimenta a Fase 7).

**Critério de aceite:** nenhuma resposta aprovada no conjunto de avaliação cita um índice fora do contexto. **Atendido** (0 citações inválidas nas 13 respostas da avaliação).

### Fase 5. Tratamento de falta de evidência

**Objetivo:** o sistema recusa quando não há suporte, em vez de inventar. É critério explícito de avaliação.

**5.1 Regras determinísticas (**`guardrails.py`**), antes de chamar o LLM**

- [x] **Nenhum resultado relevante:** `max_score < T_MAX` → status `sem_evidencia`, sem chamar o LLM.
- [x] **Evidência insuficiente:** menos de `MIN_DOCS` (ex.: 3) reviews com `score ≥ SCORE_MIN` → status `sem_evidencia`.

**5.2 Regras no LLM (segunda barreira)**

- [x] O prompt manda retornar `SEM_EVIDENCIA` quando o contexto não responde à pergunta.
- [x] Detectar o token na saída e converter para status `sem_evidencia` ou `fora_de_escopo`.

**5.3 Calibração dos limiares (notebook)**

- [x] Montar dois conjuntos: ~15 perguntas **dentro** do escopo e ~15 **fora** (ex.: "Qual a capital da França?", "Como funciona um motor de carro?", "Quem ganhou a Copa de 2018?").
- [x] Para cada pergunta, registrar `max_score` e a quantidade de docs acima de cada limiar candidato.
- [ ] Plotar as duas distribuições e escolher `T_MAX` e `SCORE_MIN`. O cosseno varia de modelo para modelo, então **calibrar empiricamente e não assumir um valor padrão**. Sem reescrita, perguntas abstratas ("padrões em clientes insatisfeitos", score máximo ≈ 0,50) pontuam no mesmo patamar de casos sem evidência (0,48 a 0,52): os conjuntos se sobrepõem. Por isso o gate é calibrado para **não recusar perguntas legítimas** e o julgamento semântico fica com o token `SEM_EVIDENCIA` / `FORA_DE_ESCOPO` do LLM. **Parcial:** a calibração saiu em tabelas (`python -m src.calibrate`), sem gráfico das distribuições.
- [x] Conferir o gate nas perguntas reais de `eval/perguntas.csv` (as 13 que devem ser respondidas precisam passar).
- [x] Gravar os valores em `config.py` com um comentário apontando o notebook.

**5.4 Mensagens**

- [x] Texto padrão para sem evidência: "Não encontrei avaliações suficientes para responder a essa pergunta."
- [x] Texto padrão para fora de escopo: "Essa pergunta está fora do escopo das avaliações de clientes."
- [x] Na UI, ainda mostrar as reviews de maior score (quando houver) como "mais próximas encontradas", rotuladas como **não conclusivas**.

**Critério de aceite:** 100% das perguntas fora de escopo do conjunto de calibração são recusadas, e as perguntas legítimas do enunciado não são recusadas. **Parcial:** as 5 do enunciado não são recusadas e as 7 recusas da avaliação funcionaram. Das 15 fora de escopo da calibração, 13 são barradas pelo gate; clima (0,486) e futebol (0,617) passam e dependem do LLM, ainda não testados ponta a ponta.

### Fase 6. Site (Streamlit)

**Objetivo:** interface de busca utilizável pela equipe, sobre o pipeline das fases 3 a 5.

**6.1 Infraestrutura**

- [x] `app.py` com `st.set_page_config(layout="wide")`.
- [x] Carregar modelo de embeddings e collection do Chroma com `@st.cache_resource`, para não recarregar a cada interação.
- [x] Cachear respostas repetidas com `@st.cache_data` (chave: pergunta + `top_k`). (feito com cache em `st.session_state`, chave pergunta + `top_k`, sem cachear erros; não usa `@st.cache_data`)
- [x] Guardar o histórico de buscas em `st.session_state`.

**6.2 Layout**

- [x] **Sidebar:** slider de `top_k`.
- [x] **Área principal:** `st.text_input` ou `st.chat_input` para a pergunta, botões de exemplo, bloco de resposta, lista de evidências.
- [x] **Evidência:** cartão com texto, nota (estrelas), data e score, âncora `[n]`.
- [x] **Estados:** `st.spinner` durante a geração, `st.warning` para sem evidência, `st.info` para fora de escopo, `st.error` para falha de API.
- [x] **Debug:** `st.expander("Como a resposta foi gerada")` com a pergunta usada na busca, scores, gate, latência por etapa e prompt completo.

**6.3 Robustez**

- [x] Validar entrada vazia ou muito longa.
- [x] Tratar falha do LLM sem derrubar a página.
- [x] Mensagem clara se `chroma_db/` não existir ("rode `python build_index.py`").
- [x] Nenhuma chave de API exposta na UI ou nos logs.

**6.4 Verificação manual**

- [ ] Rodar `streamlit run app.py` do zero, a partir de um clone limpo. **Parcial:** num clone limpo (venv novo, sem `.env`, `data/` nem `chroma_db/`) foram validados: instalação do `requirements.txt`, 69 testes sem chave, download dos CSVs, EDA, `build_index.py` (com o cache de embeddings copiado, sem API) e a renderização do app pelo `AppTest`, sem exceções. Falta repetir com a geração real de embeddings e usar o site no navegador.
- [ ] Testar as 5 perguntas do enunciado, uma pergunta sem evidência e uma fora de escopo. **Parcial:** no navegador foram testadas 2 das 5 do enunciado, uma recusa pelo gate (França) e uma recusa pelo LLM (streaming). As 5 do enunciado passaram pelo pipeline em lote (Fase 7).
- [x] Conferir que os links `[n]` levam à evidência certa. (verificado no navegador: 16 links `[n]`, todos com cartão de destino)

**Critério de aceite:** um colega consegue fazer uma busca e ler a resposta com as evidências sem instrução adicional. **Pendente** (falta o teste com um colega).

### Fase 7. Avaliação

**Objetivo:** evidência quantitativa e qualitativa de que o sistema funciona, para o README e o vídeo.

- [x] Montar `eval/perguntas.csv` com ~15 a 20 perguntas e colunas `id, pergunta, tipo, esperado`:
  - as 5 do enunciado (reclamações gerais, entrega, elogios, insatisfação, evidência de prazo);
  - perguntas de tema específico (qualidade do produto, atendimento, embalagem, produto errado);
  - perguntas **sem evidência** e **fora de escopo**.
- [x] Rodar o pipeline em lote e gravar em `eval/results.csv`: resposta, status, evidências, scores, latência.
- [x] Avaliar manualmente cada pergunta: **Parcial:** há uma pré-avaliação preenchida em `eval/results.csv` (feita por leitura das respostas); falta a **conferência da equipe**.
  - **Precisão@k da recuperação:** reviews relevantes ÷ k;
  - **Fidelidade:** a resposta está sustentada pelas evidências? (0/1);
  - **Citações válidas:** % dos `[n]` que apontam para review recuperada;
  - **Acerto da recusa:** recusou quando devia e respondeu quando devia.
- [ ] Listar as falhas encontradas e o que foi ajustado. **Parcial:** os achados estão no README (seção Avaliação). Falta listar também o que foi ajustado (validador de citações, limiares do gate).
- [x] Testes automatizados (`pytest`):
  - limpeza de texto e deduplicação;
  - validador de citações;
  - gate de score (`max_score`, `MIN_DOCS`);
  - detecção do token `SEM_EVIDENCIA`.

**Critério de aceite:** tabela de resultados pronta e `pytest` passando. **Atendido** (tabela no README e 22 testes passando; as notas manuais ainda são pré-avaliação).

### Fase 8. Entregáveis

**8.1 README.md**

- [ ] Visão geral e problema de negócio. **Parcial:** o README descreve o produto, mas falta um parágrafo curto sobre o problema de negócio.
- [x] Diagrama de arquitetura (a figura da seção 2 serve de base).
- [x] Tecnologias e bibliotecas, com versões.
- [x] Preparação da base: cada decisão da Fase 1 com números (volume, % sem texto, duplicatas, corte de tamanho).
- [x] Estratégia do pipeline: processamento da pergunta, recuperação, seleção, contexto, prompt e geração.
- [x] Tratamento de falta de evidência: regras, limiares e como foram calibrados.
- [x] Decisões técnicas e justificativas (modelo de embeddings, Chroma, sem chunking, Python puro, limiares, OpenAI como LLM).
- [x] Resultados da avaliação.
- [x] Instalação e execução, passo a passo: `git clone`, venv, `pip install`, download do dataset, `.env` com `OPENAI_API_KEY`, `python build_index.py`, `streamlit run app.py`.
- [x] 4 a 6 exemplos de pergunta e resposta com evidências, incluindo uma recusa.
- [x] Limitações conhecidas (perguntas agregadas baseadas em amostra, viés das reviews com comentário).
- [x] Informações de reprodutibilidade: versões fixadas, seeds, origem do dataset.

**8.2 Repositório**

- [x] Repositório público, com estrutura organizada conforme a seção 4. https://github.com/PedroVRLSaraiva/FIAP-tech-challenge-4
- [x] Nenhum segredo commitado (verificar o histórico). Verificado no histórico local (1 commit): 0 ocorrências da chave e do padrão `sk-`; `.env` fica fora do git. Repetir a varredura se houver novos commits antes de publicar.
- [x] `data/` e `chroma_db/` ignorados, com instruções para regenerar.
- [ ] Clone limpo → seguir o README → site funcionando. **Parcial:** ver Fase 6.4; o passo a passo detalhado está na seção Reprodutibilidade do README.

**8.3 Vídeo (até 7 minutos)**

| Bloco | Conteúdo | Orçamento |
|---|---|---|
| 1 | Problema de negócio e o que o site entrega | ~1 min |
| 2 | Arquitetura e preparação da base (números da EDA, por que sem chunking) | ~1,5 min |
| 3 | Estratégia de recuperação: embeddings, busca, score gate | ~1 min |
| 4 | Demo no site: duas perguntas com evidências, uma sem resposta, e o painel de debug | ~2,5 min |
| 5 | Desafios, decisões e limitações | ~1 min |

- [ ] Roteiro escrito antes de gravar. **Pendente.**
- [ ] Mostrar o painel "Como a resposta foi gerada" na demo. **Pendente.**
- [ ] Mostrar explicitamente a recusa de uma pergunta sem evidência. **Pendente.**
- [ ] Conferir que a duração total ficou em até 7 minutos. **Pendente.**

## 6. Segundo momento (opcional, fora do escopo atual)

Só começar depois que a seção 5 estiver concluída e o checklist final (seção 9) estiver marcado.

**Aplicado:** todos os itens, exceto a reescrita da query pelo LLM (6.A) e a trilha Databricks (6.F). Resultados em `Documentação/Comparativo antes e depois.md` e no README.

### A. Recuperação avançada

- [ ] **Reescrita da query via LLM (query expansion):** acrescentar sinônimos e termos de domínio (ex.: "atraso" → "demora, não chegou, prazo"). Retirada do escopo obrigatório para reduzir custo, latência e dependência do limite de requisições do modelo. Se voltar: usar um modelo mais barato só nesta etapa, cachear a reescrita em disco, **recalibrar** `T_MAX` **/** `SCORE_MIN` (a reescrita eleva o score das perguntas abstratas) e expor a query reescrita no modo debug. **Não será feito (decisão da equipe).**
- [x] **Busca híbrida:** `rank_bm25` sobre os mesmos documentos, com fusão por **Reciprocal Rank Fusion** (`1/(k+rank)`, k=60). (`src/lexical.py` + `retrieval.hybrid_fuse`. Medida: **reduz** cosseno e cobertura de temas neste conjunto, por isso fica desligada por padrão.)
- [x] **MMR** (λ ≈ 0,5 a 0,7) para reduzir redundância entre as reviews selecionadas. (`src/diversify.py`, λ = 0,5 por varredura em `eval/retrieval_comparison.md`. Ligado por padrão.)
- [x] **Reranking** com cross-encoder. (`src/rerank.py`, cross-encoder `mmarco-mMiniLMv2-L12-H384-v1`. Ligado por padrão.)
- [x] Toggles na sidebar para cada técnica. (agrupar, MMR com λ, reranking, híbrida, temas e filtros automáticos.)
- [x] Na avaliação, comparar: vetorial puro × híbrido, com e sem MMR. (`src/diversify.py`, λ = 0,5 por varredura em `eval/retrieval_comparison.md`. Ligado por padrão.)

### B. Metadados e filtros (desafio adicional)

- [x] Join `reviews ⋈ orders` por `order_id`: `order_status`, `order_purchase_timestamp`, `order_delivered_customer_date`, `order_estimated_delivery_date`. (`data_prep.enrich_with_orders`; todos os 35.702 documentos casam com um pedido.)
- [x] Calcular `atraso_dias = delivered_customer_date − estimated_delivery_date` e `entregue_atrasado` (bool). Pedidos não entregues ficam com `NaN`. (11,7% dos pedidos entregues chegaram atrasados.)
- [x] `⋈ order_items ⋈ products ⋈ category_translation`, com **uma categoria por pedido** (primeiro item ou mais frequente), regra documentada. (73 categorias; 2 sem tradução mantêm o nome em português; 993 documentos sem categoria.)
- [x] Derivar `ano_mes` (int `YYYYMM`) e `data_compra_ts` (int epoch), já que o Chroma só filtra numericamente com números.
- [x] Reconstruir o índice com `python build_index.py --rebuild`. (`python build_index.py --reindex`, que reaproveita o cache e não gera custo de embeddings.)
- [x] Construtor de filtros para o `where` do Chroma (`$and`, `$gte`/`$lte`, `$in`): nota, categoria, período e entrega atrasada. (`src/filters.py`; testes verificam que o `where` do Chroma e a máscara em pandas dão o mesmo resultado.)
- [x] Aplicar o filtro **na consulta ao vetor** (pre-filter), e não depois do top-k.
- [x] Extração automática de filtros a partir da pergunta (LLM com saída JSON estruturada, validada contra schema). Exemplo: "nota baixa" → `{"review_score": {"$lte": 2}}`. (`src/filter_extraction.py`; só consulta o LLM quando a pergunta tem pista de filtro; resultado em cache. Acertou 26 de 28.)
- [x] Filtros na sidebar (nota, categoria, período, entregue com atraso), mesclados com os automáticos (os manuais têm prioridade), exibidos na UI.
- [x] Regra de guardrail: se os filtros zerarem os candidatos, informar que não há reviews naquele recorte. (status `sem_evidencia` com a mensagem `MSG_SEM_RECORTE`, sem chamar o LLM.)
- [ ] Comparação antes e depois: 5 a 8 perguntas com e sem filtro, com precisão@k manual e justificativa de como cada metadado melhora a recuperação. **Parcial:** a conformidade com o filtro foi medida com e sem filtro nas 8 perguntas (0,51 → 1,00) e há pré-avaliação de precisão@10 com filtro (0,91), mas **não** a precisão@10 manual das mesmas perguntas sem filtro.
- [x] Adicionar perguntas com filtro ao `eval/perguntas.csv` e testes do construtor de filtros. (perguntas 21 a 28, coluna `filtro_esperado`; testes em `tests/test_filters.py`.)
- [x] Critério de aceite: "principais problemas em avaliações com nota baixa" devolve só reviews com nota ≤ 2. (pergunta 21: as 10 evidências têm nota 1 ou 2.)

### C. Interface

- [x] Exportar as evidências em CSV (`st.download_button`).
- [x] Mostrar categoria e atraso na entrega nos cartões de evidência (depende do item B).

### D. Análise por temas

- [x] Clusterização dos embeddings (KMeans) para sumarizar perguntas agregadas por tema. (`src/themes.py`; conjunto de até 600 vizinhos, `k = avaliações // 40`.)

### E. Qualidade da seleção das evidências (diversidade e utilidade)

**Por que existe:** hoje o top-k são as *k reviews mais parecidas com a pergunta*. `select` em `src/retrieval.py` descarta quem está abaixo de `SCORE_MIN` e pega as `top_k` primeiras por score. Nada verifica se as reviews trazem pontos de vista diferentes, se são cópias do mesmo texto ou se representam o conjunto. "Mais parecida" não é o mesmo que "mais útil".

**Medição atual** (top-10 e top-50 recuperados, sem chamar o LLM de geração):

| Pergunta | Textos distintos no top-10 | Textos distintos no top-50 | Texto mais repetido |
|---|---|---|---|
| Reclamações sobre prazo de entrega | 5 de 10 | 34 de 50 | "Muita demora na entrega" ×9 |
| Elogios à rapidez da entrega | 9 de 10 | 42 de 50 | "Entrega muito rápida" ×7 |
| Qualidade dos produtos | 8 de 10 | 22 de 50 | "Produto de qualidade" ×23 |

A precisão@10 da avaliação (0,95) mede relevância, não diversidade nem representatividade. Possibilidades, da mais barata à mais cara:

- [x] **Agrupar textos idênticos e informar a contagem:** o contexto passa a dizer `[1] "Muita demora na entrega" (9 avaliações iguais)`. Os slots cobrem mais variedade e a resposta ganha um sinal real de frequência, em vez de repetição apresentada como evidências independentes. Determinístico, sem chamada extra ao LLM. Exige novo teste e nova rodada de avaliação. (`diversify.group_duplicates`; textos distintos 9,08 → 10,00.)
- [ ] **Aumentar o** `top_k` **padrão** (ou sugerir valores maiores em perguntas agregadas): a faixa já vai até 50 no slider; avaliar o custo em tokens e a qualidade da resposta com 20 a 30 reviews.
- [x] **MMR** (λ ≈ 0,5 a 0,7): escolhe cada nova review equilibrando relevância e diferença em relação às já escolhidas. Já listado em 6.A; aqui entra como solução do problema de diversidade. (`src/diversify.py`, λ = 0,5 por varredura em `eval/retrieval_comparison.md`. Ligado por padrão.)
- [x] **Reranking com cross-encoder:** reordena os candidatos olhando pergunta e review juntas, o que aproxima de "útil" mais do que de "parecido". Já listado em 6.A. (`src/rerank.py`, cross-encoder `mmarco-mMiniLMv2-L12-H384-v1`. Ligado por padrão.)
- [x] **Clusterização (KMeans):** para perguntas agregadas, agrupa as reviews por tema e informa o tamanho de cada grupo, o que permite responder sobre a base inteira em vez de uma amostra. Já listado em 6.D.
- [x] **Medir o efeito:** acrescentar à avaliação a métrica "textos distintos no top-k" e a cobertura de temas, comparando antes e depois de cada técnica. (`eval/compare_retrieval.py` e a coluna `n_textos_distintos` do `run_eval`.)

### F. Fora do plano

- [ ] Trilha Databricks (+10 pontos), se alguém da equipe a conduzir em paralelo. **Não será feito (decisão da equipe):** depende de ambiente Databricks e estava marcado como fora do plano.

## 7. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Alucinação | Prompt restritivo, score gate antes do LLM, token `SEM_EVIDENCIA`, validador de citações |
| Reviews sem texto (maioria da base) | Documentar o percentual descartado e a justificativa |
| Gate com scores sobrepostos (pergunta abstrata × sem evidência) | Gate permissivo (`T_MAX` 0,48), segunda barreira no LLM e verificação nas perguntas de `eval/perguntas.csv` |
| Top-k redundante: reviews quase idênticas ocupam os slots (ver 6.E) | **Mitigado:** agrupar textos idênticos com contagem, MMR e reranking (textos distintos 9,08 → 10,00). Paráfrases continuam separadas (limitação registrada) |
| O enunciado pede demonstrar "a transformação da pergunta" e, sem reescrita, ela é normalização + embedding | Explicar essa etapa no README e no vídeo como normalização e vetorização; a query expansion está disponível como opcional em 6.A |
| Perguntas agregadas ("principais problemas"): o top-k é só uma amostra | `top_k` maior e aviso explícito de que a resposta se baseia numa amostra |
| Faixa de scores do modelo de embeddings difícil de prever, com limiar difícil de escolher | Calibração empírica com conjuntos dentro e fora de escopo (Fase 5.3) |
| Prompt injection via texto de review | Contexto delimitado e tratado como dado no prompt |
| Custo e tempo de embeddings | Cache em disco por modelo de embedding, lotes de 256, indexação idempotente |
| Falha, rate limit ou custo da API da OpenAI | Retry com backoff em limites por minuto; falha imediata e mensagem clara em créditos esgotados ou limite diário (RPD); cache de respostas na sessão; `max_tokens` limitado. O pipeline gasta no máximo 1 requisição de LLM por pergunta (o gate recusa antes da geração) |
| Vazamento da chave da OpenAI | Chave só no `.env` ignorado pelo git, `.env.example` sem valor, verificação do histórico antes de publicar |
| Reprodutibilidade | Versões fixadas, seeds, `build_index.py` idempotente, README testado em clone limpo |

## 8. Decisões já tomadas

| Pergunta | Decisão |
|---|---|
| Qual LLM? | OpenAI, modelo `gpt-6-luna`, com chave em `OPENAI_API_KEY` e a interface `LLMClient` isolando o provedor |
| Reescrita da query pelo LLM? | **Retirada** do escopo obrigatório; item opcional do segundo momento (seção 6, A) |
| Textos curtos ("ok", "Muito bom")? | Descartados (< 15 caracteres ou < 3 palavras): poucos insights sobre a real opinião do cliente |
| Embeddings? | OpenAI `text-embedding-3-small`, mesma chave do LLM |
| Interface? | Site Streamlit |
| Itens opcionais entram agora? | **Sim, todos da seção 6, exceto a reescrita da query pelo LLM e a trilha Databricks** (decisão posterior à primeira entrega) |
| Seleção das evidências | Padrão: agrupar idênticos + MMR (λ = 0,5) + reranking + filtros automáticos. Híbrida e temas disponíveis, desligados (híbrida piorou as métricas; temas é opt-in para perguntas agregadas) |
| Trilha Databricks (+10 pontos)? | Fora do escopo atual |

## 9. Checklist final de entrega

- [x] Repositório público no GitHub com código, README e estrutura organizada. https://github.com/PedroVRLSaraiva/FIAP-tech-challenge-4
- [ ] `python build_index.py` e `streamlit run app.py` funcionam a partir de um clone limpo. **Parcial:** validado com o cache de embeddings e o `AppTest` (ver Fase 6.4); falta a geração real de embeddings e o uso no navegador.
- [x] Site responde às 5 perguntas do enunciado com citações e evidências.
- [x] Perguntas sem evidência e fora de escopo são recusadas.
- [x] `eval/results.csv` e a tabela de avaliação incluídos no README.
- [x] `pytest` passando.
- [x] Nenhum segredo no repositório. Verificado (ver Fase 8.2).
- [ ] Vídeo de até 7 minutos gravado e o link anexado à entrega. **Pendente.**
