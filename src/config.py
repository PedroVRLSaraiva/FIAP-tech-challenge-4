"""Configuração central do projeto. Todos os parâmetros ajustáveis ficam aqui."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # silencia a telemetria do Chroma

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --- Caminhos ---------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_REVIEWS_CSV = DATA_DIR / "olist_order_reviews_dataset.csv"
ORDERS_CSV = DATA_DIR / "olist_orders_dataset.csv"
ORDER_ITEMS_CSV = DATA_DIR / "olist_order_items_dataset.csv"
PRODUCTS_CSV = DATA_DIR / "olist_products_dataset.csv"
CATEGORY_TRANSLATION_CSV = DATA_DIR / "product_category_name_translation.csv"
PROCESSED_PARQUET = DATA_DIR / "processed" / "reviews_docs.parquet"
CACHE_DIR = ROOT / ".cache"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "olist_reviews"

# --- Modelos ----------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # OpenAI, 1536 dim
EMBEDDING_BATCH_SIZE = 256  # textos por requisição à API de embeddings
EMBEDDING_MAX_CHARS = 8000  # truncamento de segurança (limite da API: 8191 tokens)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-6-luna")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 800
LLM_TIMEOUT_S = 60
LLM_MAX_RETRIES = 3

# --- Preparação da base -----------------------------------------------------
MIN_CHARS = 15          # textos menores que isso são descartados (ver EDA)
MIN_WORDS = 3
URL_TOKEN = "[url]"

# --- Recuperação ------------------------------------------------------------
TOP_N = 50              # candidatos buscados no Chroma
DEFAULT_TOP_K = 10      # reviews enviadas ao LLM
MAX_TOP_K = 50
MAX_QUESTION_CHARS = 500
MAX_REVIEW_CHARS_IN_CONTEXT = 600

# --- Seleção das evidências (diversidade e utilidade) ------------------------
# Cada técnica é opcional e pode ser ligada/desligada na interface (ver SearchOptions em src/answer.py).
RRF_K = 60              # Reciprocal Rank Fusion: 1 / (RRF_K + posição)
BM25_TOP_N = 50         # candidatos lexicais (BM25) fundidos com os vetoriais na busca híbrida
MMR_LAMBDA = 0.5        # 1.0 = só relevância; 0.0 = só diversidade (varredura em eval/retrieval_comparison.md)
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")  # multilíngue
RERANK_MAX_LENGTH = 256
RERANK_POOL = 60        # candidatos (após agrupar duplicatas) enviados ao cross-encoder
THEME_POOL = 600        # vizinhos recuperados para agrupar por tema (só entram os com score >= SCORE_MIN)
THEME_K_MAX = 6         # nº máximo de temas; k = clamp(avaliações // 40, 3, THEME_K_MAX)
THEME_KEYWORDS = 5      # palavras-chave por tema
# Opções padrão do site (ver eval/compare_retrieval.py e a seção "Seleção das evidências" do README).
RECOMMENDED_OPTIONS = dict(hybrid=False, dedup=True, mmr=True, rerank=True, themes=False, auto_filters=True)

# --- Guardrails (calibrar no notebook de calibração, ver notebooks/) --------
# Calibrados com `python -m src.calibrate` (15 perguntas dentro e 15 fora do escopo) e conferidos
# nas 20 perguntas de eval/perguntas.csv, sobre o índice real de 35.702 reviews e
# text-embedding-3-small, SEM reescrita de query:
#   dentro do escopo: max_score entre 0,500 e 0,843. As mais baixas são as abstratas, ex.:
#                     "padrões identificados em clientes insatisfeitos" (0,500 a 0,538).
#   fora do escopo:   max_score entre 0,256 e 0,486, com UMA exceção ("melhor time de futebol",
#                     0,617, por causa de reviews de camisas de time).
#   sem evidência:    0,483 a 0,525 (assunto de e-commerce, mas sem reviews sobre o tema).
# Os conjuntos se sobrepõem: perguntas abstratas pontuam no mesmo patamar dos casos sem evidência.
# Por isso o gate é uma primeira barreira grossa (descarta só o que é claramente alheio à base) e
# foi calibrado para NÃO recusar perguntas legítimas; o julgamento semântico fica com a segunda
# barreira (tokens SEM_EVIDENCIA / FORA_DE_ESCOPO do LLM). Recusar errado uma pergunta válida é
# pior do que gastar uma chamada de geração num caso limítrofe.
T_MAX = 0.48            # max_score abaixo disso => sem evidência
SCORE_MIN = 0.45        # score mínimo para uma review entrar no contexto
MIN_DOCS = 3            # mínimo de reviews acima de SCORE_MIN

NO_EVIDENCE_TOKEN = "SEM_EVIDENCIA"
MSG_SEM_EVIDENCIA = "Não encontrei avaliações suficientes para responder a essa pergunta."
MSG_FORA_DE_ESCOPO = "Essa pergunta está fora do escopo das avaliações de clientes."
MSG_SEM_RECORTE = "Não há avaliações nesse recorte. Afrouxe ou remova algum filtro e tente de novo."
