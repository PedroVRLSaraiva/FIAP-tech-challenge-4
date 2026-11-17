import pandas as pd

from src.data_prep import build_documents, clean_text, is_too_short


def test_clean_text_normalizes_whitespace_and_urls():
    out = clean_text("  Chegou   rápido\r\n\tveja http://x.com/abc ok  ")
    assert out == "Chegou rápido veja [url] ok"


def test_clean_text_keeps_accents_and_case():
    assert clean_text("Não Recebi O Produto") == "Não Recebi O Produto"


def test_is_too_short():
    assert is_too_short("ok")
    assert is_too_short("muito bom")  # 2 palavras
    assert not is_too_short("produto chegou bem rápido")


def _raw():
    return pd.DataFrame(
        {
            "review_id": ["a", "b", "c", "d", "e", "a2"],
            "order_id": ["o1", "o2", "o3", "o4", "o1", "o9"],
            "review_score": [1, 5, 3, 4, 1, 2],
            "review_comment_title": [None, "Ótimo", None, None, None, None],
            "review_comment_message": [
                "Não recebi o produto no prazo",
                "chegou antes do prazo",
                None,
                "ok",
                "Não recebi o produto no prazo",  # duplicata (mesmo pedido o1)
                "Não recebi o produto no prazo",  # mesmo texto, outro pedido: mantém
            ],
            "review_creation_date": ["2018-03-01 00:00:00"] * 6,
            "review_answer_timestamp": ["2018-03-02"] * 6,
        }
    )


def test_build_documents_filters_and_stats():
    docs, stats = build_documents(_raw())
    assert stats["total"] == 6
    assert stats["com_mensagem"] == 5
    assert stats["descartadas_curtas"] == 1
    assert stats["descartadas_duplicadas_pedido"] == 1
    assert set(docs["id"]) == {"a", "b", "a2"}
    assert docs["text"].notna().all() and (docs["text"].str.len() > 0).all()
    assert docs.loc[docs["id"] == "b", "text"].iloc[0].startswith("Ótimo.")
    assert docs.loc[docs["id"] == "a", "data"].iloc[0] == "2018-03-01"


def test_review_id_unique():
    raw = _raw()
    raw.loc[5, "review_id"] = "a"  # review_id repetido
    docs, stats = build_documents(raw)
    assert docs["id"].is_unique
