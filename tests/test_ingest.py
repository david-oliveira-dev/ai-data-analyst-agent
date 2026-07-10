"""Testes da ingestão de CSV."""
from __future__ import annotations

import pytest

from src.data.ingest import clean, load_csv


def test_load_from_bytes(sample_csv_bytes):
    df = load_csv(sample_csv_bytes)
    assert len(df) == 400
    assert {"idade", "renda", "score", "cidade", "churn"}.issubset(df.columns)


def test_infers_separator():
    csv = b"a;b;c\n1;2;3\n4;5;6"      # separador ';'
    df = load_csv(csv)
    assert list(df.columns) == ["a", "b", "c"]
    assert df.shape == (2, 3)


def test_parses_brazilian_decimal():
    csv = b"valor\n1.234,56\n2.000,00\n999,90"
    df = load_csv(csv)
    assert df["valor"].dtype.kind == "f"
    assert abs(df["valor"].iloc[0] - 1234.56) < 1e-6


def test_strips_column_names():
    df = clean(load_csv(b"  a , b \n1,2"))
    assert list(df.columns) == ["a", "b"]


def test_empty_csv_raises():
    with pytest.raises(ValueError):
        load_csv(b"")


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_csv("/caminho/inexistente/arquivo.csv")
