"""Smoke tests de la demo Streamlit (`app.py`) usando `AppTest`.

Ejecutan el script real de la app (no solo verifican sintaxis) y
confirman que las tres páginas, y las interacciones básicas del usuario
(filtro de categoría, slider de K), no lanzan excepciones. Se apoya en
los mismos artefactos entrenados que usa la demo en producción
(`models/*.joblib`, `data/processed/products_master.csv`).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")

pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "models" / "content_based_recommender.joblib").exists(),
    reason="Requiere haber corrido el pipeline (python -m src.pipeline.train_pipeline) al menos una vez.",
)


def test_app_loads_default_page_without_exceptions():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert not at.exception


def test_app_evaluation_page_without_exceptions():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.sidebar.radio[0].set_value("📊 Evaluación de Modelos").run(timeout=30)
    assert not at.exception


def test_app_about_page_without_exceptions():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.sidebar.radio[0].set_value("ℹ️ Acerca del Proyecto").run(timeout=30)
    assert not at.exception


def test_app_category_filter_and_k_slider_interaction():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.selectbox[0].set_value("Baby").run(timeout=30)
    assert not at.exception
    at.slider[0].set_value(10).run(timeout=30)
    assert not at.exception
    assert len(at.dataframe) == 3  # las 3 tabs de recomendaciones
