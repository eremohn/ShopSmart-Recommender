"""Test de integración del pipeline reproducible end-to-end.

Es un test "pesado" comparado con el resto del suite (ejecuta el pipeline
completo sobre los 728 productos reales), pero es exactamente lo que se
quiere validar en CI: que `src/pipeline/train_pipeline.py` siga
corriendo de punta a punta sin errores ante cualquier cambio en `src/`.

Usa un directorio temporal para los artefactos de salida (`models_dir`,
`processed_dir`), de forma que correr este test **no pise** los
artefactos de producción (`models/`, `data/processed/`) que consume la
demo de Streamlit (`app.py`).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.train_pipeline import run_pipeline


def test_run_pipeline_end_to_end(tmp_path):
    models_dir = tmp_path / "models"
    processed_dir = tmp_path / "processed"

    result = run_pipeline(
        n_components=20,
        experiment_name="ci-smoke-test",
        models_dir=models_dir,
        processed_dir=processed_dir,
    )

    assert result["n_products"] > 0
    assert result["feature_matrix_shape"][1] == 20
    assert (models_dir / "content_based_recommender.joblib").exists()
    assert (processed_dir / "products_master.csv").exists()

    df_eval = result["evaluation"]
    assert {"content_based", "popularity"} <= set(df_eval.index)
    assert (df_eval["category_precision_at_10"] >= 0).all()
    assert (df_eval["category_precision_at_10"] <= 1).all()
