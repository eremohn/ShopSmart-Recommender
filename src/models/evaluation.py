"""Métricas de evaluación offline para los recomendadores del MVP.

Al no existir historial de usuario (ver `src/models/recommenders.py`), no
es posible calcular métricas clásicas de *precision/recall* contra
interacciones reales de un usuario. Este módulo implementa, en cambio,
un conjunto de **métricas proxy defendibles** ampliamente usadas para
evaluar sistemas de recomendación basados en contenido cuando no hay
ground truth explícito de preferencias individuales:

- **Category Precision@K**: coherencia temática de las recomendaciones.
- **Catalog Coverage@K**: qué porción del catálogo llega a recomendarse
  (mide personalización / diversidad de exposición).
- **Intra-list Diversity@K**: qué tan distintos son entre sí los ítems de
  una misma lista de recomendación (evita listas redundantes).
- **Average Recommendation Quality@K**: calidad percibida promedio
  (rating bayesiano) de los ítems recomendados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def category_precision_at_k(
    recommendations_by_query: dict[str, list[str]], product_index: pd.DataFrame
) -> float:
    """Fracción promedio de recomendaciones que comparten `main_category`
    con el producto consultado.

    Parameters
    ----------
    recommendations_by_query : dict[str, list[str]]
        Mapeo `{asin_consultado: [asin_recomendado_1, ..., asin_recomendado_k]}`.
    product_index : pd.DataFrame
        DataFrame indexado por `asin` con la columna `main_category`.
    """
    precisions = category_precision_per_query(recommendations_by_query, product_index)
    return float(np.mean(list(precisions.values()))) if precisions else 0.0


def category_precision_per_query(
    recommendations_by_query: dict[str, list[str]], product_index: pd.DataFrame
) -> dict[str, float]:
    """Igual que `category_precision_at_k`, pero devuelve la precisión
    **por consulta** en lugar del promedio agregado. Es el insumo
    necesario para calcular intervalos de confianza y tests estadísticos
    pareados entre modelos (sección de evaluación formal, notebook 04)."""
    category_map = product_index.set_index("asin")["main_category"]
    precisions = {}
    for query_asin, rec_asins in recommendations_by_query.items():
        if not rec_asins:
            continue
        query_category = category_map.loc[query_asin]
        matches = sum(category_map.loc[a] == query_category for a in rec_asins)
        precisions[query_asin] = matches / len(rec_asins)
    return precisions


def bootstrap_confidence_interval(
    values: list[float] | np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Calcula el intervalo de confianza de la media de `values` mediante
    remuestreo bootstrap (con reposición), sin asumir normalidad — más
    apropiado que un intervalo paramétrico para métricas acotadas en
    [0, 1] como las de este proyecto.

    Returns
    -------
    tuple[float, float, float]
        `(media, límite_inferior, límite_superior)`.
    """
    rng = np.random.default_rng(random_state)
    values = np.asarray(values)
    boot_means = np.array(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(values.mean()), float(lower), float(upper)


def catalog_coverage_at_k(
    recommendations_by_query: dict[str, list[str]], catalog_size: int
) -> float:
    """Porcentaje de productos del catálogo que aparecen en al menos una
    lista de recomendaciones. Un valor bajo indica que el modelo siempre
    recomienda el mismo subconjunto de productos ("popularity bias")."""
    all_recommended = {a for recs in recommendations_by_query.values() for a in recs}
    return len(all_recommended) / catalog_size


def intra_list_diversity(
    recommendations_by_query: dict[str, list[str]],
    features_matrix: np.ndarray,
    asin_to_pos: dict[str, int],
) -> float:
    """Diversidad intra-lista promedio: `1 - similitud_coseno_promedio`
    entre todos los pares de ítems dentro de cada lista recomendada.
    Valores cercanos a 1 indican listas variadas; cercanos a 0, listas
    de productos casi idénticos entre sí."""
    from sklearn.metrics.pairwise import cosine_similarity

    diversities = []
    for rec_asins in recommendations_by_query.values():
        if len(rec_asins) < 2:
            continue
        positions = [asin_to_pos[a] for a in rec_asins if a in asin_to_pos]
        if len(positions) < 2:
            continue
        sub_matrix = features_matrix[positions]
        sim = cosine_similarity(sub_matrix)
        upper_triangle = sim[np.triu_indices_from(sim, k=1)]
        diversities.append(1 - upper_triangle.mean())
    return float(np.mean(diversities)) if diversities else 0.0


def avg_recommendation_quality(
    recommendations_by_query: dict[str, list[str]],
    product_index: pd.DataFrame,
    quality_col: str = "bayesian_rating",
) -> float:
    """Calidad promedio (según `quality_col`) de todos los ítems
    recomendados a lo largo de todas las consultas."""
    quality_map = product_index.set_index("asin")[quality_col]
    all_recommended = [a for recs in recommendations_by_query.values() for a in recs]
    if not all_recommended:
        return 0.0
    return float(quality_map.loc[all_recommended].mean())


def paired_wilcoxon_test(
    values_a: dict[str, float], values_b: dict[str, float]
) -> dict[str, float]:
    """Test de rangos con signo de Wilcoxon, pareado por consulta (`asin`).

    Se usa en lugar de un t-test pareado porque las métricas de precisión
    por consulta no siguen una distribución normal (están acotadas en
    [0, 1] y suelen concentrarse en pocos valores discretos, ej. 0.0,
    0.1, 0.2, ..., 1.0 para K=10) — Wilcoxon no asume normalidad y es la
    alternativa no paramétrica estándar al t-test pareado.

    Returns
    -------
    dict
        `{"statistic": ..., "p_value": ..., "n_pairs": ...,
        "mean_diff": ...}`, donde `mean_diff = mean(A) - mean(B)`.
    """
    from scipy import stats

    common_keys = sorted(set(values_a) & set(values_b))
    a = np.array([values_a[k] for k in common_keys])
    b = np.array([values_b[k] for k in common_keys])

    diffs = a - b
    if np.all(diffs == 0):
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": len(common_keys), "mean_diff": 0.0}

    result = stats.wilcoxon(a, b)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_pairs": len(common_keys),
        "mean_diff": float(a.mean() - b.mean()),
    }
