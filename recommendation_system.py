"""Sistema sencillo de recomendacion de productos basado en contenido.

El modelo aprende una representacion numerica de las caracteristicas de cada
producto y utiliza sus vecinos mas cercanos para encontrar productos similares.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ProductRecommender:
	"""Recomienda productos similares a partir de sus caracteristicas."""

	def __init__(
		self,
		id_column: str = "product_id",
		numeric_features: list[str] | None = None,
		categorical_features: list[str] | None = None,
		text_feature: str = "description",
	) -> None:
		self.id_column = id_column
		self.numeric_features = numeric_features or ["price", "rating"]
		self.categorical_features = categorical_features or ["category", "brand"]
		self.text_feature = text_feature
		self.data: pd.DataFrame | None = None
		self._preprocessor: ColumnTransformer | None = None
		self._model: NearestNeighbors | None = None

	def load_data(self, source: str | Path | pd.DataFrame) -> pd.DataFrame:
		"""Carga productos desde un CSV o reutiliza un DataFrame existente."""
		if isinstance(source, pd.DataFrame):
			data = source.copy()
		else:
			data = pd.read_csv(source)

		if data.empty:
			raise ValueError("El conjunto de productos no puede estar vacio.")
		if self.id_column not in data.columns:
			raise ValueError(f"Falta la columna identificadora '{self.id_column}'.")

		self.data = data.reset_index(drop=True)
		return self.data

	def fit(self, data: pd.DataFrame | None = None) -> "ProductRecommender":
		"""Preprocesa las caracteristicas y entrena el modelo de vecinos."""
		if data is not None:
			self.load_data(data)
		if self.data is None:
			raise ValueError("Carga los datos con load_data antes de entrenar.")

		self._validate_features(self.data)
		features = self._prepare_features(self.data)

		# Cada tipo de dato recibe el tratamiento adecuado antes de combinarse:
		# escala para numeros, one-hot para categorias y TF-IDF para texto.
		self._preprocessor = ColumnTransformer(
			transformers=[
				("numeric", StandardScaler(), self.numeric_features),
				(
					"categorical",
					OneHotEncoder(handle_unknown="ignore"),
					self.categorical_features,
				),
				("text", TfidfVectorizer(), self.text_feature),
			]
		)
		matrix = self._preprocessor.fit_transform(features)

		# La distancia coseno mide la similitud entre perfiles de productos.
		self._model = NearestNeighbors(metric="cosine", algorithm="brute")
		self._model.fit(matrix)
		return self

	def fit_from_csv(self, path: str | Path) -> "ProductRecommender":
		"""Atajo para cargar un CSV y entrenar el recomendador."""
		return self.fit(self.load_data(path))

	def recommend_similar(self, product_id: Any, n: int = 5) -> pd.DataFrame:
		"""Devuelve los productos mas similares a un producto existente."""
		self._ensure_trained()
		assert self.data is not None
		matches = self.data.index[self.data[self.id_column] == product_id]
		if len(matches) == 0:
			raise KeyError(f"No existe un producto con id '{product_id}'.")

		row_index = int(matches[0])
		query = self._transform(self.data.iloc[[row_index]])
		distances, indices = self._model.kneighbors(
			query, n_neighbors=min(n + 1, len(self.data))
		)
		result = self._build_result(indices[0], distances[0])
		return result[result[self.id_column] != product_id].head(n).reset_index(drop=True)

	def recommend_from_features(
		self, features: dict[str, Any], n: int = 5
	) -> pd.DataFrame:
		"""Recomienda productos para un perfil que aun no esta en el catalogo."""
		self._ensure_trained()
		assert self.data is not None
		query = pd.DataFrame([features])
		prepared = self._prepare_features(query, fill_from=self.data)
		assert self._preprocessor is not None
		query_matrix = self._preprocessor.transform(prepared)
		distances, indices = self._model.kneighbors(
			query_matrix, n_neighbors=min(n, len(self.data))
		)
		return self._build_result(indices[0], distances[0]).head(n).reset_index(drop=True)

	def _validate_features(self, data: pd.DataFrame) -> None:
		required = set(self.numeric_features + self.categorical_features + [self.text_feature])
		missing = sorted(required - set(data.columns))
		if missing:
			raise ValueError(f"Faltan columnas de caracteristicas: {', '.join(missing)}")

	def _prepare_features(
		self, data: pd.DataFrame, fill_from: pd.DataFrame | None = None
	) -> pd.DataFrame:
		prepared = data.copy()
		source = fill_from if fill_from is not None else prepared
		for column in self.numeric_features:
			prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
			prepared[column] = prepared[column].fillna(source[column].median())
		for column in self.categorical_features + [self.text_feature]:
			prepared[column] = prepared[column].fillna("").astype(str)
		return prepared[self.numeric_features + self.categorical_features + [self.text_feature]]

	def _transform(self, data: pd.DataFrame):
		assert self._preprocessor is not None
		return self._preprocessor.transform(self._prepare_features(data, self.data))

	def _build_result(self, indices, distances) -> pd.DataFrame:
		assert self.data is not None
		result = self.data.iloc[indices].copy()
		result["similarity_score"] = 1 - distances
		return result

	def _ensure_trained(self) -> None:
		if self.data is None or self._preprocessor is None or self._model is None:
			raise RuntimeError("Debes entrenar el recomendador antes de recomendar.")


if __name__ == "__main__":
	# Ejemplo autocontenido: en un proyecto real, sustituye este DataFrame por
	# ProductRecommender().fit_from_csv("productos.csv").
	products = pd.DataFrame(
		[
			{"product_id": 1, "name": "Laptop Pro", "category": "informatica", "brand": "Nova", "description": "laptop profesional para trabajo", "price": 1200, "rating": 4.8},
			{"product_id": 2, "name": "Laptop Air", "category": "informatica", "brand": "Nova", "description": "laptop ligera para trabajo", "price": 950, "rating": 4.6},
			{"product_id": 3, "name": "Auriculares Studio", "category": "audio", "brand": "Sonic", "description": "auriculares inalambricos con cancelacion", "price": 220, "rating": 4.7},
			{"product_id": 4, "name": "Monitor 4K", "category": "informatica", "brand": "Vision", "description": "monitor profesional de alta resolucion", "price": 500, "rating": 4.5},
		]
	)

	recommender = ProductRecommender().fit(products)
	print("Productos similares a Laptop Pro:")
	print(recommender.recommend_similar(1, n=2)[["name", "similarity_score"]])

	print("\nRecomendaciones para un nuevo perfil:")
	print(
		recommender.recommend_from_features(
			{
				"category": "informatica",
				"brand": "Nova",
				"description": "laptop ligera para estudiar",
				"price": 900,
				"rating": 4.5,
			},
			n=2,
		)[["name", "similarity_score"]])
