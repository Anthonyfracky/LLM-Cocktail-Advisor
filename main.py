import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import json
from dataclasses import dataclass
import openai
from rich.console import Console
from rich.markdown import Markdown
import sys


@dataclass
class Cocktail:
    id: str
    name: str
    alcoholic: str
    category: str
    glassType: str
    instructions: str
    drinkThumbnail: str
    ingredients: List[str]
    ingredientMeasures: List[str]


class VectorDatabase:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.cocktails: List[Cocktail] = []
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def add_cocktails(self, cocktails: List[Cocktail]):
        self.cocktails = cocktails
        vectors = []
        for cocktail in cocktails:
            # Створюємо текстовий опис для векторизації
            text = " ".join(cocktail.ingredients)
            vector = self.encoder.encode([text])[0]
            vectors.append(vector)

        vectors_np = np.array(vectors).astype('float32')
        self.index.add(vectors_np)

    def search_similar(self, query: str, k: int = 5) -> List[Cocktail]:
        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        return [self.cocktails[idx] for idx in indices[0]]


class UserPreferences:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.preferences: List[str] = []

    def add_preference(self, preference: str):
        if preference not in self.preferences:
            self.preferences.append(preference)
            vector = self.encoder.encode([preference])[0]
            self.index.add(np.array([vector]).astype('float32'))

    def get_preferences(self) -> List[str]:
        return self.preferences


class CocktailCLI:
    def __init__(self):
        self.console = Console()
        self.vector_db = VectorDatabase()
        self.user_prefs = UserPreferences()
        self.load_data()
        self.setup_llm()

    def setup_llm(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.base_url = "https://api.groq.com/openai/v1"

    def load_data(self):
        df = pd.read_csv('cocktails_data.csv')
        cocktails = []
        for _, row in df.iterrows():
            cocktail = Cocktail(
                id=row['id'],
                name=row['name'],
                alcoholic=row['alcoholic'],
                category=row['category'],
                glassType=row['glassType'],
                instructions=row['instructions'],
                drinkThumbnail=row['drinkThumbnail'],
                ingredients=json.loads(row['ingredients']),
                ingredientMeasures=json.loads(row['ingredientMeasures'])
            )
            cocktails.append(cocktail)
        self.vector_db.add_cocktails(cocktails)

    async def process_query(self, query: str) -> str:
        # Визначення типу запиту та формування відповіді
        messages = [
            {"role": "system",
             "content": "You are a helpful cocktail expert. Provide concise and accurate responses about cocktails."},
            {"role": "user", "content": query}
        ]

        # Додаємо контекст про вподобання користувача
        if self.user_prefs.get_preferences():
            prefs_context = f"User preferences: {', '.join(self.user_prefs.get_preferences())}"
            messages.append({"role": "system", "content": prefs_context})

        # Якщо запит містить пошук схожих напоїв
        if "схожий" in query.lower():
            similar_cocktails = self.vector_db.search_similar(query, k=3)
            cocktails_context = "Found similar cocktails:\n" + "\n".join(
                [f"- {c.name} ({', '.join(c.ingredients)})" for c in similar_cocktails]
            )
            messages.append({"role": "system", "content": cocktails_context})

        response = await openai.ChatCompletion.acreate(
            model="mixtral-8x7b-32768",
            messages=messages,
            max_tokens=500
        )

        # Перевіряємо наявність нових вподобань у запиті
        if "люблю" in query.lower() or "подобається" in query.lower():
            preferences = self.extract_preferences(query)
            for pref in preferences:
                self.user_prefs.add_preference(pref)

        return response.choices[0].message.content

    def extract_preferences(self, query: str) -> List[str]:
        # Простий метод вилучення вподобань з тексту
        preferences = []
        keywords = ["люблю", "подобається"]
        for keyword in keywords:
            if keyword in query.lower():
                # Витягуємо слова після ключового слова
                words = query.lower().split(keyword)[1].strip().split()
                preferences.extend([w for w in words if len(w) > 2])
        return preferences

    async def run(self):
        self.console.print("[bold green]Вітаємо у системі рекомендації коктейлів![/bold green]")
        self.console.print("Введіть ваш запит або 'вихід' для завершення роботи.")

        while True:
            try:
                query = input("\nВаш запит: ").strip()

                if query.lower() in ['вихід', 'exit', 'quit']:
                    self.console.print("[bold red]До побачення![/bold red]")
                    break

                if not query:
                    continue

                response = await self.process_query(query)
                self.console.print(Markdown(response))

            except KeyboardInterrupt:
                self.console.print("\n[bold red]Програму завершено користувачем.[/bold red]")
                break
            except Exception as e:
                self.console.print(f"[bold red]Помилка: {str(e)}[/bold red]")


if __name__ == "__main__":
    import asyncio

    cli = CocktailCLI()
    asyncio.run(cli.run())