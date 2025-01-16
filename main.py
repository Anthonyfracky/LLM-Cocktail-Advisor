import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import json
from dataclasses import dataclass
from rich.console import Console
from rich.markdown import Markdown
import sys
import ast
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


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
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        self.load_data()

    def parse_list_field(self, field_value):
        if pd.isna(field_value):
            return []
        try:
            return json.loads(field_value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(field_value)
            except (ValueError, SyntaxError):
                return [item.strip() for item in str(field_value).split(',') if item.strip()]

    def load_data(self):
        try:
            df = pd.read_csv('cocktails_data.csv')
            cocktails = []

            for _, row in df.iterrows():
                try:
                    ingredients = self.parse_list_field(row['ingredients'])
                    measures = self.parse_list_field(row['ingredientMeasures'])

                    cocktail = Cocktail(
                        id=str(row['id']),
                        name=str(row['name']),
                        alcoholic=str(row['alcoholic']),
                        category=str(row['category']),
                        glassType=str(row['glassType']),
                        instructions=str(row['instructions']),
                        drinkThumbnail=str(row['drinkThumbnail']),
                        ingredients=ingredients,
                        ingredientMeasures=measures
                    )
                    cocktails.append(cocktail)
                except Exception as e:
                    self.console.print(f"[yellow]Skipped cocktail {row.get('name', 'Unknown')}: {str(e)}[/yellow]")

            if not cocktails:
                raise ValueError("No cocktails were loaded from the dataset")

            self.vector_db.add_cocktails(cocktails)
            self.console.print(f"[green]Loaded {len(cocktails)} cocktails[/green]")

        except FileNotFoundError:
            self.console.print("[red]Error: cocktails_data.csv file not found[/red]")
            sys.exit(1)
        except Exception as e:
            self.console.print(f"[red]Error loading data: {str(e)}[/red]")
            sys.exit(1)

    async def process_query(self, query: str) -> str:
        system_prompt = """You are a cocktail expert assistant. Provide concise and accurate responses about cocktails.
        Focus on being helpful and informative while maintaining a natural conversational tone."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        if self.user_prefs.get_preferences():
            prefs_context = f"User preferences: {', '.join(self.user_prefs.get_preferences())}"
            messages.append({"role": "system", "content": prefs_context})

        if "similar" in query.lower():
            similar_cocktails = self.vector_db.search_similar(query, k=3)
            cocktails_context = "Found similar cocktails:\n" + "\n".join(
                [f"- {c.name} ({', '.join(c.ingredients)})" for c in similar_cocktails]
            )
            messages.append({"role": "system", "content": cocktails_context})

        response = self.client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )

        if "like" in query.lower() or "love" in query.lower():
            preferences = self.extract_preferences(query)
            for pref in preferences:
                self.user_prefs.add_preference(pref)

        return response.choices[0].message.content

    def extract_preferences(self, query: str) -> List[str]:
        preferences = []
        keywords = ["like", "love"]
        for keyword in keywords:
            if keyword in query.lower():
                words = query.lower().split(keyword)[1].strip().split()
                preferences.extend([w for w in words if len(w) > 2])
        return preferences

    async def run(self):
        self.console.print("[bold green]Welcome to the Cocktail Recommendation System![/bold green]")
        self.console.print("Enter your query or type 'exit' to quit.")

        while True:
            try:
                query = input("\nYour query: ").strip()

                if query.lower() in ['exit', 'quit']:
                    self.console.print("[bold red]Goodbye![/bold red]")
                    break

                if not query:
                    continue

                response = await self.process_query(query)
                self.console.print(Markdown(response))

            except KeyboardInterrupt:
                self.console.print("\n[bold red]Program terminated by user.[/bold red]")
                break
            except Exception as e:
                self.console.print(f"[bold red]Error: {str(e)}[/bold red]")


if __name__ == "__main__":
    import asyncio

    cli = CocktailCLI()
    asyncio.run(cli.run())
