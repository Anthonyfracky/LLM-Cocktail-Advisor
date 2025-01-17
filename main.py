import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.markdown import Markdown
import sys
import ast
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.schema.runnable import RunnableSequence
from langchain.schema.output_parser import StrOutputParser
import json

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

    def to_text(self) -> str:
        return f"""Name: {self.name}
Alcoholic: {self.alcoholic}
Category: {self.category}
Glass: {self.glassType}
Instructions: {self.instructions}
Ingredients: {', '.join(f'{measure} {ingredient}' for measure, ingredient in zip(self.ingredientMeasures, self.ingredients))}"""

class CocktailVectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        self.cocktails: List[Cocktail] = []
        self.vector_store = None

    def add_cocktails(self, cocktails: List[Cocktail]):
        self.cocktails = cocktails
        documents = [
            Document(
                page_content=cocktail.to_text(),
                metadata={
                    "id": cocktail.id,
                    "name": cocktail.name,
                    "ingredients": cocktail.ingredients,
                    "category": cocktail.category,
                    "alcoholic": cocktail.alcoholic
                }
            )
            for cocktail in cocktails
        ]

        self.vector_store = LangchainFAISS.from_documents(
            documents,
            self.embeddings
        )

    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)

class UserPreferenceStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        self.vector_store = None
        self.preferences: List[Dict] = []
        self.chat_history: List[str] = []

    def save_user_preferences(self, preference_text: str, metadata: Optional[Dict] = None):
        if not preference_text.strip():
            return

        if not self.vector_store:
            self.vector_store = LangchainFAISS.from_documents(
                [Document(page_content=preference_text, metadata=metadata or {})],
                self.embeddings
            )
        else:
            self.vector_store.add_documents(
                [Document(page_content=preference_text, metadata=metadata or {})]
            )

        self.preferences.append({
            "text": preference_text,
            "metadata": metadata or {}
        })

    def get_relevant_preferences(self, query: str, k: int = 3) -> List[Document]:
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)

class CocktailSystem:
    def __init__(self):
        self.cocktail_store = CocktailVectorStore()
        self.preference_store = UserPreferenceStore()

        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            model_name="mixtral-8x7b-32768",
            temperature=0.7
        )

        self.load_data()
        self.setup_chains()

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
                    print(f"Skipped cocktail {row.get('name', 'Unknown')}: {str(e)}")

            if not cocktails:
                raise ValueError("No cocktails were loaded from the dataset")

            self.cocktail_store.add_cocktails(cocktails)
            print(f"Loaded {len(cocktails)} cocktails")

        except FileNotFoundError:
            print("Error: cocktails_data.csv file not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            sys.exit(1)

    def setup_chains(self):
        preference_prompt = PromptTemplate.from_template(
            "Extract user preferences from the following text. "
            "Focus on ingredients, types of cocktails, and flavors they like. "
            "If no clear preferences are found, return 'No specific preferences found.':\n\n"
            "Text: {text}\n\n"
            "Preferences:"
        )

        self.preference_chain = (
            preference_prompt
            | self.llm
            | StrOutputParser()
        )

        qa_prompt = PromptTemplate.from_template(
            """You are a knowledgeable cocktail expert. Use the following pieces of context and user preferences to answer the question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Always be specific about cocktail recommendations and include ingredients when relevant.

            Context: {context}
            User Preferences: {preferences}
            Question: {question}
            Chat History: {chat_history}

            Answer:"""
        )

        self.qa_chain = (
            qa_prompt
            | self.llm
            | StrOutputParser()
        )

    async def extract_preferences(self, query: str) -> str:
        try:
            response = self.preference_chain.invoke({"text": query})

            if response.strip() and "no specific preferences found" not in response.lower():
                self.preference_store.save_user_preferences(
                    response,
                    metadata={
                        "original_query": query,
                        "timestamp": pd.Timestamp.now().isoformat()
                    }
                )
                self.preference_store.chat_history.append(f"User preference: {response}")

            return response
        except Exception as e:
            print(f"Warning: Could not extract preferences: {str(e)}")
            return ""

    async def process_query(self, query: str) -> str:
        try:
            preferences = ""
            if any(word in query.lower() for word in ['like', 'love', 'prefer', 'favorite', 'enjoy']):
                preferences = await self.extract_preferences(query)

            relevant_cocktails = self.cocktail_store.search_similar(query)
            relevant_preferences = self.preference_store.get_relevant_preferences(query)

            context = "\n\n".join([doc.page_content for doc in relevant_cocktails])
            preferences_context = "\n\n".join([doc.page_content for doc in relevant_preferences])
            chat_history = "\n".join(self.preference_store.chat_history[-5:])

            response = self.qa_chain.invoke({
                "context": context,
                "preferences": preferences_context,
                "question": query,
                "chat_history": chat_history
            })

            self.preference_store.chat_history.append(f"User: {query}")
            self.preference_store.chat_history.append(f"Assistant: {response}")

            return response

        except Exception as e:
            print(f"Error in query processing: {str(e)}")
            return "I apologize, but I encountered an error processing your request. Could you please try rephrasing your question?"