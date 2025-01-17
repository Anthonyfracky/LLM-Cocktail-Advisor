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
from typing import List, Dict, Optional, TypedDict
from typing_extensions import Annotated
from langchain_core.tools import tool
from langchain.tools.render import format_tool_to_openai_function
from langchain_openai import ChatOpenAI

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

class PreferenceInfo(TypedDict):
    has_preferences: bool
    preferences: Optional[str]
    confidence: float

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

        self.setup_chains()
        self.setup_tools()

    def setup_tools(self):
        # Define the schema for the function
        self.preference_analysis_function = {
            "name": "analyze_preferences",
            "description": "Analyzes user input to detect and extract cocktail preferences",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "The user's message to analyze"
                    }
                },
                "required": ["user_input"]
            }
        }

    async def analyze_preferences(self, user_input: str) -> dict:
        """Analyzes user preferences using a separate LLM call"""
        messages = [
            {
                "role": "system",
                "content": """Analyze the user's message for cocktail preferences. Return a JSON object with:
                - has_preferences: boolean indicating if preferences were found
                - preferences: string describing the preferences or null if none found
                - confidence: number between 0 and 1 indicating confidence in the analysis

                Example preferences to look for:
                - Specific ingredients they like/dislike
                - Types of cocktails they enjoy
                - Flavor preferences (sweet, sour, bitter, etc.)
                - Strength preferences
                - Style preferences"""
            },
            {
                "role": "user",
                "content": user_input
            }
        ]

        response = await self.llm.ainvoke(messages)

        try:
            # Try to parse the response as a structured preference analysis
            content = response.content
            if isinstance(content, str):
                # If it's a string, try to extract JSON from it
                try:
                    # Look for JSON-like structure in the response
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start >= 0 and end > start:
                        json_str = content[start:end]
                        result = json.loads(json_str)
                        return {
                            "has_preferences": bool(result.get("has_preferences", False)),
                            "preferences": result.get("preferences"),
                            "confidence": float(result.get("confidence", 0.0))
                        }
                except (json.JSONDecodeError, ValueError):
                    pass

            # If parsing fails, return a default structure
            return {
                "has_preferences": False,
                "preferences": None,
                "confidence": 0.0
            }
        except Exception as e:
            print(f"Error analyzing preferences: {e}")
            return {
                "has_preferences": False,
                "preferences": None,
                "confidence": 0.0
            }

    async def process_query(self, query: str) -> str:
        try:
            preferences = ""
            should_analyze_preferences = False

            # Check if this is a recommendation request without preference
            is_recommendation_request = any(
                word in query.lower() for word in ['recommend', 'recomend', 'suggestion', 'suggest'])
            contains_with = 'with' in query.lower()

            if is_recommendation_request and contains_with:
                # For queries like "recommend cocktail with vodka", treat the ingredient as context
                # but don't save it as a preference
                preferences = ""
            else:
                # For other queries, check for preferences
                messages = [
                    {
                        "role": "system",
                        "content": """You are a cocktail expert assistant. Determine if the user's message contains 
                        preferences about cocktails, ingredients, or drinking preferences. Only identify preferences
                        when users explicitly express likes/dislikes (e.g., "I like", "I love", "I prefer", "I enjoy").
                        Do not treat ingredient requests or questions as preferences."""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ]

                response = await self.llm.ainvoke(
                    messages,
                    functions=[self.preference_analysis_function],
                    function_call="auto"
                )

                if response.additional_kwargs.get("function_call"):
                    preference_result = await self.analyze_preferences(query)

                    if preference_result["has_preferences"] and preference_result["confidence"] > 0.6:
                        extracted_prefs = await self.extract_preferences(query)
                        if extracted_prefs and "no specific preferences found" not in extracted_prefs.lower():
                            preferences = extracted_prefs
                            should_analyze_preferences = True

            # Process the query with context
            relevant_cocktails = self.cocktail_store.search_similar(query)
            relevant_preferences = self.preference_store.get_relevant_preferences(
                query) if should_analyze_preferences else []

            context = "\n\n".join([doc.page_content for doc in relevant_cocktails])
            preferences_context = "\n\n".join([doc.page_content for doc in relevant_preferences])
            chat_history = "\n".join(self.preference_store.chat_history[-5:])

            qa_prompt = PromptTemplate.from_template(
                """You are a knowledgeable and friendly cocktail expert. Use the following pieces of context and user preferences to answer the question.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Always be specific about cocktail recommendations and include ingredients when relevant.
                Do not include any confidence scores or technical details in your response.
                Keep your tone conversational and helpful.

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

            final_response = self.qa_chain.invoke({
                "context": context,
                "preferences": preferences_context,
                "question": query,
                "chat_history": chat_history
            })

            # Update chat history
            self.preference_store.chat_history.append(f"User: {query}")
            self.preference_store.chat_history.append(f"Assistant: {final_response}")

            return final_response

        except Exception as e:
            print(f"Error in query processing: {str(e)}")
            return "I apologize, but I encountered an error processing your request. Could you please try rephrasing your question?"

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

