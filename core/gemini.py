from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_answer(question: str, context_chunks: list[str]) -> str:
    try:
        context = "\n".join(context_chunks)
        
        prompt = f"""You are a personal finance assistant.
Answer ONLY based on the context provided below.
If the answer is not in the context, say "I don't have enough data to answer that."

Context (relevant transactions):
{context}

User Question: {question}

Answer clearly and concisely."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    except Exception as e:
        error = str(e)
        if "429" in error or "rate_limit" in error.lower():
            return "⚠️ Too many requests. Please wait a moment and try again."
        elif "401" in error or "auth" in error.lower():
            return "⚠️ API key invalid. Please check your configuration."
        elif "connection" in error.lower():
            return "⚠️ Connection error. Please check your internet connection."
        else:
            return f"⚠️ Something went wrong. Please try again."