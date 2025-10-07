import openai, os

from core.llm.providers import OpenAIChat

openai.api_key = "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"
openai.api_base = "https://api.openai.com/v1"


if __name__ == "__main__":
    resp = openai.embeddings.create(
        model="text-embedding-3-large",
        input="Hello world"
    )
    print(resp.data[0].embedding[:5])  # should print floats



