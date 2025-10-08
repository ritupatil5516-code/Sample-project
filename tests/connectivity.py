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



