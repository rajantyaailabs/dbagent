import os

from google import genai

# Ensure your API key is set as an environment variable
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

client = genai.Client()

for model in client.models.list():
    print(model.name)