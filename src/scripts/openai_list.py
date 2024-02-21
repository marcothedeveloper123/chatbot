from openai import OpenAI
import ollama

client = OpenAI()
clients = client.models.list()
ids = [model.id for model in clients]

print(ids)

list = ollama.list()
# models = [model.name for model in list['models']]
model_names = [model["name"] for model in list["models"]]

print(model_names)
