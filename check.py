import google.generativeai as genai


genai.configure(api_key="")

print("--- Active Chat Models for your API Key ---")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)