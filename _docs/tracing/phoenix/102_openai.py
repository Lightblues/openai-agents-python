""" 
OpenAI
    - ChatCompletion
    - Responses

pip install openinference-instrumentation-openai
"""
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register

tracer_provider = register(
    protocol="grpc", # "http/protobuf",
    project_name="default",
    batch=True,
    # auto_instrument=True,
    endpoint="http://9.134.230.111:4317" # 6006"
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


import openai
client = openai.OpenAI()
tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            }
        },
        "required": [
            "location"
        ],
        "additionalProperties": False
    }
}]
response = client.responses.create(
    model="gpt-4o",
    input=[{"role": "user", "content": "What is the weather like in Paris today?"}],
    tools=tools
)
print(response)


tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia"
                }
            },
            "required": [
                "location"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}]
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the weather like in Paris today?"}],
    tools=tools
)
print(response)