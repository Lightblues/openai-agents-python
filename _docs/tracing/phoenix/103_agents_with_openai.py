""" 
pip install openinference-instrumentation-openai-agents openai-agents
"""
# NOTE: 这里强制用 OpenAIInstrumentor 而非 Agents 的也可以! 
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

from agents import Agent, Runner, function_tool
@function_tool
def get_weather(city: str) -> str:
    """Get current temperature for a given location."""
    return f"The weather in {city} is sunny."

agent = Agent(name="Assistant", instructions="You are a helpful assistant", tools=[get_weather])
result = Runner.run_sync(agent, "What is the weather like in Paris today?")
print(result.final_output)
