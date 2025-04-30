from phoenix.otel import register

# configure the Phoenix tracer
tracer_provider = register(
    protocol="grpc", # "http/protobuf",
    project_name="default",
    batch=True,
    auto_instrument=True,
    endpoint="http://9.134.230.111:4317" # 6006"
)
tracer = tracer_provider.get_tracer(__name__)

@tracer.chain
def my_func(input: str) -> str:
    return "output"

my_func("input")

# Add OpenAI API Key
import openai

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a haiku."}],
)
print(response.choices[0].message.content)