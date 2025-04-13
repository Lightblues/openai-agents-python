import openai

client = openai.OpenAI(
  api_key="sk-2mwOZVHGeIRplsXZCh4xXuNX8KB7ncohakn1CYLg03r8nQWY",
  base_url="https://testapi.lkeap.cloud.tencent.com/v1/",
)
model = "deepseek-v3-0324-function-call"
messages = [
    {"role": "user", "content": "你是谁？查一下明后天北上广深的天气"},
    {"role": "assistant", "content": "好的，我来帮你查一下。"*10000},
    {"role": "user", "content": "告诉我你是谁？"},
]
print(messages)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast"
                    }
                },
                "required": [
                    "location", "num_days"
                ]
            }
        }
    }
]
def test_non_stream():
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        extra_body={"truncate_after_first_fc": True},
    )
    print(response.choices[0].message.tool_calls)
    print(response)

def test_stream():
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        stream=True,
        extra_body={"truncate_after_first_fc": True},
    )
    for chunk in stream:
        print(chunk)

# test_non_stream()
test_stream()
print()
