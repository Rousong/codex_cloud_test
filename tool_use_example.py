import os
import subprocess

from anthropic import Anthropic


client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

TOOLS = [
    {
        "name": "run_command",
        "description": "Run a shell command and return stdout/stderr.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"}
            },
            "required": ["command"],
        },
    }
]


def run_command(command: str) -> str:
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout or result.stderr or "(no output)"


messages = [{"role": "user", "content": "Please show files in current folder."}]

response = client.messages.create(
    model=MODEL,
    max_tokens=300,
    tools=TOOLS,
    messages=messages,
)

messages.append({"role": "assistant", "content": response.content})

if response.stop_reason == "tool_use":
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            output = run_command(block.input["command"])
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                }
            )

    messages.append({"role": "user", "content": tool_results})
    final_response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        tools=TOOLS,
        messages=messages,
    )
    print(final_response.content[0].text)
else:
    print(response.content[0].text)
