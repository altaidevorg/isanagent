import asyncio
import os
from enum import Enum
from pydantic import BaseModel, Field
from afterimage import (
    AsyncStructuredGenerator,
    ContextualInstructionGeneratorCallback,
    InMemoryDocumentProvider,
    WithContextRespondentPromptModifier
)

class Severity(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"
    NONE = "None"

class CodeReviewOutput(BaseModel):
    bug_found: bool = Field(description="Whether a bug or security vulnerability was found in the code.")
    severity: Severity = Field(description="The severity of the issue.")
    explanation: str = Field(description="Detailed explanation of the issue and why it occurs.")
    suggested_fix: str = Field(description="The corrected code snippet.")

CODE_SNIPPETS = [
    "def calculate_discount(price, discount):\n    return price - (price * discount / 100)",
    "import sqlite3\ndef get_user(username):\n    conn = sqlite3.connect('users.db')\n    cursor = conn.cursor()\n    cursor.execute(f\"SELECT * FROM users WHERE username = '{username}'\")\n    return cursor.fetchone()",
    "def divide_numbers(a, b):\n    return a / b"
]

async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY")
        return

    docs = InMemoryDocumentProvider(CODE_SNIPPETS)

    instruction_cb = ContextualInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs,
        model_name="gemini-2.5-flash",
        num_random_contexts=1,
        n_instructions=1,
    )

    respondent_prompt = "You are an expert senior software engineer conducting a code review. Analyze the provided code for bugs, security vulnerabilities (like SQL injection), or edge cases (like division by zero)."

    gen = AsyncStructuredGenerator(
        output_schema=CodeReviewOutput,
        respondent_prompt=respondent_prompt,
        api_key=api_key,
        model_name="gemini-2.5-flash",
        instruction_generator_callback=instruction_cb,
    )
    
    await gen.generate(num_samples=3)

if __name__ == "__main__":
    asyncio.run(main())
