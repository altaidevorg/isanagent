import asyncio
import os
from enum import Enum
from typing import List
from pydantic import BaseModel, Field

from afterimage import (
    AsyncStructuredGenerator,
    PersonaGenerator,
    PersonaInstructionGeneratorCallback,
    InMemoryDocumentProvider,
)

# --- Schema Definitions ---


class SupportIntent(str, Enum):
    REFUND = "Refund Request"
    TECHNICAL_SUPPORT = "Technical Support"
    BILLING = "Billing Inquiry"
    PRODUCT_INFO = "Product Information"
    WARRANTY = "Warranty Claim"
    COMPLAINT = "General Complaint"
    OTHER = "Other"


class UrgencyLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class ActionType(str, Enum):
    CLOSE = "Close"
    ESCALATION = "Escalation"
    KEEP_OPEN = "Keep Open"


class ToolCall(str, Enum):
    KNOWLEDGE_BASE_SEARCH = "Knowledge Base Search"
    NONE = "none"


class CustomerSupportInteraction(BaseModel):
    agent_reasoning: str = Field(
        description="Step-by-step reasoning to reach the final response. Explain the diagnosis and decision process."
    )
    intent: str = Field(description="Primary intent of the customer")
    urgency: str = Field(description="Assessed urgency level")
    sentiment_score: float = Field(
        description="Sentiment score from -1.0 (Very Negative) to 1.0 (Very Positive)"
    )
    key_entities: List[str] = Field(
        description="Key entities extracted (Product names, Order IDs, Dates)"
    )
    missing_information: List[str] = Field(
        description="Information missing to resolve the query"
    )
    action: ActionType = Field(
        description="The action taken by the agent. Close if it's resolved, escalade if it's urgent, and keep it open if it's pending customer."
    )
    action_reason: str = Field(description="Reason for the action taken by the agent.")
    query: str = Field(
        description="The search query that you would need to run against the knowledge base to resolve the customer request."
    )
    response: str = Field(
        description="The final natural language response to the customer"
    )


# --- Knowledge Base (Context) ---

TECH_GADGET_POLICIES = [
    """
    # Refund Policy for TechGadget Inc.
    - Standard Refund: Items can be returned within 30 days of purchase for a full refund.
    - Defective Items: Defective items have a 1-year warranty. We will ship a replacement immediately upon proof of defect.
    - Digital Goods: No refunds on software keys once redeemed.
    - Restocking Fee: Open box items (non-defective) are subject to a 15% restocking fee.
    """,
    """
    # Troubleshooting Guide: TechGadget 3000
    - Screen Won't Turn On: Hold the power button for 15 seconds to force reset. Check the charging port for debris.
    - Bluetooth Audio Lag: Enhance connection by turning off WiFi on nearby devices (interference). Firmware v2.1 fixes this.
    - Battery Draining Fast: Turn off 'Always-On Display' in Settings > Battery. Replace battery if health < 80%.
    """,
    """
    # Shipping & Delivery Guidelines
    - Express Shipping: 1-2 business days. Cost: $15.00.
    - Standard Shipping: 3-5 business days. Free for orders over $50.
    - International: 7-14 business days. Customs duties are the responsibility of the recipient.
    - Lost Packages: Claims must be filed within 48 hours of marked delivery.
    """,
]


async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    # api_key = "chengethis"
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable.")
        return

    # 1. Setup Document Provider
    docs = InMemoryDocumentProvider(TECH_GADGET_POLICIES)

    # 2. Setup Persona Generator
    persona_gen = PersonaGenerator(api_key=api_key)

    # Generate personas for the documents
    # This will populate the .personas attribute of each Document in the provider
    await persona_gen.generate_from_documents(docs)

    # 3. Setup Instruction Generator (The "Simulator")
    # This generates the user query based on the docs + random personas
    instruction_callback = PersonaInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs,
        num_random_contexts=1,  # Create varied scenarios
    )

    # 4. Setup Structured Generator (The "Agent")
    # This acts as the AI Support Agent processing the simulated query
    respondent_prompt = """
    You are an advanced AI Customer Support Agent for "TechGadget Inc".
    Your goal is to triage incoming queries, analyze them deeply, and provide helpful, accurate responses based *strictly* on the provided context.
    
    - Be empathetic but professional.
    - If a user claims a defect, check the warranty policy.
    - If a user wants a refund, check the 30-day window.
    """

    generator = AsyncStructuredGenerator(
        output_schema=CustomerSupportInteraction,
        respondent_prompt=respondent_prompt,
        api_key=api_key,
        model_name="gemini-2.0-flash",
        instruction_generator_callback=instruction_callback,
    )

    print("Starting generation of synthetic customer support dataset...")
    # Generate 15 samples (5 per document roughly)
    await generator.generate(num_samples=15)
    print("Generation complete.")


if __name__ == "__main__":
    asyncio.run(main())
