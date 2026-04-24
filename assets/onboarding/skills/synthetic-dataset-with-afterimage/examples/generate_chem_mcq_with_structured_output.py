import asyncio
import os
from pydantic import BaseModel, Field
from typing import List

from afterimage import (
    AsyncStructuredGenerator,
    ContextualInstructionGeneratorCallback,
    InMemoryDocumentProvider,
)

class ChemistryMCQ(BaseModel):
    question: str = Field(description="The multiple choice question text. Must be a top-tier, high-quality question assessing deep understanding.")
    option_a: str = Field(description="Option A")
    option_b: str = Field(description="Option B")
    option_c: str = Field(description="Option C")
    option_d: str = Field(description="Option D")
    correct_option: str = Field(description="The correct option letter (A, B, C, or D)")
    explanation: str = Field(description="Detailed step-by-step explanation of why the correct option is right and why the others are wrong.")
    subfield: str = Field(description="The subfield of chemistry (e.g., Organic, Physical, Inorganic, Analytical, Biochemistry)")
    difficulty: str = Field(description="Difficulty level (e.g., Advanced Undergraduate, Graduate, Professional)")

CHEMISTRY_TOPICS = [
    "Quantum Chemistry and Molecular Orbital Theory",
    "Thermodynamics and Statistical Mechanics",
    "Chemical Kinetics and Reaction Dynamics",
    "Coordination Chemistry and Ligand Field Theory",
    "Organometallic Chemistry and Catalysis",
    "Stereochemistry and Asymmetric Synthesis",
    "Advanced Reaction Mechanisms in Organic Chemistry",
    "Spectroscopy (Advanced NMR, IR, Mass Spec, EPR)",
    "Electrochemistry and Battery Technologies",
    "Solid State Chemistry and Crystallography",
    "Polymer Chemistry and Macromolecules",
    "Biochemistry, Enzymology, and Metabolic Pathways",
    "Advanced Analytical Chemistry and Instrumental Methods",
    "Photochemistry and Pericyclic Reactions",
    "Nuclear Chemistry and Radiochemistry",
    "Computational Chemistry and Density Functional Theory",
    "Supramolecular Chemistry and Host-Guest Interactions",
    "Bioinorganic Chemistry and Metalloenzymes",
    "Materials Chemistry and Nanomaterials",
    "Green Chemistry and Sustainable Synthesis"
]

async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable.")
        return

    docs = InMemoryDocumentProvider(CHEMISTRY_TOPICS)

    instruction_callback = ContextualInstructionGeneratorCallback(
        api_key=api_key,
        documents=docs,
        num_random_contexts=1,
        n_instructions=5,
        model_name="gemini-2.5-flash",
        prompt="Generate a prompt asking an expert chemist to create a highly challenging, graduate-level multiple-choice question about the following chemistry topic: {context}. The prompt should specify that the question must require deep conceptual understanding or complex problem-solving, not just rote memorization."
    )

    respondent_prompt = """
    You are a world-class chemistry professor and researcher.
    Your task is to create top-tier, graduate-level multiple-choice questions in chemistry.
    The questions should assess real, deep understanding of chemistry concepts, mechanisms, or calculations, avoiding simple trivia or high-school level material.
    Provide the question, four plausible options (A, B, C, D), the correct option, a detailed explanation, the subfield, and the difficulty level.
    """

    generator = AsyncStructuredGenerator(
        output_schema=ChemistryMCQ,
        respondent_prompt=respondent_prompt,
        api_key=api_key,
        model_name="gemini-2.5-flash",
        instruction_generator_callback=instruction_callback,
    )

    print("Starting generation of 100 chemistry MCQs...")
    await generator.generate(num_samples=100, max_concurrency=10)
    print("Generation complete.")

if __name__ == "__main__":
    asyncio.run(main())
