import asyncio
import os
from afterimage.providers import LLMFactory
from afterimage.simula import OpenSimula

INSTRUCTION_Y = """\
You are generating synthetic **math word problems** for middle school students.
Each problem must require at least two steps of reasoning (e.g., addition then division).
The answer must include a step-by-step Chain-of-Thought explanation before providing the final numerical answer.
"""

async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY")
        return

    llm = LLMFactory.create(
        provider="gemini",
        model_name="gemini-2.5-flash",
        api_key=api_key,
    )
    
    sim = OpenSimula(llm, temperature=0.5)
    
    # Build taxonomy and generate questions
    bundle = await sim.build_taxonomy(
        INSTRUCTION_Y,
        target_depth_D=2,
        proposal_N=2,
        max_factors=3,
        max_children_per_node=3,
        max_frontier_per_depth=5,
        show_progress=True,
    )
    
    OpenSimula.validate_taxonomy_bundle(bundle)
    
    # Generate samples based on the taxonomy
    print("Inferring strategies...")
    spec = await sim.infer_strategies(bundle)
    
    print("Generating 1 sample...")
    for i in range(1):
        mix = sim.sample_mix(bundle, spec)
        meta = await sim.draw_meta_prompt(
            instruction_y=bundle.instruction_y,
            bundle=bundle,
            mix=mix,
            K=3,
            complexify_c=0.2,
            sequential=False,
        )
        row = await sim.generate_single_qa_datapoint(
            instruction_y=bundle.instruction_y,
            bundle=bundle,
            mix=mix,
            meta=meta,
        )
        
        print(f"--- Sample {i+1} ---")
        if row:
            print(row.model_dump_json(indent=2))
        else:
            print("Row rejected by critic.\n")

if __name__ == "__main__":
    asyncio.run(main())
