# **MaxEvolve: Agentic and Evolutionary Kernel Porting Framework for isanagent**

## **1. Executive Summary**

As machine learning models scale and transition toward highly irregular, sparse, or dynamic architectures (e.g., Mixture of Experts, Block-Sparse Attention, Ragged Batching), standard compiler heuristics in XLA and TorchInductor frequently fail to achieve optimal hardware utilization. Custom kernel development is required to bypass these limitations, but porting kernels between ecosystems—specifically from PyTorch/Triton to JAX/Pallas—presents a formidable engineering bottleneck. 

This document outlines the implementation plan for **MaxEvolve**, a framework designed to run directly within the **isanagent** workspace. MaxEvolve combines the multi-agent orchestration of Google's open-source **MaxKernel** (from the `AI-Hypercomputer/accelerator-agents` repository) with the closed-loop evolutionary optimization of **AlphaEvolve**. By treating kernel porting as a guided, iterative search problem, MaxEvolve automates:
1. **Syntactical and Paradigm Translation:** Converting Triton's imperative pointer arithmetic into Pallas's declarative `BlockSpec` and `pallas_call` orchestration via a rigorous 12-step pipeline.
2. **Numerical Validation:** Generating robust test harnesses using JAX's `interpret=True` CPU emulation to catch out-of-bounds memory accesses and race conditions.
3. **Hardware-in-the-Loop Optimization:** Iteratively mutating scheduling parameters (e.g., tiling dimensions, pipeline steps, layout casting) and benchmarking on actual hardware (GPU/TPU) to maximize Model FLOPs Utilization (MFU) and minimize latency.

---

## **2. System Architecture & Agent Hierarchy**

MaxEvolve inherits and extends the exact agent hierarchy of the MaxKernel codebase, wrapping its core capabilities in an evolutionary optimization loop.

```
KernelGenerationOrchestrationAgent (root_agent)
├── ExplanationAgent - Explains TPU/Pallas concepts
├── PlanKernelAgent - Creates/revises optimization plans
├── ImplementKernelAgent - Implements approved plans
├── ValidatedTestGenerationAgent
│   ├── GenerateTestFileAgent - Creates pytest test files
│   ├── TestValidationLoopAgent - Validates test syntax/structure
│   └── ValidationSummaryAgent - Reports validation results
├── UnifiedTestAgent
│   ├── ReadFileForTestingAgent - Locates test files
│   ├── RunTestsAgent - Executes pytest with server management
│   └── SummarizeTestResultsAgent - Analyzes and reports results
├── ProfileAgentOrchestrator
│   ├── ReadFileForProfilingAgent - Locates kernel files
│   ├── GenerateProfilingScriptAgent - Creates profiling scripts
│   ├── EvalProfileAgent - Executes profiling
│   └── SummarizeProfileAgent - Analyzes bottlenecks
├── AutotuneAgent - Automated parameter tuning
│   ├── AutotunePlannerAgent - Prepares specs and search space
│   ├── AutotuneRunner - Manages server and executes grid search
│   └── AutotuneSummaryAgent - Reports results to user
├── GpuToJaxAgent - GPU-to-JAX conversion pipeline (12-step pipeline)
└── EvolveKernelAgent (AlphaEvolve Extension)
    ├── ProgramDatabaseManager - Manages MAP-Elites candidate database
    ├── MutationOperatorAgent - Applies domain-specific LLM mutations
    └── EvolutionOrchestrator - Manages the closed-loop selection-mutation-evaluation flywheel
```

---

## **3. The GpuToJaxAgent 12-Step Conversion Pipeline**

The `GpuToJaxAgent` sub-agent executes a highly structured, multi-stage pipeline to convert GPU code (CUDA, Triton, or PyTorch CUDA) into validated JAX Pallas code:

1. **IdentifyFrameworkAgent:** Reads the GPU source file and identifies the framework (CUDA, Triton, or PyTorch CUDA), saving the detected framework to the agent state.
2. **AnalyzePlanAndWriteAgent:** Analyzes the GPU code, infers memory access patterns, creates a detailed simplification plan, and writes it to `SIMPLIFICATION_PLAN.md`.
3. **OrganizeGpuCodeAgent:** Simplifies the GPU code based on the approved plan, stripping hardware-specific optimizations, and writes it to a file with the appropriate extension.
4. **WriteSimplificationReadmeAgent:** Writes a README explaining the original code and the simplification steps taken.
5. **ConvertToJaxAgent:** Converts the organized code to JAX Pallas and writes it to `converted_jax.py`.
6. **ValidateSyntaxAgent:** Validates JAX syntax using `JaxSyntaxChecker` and routes to the fix or compilation stage based on results.
7. **FixConversionAgent:** Fixes syntax errors in the JAX conversion and writes the corrected code back to `converted_jax.py`.
8. **ValidateCompilationAgent:** Validates JAX compilation using `JaxCompilationChecker` (with automated server management) and proceeds to shape validation.
9. **ValidateShapesAgent:** Validates tensor shapes using `ShapeValidator` and proceeds to test generation.
10. **GenerateCorrectnessTestAgent:** Generates a validation test for the JAX code and writes it to `test_correctness.py`.
11. **RunCorrectnessTestAgent:** Runs the correctness test using `JaxCorrectnessChecker` (with automated server management) and routes based on results.
12. **GenerateSummaryAgent:** Generates a final summary of the conversion process, reporting compilation, syntax, shape, and correctness results.

---

## **4. Evolutionary Search Engine (AlphaEvolve Integration)**

To achieve state-of-the-art performance, MaxEvolve implements an evolutionary search loop (`EvolveKernelAgent`) that extends the `AutotuneAgent` beyond standard grid search.

### **4.1. Program Database & MAP-Elites**
* **Storage:** A structured JSON database stored in the `kernels/database/` directory.
* **Categorization:** Candidates are mapped onto a multi-dimensional grid (MAP-Elites) based on:
  * **Code Complexity:** Number of lines, AST depth.
  * **Register Pressure:** Estimated register usage or spilling.
  * **Performance (Fitness):** Execution latency or MFU.
* **Diversity Preservation:** By maintaining a diverse population of kernels across these dimensions, the search avoids converging prematurely on suboptimal local minima.

### **4.2. Mutation Operators**
The evolutionary engine prompts the LLM to apply highly targeted, domain-specific mutations to the scheduling parameters of the best-performing kernels:
* **Tiling Mutations:** Adjusting block sizes (e.g., mutating a tile size from `128x128` to `128x64` or `64x256`) to balance register pressure against compute intensity.
* **Pipelining Mutations (Mosaic GPU):**
  * Mutating `max_concurrent_steps` to overlap memory transfers (DMA) with compute (MXU/Tensor Cores).
  * Mutating `delay_release` to prevent data races during asynchronous WGMMA operations.
* **Layout Casting Mutations:** Casting references to `Layout.WGMMA` (for Hopper Tensor Cores) or `Layout.WG_STRIDED` (for vector units) to maximize instruction throughput.
* **Sparsity Mutations:** Refactoring dense masks into Block-Coordinate (Block-COO) formats and integrating `PrefetchScalarGridSpec` for asynchronous scalar prefetching.

### **4.3. Closed-Loop Hardware Evaluation**
1. **Selection:** A prompt sampler selects a high-performing kernel from the database.
2. **Mutation:** The LLM applies a mutation operator to generate a candidate.
3. **Correctness Filter:** The candidate is passed to the `ValidatedTestGenerationAgent`. If it fails numerical equivalence, it is immediately discarded.
4. **Performance Profiling:** If correct, the candidate is profiled by the `ProfileAgentOrchestrator` on actual hardware.
5. **Database Update:** The candidate's performance is recorded, and it is inserted into the MAP-Elites database, potentially replacing a slower kernel in its category.

---

## **5. Integration with the isanagent Harness**

MaxEvolve is designed to run natively within the `isanagent` workspace, leveraging its built-in tools and skills.

### **5.1. Mapping to isanagent Tools**
* **Sub-Agent Spawning (`subagent_spawn`):** Used to launch the specialized sub-agents in parallel or sequential pipelines.
* **Execution Harness (`execution_run_background`):** Used to run long-running evolutionary search loops and execute hardware-in-the-loop profiling on GPU/TPU environments.
* **Colab CLI (`colab-cli`):** For environments where local hardware is unavailable, the evolutionary loop can orchestrate remote execution on Google Colab T4/L4/A100 instances.
* **Web & arXiv Search (`web_search`, `arxiv_search`):** Used by the agents to fetch the latest JAX/Pallas documentation, API updates, and hardware-specific optimization papers.

### **5.2. Step-by-Step Execution Workflow**

```
+-----------------------------------------------------------------------------------+
| Step 1: Initialization                                                            |
| - User provides Triton/PyTorch kernel and target hardware.                        |
| - isanagent initializes the MAP-Elites database.                                  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Step 2: Agentic Translation (GpuToJaxAgent 12-Step Pipeline)                      |
| - Identify framework, analyze plan, simplify GPU code, and convert to JAX.        |
| - Validate syntax, compilation, and shapes.                                       |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Step 3: Numerical Verification                                                    |
| - ValidatedTestGenerationAgent runs pytest with interpret=True.                   |
| - If tests fail, GpuToJaxAgent's FixConversionAgent refactors the code.           |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Step 4: Evolutionary Optimization Loop (EvolveKernelAgent)                        |
| - isanagent launches execution_run_background for the evolutionary flywheel.      |
| - LLM mutates scheduling parameters (pipelining, layouts, tiling).                |
| - Candidates are compiled, validated, and profiled on actual hardware.            |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Step 5: Delivery & Handover                                                       |
| - The best-performing kernel is saved to the workspace.                           |
| - A comprehensive performance report (latency, MFU, speedup) is generated.        |
+-----------------------------------------------------------------------------------+
```

---

## **6. Technical Challenges & Mitigations**

### **6.1. Uncoalesced Gather Operations (Triton 3.x Regression)**
* **Challenge:** Naive translation of Triton pointer math in Pallas leads to uncoalesced scalar memory loads, causing a 5x-10x performance regression on newer Triton backends.
* **Mitigation:** The `GpuToJaxAgent` strictly forbids the generation of manual pointer offsets inside the kernel body. The `ImplementKernelAgent` is forced to use declarative `BlockSpec` objects, ensuring that the hardware DMA engine handles memory transfers in contiguous, coalesced blocks.

### **6.2. The Autodiff Transposition Paradox**
* **Challenge:** Automatically transposing a highly optimized forward-pass kernel (`jax.grad`) alters memory access patterns, converting fast parallel reads into slow, overlapping atomic writes.
* **Mitigation:** MaxEvolve does not rely on JAX's native automatic differentiation for custom kernels. The `ImplementKernelAgent` is instructed to generate a `jax.custom_vjp` wrapper, explicitly defining a manually engineered and optimized backward-pass kernel.

### **6.3. Dynamic Sparsity and Lexicographical Constraints**
* **Challenge:** Pallas's `BlockSpec` strictly enforces a sequential, lexicographical traversal through non-overlapping blocks, which struggles with dynamic sparsity (e.g., Top-K routing in Neural Sparse Attention).
* **Mitigation:** The framework integrates `PrefetchScalarGridSpec` to load sparse coordinates directly into Scalar Memory (SMEM) ahead of time. The `index_map` reads these coordinates dynamically, allowing the DMA engine to fetch disjoint blocks asynchronously in the background.

---

## **7. Phased Implementation Roadmap**

**Status (implemented in isanagent):**

| Phase | Scope | Status |
|-------|--------|--------|
| 1 | Named agents, skill, validators, plan-step `agent`, model/temperature at spawn | Done |
| 2 | MAP-Elites tools, evolve_runner.py, kernel_mutator / evolve_orchestrator | Done |
| 3 | Colab TPU profiling path, roofline_mfu.py, background evolution | Done |
| 4 | Benchmark ladder docs, REPORT template, user guide | Done |

See `docs/kernel-porting-user-guide.md` for operator instructions.

### **Phase 1: Foundation & Agent Setup (Weeks 1-2)**
* Author the system prompts and RAG pipelines for the specialized sub-agents (`PlanKernelAgent`, `GpuToJaxAgent`'s 12 steps, `ImplementKernelAgent`, `ValidatedTestGenerationAgent`).
* Implement the JAX CPU interpretation harness (`interpret=True`) for safe, local validation.

### **Phase 2: Evolutionary Engine & Database (Weeks 3-4)**
* Build the MAP-Elites program database structure in the workspace.
* Implement the mutation operators for tiling, pipelining, and layout casting.
* Integrate the evolutionary loop with `isanagent`'s `execution_run_background` tool.

### **Phase 3: Hardware Integration & Profiling (Weeks 5-6)**
* Connect the `ProfileAgentOrchestrator` to actual GPU/TPU hardware (or remote Colab instances via `colab-cli`).
* Implement automated roofline model calculations to measure MFU and memory bandwidth.

### **Phase 4: Validation & Production Deployment (Weeks 7-8)**
* Validate the complete MaxEvolve framework on complex kernels (e.g., FlashAttention, Block-Sparse Attention, Deepseek MLA).
* Generate comprehensive documentation and handover materials.
