# **From Triton to Pallas: Agentic, Evolutionary Kernel Porting from PyTorch to JAX**

The evolution of machine learning infrastructure is increasingly defined by the tension between high-level framework ergonomics and low-level hardware utilization. Frameworks such as PyTorch and JAX have successfully abstracted the immense complexity of neural network execution through just-in-time (JIT) compilation, automatic differentiation, and sophisticated intermediate representations powered by the Accelerated Linear Algebra (XLA) compiler1. For dense, highly regular computations, these compiler stacks achieve near-peak hardware utilization. However, as the field pivots toward novel, dynamic architectures—such as block-sparse attention mechanisms, Mixture of Experts (MoE), and ragged batch dimension processing—generic compiler heuristics frequently fail to map irregular memory access patterns optimally onto accelerator hardware, resulting in severely memory-bandwidth-bound execution3.  
To bridge this optimization gap, custom kernel languages have emerged, affording engineers fine-grained control over static RAM (SRAM) and High-Bandwidth Memory (HBM) interactions. OpenAI’s Triton has become the standard for the PyTorch ecosystem, heavily integrated into the torch.compile stack via TorchInductor6. Conversely, Pallas is JAX’s native extension for hardware-accelerated kernel programming, capable of lowering to Mosaic for Google Cloud Tensor Processing Units (TPUs) or to Triton for NVIDIA Graphics Processing Units (GPUs)8.  
Translating GPU kernels from Triton to JAX Pallas presents a formidable engineering bottleneck. The migration is not a simple syntactical translation; it demands a comprehensive paradigm shift from Triton’s imperative pointer arithmetic to Pallas’s declarative orchestration10. The complexity is further compounded by JAX’s functional purity constraints and its interaction with core transformations like vmap, jit, and pmap6. This report provides an exhaustive investigation into an agentic, evolutionary approach to migrating PyTorch and Triton kernels to JAX Pallas. By treating kernel porting not as manual engineering but as a guided, iterative search problem, AI-driven agents can automatically extract memory access patterns, validate numerical equivalence, and iteratively mutate scheduling parameters to optimize throughput13.

## **The Architectural Chasm: Why Triton-to-Pallas Porting is Difficult**

The difficulty of porting kernels between these two ecosystems stems from foundational differences in how each compiler models memory logistics and parallel execution. Modern ML accelerators possess staggering compute capabilities—a single TPU v5e can execute 197 TFLOP/s of BF16 compute—but their core computational engines, such as Matrix Multiply Units (MXUs) or Tensor Cores, must be continuously fed by local, high-speed memory3. Fetching data dynamically from the larger, slower HBM causes hardware stalls. Optimizing workloads requires bypassing the XLA safety net to explicitly manage the staging of memory blocks4.

### **Imperative Pointer Arithmetic vs. Declarative Block Specifications**

In Triton, developers imperatively compute memory offsets for each block using multi-dimensional pointer arithmetic within the kernel logic itself. A typical Triton matrix multiplication kernel manually calculates the starting pointer for a block of data, iterating over offsets and relying on aggressive compiler passes to recognize contiguous memory access patterns. Older versions of the Triton compiler (Triton 2.x) utilized heuristic passes that analyzed these tensors of pointers, recognized perfectly contiguous offsets, and silently optimized them into lightning-fast vectorized block loads11.  
Pallas explicitly eliminates in-kernel pointer arithmetic, moving the addressing logic into the higher-level orchestration layer10. Pallas functions do not accept raw JAX arrays; instead, they operate on pallas.Ref objects, which serve as explicit references to memory buffers4. To define how massive global tensors are sliced into blocks for each kernel instance, Pallas utilizes BlockSpec objects4. A BlockSpec requires a block\_shape and an index\_map—a function mapping the program's multidimensional grid coordinates to block indices16.  
This structural difference creates a significant translation trap. Attempting a naive, one-to-one translation of Triton pointer math directly into Pallas leads to catastrophic performance regressions. When Pallas generates explicit pointer offsets and lowers them to newer Triton backends (Triton 3.x), the compiler fails to trigger the deprecated contiguous-memory heuristics. As a result, the GPU executes thousands of scalar, uncoalesced memory loads (known as a "Gather" operation) instead of a single coalesced block load, devastating memory bandwidth and causing massive register spilling, leading to observed 5x to 10x performance regressions15.

### **Differing Grid Semantics and Functional Constraints**

Triton inherits much of its execution philosophy from CUDA, utilizing thread blocks and program IDs to dictate execution order and shared memory allocation10. Developers must explicitly synchronize these threads to avoid race conditions. Pallas, operating within the JAX ecosystem, enforces a single-program multiple-data (SPMD) paradigm mapped across multidimensional tuples known as grids16. The Pallas grid abstracts away thread-level synchronization by launching a programmatic instance for every element in the iteration space, executing kernels as isolated functional units that read from and write to predefined pallas.Ref buffers17.  
Furthermore, JAX arrays are inherently immutable, requiring state updates to occur via .at\[\] index mapping in standard JAX code. Pallas breaches this immutability intentionally by allowing in-place mutation of Ref objects, but this mutation must be strictly managed to ensure compatibility with JAX's broader functional expectations17.

### **The Interaction with JAX Transformations**

A core superpower of JAX is its composable function transformations. Pallas kernels must integrate seamlessly with these transformations, presenting unique challenges during the migration of Triton kernels1.  
The vmap transformation automatically vectorizes JAX programs, allowing operations written for single inputs to be mapped across a batch dimension12. When a pallas\_call is vectorized using vmap, JAX automatically augments the grid with an additional dimension corresponding to the new batch size6. It seamlessly transforms the BlockSpec mappings to handle indexing along this new batch dimension, achieving automated parallelization without requiring manual code rewrites6. Similarly, standard compilation via jax.jit functions elegantly, lowering the pallas\_call natively to the underlying accelerator6. Distributed execution transformations, such as pmap and shard\_map, also integrate with Pallas, enabling the scaling of custom kernels across heterogeneous clusters by managing collective communication and data sharding primitives explicitly6.  
However, reverse-mode automatic differentiation (jax.grad) represents a critical paradox in kernel porting. The jax.grad transformation decomposes into Jacobian-vector products (jvp), partial evaluation, and transposition rules6. Transposing a highly optimized forward-pass kernel fundamentally alters its memory access patterns. For example, a forward kernel featuring parallel, disjoint writes and overlapping reads, when automatically transposed for the backward pass, mutates into a kernel with parallel, disjoint reads and overlapping writes. Overlapping writes necessitate atomic operations, which severely degrade throughput and trigger memory contention6. Pallas does not currently possess a program representation amenable to automatic loop reordering for optimal transposition. Consequently, an automated porting system cannot rely on native autodiff; it must explicitly generate a jax.custom\_vjp to define an optimized, manually engineered backward pass5.

| Component / Mechanism | Triton Implementation | JAX Pallas Implementation | Migration Challenge |
| :---- | :---- | :---- | :---- |
| **Memory Loading** | Imperative pointer arithmetic (tt.load) | Declarative BlockSpec orchestration | Avoiding uncoalesced Gather operations15 |
| **Execution Model** | Thread blocks and warps | Multidimensional Tuple Grids | Refactoring synchronization paradigms |
| **Batch Processing** | Manual nested loops / Striding | Automatic jax.vmap grid augmentation | Resolving rank mismatches in legacy loops6 |
| **Automatic Differentiation** | PyTorch autograd integration | jax.grad transposition paradox | Mandates automated generation of custom\_vjp \[cite: 6, 20\] |
| **Memory Types** | Shared Memory (\_\_shared\_\_) | SRAM / VMEM / SMEM explicitly defined | Aligning hardware-specific byte boundaries3 |

## **Designing Agent Skills for Kernel Migration**

Because of the architectural discrepancies between Triton and Pallas, standard rule-based transpilers fail. A static syntax parser cannot infer the high-level geometric intent behind a block of low-level pointer arithmetic to refactor it into a BlockSpec, nor can it automatically balance pipeline parameters based on targeted hardware topologies10. To solve this, advanced Human-in-the-Loop (HITL) agentic frameworks, such as the open-source MaxKernel system, have been engineered. These AI-driven systems conceptualize kernel translation as a multi-stage reasoning and optimization pipeline managed by specialized sub-agents13.

### **Extracting Memory Access Patterns and Inferring Tiling**

Before generating code, an orchestration agent (frequently termed the PlanKernelAgent) analyzes the original Triton or CUDA source to infer the underlying mathematical operations. It formulates an optimization plan, extracting the implicit memory access patterns buried within the imperative code13. The agent infers tiling strategies tailored to the target hardware. For instance, when targeting Google Cloud TPUs, the agent recognizes that Matrix Multiply Units (MXUs) natively process tiles of 128x128 elements. It suggests block dimensions that are multiples of 128 to ensure maximal hardware occupancy, preventing the occurrence of padded, idle compute cycles16.

### **Rewriting Pointer Arithmetic into Structured Indexing**

Once the geometric plan is approved, the translation agent (GpuToJaxAgent) tackles the core syntactical migration. It automatically converts existing GPU kernel code into JAX by intelligently stripping out hardware-specific optimization artifacts—such as NVIDIA-specific warp-level synchronization or explicit shared memory staging—that are counterproductive in the Pallas orchestration layer13. The implementation agent (ImplementKernelAgent) then drafts the idiomatic Pallas code. Leveraging a Retrieval-Augmented Generation (RAG) pipeline connected to the latest JAX and Pallas documentation, it maps the extracted memory logic directly into BlockSpec objects, establishing a functional mapping between the loop iteration indices and the corresponding memory slices13.

### **Detecting Compilation Issues and Numerical Equivalence**

The generated code must be rigorously validated before deployment to prevent silent failures or catastrophic NaN explosions on the hardware. The validation agent (ValidatedTestGenerationAgent) generates extensive pytest suites to verify numerical equivalence between the original PyTorch model and the new Pallas implementation13. If compilation errors occur, the agent captures the tracebacks and iterates on the code. A critical skill in this phase is the deployment of JAX’s interpret=True debugging mode24. By running the Pallas call through a JAX interpretation path on the CPU, the agent can simulate the shared memory environments (HBM, VMEM) and aggressively detect race conditions or out-of-bounds memory accesses before the code ever touches an accelerator24.

| Agent Specialization | Primary Skills | Role in Kernel Migration Workflow |
| :---- | :---- | :---- |
| **PlanKernelAgent** | Tiling inference, architecture mapping | Extracts memory patterns and proposes hardware-aligned block geometries13. |
| **GpuToJaxAgent** | CUDA artifact removal, syntactical parsing | Strips Triton/CUDA specific pointer math and warp-synchronization logic13. |
| **ImplementKernelAgent** | RAG-assisted code drafting | Translates logic into declarative BlockSpec and pallas\_call structures13. |
| **ValidatedTestGenerationAgent** | Pytest generation, OOB detection | Evaluates numerical equivalence and uses interpret=True to catch memory bounds errors13. |
| **ProfileAgentOrchestrator** | DMA analysis, MFU calculation | Profiles JIT-compiled kernels on-device to measure memory vs. compute ratios13. |

## **The Evolutionary Workflow: Guided Optimization**

While the multi-agent pipeline can reliably generate a functionally correct Pallas kernel, achieving state-of-the-art performance—parity with or outperforming hand-tuned Triton—requires exploring an immense hyperparameter space. Real-world performance is heavily dictated by interacting variables such as pipeline overlaps, register usage, and DMA transfer rates. Static code generation cannot predict these nonlinear hardware dynamics. Therefore, the optimization process is framed as an evolutionary search problem, spearheaded by algorithmic discovery frameworks such as AlphaEvolve14.

### **Beyond Zero-Shot Prompting**

Standard Large Language Models (LLMs) often fail to produce optimal, bug-free low-level kernels on the first attempt due to the fragility of hardware constraints. AlphaEvolve mitigates this limitation by orchestrating a continuously running flywheel of code generation, execution, and feedback, transforming code optimization into a biological evolutionary process27. This approach substantially enhances the capabilities of base LLMs by grounding their output entirely in empirical execution metrics29.  
The workflow begins by establishing a program database of candidate Pallas kernels. The system utilizes Quality-Diversity algorithms, such as MAP-Elites, to maintain a diverse population of solutions spanning the fitness landscape14. Programs are categorized by inherent metrics—such as code complexity, register spilling, and instruction length—and scored by a deterministic fitness function14. Maintaining genetic diversity prevents the search algorithm from converging prematurely on suboptimal local minima, allowing it to explore radically different algorithmic approaches.

### **Iterative Mutation of Scheduling Parameters**

A prompt sampler selects high-performing kernels from the database and constructs context-rich prompts. The LLM is instructed to apply domain-specific mutations. The sampler balances exploitation (refining the best programs) with exploration (mutating diverse programs to escape local optima) and recombination (merging concepts from multiple high-performing kernels)14.  
In the context of Pallas, these mutations target the most sensitive scheduling parameters. When targeting the Mosaic GPU backend, explicit pipelining is controlled via the plgpu.emit\_pipeline API30. The agent iteratively mutates configurations such as max\_concurrent\_steps (the maximum number of sequential stages active concurrently) and delay\_release (iterations to wait before reusing a buffer to prevent data races)21. It explores the parameter space of num\_compute\_wgs (warpgroups) to balance compute capacity against memory thread allocation, ensuring that neither register spilling nor pipeline bubbling degrades overall throughput30.

### **Hardware-in-the-Loop Evaluation**

Because static analysis cannot accurately predict cycle-level stalls caused by memory contention, candidates generated by the LLM are immediately JIT-compiled via XLA and benchmarked on actual TPU or GPU hardware14. This hardware-in-the-loop evaluator rejects candidates that fail robust functional correctness tests, immediately discarding any mutations that break numerical parity. For the surviving candidates, the evaluator measures raw execution latency and Model Flops Utilization (MFU)13. The top-ranked programs are reinserted into the evolutionary database, and the cycle continues autonomously27.

## **Lessons from Large-Scale PyTorch-to-JAX Migrations**

The practical application of these agentic frameworks has yielded significant improvements in large-scale machine learning systems, successfully mitigating memory bottlenecks, resolving critical crashes, and discovering non-intuitive heuristics that accelerate production workloads.

### **Case Study: Block-Sparse Attention and Scalar Prefetching**

Standard attention mechanisms face a quadratic memory and compute bottleneck. To implement Block-Sparse Attention efficiently, operations must skip computations for blocks explicitly zeroed out by a sparse routing mask3. Standard XLA heuristics fail in this scenario because issuing on-the-fly, unpredictable memory fetches stalls the MXUs. A human attempting to write this in Triton might use complex boolean masking over massive matrices, which results in unnecessary memory traffic3.  
During a migration to Pallas, the agentic system identifies the sparsity pattern and deploys a highly specialized TPU feature: PrefetchScalarGridSpec3. Instead of utilizing a massive boolean matrix, the agent refactors the routing mask into a Block-Coordinate (Block-COO) format. The generated Pallas kernel pre-loads these small coordinates directly into Scalar Memory (SMEM) immediately before the main compute pipeline begins3. The dynamically generated index\_map then utilizes these scalar values to instruct the hardware's DMA engine exactly which blocks of the Query and Key matrices to fetch from HBM ahead of time. This guarantees that the memory transfers occur asynchronously in the background while the MXU is busy performing math on the previous block, entirely resolving the memory bandwidth bottleneck3.

### **Case Study: Tiling Heuristics and Deepseek MLA**

Evolutionary search routinely discovers hardware heuristics that human engineers overlook. Traditional engineering relies on static rules of thumb, but agentic search empirically determines optimal shapes based on dynamic input configurations14. When AlphaEvolve was deployed to optimize tiling heuristics for the critical matrix multiplication kernels used to train Google's Gemini LLMs, it collected real kernel invocation shapes from the training pipeline (e.g., embedding lookups, attention projections)14. The system evolved custom TileConfig heuristics that dynamically altered block dimensions based on sequence length boundaries, achieving an average kernel speedup of 23% and contributing to a 1% reduction in overall Gemini training time14.  
Similarly, during the optimization of the Deepseek Multi-Head Latent Attention (MLA) kernel for inference on the v5p TPU platform, the MaxKernel agent achieved an 8.7% latency reduction (dropping from 3.12 ms to 2.856 ms) and a 9% increase in throughput (climbing from 116.73 TFLOPS to 127.82 TFLOPS) compared to a highly optimized, human-written baseline Pallas kernel13.

| Workload / Model | Optimization Applied | Performance Outcome |
| :---- | :---- | :---- |
| **Deepseek MLA Inference** | Agentic hardware-aware tuning | 8.7% latency reduction, 9% throughput increase13. |
| **Gemini Matrix Multiplications** | Evolutionary tiling heuristics | 23% kernel speedup, 1% total training time reduction14. |
| **Fully Homomorphic Encryption (FHE)** | Evolution of FHE TPU primitives | 2.5x speedup for TFHE bootstrap latency27. |

### **Surprising Failure Modes and Debugging Strategies**

Deploying agentic systems is not without friction. LLMs are prone to hallucinating hardware-incompatible memory access patterns if left unchecked. A common failure mode during migration is the generation of out-of-bounds index mappings due to incorrect stride assumptions pulled from PyTorch layouts. This results in silent NaN explosions or catastrophic segmentation faults on the hardware4.  
To counter this, robust debugging strategies are strictly enforced. The use of the interpret=True parameter in pallas\_call is critical. This mode runs the Pallas kernel through a sequential JAX interpretation path on the CPU, padding out-of-bounds floating-point reads with NaN values, which explicitly flags indexing errors before the code is lowered to LLO or PTX16.  
Furthermore, agents have demonstrated the ability to resolve complex state-machine crashes autonomously. In one instance involving the RPAv3 Kernel, unpadded inputs during the prefill phase caused persistent crashes. The agent correctly diagnosed the DMA out-of-bounds errors, implemented deadlock-prevention logic, integrated defensive clamping to handle edge cases, and secured the correct processing of chunked prefill sizes with negligible performance impact13. By grounding the agent in scoped file-system constraints and enforcing rigid trace-analysis feedback loops, the risk of deploying hallucinated kernels is entirely mitigated.

## **Presentation Plan: A Blueprint for Agent-Assisted Workflows**

To effectively convey these complex engineering concepts to an audience of AI infrastructure developers and ML researchers, the following presentation plan details a structured, narrative-driven blueprint for the accompanying technical talk.

### **Slide 1: Title and Abstract**

* **Title:** From Triton to Pallas: Agentic, Evolutionary Kernel Porting from PyTorch to JAX.  
* **Content:**  
  * Overview of the speaker's background and the core premise.  
  * Transitioning from manual kernel engineering to automated search space exploration.  
* **Speaker Notes:** "Welcome. Today we are discussing a paradigm shift in ML infrastructure. Writing custom GPU and TPU kernels is notoriously difficult, and porting them between ecosystems like PyTorch and JAX is even harder. What if we stop treating this as a manual translation problem and start treating it as an iterative, AI-driven optimization problem?"

### **Slide 2: The Limits of the Compiler Safety Net**

* **Title:** Why We Write Custom Kernels.  
* **Content:**  
  * XLA and TorchInductor provide near-perfect performance for standard dense operations.  
  * The breakdown occurs during irregular compute (MoE), ragged batches, and sparse matrices.  
  * The memory bottleneck: MXUs stall while waiting for High-Bandwidth Memory (HBM).  
* **Speaker Notes:** "Standard compilers are incredible tools, but they fail when workloads become irregular. Our compute capability is scaling much faster than our memory bandwidth. To keep our Matrix Multiply Units fed, we have to bypass XLA's safety net and take explicit control of how data moves from main memory to local SRAM."

### **Slide 3: The Architectural Chasm**

* **Title:** Triton vs. Pallas: Imperative vs. Declarative.  
* **Content:**  
  * **Triton:** Heavily relies on thread-level coordination and explicit pointer arithmetic (tt.load(ptrs)).  
  * **Pallas:** Enforces declarative orchestration. BlockSpec handles dicing tensors into SRAM tiles via an index\_map.  
  * **The Translation Trap:** Generating Triton-style pointer math in Pallas destroys memory coalescing, leading to massive register spilling and 5x-10x performance regressions.  
* **Speaker Notes:** "You cannot just run a regex script to convert Triton to Pallas. If you attempt to force Triton's imperative pointer math into Pallas, the compiler will execute thousands of uncoalesced scalar loads instead of a clean block load. You must fundamentally rewrite the logic."

### **Slide 4: JAX Transformations and the AD Paradox**

* **Title:** Integrating with the JAX Ecosystem.  
* **Content:**  
  * jax.vmap: Automatically vectorizes grids and block specs for batching.  
  * jax.grad: The autodiff paradox. Transposing optimized forward-pass kernels creates slow, overlapping atomic writes.  
  * The mandatory requirement to engineer jax.custom\_vjp for backward passes.  
* **Speaker Notes:** "JAX transformations are powerful, but they complicate kernel writing. Vectorization works beautifully out of the box, but automatic differentiation does not. Transposing a fast read creates a slow atomic write, forcing us to manually author custom backward passes for almost every kernel."

### **Slide 5: Enter the Agent**

* **Title:** A Hierarchy of Kernel Translation Skills.  
* **Content:**  
  * PlanKernelAgent: Analyzes shapes and hardware topology to propose tiling strategies.  
  * GpuToJaxAgent: Strips out CUDA-specific synchronization artifacts.  
  * ImplementKernelAgent: Drafts Pallas code using RAG-assisted documentation retrieval.  
  * ValidatedTestGenerationAgent: Writes pytest equivalence tests.  
* **Speaker Notes:** "To automate this, we use multi-agent systems like MaxKernel. We break the problem down into distinct skills: one agent plans the memory geometry, another strips out the CUDA artifacts, a third writes the Pallas logic, and a fourth validates the math."

### **Slide 6: Escaping Local Minima with Evolution**

* **Title:** AlphaEvolve and Hardware-in-the-Loop Search.  
* **Content:**  
  * Zero-shot prompting fails due to strict hardware boundaries.  
  * Implementation of MAP-Elites Program Databases to maintain genetic diversity.  
  * LLMs mutate scheduling parameters iteratively (e.g., max\_concurrent\_steps, delay\_release).  
* **Speaker Notes:** "LLMs are creative, but they are bad at hardware constraints. We solve this by wrapping the LLM in an evolutionary algorithm. We maintain a diverse database of candidate kernels, prompt the LLM to mutate their scheduling parameters, and evaluate them dynamically."

### **Slide 7: Grounding AI in Physical Reality**

* **Title:** The Hardware-in-the-Loop Evaluator.  
* **Content:**  
  * Real-time JIT compilation on actual GPU/TPU silicon.  
  * Filtering silent failures via interpret=True CPU execution to detect OOB memory access.  
  * Scoring survival based on Model Flops Utilization (MFU) and raw latency.  
* **Speaker Notes:** "The fitness function must be grounded in reality. The generated kernels are compiled and executed on actual silicon. If they trigger an out-of-bounds error, they are killed. If they survive, their throughput determines if they breed the next generation of optimizations."

### **Slide 8: Case Study: Sparse Attention and Scalar Prefetch**

* **Title:** Discovering Hardware-Specific Paradigms.  
* **Content:**  
  * **The Problem:** ![][image1] memory loads in standard attention.  
  * **The Discovery:** The agent deploys the TPU PrefetchScalarGridSpec.  
  * **The Mechanism:** Converts masks to Block-COO format, feeding scalar coordinates to the DMA forklift for asynchronous background loading.  
* **Speaker Notes:** "Here is a real example. For block-sparse attention, the agent correctly identified that it could avoid massive boolean masks by utilizing a TPU-specific feature called scalar prefetching. It routed the DMA engine asynchronously, entirely removing the memory bottleneck."

### **Slide 9: Case Study: Superhuman Tiling Heuristics**

* **Title:** Outperforming Hand-Tuned Kernels.  
* **Content:**  
  * Gemini Training: 23% kernel speedup, 1% total training time reduction.  
  * Deepseek MLA Inference: 8.7% latency reduction, 9% throughput increase.  
  * Agents discover dynamic, sequence-length-dependent tiling configurations that human engineers lack the time to manually hardcode.  
* **Speaker Notes:** "The evolutionary search isn't just matching human baselines; it is surpassing them. By evolving shape-dependent heuristics for Gemini and Deepseek inference, these agents are shaving milliseconds off every forward pass, which translates to massive cost savings at datacenter scale."

### **Slide 10: Building Your Own Agentic Workflow**

* **Title:** A Blueprint for the Future of Infrastructure.  
* **Content:**  
  * Establish your baseline XLA profiles.  
  * Implement strict debugging harnesses using JAX interpretation modes and Perfetto traces.  
  * Deploy open-source agent frameworks to automate your tuning scripts.  
* **Speaker Notes:** "You can implement this in your own workflows today. Build strict validation harnesses, utilize JAX's interpretation mode to catch silent bugs, and let the evolutionary search handle the parameter tuning. The future of kernel engineering is guided optimization."

#### **Works cited**

1. JAX, XLA, and Pallas — MaxText documentation \- Read the Docs, [https://maxtext.readthedocs.io/en/latest/reference/core\_concepts/jax\_xla\_and\_pallas.html](https://maxtext.readthedocs.io/en/latest/reference/core_concepts/jax_xla_and_pallas.html)  
2. PolyBlocks: A Compiler Infrastructure for AI Chips and Programming Frameworks \- arXiv, [https://arxiv.org/html/2603.06731v1](https://arxiv.org/html/2603.06731v1)  
3. Breaking the O(N^2) Bottleneck: Implementing High-Performance Block-Sparse Attention with JAX/Pallas \- Hugging Face, [https://huggingface.co/blog/rishiraj/block-sparse-attention-with-jaxpallas](https://huggingface.co/blog/rishiraj/block-sparse-attention-with-jaxpallas)  
4. Writing Pallas Kernels for JAX: Stepping Outside the XLA Safety Net \- Rajat Pandit, [https://rajatpandit.com/ai-infrastructure/writing-pallas-kernels-for-jax/](https://rajatpandit.com/ai-infrastructure/writing-pallas-kernels-for-jax/)  
5. Optimizing with Pallas kernels — MaxText documentation \- Read the Docs, [https://maxtext.readthedocs.io/en/latest/guides/optimization/pallas\_kernels\_performance.html](https://maxtext.readthedocs.io/en/latest/guides/optimization/pallas_kernels_performance.html)  
6. Pallas Design \- JAX documentation, [https://docs.jax.dev/en/latest/pallas/design/design.html](https://docs.jax.dev/en/latest/pallas/design/design.html)  
7. Deep Learning in Practice: A Technical Comparison of PyTorch and JAX | by Nijesh Kanjinghat | Medium, [https://medium.com/@nijesh-kanjinghat/deep-learning-in-practice-a-technical-comparison-of-pytorch-and-jax-6458a115dcde](https://medium.com/@nijesh-kanjinghat/deep-learning-in-practice-a-technical-comparison-of-pytorch-and-jax-6458a115dcde)  
8. Pallas: a JAX kernel language, [https://docs.jax.dev/en/latest/pallas/index.html](https://docs.jax.dev/en/latest/pallas/index.html)  
9. The Rise of Pallas: Unlocking TPU Potential with Custom Kernels \- Medium, [https://medium.com/data-science/the-rise-of-pallas-unlocking-tpu-potential-with-custom-kernels-67be10ab846a](https://medium.com/data-science/the-rise-of-pallas-unlocking-tpu-potential-with-custom-kernels-67be10ab846a)  
10. Unlocking Kernel-Level Optimizations on TPUs using Pallas: A Getting Started Guide, [https://medium.com/@engineerbharath/unlocking-kernel-level-optimizations-on-tpus-using-pallas-a-getting-started-guide-ae47a3ad5bb1](https://medium.com/@engineerbharath/unlocking-kernel-level-optimizations-on-tpus-using-pallas-a-getting-started-guide-ae47a3ad5bb1)  
11. triton/python/tutorials/03-matrix-multiplication.py at main \- GitHub, [https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py](https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py)  
12. Automatic vectorization \- JAX documentation, [https://docs.jax.dev/en/latest/automatic-vectorization.html](https://docs.jax.dev/en/latest/automatic-vectorization.html)  
13. MaxKernel: Automating Pallas Kernel Generation and Optimization via Agentic Systems, [https://discuss.google.dev/t/maxkernel-automating-pallas-kernel-generation-and-optimization-via-agentic-systems/366686](https://discuss.google.dev/t/maxkernel-automating-pallas-kernel-generation-and-optimization-via-agentic-systems/366686)  
14. LLM-Guided Kernel Optimization \- Yet Another AI Blog, [https://mlai.blog/2025-12-20-llm-kernel-optimization](https://mlai.blog/2025-12-20-llm-kernel-optimization)  
15. Severe (5-10x) performance regression in Triton kernel via JAX/Pallas: Triton 2.x vs 3.6 · Issue \#9640 \- GitHub, [https://github.com/triton-lang/triton/issues/9640](https://github.com/triton-lang/triton/issues/9640)  
16. Grids and BlockSpecs \- JAX documentation, [https://docs.jax.dev/en/latest/pallas/grid\_blockspec.html](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)  
17. Pallas Quickstart \- JAX documentation, [https://docs.jax.dev/en/latest/pallas/quickstart.html](https://docs.jax.dev/en/latest/pallas/quickstart.html)  
18. Quickstart: How to think in JAX \- JAX documentation, [https://docs.jax.dev/en/latest/quickstart.html](https://docs.jax.dev/en/latest/quickstart.html)  
19. Thinking in Pallas \- Sharded MatMuls \- Consider the Bulldog, [https://considerthebulldog.com/pallas-sharded-matmuls/](https://considerthebulldog.com/pallas-sharded-matmuls/)  
20. gist:fc40a161b4081b2828adac41070c3c40 \- GitHub, [https://gist.github.com/vanbasten23/fc40a161b4081b2828adac41070c3c40](https://gist.github.com/vanbasten23/fc40a161b4081b2828adac41070c3c40)  
21. jax.experimental.pallas.mosaic\_gpu.CompilerParams \- JAX documentation, [https://docs.jax.dev/en/latest/\_autosummary/jax.experimental.pallas.mosaic\_gpu.CompilerParams.html](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.mosaic_gpu.CompilerParams.html)  
22. AI-Hypercomputer/accelerator-agents \- GitHub, [https://github.com/AI-Hypercomputer/accelerator-agents](https://github.com/AI-Hypercomputer/accelerator-agents)  
23. Define a Custom TPU/GPU Kernel \- Keras, [https://keras.io/guides/define\_custom\_kernel/](https://keras.io/guides/define_custom_kernel/)  
24. Pallas for people who know JAX but not kernels yet \- Hugging Face, [https://huggingface.co/blog/ariG23498/pallas-for-beginners](https://huggingface.co/blog/ariG23498/pallas-for-beginners)  
25. jax.experimental.pallas.pallas\_call \- JAX documentation, [https://docs.jax.dev/en/latest/\_autosummary/jax.experimental.pallas.pallas\_call.html](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.pallas_call.html)  
26. jax.experimental.pallas.tpu.InterpretParams \- JAX documentation, [https://docs.jax.dev/en/latest/\_autosummary/jax.experimental.pallas.tpu.InterpretParams.html](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.InterpretParams.html)  
27. Adapting AlphaEvolve to Optimize Fully Homomorphic Encryption on TPUs \- arXiv, [https://arxiv.org/html/2605.14718v1](https://arxiv.org/html/2605.14718v1)  
28. AlphaEvolve: A coding agent for scientific and algorithmic discovery \- Googleapis.com, [https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)  
29. (PDF) AlphaEvolve: A coding agent for scientific and algorithmic discovery \- ResearchGate, [https://www.researchgate.net/publication/392736011\_AlphaEvolve\_A\_coding\_agent\_for\_scientific\_and\_algorithmic\_discovery](https://www.researchgate.net/publication/392736011_AlphaEvolve_A_coding_agent_for_scientific_and_algorithmic_discovery)  
30. Mosaic GPU Pipelining \- JAX documentation, [https://docs.jax.dev/en/latest/pallas/gpu/pipelining.html](https://docs.jax.dev/en/latest/pallas/gpu/pipelining.html)  
31. jax.experimental.pallas.mosaic\_gpu.emit\_pipeline\_warp\_specialized \- JAX documentation, [https://docs.jax.dev/en/latest/\_autosummary/jax.experimental.pallas.mosaic\_gpu.emit\_pipeline\_warp\_specialized.html](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized.html)  
32. Google: AlphaEvolve \- Next-Generation Algorithm Design Leveraging Gemini \- note, [https://note.com/repkuririn7/n/nc082aab192ff?hl=en](https://note.com/repkuririn7/n/nc082aab192ff?hl=en)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAaCAYAAAAJ1SQgAAADPUlEQVR4Xu2XS6hOURTHl1De7zyipAyIQkYeg6soJRRmzAwMyEAhE92JofJK8kgG8shEIWEgA5SR8hipSyRJSpE363fXWX37W985557ru9+9Bt+v/n1nP75z9lp77bX3FmnzXzFYtUZ1XHVENai+eWAZqxoaK5ugS7Utex6Slad7o9j3+gQGPUU1TTUqtOWxQ3VU+tbYd6qDSfmzanFSvqrakJR7DaHyQPVF9Ur1WvVTtSztlMMj1eRYqYwTC0Pe9UK1X+odMkwsRM+oTmbandWPEAtlYFyMZVZWhpmqx6qNSV1lZoh5a73UPgITVL+k+KWzVatiZcZIsXX3RvVH9U21JGknPDvEnPBEtVW1VBrX5zzVnpz6LarnYoZXZqGY95mhPBjIR9WiUM/HD6mGh/rIfdUzMYOPSeOgJ6nmhzqHtXlb6ifAoe2h2BjiO3O5JzYIvFTEWrE+OGN8Us9afZuUi9gslgOYBd7D/1JojzB4+m3KnrerFtT1MHASE+HJrBQ+jnfKspsb+1IsaQHr6rrqjncqAOfMzZ47pfY9B0POJmWHsD0lZiy6JeawCO9nEq7EhggLPs/TkV3SaCy/lE97pwIw1KPBZ+F3rbm77UZSdj6JfdOFU8kBEZx1XvU0NkQIHz5etF4cPB/DmG2A7WCvdyogDVES0gWxd/EMvIc12QwHVB9iZQr7510xr/S0uJkJBrguqfPQ5reIvFnzpNIptRBuar8Uc3gaLQ24sedCfYQB5a3rKsYSwtFYYNmQrOaItfua/lcwlrEU4gmmJ2MJM/bZuJdWMZYQzlvTnisOq66pRtc39xqM5SBUCrGOwRieB7NPFszb0JerfoglryI4HeVtK+CJhz7NgrHkj1I8RC9KfaYbIzbjX1U7k/oUDgJkQNZ8ZKJqn+q9akVoc0hUJJVmQ7jqFtgN64cQYBthpi+pvqsuS/1ZNOLJJaZ8Tjo+a668/XG1WNRUuWiUwU2oS2zslcA7HWIhx2/VAZCdOe8OJCvF9uT0zN0SuOlwbh4oiC7O2jel5/N5n8CZuizcWwm3IS4wcadoKay9oitgq+B7JKayM31LmKo6IXYn7i+41vW7oW3atGk9fwEp36S0X6yxtQAAAABJRU5ErkJggg==>