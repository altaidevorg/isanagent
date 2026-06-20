# **Advanced Hardware-Software Co-Design: A Comprehensive Guide to Converting from Triton to Pallas**

## **Introduction to the Hardware-Software Co-Design Paradigm**

As the computational requirements for modern machine learning, particularly the training and inference of large language models (LLMs), continue to scale exponentially, the gap between high-level algorithmic expression and low-level hardware execution has become a central bottleneck. High-level mathematical frameworks, such as standard PyTorch and Just-In-Time (JIT) compiled JAX, offer highly ergonomic array-processing environments. However, they frequently fail to extract peak computational performance from underlying artificial intelligence accelerators. This operational inefficiency primarily stems from the "memory wall"—the physical hardware limitation wherein High-Bandwidth Memory (HBM) data transfer rates lag significantly behind the arithmetic throughput of localized Matrix Multiply Units (MXUs) and Tensor Cores1. Standard compilers, including standard XLA (Accelerated Linear Algebra), generally process computations sequentially across bulk arrays, often unnecessarily routing intermediate tensor data back to distant HBM, which severely diminishes arithmetic intensity2.  
To circumvent these fundamental hardware limitations, the industry relies increasingly on specialized, low-level kernel programming languages. OpenAI's Triton emerged as a dominant domain-specific language (DSL) for NVIDIA Graphics Processing Units (GPUs), providing a critical abstraction layer just above raw CUDA C++3. Triton allows developers to write block-level, Single-Program Multiple-Data (SPMD) Python code that a custom compiler lowers into highly optimized Parallel Thread Execution (PTX) instruction sets4. It abstracts away thread-level synchronization while exposing block-level memory management.  
Simultaneously, the Google JAX ecosystem introduced Pallas, an advanced extension designed to bring Triton-like block-level control to both NVIDIA GPUs and Google Tensor Processing Units (TPUs)3. Pallas operates by combining the functional, pure-array philosophy of the JAX framework with the explicit, granular hardware memory management required for optimal kernel development4. While Triton traditionally mandates manual pointer arithmetic and explicit memory loading strategies, Pallas introduces a "software-defined architecture." It utilizes declarative grid mappings that cleanly separate the orchestration of memory transfers from the core mathematical logic executing on the chip2.  
This report provides an exhaustive analysis of the conversion pathways, philosophical distinctions, grid shape specifications, lowering logic, and hardware communication protocols distinguishing Triton and Pallas. It is designed to equip developers and systems researchers with the architectural knowledge required to port algorithms from Triton to Pallas, optimizing for both GPU and TPU topologies.

## **General Philosophy and Programming Models**

The foundational philosophy of a kernel language dictates how the programmer interacts with the hardware's memory hierarchy, arithmetic logic units, and execution scheduler. Triton and Pallas approach this multi-dimensional problem from fundamentally different conceptual angles, despite both leveraging Python as their high-level host language.

### **The Triton Philosophy: Imperative Pointer Manipulation**

Triton is engineered to sit exactly one abstraction layer above native CUDA programming. It models parallel execution around blocks of threads and forces the programmer to explicitly compute memory addresses using pointer arithmetic7. The philosophical core of Triton assumes that the developer must explicitly define the specific offsets for every tensor element loaded from global device memory into localized shared memory9.  
In a standard Triton kernel architecture, the programmer utilizes internal primitives to determine the current execution block's spatial coordinates. This is typically achieved by calculating the base pointer for a tensor, calculating the specific offset range based on the block index, and executing a load operation combined with a boolean mask to prevent catastrophic out-of-bounds memory accesses9. This imperative style provides immense, granular flexibility, allowing for arbitrary memory gathers and scatters, but it couples the memory addressing logic tightly with the actual mathematical operations. The Triton compiler is subsequently tasked with analyzing these manual pointer calculations and utilizing complex heuristic compiler passes to optimize them into coalesced, efficient block loads5.

### **The Pallas Philosophy: Declarative Tiling and Reference Types**

Pallas, conversely, embeds itself deeply within the pure functional ecosystem of JAX4. Rather than treating memory as an array of raw integer pointers requiring arithmetic tracking, Pallas introduces the concept of Reference types, denoted as Ref4. A Ref represents a mutable buffer residing in physical memory, mapping directly to physical layouts in Static Random-Access Memory (SRAM), Virtual Memory (VMEM) on TPUs, or Shared Memory (SMEM) on GPUs4.  
The primary philosophical departure in Pallas is the systematic eradication of manual pointer arithmetic inside the active kernel body. Pallas forces a rigorous separation of computational concerns into two distinct domains: the kernel logic and the orchestration layer. The kernel logic is the Python function executed directly on the accelerator core. It receives Ref objects, reads from them using standard array indexing to generate temporary, immutable jax.Array objects, performs the necessary computations, and writes the results back to an output Ref7. The orchestration layer is managed by the pallas\_call wrapper, which dictates exactly how massive global tensors residing in HBM are diced into manageable blocks and fed to the kernel grid2.  
By completely removing pointer mathematics from the kernel definition, Pallas allows the underlying hardware Direct Memory Access (DMA) engine to deterministically anticipate memory access patterns long before the mathematical execution begins2. This predictability enables seamless, automated prefetching and advanced software pipelining2. The developer declares the mathematical shapes and topological mappings, and the underlying compiler handles the physical mapping to the hardware. Furthermore, Pallas introduces an interpret mode (interpret=True), which allows the compiler to run the kernel on a standard CPU by emulating the execution grid as a jax.lax.scan loop, providing a robust environment for debugging without requiring immediate accelerator access4.

### **High-Level Comparison of Hardware Paradigms**

To contextualize the architectural distinctions between these frameworks, the following table compares their core features, hardware targets, and memory management paradigms.

| Architectural Feature | Triton Paradigm | Pallas Paradigm |
| :---- | :---- | :---- |
| **Host Ecosystem** | PyTorch / Standalone | JAX |
| **Primary Hardware Targets** | NVIDIA GPUs, AMD GPUs | Google TPUs, NVIDIA GPUs |
| **Addressing Model** | Imperative pointer arithmetic (tl.load) | Declarative orchestration (BlockSpec) |
| **Memory Representation** | Raw Memory Pointers | Ref objects (mutable arrays) |
| **Out-of-bounds Handling** | Manual Boolean masks (mask=offsets \< length) | Automated compiler padding / discard |
| **State Mutability** | In-place modification via pointer overrides | Read from Ref to jax.Array, compute, write back |
| **Vectorization Concept** | Implicit block vectorization | Explicit bulk array operations compiled to vector math |

## **Translating Concepts: The Triton to Pallas Conversion Guide**

Converting a highly optimized Triton kernel to a Pallas kernel requires a fundamental paradigm shift in how the developer approaches memory iteration and bounds checking. The translation process involves dismantling the explicit pointer mathematics of Triton and reassembling the logic into Pallas's declarative BlockSpec format.

### **Step 1: Eliminating Pointer Mathematics and Masking**

In Triton, fetching a block of data requires defining the block's starting point and generating a sequence of offsets. A developer typically utilizes tl.program\_id(axis) multiplied by a constant block size, followed by tl.arange to generate the sequence9. Because tensors are rarely perfectly divisible by the execution block size, the developer must also generate a boolean mask to prevent segmentation faults during memory reads9.  
When converting to Pallas, all of this logic is deleted. Pallas kernels do not require masking because the orchestration layer inherently understands the global tensor boundaries14. If a block shape does not evenly divide the global tensor shape, the final execution iteration still receives a reference to a perfectly sized block. The Pallas compiler automatically pads the out-of-bounds elements on the input side (often with garbage values or NaNs in interpretation mode) and safely discards the out-of-bounds outputs upon writing back to global memory14. Consequently, the developer can assume that all inputs provided to the Pallas kernel are valid, dense blocks ready for immediate computation.

### **Step 2: Translating Memory Operations**

Triton relies on explicit functions such as tl.load and tl.store to move data between HBM and SMEM9. In Pallas, memory loading is achieved seamlessly through reference indexing. An input pointer in Triton becomes an input Ref in Pallas. The Pallas developer loads the data into local registers by executing standard slice indexing, such as x \= x\_ref\[...\], which yields a standard jax.Array7.  
For advanced, dynamic addressing where standard slice indexing is insufficient, Pallas provides pallas.load and pallas.store4. These primitives are closer to Triton's paradigm and allow for complex, non-contiguous memory access. However, their use is generally discouraged in favor of pure BlockSpec orchestration unless the algorithm strictly requires dynamic, runtime-computed slicing4.

### **Step 3: Refactoring Computations**

Because Pallas executes within the JAX environment, the computational logic inside the kernel relies heavily on jax.numpy (abbreviated as jnp)16. Triton operations, such as tl.dot, translate almost directly to Pallas equivalents, such as pl.dot, which specifically leverages the hardware's specialized matrix units (MXUs on TPU or Tensor Cores on GPU)16. However, developers must be acutely aware of precision constraints. Matrix multiplications utilizing pl.dot on a TPU generally default to producing float32 output in the MXU12. Inputs passed as 32-bit operands will often be silently down-casted and rounded to bfloat16 during computation to maximize MXU throughput, unless strict float32 execution is explicitly requested via configuration flags12.  
Furthermore, standard JAX array operations natively decompose into vectorized instructions. If a developer needs to perform an elementwise activation function, such as a ReLU, they simply apply jnp.maximum(acc, 0\) directly to the accumulated jax.Array while it resides in local memory16. This represents a significant ergonomic upgrade over manual vectorized iteration.

## **Grid Shape Specifications and Data Partitioning**

Both frameworks rely heavily on an SPMD execution model where a single kernel program is deployed across a massive grid of hardware execution units. However, the mechanism by which the global iteration space is mapped to data slices diverges significantly, representing the most complex aspect of the conversion process.

### **Triton Grid Instantiation**

In Triton, the grid is fundamentally tied to the physical dispatch of thread blocks on the GPU. It is typically defined as a one-, two-, or three-dimensional tuple. The function signature of a Triton call accepts the grid as a meta-parameter. Inside the Triton kernel, the programmer accesses the grid coordinates to calculate the base data pointer9. This imperative grid definition tightly couples the hardware execution topology to the specific dimensionality of the data being processed. If the developer wishes to change how a tensor is tiled, they must rewrite the internal pointer arithmetic of the kernel itself9.

### **Pallas Grid and BlockSpec Semantics**

Pallas utilizes a significantly more abstracted grid specification system composed of two interacting components defined at the API level: the grid tuple and the BlockSpec object14.  
The grid in Pallas defines a discrete loop iteration space14. For example, grid=(4, 5\) yields 20 distinct program invocations. Inside the kernel, pl.program\_id(axis) functions analogously to Triton, yielding the current execution index, while pl.num\_programs(axis) returns the total grid size for a specific dimension14.  
The critical architectural innovation is the BlockSpec6. A BlockSpec specifies exactly how a massive HBM array should be sliced and routed for each kernel invocation. It requires a block\_shape parameter and an index\_map lambda function14.  
The index\_map acts as the routing logic. It accepts the current program's invocation indices (matching the dimensionality of the grid) and returns a tuple of block indices mapping to the tensor14. The Pallas compiler internally calculates the actual memory slices by multiplying the returned block indices by the respective dimensions defined in the block\_shape14.  
Mathematically, if the grid invocation indices are given as the vector ![][image1], the index\_map function ![][image2] outputs block indices ![][image3]. The physical memory bounds for the ![][image4]\-th dimension of the input tensor are then calculated by the compiler as:  
![][image5]  
![][image6]  
This separation of concerns allows a developer to write a purely mathematical block-level kernel once, and entirely alter the data partitioning strategy simply by modifying the BlockSpec in the orchestration layer, without ever touching the kernel logic6.

#### **Example: Matrix Multiplication Tiling Mapping**

Consider an ![][image7] matrix ![][image8] multiplied by a ![][image9] matrix ![][image10], computed over a parallel grid sized precisely to handle the output tiles: grid=(M / tile\_M, N / tile\_N). The corresponding BlockSpec definitions passed to pallas\_call are:

* **Matrix A:** pl.BlockSpec(index\_map=lambda i, j: (i, 0), block\_shape=(tile\_M, K))  
* **Matrix B:** pl.BlockSpec(index\_map=lambda i, j: (0, j), block\_shape=(K, tile\_N))  
* **Matrix C (Output):** pl.BlockSpec(index\_map=lambda i, j: (i, j), block\_shape=(tile\_M, tile\_N))

In this architecture, the grid iteration over indices i and j defines the output tile space, while the index\_map uniquely instructs the Pallas DMA engine to fetch the corresponding full row-slice of ![][image8] and column-slice of ![][image10] required to compute that specific tile16.

### **Addressing Sparsity: Scalar Prefetch Grids**

A major advantage of Pallas's separated orchestration layer becomes evident in highly sparse workloads, such as Block-Sparse Attention for LLMs1. Standard XLA compilers struggle immensely with sparsity because fetching non-contiguous blocks dynamically from HBM stalls the MXU, destroying arithmetic intensity1.  
Pallas bypasses this bottleneck via the PrefetchScalarGridSpec abstraction1. Instead of passing an attention mask as a massive, wasteful boolean matrix, the developer passes the mask in a highly compressed Block-Coordinate format. PrefetchScalarGridSpec allows the compiler to load this small list of coordinates directly into the accelerator's Scalar Memory (SMEM) immediately before the main compute pipeline activates. The kernel's index\_map lambda functions read these scalar coordinates on-the-fly, instructing the DMA engine exactly which disjoint blocks to fetch from HBM1. The DMA controller successfully pipelines these unpredictable HBM fetches while the MXU simultaneously operates on the prior block, effectively masking the extreme latency associated with sparse tensor access1.

## **Deep Dive: Compiler Lowering Logic and Interoperability**

The "lowering" process refers to the intricate translation pipeline that converts high-level Python code into optimized machine instructions. The multi-stage pipelines for Triton and Pallas highlight their differing structural priorities and integration strategies.

### **The Triton Lowering Pipeline**

Triton's compilation pipeline is traditionally triggered via a JIT decorator. The process relies entirely on intermediate representations (IR) based heavily on the open-source Multi-Level Intermediate Representation (MLIR) framework5.  
The process begins with Abstract Syntax Tree (AST) Parsing. The Python AST of the kernel function is parsed and converted directly into a high-level, hardware-agnostic Triton-IR5. Next, the optimizer module ingests the Triton-IR and lowers it into Triton-GPU IR (TTGIR)5. This stage is deeply hardware-aware. Specific layout configurations are injected into the IR, mathematically representing how tensor data is distributed across physical Streaming Multiprocessors (SMs) and warps5. For NVIDIA architectures, these layouts might include \#blocked (contiguous warp assignment) or highly specific layouts tailored for hardware Tensor Cores, such as dot\_op5.  
Following the generation of TTGIR, a comprehensive suite of optimization passes is executed. These include general MLIR passes, such as Common Subexpression Elimination and Dead Code Elimination, alongside highly specialized GPU passes, such as Memory Coalescing, Software Pipelining, and hardware-specific instruction insertion (e.g., AMD's OptimizeLDSUsage or NVIDIA's asynchronous dot operations)5. Finally, the optimized TTGIR is lowered into LLVM-IR, which is subsequently compiled by the backend machine code generation modules into Parallel Thread Execution (PTX) format for NVIDIA, or specific ISA assembly formats for AMD GPUs5.

### **The Pallas Lowering Pipeline and JAX Interoperability**

Because Pallas is a native extension of JAX, it bypasses standard AST parsing and instead integrates deeply with the XLA compiler stack2. A Pallas kernel is lowered dynamically depending on the hardware target—either to the internal Mosaic dialect for Google TPUs, or to Mosaic GPU/Triton for NVIDIA hardware4.  
The pipeline initiates with JAX Tracing and Jaxpr Generation. When pallas\_call is executed, JAX utilizes its standard dynamic tracing mechanism19. The kernel function is passed specialized proxy objects that aggressively record the mathematical operations applied to them. This creates a jaxpr (JAX Expression), which is an intermediate, strongly-typed functional representation of the kernel logic4. Because Pallas relies on this tracing infrastructure, it natively supports functional transformations like jax.vmap (vectorized map). A batched version of a Pallas kernel can be constructed effortlessly by vmap-ing the pallas\_call, which the compiler resolves by augmenting the grid dimensions and adjusting the BlockSpec mappings automatically4.  
During the tracing phase, primitive lowering occurs. Pallas code utilizes a restricted subset of JAX primitives alongside Pallas-specific primitives like pallas.load and pallas.store4. Operations that do not map efficiently to hardware accelerators, such as high-level convolutions (conv\_general) or arbitrary scattered memory writes, are strictly prohibited and will trigger compilation failures4.  
The generated jaxpr is then lowered into StableHLO (High-Level Optimizer format), which is XLA's standard dialect4. The Pallas kernel itself is mathematically wrapped as a lax.scan over the multidimensional grid or represented as a custom call node within the broader XLA execution graph4.  
The final stage is backend dispatch, which varies drastically based on the physical hardware:

* **TPU Dispatch (Mosaic):** For TPUs, the StableHLO is lowered directly to Mosaic4. Mosaic physically maps the operations to the specific topographical traits of the TPU, partitioning the required data across the Vector Processing Unit (VPU) and Matrix Multiply Unit (MXU). Operations are mapped tightly to 2D vector registers, which, as of recent TPU generations, are sized at 8x128 lanes for standard 32-bit floating-point values12.  
* **GPU Dispatch (Mosaic GPU or jax-triton):** By default on newer JAX builds designed for NVIDIA's Hopper architecture and beyond, Pallas lowers to the internal Mosaic GPU dialect, bypassing Triton entirely3. For older architectures, or if explicitly forced via the backend='triton' compiler flag, Pallas utilizes the jax-triton plugin. This plugin translates the Pallas IR directly into Triton IR, effectively utilizing Pallas as a high-level front-end generator for the Triton compiler21.

#### **The jax-triton Custom Call Complexities**

The integration of Pallas with Triton via the jax-triton repository introduces significant toolchain complexities. The PTX generated by the Triton compilation pass is explicitly passed into a custom call that lives within the jaxlib GPU bindings21. A common failure mode occurs due to version mismatches. jax-triton depends on the version of Triton currently mirrored internally at Google, which frequently lags behind the public Triton HEAD by several weeks21.  
Furthermore, Triton historically ships with its own bundled CUDA toolkit, emitting PTX targeted at version 8.0. If the underlying jaxlib attempts to compile this PTX utilizing a bundled ptxas assembler that is older, the compilation will catastrophically fail21. Developers must therefore ensure strict synchronization between the locally installed CUDA toolkit, the jaxlib version, and the explicitly pinned Triton nightly build to maintain a functional lowering pipeline21.

## **Hardware Communication and Memory Hierarchy Orchestration**

Maximizing compute utilization on multi-million dollar accelerator clusters dictates that data must reside in fast SRAM or operational Registers exactly when the MXU or Tensor Cores require it. The strategies employed by Triton and Pallas reflect their differing compiler philosophies regarding software pipelining and data layouts.

### **Physical Hardware Constraints and the Roofline Model**

Modern accelerators operate across a strict, unforgiving memory hierarchy. High-Bandwidth Memory (HBM) serves as massive global storage but is characterized by high latency and structural bottlenecks2. Direct Memory Access (DMA) engines act as asynchronous controllers capable of moving large, contiguous data blocks between HBM and local memory without tying up precious compute cycles1. Local Memory (Shared Memory on GPU, Virtual Memory on TPU) serves as small, ultra-fast staging areas. Data must physically reside here before computation can commence2. Finally, the Compute Units (MXU / Tensor Cores / VPU) consist of dense transistor arrays capable of massive parallel math, provided the incoming data is perfectly aligned to the unit's width2.  
To understand kernel optimization, developers rely on the roofline model, specifically monitoring arithmetic intensity. Arithmetic intensity is mathematically defined as the ratio of floating-point operations (FLOPs) performed to the bytes of memory transferred from HBM22. For example, on a TPUv5e, the arithmetic intensity magic number is 240; algorithms failing to reach 240 FLOPs per byte are strictly memory-bound, leaving the computational MXU starved and underutilized22. Efficient hardware communication aims to perfectly balance DMA fetching with MXU execution to achieve compute-bound status.

### **Triton: Explicit Mechanisms and TMA Exploitation**

Triton leaves memory orchestration largely up to the compiler's automated optimization passes, but gives the advanced programmer explicit mechanisms to assist the compiler. The primary method is software pipelining, where the Triton compiler analyzes the AST to overlap the memory loading for block ![][image11] with the mathematical execution of block ![][image12]5.  
With the advent of NVIDIA's Hopper architecture (H100), Triton introduced deep support for the Tensor Memory Accelerator (TMA) and Warpgroup Matrix Multiply Accumulate (WGMMA) instructions13. The TMA is a specialized hardware unit allowing asynchronous, fully hardware-managed multi-dimensional memory transfers directly from HBM to SMEM. To leverage this paradigm in Triton, the programmer must explicitly utilize specialized block pointers (tt.make\_tensor\_ptr), defining the global tensor shape, the block shapes, and the exact dimensional strides11. This explicitly commands the hardware to bypass traditional scalar load instructions and utilize the TMA, ensuring that memory bandwidth is maximally utilized.

### **Pallas: Declarative Software Pipelining and Advanced Precision**

Pallas views memory transfers and compute execution as a unified, mathematically scheduled pipeline. Memory is transferred from HBM to VMEM/SMEM automatically based entirely on the BlockSpec definitions2.  
On Hopper GPUs specifically, Pallas utilizes the Mosaic GPU backend, which exposes direct, explicit software pipelining APIs. The programmer uses the plgpu.emit\_pipeline function, defining sequential pipelines with highly specific configurations13.  
Two critical configuration parameters dictate the hardware communication efficiency:

* max\_concurrent\_steps: Dictates the maximum number of concurrent memory transfers actively in flight. Autotuning this value is critical; increasing the pipeline depth improves utilization of the memory subsystem but linearly consumes additional SMEM to hold the temporary data buffers, potentially reducing the total active occupancy of the Streaming Multiprocessor13.  
* delay\_release: Controls the reuse of SMEM buffers. Setting this parameter delays the pipeline from overwriting a buffer13. This is absolutely necessary when using asynchronous WGMMA operations, as the localized compute core might still be actively reading the SMEM buffer even after the primary pipeline orchestration iteration advances. Omitting this parameter results in catastrophic silent data races13.

When data reaches local memory, Pallas requires specific layout casting for hardware utilization. For instance, interacting with the Hopper Tensor Core requires casting references to Layout.WGMMA, which formats the 16-bit input operands specifically for the accumulator registers24. Conversely, for operations not utilizing the Tensor Cores, developers cast to Layout.WG\_STRIDED, which partitions values equally across the 128 CUDA lanes comprising a Pallas thread in a round-robin fashion to maximize vector unit throughput24.  
Advanced kernels must also manage distributed hardware communication. In Pallas, multi-device sharding is handled seamlessly by underlying JAX primitives. Algorithms rely on All-Gather operations to gather sharded matrices, and Reduce-Scatter operations to sum shards over an axis and redistribute the results, effectively mapping massive multi-device communications directly into the array compilation graph22. Furthermore, when operating in low-precision training environments, Pallas integrates with concepts like FP8 Delayed Scaling25. Instead of reading a tensor twice to calculate scale factors, the quantization process calculates a scaling factor directly from a historical amax buffer (reducing tensor reads to one), applies the scale, casts the tensor to FP8, and synchronously updates the history buffer25.

## **Concrete Code Comparison: Architecture and Syntax**

To synthesize the philosophical, structural, and hardware-level differences, analyzing concrete implementation syntax is highly instructive.

### **Case Study 1: Vector Addition**

Vector addition is a fundamentally memory-bound operation that serves as the baseline for evaluating kernel language syntax.

#### **The Triton Approach**

Python  
import triton  
import triton.language as tl

@triton.jit  
def add\_kernel(x\_ptr, y\_ptr, output\_ptr, n\_elements, block\_size: tl.constexpr):  
    \# 1\. Identify the block position in the hardware grid  
    pid \= tl.program\_id(axis=0)  
      
    \# 2\. Calculate manual memory offsets based on block size  
    block\_start \= pid \* block\_size  
    offsets \= block\_start \+ tl.arange(0, block\_size)  
      
    \# 3\. Create boolean masks to handle boundary conditions  
    mask \= offsets \< n\_elements  
      
    \# 4\. Load from global memory using pointer arithmetic  
    x \= tl.load(x\_ptr \+ offsets, mask=mask)  
    y \= tl.load(y\_ptr \+ offsets, mask=mask)  
      
    \# 5\. Compute the vector addition  
    output \= x \+ y  
      
    \# 6\. Store back to global memory, masking out-of-bounds writes  
    tl.store(output\_ptr \+ offsets, output, mask=mask)

The Triton kernel is heavily burdened with boilerplate addressing logic9. Calculating block\_start, defining offsets, and generating the mask are strictly mandatory to interact safely with the \_ptr objects without crashing the GPU9.

#### **The Pallas Approach**

Python  
import jax  
import jax.numpy as jnp  
from jax.experimental import pallas as pl

\# Kernel Definition (Executes on the hardware core)  
def add\_vectors\_kernel(x\_ref, y\_ref, o\_ref):  
    \# 1\. Load entire reference block directly into a local jax.Array  
    x \= x\_ref\[...\]  
    y \= y\_ref\[...\]  
      
    \# 2\. Compute and Store directly back to the output reference  
    o\_ref\[...\] \= x \+ y

\# Orchestration Layer (Executes on the host)  
@jax.jit  
def add\_vectors(x: jax.Array, y: jax.Array) \-\> jax.Array:  
    return pl.pallas\_call(  
        add\_vectors\_kernel,  
        out\_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),  
        grid=(x.size // 128,), \# Explicit iteration space  
        in\_specs=\[  
            pl.BlockSpec(block\_shape=(128,), index\_map=lambda i: (i,)),  
            pl.BlockSpec(block\_shape=(128,), index\_map=lambda i: (i,))  
        \],  
        out\_specs=pl.BlockSpec(block\_shape=(128,), index\_map=lambda i: (i,))  
    )(x, y)

The Pallas kernel logic (add\_vectors\_kernel) is entirely decoupled from the hardware mapping, containing zero pointer math and zero explicit bounds-checking7. The orchestration wrapper (add\_vectors) handles the complex iteration logic. If x.size is not a perfect multiple of 128, the pallas\_call infrastructure silently pads the inputs and safely drops the out-of-bounds outputs14.

### **Case Study 2: Tiled Matrix Multiplication (Fused ReLU)**

The disparity becomes drastically more pronounced in compute-bound matrix multiplications, where two-dimensional tiling and pipeline scheduling are strictly required to keep the MXU active.

#### **The Pallas Implementation**

Pallas leverages its grid mapping to entirely abstract the asynchronous fetching of the ![][image13] dimension, allowing the kernel to focus solely on the dense MXU calculation.

Python  
def matmul\_relu\_kernel(a\_ref, b\_ref, c\_ref):  
    \# Perform matrix multiplication via MXU utilizing the localized references  
    acc \= pl.dot(a\_ref\[...\], b\_ref\[...\])  
      
    \# Fusion occurs here: Apply activation while data remains entirely in fast VMEM/SMEM  
    result \= jnp.maximum(acc, 0)  
      
    \# Write the materialized tile to the output Ref  
    c\_ref\[...\] \= result

@jax.jit  
def fused\_matmul(a, b):  
    m, k \= a.shape  
    \_, n \= b.shape  
    tile\_m, tile\_n \= 128, 128  
      
    return pl.pallas\_call(  
        matmul\_relu\_kernel,  
        out\_shape=jax.ShapeDtypeStruct((m, n), a.dtype),  
        in\_specs=\[  
            \# Instruct DMA to fetch row slices of A and column slices of B  
            pl.BlockSpec(index\_map=lambda i, j: (i, 0), block\_shape=(tile\_m, k)),  
            pl.BlockSpec(index\_map=lambda i, j: (0, j), block\_shape=(k, tile\_n)),  
        \],  
        out\_specs=pl.BlockSpec(index\_map=lambda i, j: (i, j), block\_shape=(tile\_m, tile\_n)),  
        grid=(m // tile\_m, n // tile\_n),  
    )(a, b)

The index\_map mechanism allows Pallas to define the memory data pipeline entirely dynamically16. The lambda function lambda i, j: (i, 0\) pulls the precise block slice required for the local computation directly from HBM via DMA16. The compute core is solely responsible for executing the mathematical pl.dot and the jnp.maximum activation, resulting in highly readable, mathematically dense kernel code.

## **Edge Cases, Performance Regressions, and Sparsity Limitations**

The abstraction choices made by these complex compiler frameworks naturally lead to highly specific edge cases and catastrophic performance regressions when deployed at scale, particularly when traversing multiple intermediate representations.

### **The JAX 0.8.0 Triton Pointer Regression**

A highly critical case study regarding Triton's lowering logic occurred between JAX versions 0.6.2 and 0.8.011. When JAX developers updated the internally bundled version of Triton (upgrading to the Triton 3.x branch), execution times for custom pallas\_call kernels relying on static unrolling collapsed drastically—benchmarks showed regression from 0.11 milliseconds per block to a catastrophic 0.57 milliseconds on NVIDIA A100 GPUs11.  
The regression traced directly to how the Pallas-to-Triton mapping interacted with Triton's intermediate representation compiler passes11. Under the hood, when utilizing the jax-triton bridge, Pallas generates mathematical base pointers and aggressively broadcasts them into a tensor of flat, individual pointers, computing standard scalar arithmetic to mathematically define individual element addresses11.

* **In Triton 2.x (JAX 0.6.2):** The Triton compiler utilized heuristic passes that recognized that these thousands of computed offsets were mathematically contiguous. An aggressive optimization pass silently fused them into a single, lightning-fast coalesced memory fetch11.  
* **In Triton 3.x (JAX 0.8.0):** Triton developers aggressively deprecated this heuristic behavior, strongly favoring explicit hardware instructions (tt.make\_tensor\_ptr) to utilize the Tensor Memory Accelerator (TMA)11. Consequently, the legacy Pallas-generated pointer math bypassed the TMA entirely. This forced the GPU to blindly execute thousands of uncoalesced, scalar memory loads per block instead of one coalesced block load, thoroughly destroying memory bandwidth and inducing massive hardware register spilling11.

This edge case explicitly demonstrates the fundamental vulnerability of intermediate compilation layers: algorithmic changes in a downstream compiler's optimization assumptions (Triton) can critically impair an upstream language's (Pallas) performance profile, forcing developers to rapidly pivot to newer native backends (e.g., Mosaic GPU) to ensure stability3.

### **Dynamic Sparsity and Lexicographical Traversal Constraints**

While Pallas excels at static, predictable block-sparse tasks, highly dynamic, unpredictable algorithmic sparsity—such as Top-K routing in Neural Sparse Attention (NSA)—presents a massive computational challenge across the entire TPU software stack (JAX/XLA, Pallas, and the hardware itself)26.  
Pallas execution grids are, by fundamental definition, strictly structured. The BlockSpec abstraction strictly enforces a sequential, lexicographical traversal through non-overlapping physical data blocks26. Dynamic top-K routing relies heavily on unpredictable runtime variables and dynamic branching paths, which conflicts directly with the XLA JIT compilation model that prefers perfectly static tensor shapes26.  
Furthermore, if algorithms mathematically require overlapping memory blocks (e.g., a sliding convolution window where the stride is significantly shorter than the block size), Pallas will blindly fetch highly redundant elements from HBM for every adjacent block, severely lowering the system's arithmetic intensity26. In contiguous block setups, elements are shared, but independent fetching triggers redundant memory access. Modifying advanced AI algorithms to prevent this redundancy overlap forces a severe departure from the desired architectural mathematics, highlighting a current hard boundary in Pallas's declarative memory management capabilities when compared to Triton's granular pointer control26. Debugging these precision issues requires extensive numerical stability monitoring, checking relative tolerance (rtol) and absolute tolerance (atol) across various axes to ensure the MXU accumulators maintain mathematical fidelity during complex unrolled executions26.

## **Conclusion**

The transition from raw CUDA or XLA manipulation to high-level, domain-specific kernel languages like Triton and Pallas represents a critical, paradigm-shifting evolution in machine learning infrastructure. As hardware topologies scale in complexity, managing the HBM-to-SRAM memory wall requires sophisticated software abstraction.  
Triton remains the heavily entrenched industry standard for NVIDIA GPUs, providing explicit, uncompromising low-level control over block configurations and pointer arithmetic. It is highly optimized for extracting absolute maximum throughput via mechanisms like explicit Tensor Memory Accelerator (tt.make\_tensor\_ptr) directives. However, its deeply imperative nature requires substantial boilerplate logic to manually manage memory indices, block masking, and out-of-bounds error handling.  
Pallas approaches the hardware problem by treating the kernel as a pure, mathematical array function mapped to physical memory via the BlockSpec orchestration layer. By completely isolating addressing logic from core compute logic, Pallas provides a radically cleaner, JAX-native syntax while allowing the compiler to preemptively prefetch memory and handle complex padding semantics automatically. This model enables seamless integration with standard functional transformations like vmap, and provides a unified, highly ergonomic pathway to program both Google TPUs (via the internal Mosaic dialect) and NVIDIA GPUs (via Mosaic GPU).  
While both advanced frameworks routinely achieve near hardware-limit performance, Pallas's software-defined memory orchestration presents a more robust, scalable solution for static sparsity and multi-platform compilation workloads. Conversely, Triton remains unmatched for bleeding-edge, fine-grained register and pointer control on specialized NVIDIA architectures. The ongoing hardware architectural shifts towards explicitly managed memory pipelines—such as Hopper's TMA and the TPU's asynchronous DMA engines—will undoubtedly require highly specialized systems engineers to master both paradigms to fully utilize next-generation artificial intelligence accelerators.

#### **Works cited**

1. Breaking the O(N^2) Bottleneck: Implementing High-Performance Block-Sparse Attention with JAX/Pallas \- Hugging Face, [https://huggingface.co/blog/rishiraj/block-sparse-attention-with-jaxpallas](https://huggingface.co/blog/rishiraj/block-sparse-attention-with-jaxpallas)  
2. Unlocking Kernel-Level Optimizations on TPUs using Pallas: A Getting Started Guide, [https://medium.com/@engineerbharath/unlocking-kernel-level-optimizations-on-tpus-using-pallas-a-getting-started-guide-ae47a3ad5bb1](https://medium.com/@engineerbharath/unlocking-kernel-level-optimizations-on-tpus-using-pallas-a-getting-started-guide-ae47a3ad5bb1)  
3. Pallas-Triton kernels and kernel auto-tuning \- Robert Dyro, [https://robertdyro.com/articles/pallas-triton\_kernels/](https://robertdyro.com/articles/pallas-triton_kernels/)  
4. Pallas Design \- JAX documentation, [https://docs.jax.dev/en/latest/pallas/design/design.html](https://docs.jax.dev/en/latest/pallas/design/design.html)  
5. Unlock Peak Performance on AMD GPUs with Triton Kernel Optimizations \- ROCm™ Blogs, [https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html](https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html)  
6. Writing Pallas Kernels for JAX: Stepping Outside the XLA Safety Net \- Rajat Pandit, [https://rajatpandit.com/ai-infrastructure/writing-pallas-kernels-for-jax/](https://rajatpandit.com/ai-infrastructure/writing-pallas-kernels-for-jax/)  
7. Pallas for people who know JAX but not kernels yet \- Hugging Face, [https://huggingface.co/blog/ariG23498/pallas-for-beginners](https://huggingface.co/blog/ariG23498/pallas-for-beginners)  
8. Lab 11: Meet the TPU \- 6.S894, [https://accelerated-computing.academy/fall25/labs/lab11/](https://accelerated-computing.academy/fall25/labs/lab11/)  
9. jax-triton contains integrations between JAX and OpenAI Triton \- GitHub, [https://github.com/jax-ml/jax-triton](https://github.com/jax-ml/jax-triton)  
10. JAX-Triton documentation, [https://jax-ml.github.io/jax-triton/](https://jax-ml.github.io/jax-triton/)  
11. Severe (5-10x) performance regression in Triton kernel via JAX/Pallas: Triton 2.x vs 3.6 · Issue \#9640 \- GitHub, [https://github.com/triton-lang/triton/issues/9640](https://github.com/triton-lang/triton/issues/9640)  
12. Writing TPU kernels with Pallas \- JAX documentation, [https://docs.jax.dev/en/latest/pallas/tpu/details.html](https://docs.jax.dev/en/latest/pallas/tpu/details.html)  
13. Mosaic GPU Pipelining \- JAX documentation, [https://docs.jax.dev/en/latest/pallas/gpu/pipelining.html](https://docs.jax.dev/en/latest/pallas/gpu/pipelining.html)  
14. Grids and BlockSpecs \- JAX documentation, [https://docs.jax.dev/en/latest/pallas/grid\_blockspec.html](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)  
15. jax.experimental.pallas.pallas\_call \- JAX documentation, [https://docs.jax.dev/en/latest/\_autosummary/jax.experimental.pallas.pallas\_call.html](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.pallas_call.html)  
16. Define a Custom TPU/GPU Kernel \- Keras, [https://keras.io/guides/define\_custom\_kernel/](https://keras.io/guides/define_custom_kernel/)  
17. Pallas Quickstart \- JAX documentation, [https://docs.jax.dev/en/latest/pallas/quickstart.html](https://docs.jax.dev/en/latest/pallas/quickstart.html)  
18. jax.experimental.pallas module \- JAX documentation, [https://docs.jax.dev/en/latest/jax.experimental.pallas.html](https://docs.jax.dev/en/latest/jax.experimental.pallas.html)  
19. Ahead-of-time lowering and compilation \- JAX documentation, [https://docs.jax.dev/en/latest/aot.html](https://docs.jax.dev/en/latest/aot.html)  
20. Pallas Changelog \- JAX documentation, [https://docs.jax.dev/en/latest/pallas/CHANGELOG.html](https://docs.jax.dev/en/latest/pallas/CHANGELOG.html)  
21. How to get started with Pallas? · jax-ml jax · Discussion \#17367 \- GitHub, [https://github.com/jax-ml/jax/discussions/17367](https://github.com/jax-ml/jax/discussions/17367)  
22. Thinking in Pallas \- Sharded MatMuls \- Consider the Bulldog, [https://considerthebulldog.com/pallas-sharded-matmuls/](https://considerthebulldog.com/pallas-sharded-matmuls/)  
23. Async/await on the GPU \- VectorWare, [https://www.vectorware.com/blog/async-await-on-gpu/](https://www.vectorware.com/blog/async-await-on-gpu/)  
24. Writing Mosaic GPU kernels with Pallas \- JAX documentation, [https://docs.jax.dev/en/latest/pallas/gpu/reference.html](https://docs.jax.dev/en/latest/pallas/gpu/reference.html)  
25. FP8 Delayed Scaling — Transformer Engine 2.16.0 documentation, [https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low\_precision\_training/fp8\_delayed\_scaling/fp8\_delayed\_scaling.html](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html)  
26. Optimizing NSA for TPUs \- Kernel Worklog \- Henry Ko, [https://henryhmko.github.io/posts/nsa\_tpu/nsa\_tpu.html](https://henryhmko.github.io/posts/nsa_tpu/nsa_tpu.html)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJcAAAAaCAYAAAC6sc5/AAAEpElEQVR4Xu2aX6hNWRzHfzJkGjUM0US5PJCGhvyZaB4QIpHwYGrMAw/mQRE1Zh7UfZF5GMmUP0lJkpLy4n/KffBAlCh5miaiedB4QlH+/D7WXp3td/Y6Z+9zz93n7tP61LfbXevcddZe+7t+v99e+4pEIpFIJBIZ5AxTfWMbI6Xytbj70FVwUedV621HpFRWqa6rJtqOouDQd6pHqlmmr0ww1mXVH6ohpg9GqUbYxjYxUOMOFljPotngZ9V921iUOaoPic6pvvi8uzR+U91SjbEdCWwALjbU3yrsTjbWItPeTewSd3+LbCKCztnkZ8vwxy9U91TTTV9ZsLO4weNtR4rRkh3R2gFRs5vhHrdyjV+qriQ/K8sE1d+2MTIoeKNaYBvzMFQ1TvWtuLTQqci1RrXONiYQrdqdCj0jxY09UBGx0/j7WyQdWl6rdtvGPGCqx1KruVoapA3sVX1vG8UVoeT9/8TVDe00wXbVM9VLcQ8S3chNcWuHQUKbtxn4o1+1eF5zjVWtVW0ooJmf/jIM0aNPXE2VhlDMRZHvZ4gr9lupG7JgHMb0NQWhv9vYKG4zrlC9V535vLsOolxW8X5RnMEIRC2R11wDgTcXP9OsTAS9qm21rk9wsXtUx1Q/mb5m9IhLFRgYY2Ewy3Lp7NFMf9kiLtpgKsyFybJYrHou7jNLTR+clAqbiyhCVLLm8vh+Ww8S8heK23GcjWXtumb0irvuram2HtWJpH11qr2KTFY9VT0Ul3VCsIkxUNahaaXNFYpcHh/Wbc4/LbUajEUp+kTDsQfHHyw+N8GDSYerXkn1zUW05772mnYLNS/pL6vwZ50ray5gd0yxjVLbeZiA4woWAFNwoem5YkzGSHNYdUM11bR70ukCk9q00chcpMy7Eo6WX6kOSHYk8NDHHBvVkUTszRJ+kGEOPOiE5uFTPrUlh+X+vIrxiFb0XRN3zpmVEuGOuFdBXFMh+DK+xJsLl/o6p0x2qn60jeIWhJt8XPW76tekPY+5/DVhIhv1oE/1jziz/iD1B4Uhc7HILDZjh6KlX9P9tiMF8+czofXme+gPRQ3fj4FC86CftcV8R1PtvLvlbcckcdGNJ8qszQ30tXQGyaSZvL8R/mLKhsVJ1z0eTHNJ9a/qkNR2aB5zUZMRmYheWSl3hzgDMT4RwhIyFzDXtxLuZ358/zLbkWKuuBvcKLox/yMSjkzMgc+E5sH4RDfevhDhwJcDfv0apUTgtZuN6pWC1BC6QH8QmIYFsubK2l3sRiJXlrmAl+GMzXdYGpnL06y/DDBHaB6YkrVKXz8H1pztzUt+JwozRgiMyBiV5n9xZ095wISkSg+RIivyseP+so05aWYu0qh9gi0b5nBKis2DTenLASBTUW9Nk/r6j8/0mrZKMl/cRVP/5OGC6k9xn6cotUUv/2VBKs2qtxrBoe8DcSUCO/yJ1Be71CqkvU7DHGbbxiZgIGrrfaqDqtviov4v6Q+JO5LhCKNrSBeazSCVsWN5crM7Dr6T+iK9XXBm1GMbO0CPbcgJG5F3qmQAUqddP36/KrU6rWtYIo1rgMjAQ9TfJPXZIBKJRCKRSHE+Avbe25D1cPgmAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAaCAYAAAA9rOU8AAACBklEQVR4Xu2WPUtcURCGJ5iAoiEEgh8ohHQBC4tgQLQUu6TQViEIIYJdmvwCixSChBQWgWBhZyN+oJhCSGmjhVgJCnYhCILpEn3fzL149t1zv2AFwX3gKZy5u86dM+ecNWtyT3gEn2hQKPNMIS2wE7ZqIqEPbsHXmhBYyCoc10QV9uEm/AXfSI7swBmJfYX/4DbsDuLP4UHwdyXYkWW4CK/gUm3aHsA12CZxFsLn6UfJTZoXVZlvcESDCSyEHYjNwRz8Y/4isfwZfKHBPB7Dn+YzEaMLHmmwJOfwgwbzYBF8AxYVYxT+1aD5wPeYf34M9tam/7MHV+BDTSjzdrPeodPhQ+ATvJAYYSHh52JDz+U7hM80kcUX8y+L0Q5/mC9jDBaaVwzzl/CVJmKk/+y3JhI64G5ijDLFcMdxqQvhcB6btzJGI4rJytXB9rGNHLIYjSiGW39QEzF4TmR9UQpnirstRlExPL9OzYc9F243doQ7ZUByITwn+HYKT2PulrSYz1a/hTfMZ5KzmQu3G2eFZ8FTyYWkS6no1qZc1pAT8+4Xwm6wK9/Nj/ws0rnRty4Dv39IgyG8XYfNr3i+zdvadBTe1pXuGPMX5E8OvVxr4FC9N1+iMoWk8Ajo12AG/IHFjhd2k51ZN+9OFd6ZD2TsdlYmzJ+9Ndj2KTirCeElXLByRTe521wDlhFwE2o6RqQAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKQAAAAaCAYAAAA5dfdEAAAFRklEQVR4Xu2aach1UxTH/zJEpgyRKG8+yDxkiqg3GTMV4i3lgxIfRZm+ygellDkpJCHKB4kQjyGEFBGRXmQoQoQMGdbPuqu773rOee5z7j13eO57fvXvfc/e956z79prr7X2Po/U0dHR0dHR0dE6W5p2Nm2WOzpmCvOxeW5cdHY0PWE6L3d0zBwc8iH5HI3NSaYvTL+Y/jWdNdg9Nzxtuk796Liv6R352H8wbddrb4utTffL7/+16YiB3sUCW4YdlzSaLW+Qz9HYTnmY6TG5M86rQ+KEb5h2KdpI3efIVybjbpstTOtN15j+Mu060LtYYMuw462pb7UwN8zR1WqppJpXh+TH3W7aPXcYO5neMn2fO1rkRk3G4eeNsOP+uaMBzNGH8vkam3l1yD1Nn+XGHqTRX+VpZhKQupbkz1h0wo6jpOsSIuyXuXEUSodkx8S/aK/e9aw4W/UR6jJ5H1GMlDGuMTNEC6IG0YN784xW0tEcEnbk95WlUVPOVf18NSIc8gXTb/JCPtq+0uwmAiP9nBvV39n9Y7pTvropyh9RC4V1jzAuxTobqI9Mn8jr7kUCW4YdN8ptOaodI2uNTTgfRXzJ8fKi/n3Tbqmv5DjT+Q10hmnb/79ZT6RMIlQmxvWy+obbw/S56lN8E6I+xbjHFO0PqKUIMEdgy9KOEHakZAo4Bx6WLdn8faAWslWdQ+5j+ka+gjgimibhkChTpuuA8oL6BSdi1eNI1DSj7BzLdI1zQkTl0iFpO6W4Xotgy9KOEHaM464/5Y427LQh5mxiDhlRh75pb3hYsRwlLKV2eFzLFwn/p40FxAE6qZYjDdT0QP0i+W8unTlWP+3rTPfJU9xajpgcb2HLHGzCjgQkwA4sxmGl20I75EoREsdgFRMVA8bOOJfkjkwNGHDdpCZiArhXeY+ojxDpa6uiba0Si6y0I4Qdw7HIFETSYWxvekUtOuT1qX1Dr/1J0zapbxpQs7FSM59qcMWeYPrD9KzcGGW6Aa7LBYVzUsS/KH9TkblXgw7PzvNN04+mw+NDWtkhSeVvm67KHT2ooW8xPZc7Cng+r0xXWkw85xLVR69D5M+pqtk5O8SW5XexZdgxiHS9n+lu+UuDKqLEGxucjjD9u+laeQQ4Vf5Kib69+x+dKlfKNy8ZHIZD2Dgw/1i+isOBhjnk6eovwoeL9oDISN10Wu/6Hvk92IyV1Dkkk/+8/P4slCrKMdQRUZ/PVhHPIYuRzarAVtyD4JIhZdMfdmSesWUZMXFW0jo1+aWmM1U/nthsjs3f8jrrANPr6huKiHBT8blpc6yqJxSDvSbf/T9qus20Q9E/zCGZvFfli5AomWFB8rvZ2BCJiY7YJlPnkECKw6l5RhUxhp9yR8GRpu+0PKWWXGG6Sz7mKk6WjyGXYwH3DjtyxIYtS9jUvWS6WO6c2LnuWTwDn2kdVgx/YDBrYmNTNRaMQyqtigwYtnRIzihz4Q6kmKoIGZC2qu4frOSQQd7BzgI2aXUOCWHHKjsfKi+dNsp/S50z8t2nTM/kjkXjctNBuXEIRJUyrdS9pyUl35wbGzDMIam7H8yNM4AasmpBrgacmfrxKHlpgIOW55MBc4Sdq0qDhYPCmxpmtZDq3jNdYLpQy9+uEF35y5Q75HVUUw6W3/9beWlDrZ0nnHqMlFxugmYBWYZXsKNAhoiShtROOccCzhso5oY5Ojq1LyzUt++q2eaKFLJe7pwZDHqgJntyQFRZlxtnABuf7EBN4CgnIF1zrpvh1Wp5RLZJcKK8hhklonVMDs5jmZuOjo6Ojo6OTYP/AOb9Ma8OuJQQAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAcCAYAAAC3f0UFAAAA50lEQVR4Xu2SqwpCQRCGR1BQvIMogu9gspgtFrMYbOIj+AQWu5itYrcYThRMBqNBsVoNgpf/dy8c1j1dwQ++sDNzdmdnj8hvU4cneINDJ/dBCW7hFTacnBcWH2DFTfi4wCWMuwkfTzhyg1E8YCu0jsFEaG3JwjOs6TUvuYMrmDFFhg4cwyacwBRsww3Mh+resHABpzCtY2yhais0TK5FXZDj64na2Qv7ZL9FOIB3OJOIEXICnAThZQJ4FNVCX8csnC1bIKZ4L+oXmMOkzr2P4qvx9YjpP4A52NVxS0FCX2t4QtmJ/fk2XgTqIybkPiUQAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,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>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEIAAAAaCAYAAAADiYpyAAACoklEQVR4Xu2WzavNQRjHH3mJEEVdhYiVd3Vl4XVjwYIQpZT8AaKUlI27kY0sRMlGFBZsbVCulEghJbK6pKywspKX76eZJ/Mb9/zOmVPOaj717Zwzz3Rm5nmbMatUKpXe2SJdk95Jr6Q9TXODydIZ6WMU8xenEwbAUum6dCXTCWlqnHMgsx2XpkRbRxZJ+6Tz0m8LTunEfumrhXlPLCyIcwbJXGm3dFL6If2ysP9VyZxN0ksL+7yf2bpCpL9LD6Tpmc25Kj20sMDBzDZotllwwqfcIA5Jz6XVuaEbC6RR6Y30WVrSsAY2RxGFx9LMprkjzJuYDybg9NKsogTuWggImeHMi+Nt67WCd29ZcAZZMdywms2yUGssyuIXmuZWmHvJOh/2mbQmH+wCgSMTyAj2DlulR/Gzb05ZOCT9gYPubJrtqIX+cMeCfW/T3AoO8KaVO2OttD4b6wUvizFpvnTMQiYXl0IKaXbbQoOhT3BQuqyzQrpo4RBj0hdpWWLvhdQZDk54kfwugcCxTwIzO35HR9JJpXhZTLKQCfzhDWmChSwgG5zSssjBIaQvDugnE8DLgr28tRCUldK3OMZN1hdeFkBvoEeMSjMsRJD+AGROaVmMB/9/U5qWG3rEy+KDtDCOEUSCyf74LCYtC+C2oNZ8EW4Jh1rspyxSyIKN0jnpsv3bM3rBy5eywAHOdgsOwtHF+LXJIYHHClcoB6ZPUB4OkSi5NnNwwvv4HQe4M0rxazPtY0DmPo22dN9dYTO8Du9JQ8k4N8dPaVcyNsfCC+2wld/R66TX8TOHPZRkBvM4KFm7PLNxeLKCd85p+/vcboX0JurebdMmSL/wq25HNseVR6ONsza+E5wRaUM+mJFGO5WXavrASsXTu1KpVCqVSqXyv/gDyKSauVJhW6kAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAbCAYAAACjkdXHAAAA50lEQVR4XmNgGNaAFYidgVgaXYIYEAPE/4HYF12CEBAD4isMEM1FaHIEQTMDRCMIl6PJEQQnoRikeSGaHF7gB8Q8DBAbQZoPQPkEgTAQ74OygxhI1AyybQKUDQplkOaHQCwJV4EDqDBAbFGA8k2B+BsQPwFiGagYVsACxLOAOANJzBiIv0IxiI0T2ALxTwZE9CBjkO0gV2AFINvmMkCSIzIQBOLTDBADotHk4OAMEGujCzJAQvgAA46EArLJCIjPM2BP/BIMkGgDaZ6JLKEPxJ+gEiAM8q8lkvxFJDkYnoQkPwpGAW4AAO2uNQEiybs+AAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD8AAAAaCAYAAAAAPoRaAAACZ0lEQVR4Xu2WTUiUURSGT2iQ/ZIKQkhN7YKiRbgI3CQJuUikHxHaGfgDQdKibW5ctilXLgqCCArdpBEVMtAiwQhauQg3Ee5ECApcmL4v9x7nfmf+vm8GZjP3gQfHe+7cud+5Z84dkUgk0szchq/gnHHAx4/CxybW72ONIgdfw1/eUXggnCDFz3AvGS7NRSkkYANOwCHY6eNtcAz+83KuxhoFP497eiBuD7/h2cQMd1jP4A5cFvdcqXkqLgE2o0fgc/gCHjexRnMNrsJdeN/EyHmYh11mvCJ80ya8bMb7YI8ZSwMTdtAOBhzyZoHz38JeuAX/w+uJGSIzcNyMVeWmuGxqOXPjj+Cb/RnZmIKL8IQNePLwhh2swjn4Udwe+f3nfvm31cfZm96LS04mWPJcjCXfDufFfdDhcFIGuA6T98kGQDe8ZQdTwJLXh+WJ8+RZARd8PExOao7BL+IenuW/5l+HC9eCJiA8fT44P6sWwpLmmivi9jntxwbFVapWQiq05Cm/q4TNhP+zq9YLE/AdfpPaTpyUOlV2e+6P+xyB76SG/qQl/zIYCxeulxxch19hRzKUmrDkFVbWrLg9fpDi5FTlpBSujrvBeLgwG0k9/IB3xFXTkpRvgpVgyT+0g+AK3Pby8Ow1XRG94v7ASyamC9vrJAtnpFDq3JgmICt6xVlYCdr5M19x7Op844IUd3Zed0/gX3gVtiTDFeFDsyGxn1iYhLQVwLmn4E9xzbfUyWrzS13yeqra6FT94aEVEcb4m/q0j1eD9zx/jpZjEg7bQUPY0dXPUmjKIayoUomJRCKRSCQSaQ72AB0pgE7HAbMNAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAZCAYAAADXPsWXAAAA6ElEQVR4Xu2SwQpBQRSGj6QoNnZWCiVlS9lYWdjwDN5EFl7AguIBlBXJXnkTG6VkzQL/MXfGzOmOa2V1v/oWd/6/e+fMXKKYKNpwDY+W/DwPHMGKaUcwhhO5CLrwAW+wJTKHNNzBvgxAHV7hE/ZE5lCCJ1iTAejQjzvRxawMwJbULvh8UiIz6FF0UbuBd1j9VP3oUc6wYFmEQ7iEedP2oEfZi3UmQWqHK/oyCsNXy8Ww62U4u1D4ob/Ro/hK/HV+CR9uRmQG/i+4dIA5kfEoA1L/SNONXBakXjK11pKwDGdB1rCymJj/8wKOpzDUOmiw4wAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAaCAYAAAD8K6+QAAABoUlEQVR4Xu2WvytGURjHH0UpvxIlmQwG5VeZDGYZSDIofwCDyR+g92/AICWxMNhIoqQsshhNFpmUTAby6/vtvKfOebqv995zdV/D+dRnOc+5p/O99+l5X5FIJFIkk/AaPsArOOSXZRwewi3HEW9HcXTDVr1YiUE4BzfhN9yH9U69By7AC/gJZ2CnUy+CDrgK3+GUqlVlDX7BFzigamQRbujFlPTBJr2YAj5zAD/EvFS++EzB+uEJXBbz8B3scurt5Tr3hbAOR/ViBprhpQQEY6vxi/XCRzEHzDt1XupcTMAQahaMoRiuTky78YBTp26Dh1KTYLrNxuBbWcKwO2LChVKTYElfg23IQ9iavNAxbPF2VIYjWbsNJxLWaRqCgtk2dOHg4ADhMEkKXolGMb+HWn79p4T1e/NYVTIHYxuewWFdACV4A4/grF/KTOGtaMd80rRjG/KgZwkf85ZCgzXAW7gH21TNwhZK24a/kTcY//3wrgy2pGoedvJxo3XF22Hg15zWiwHkCbYr/j3pq4Sf96fkCfavKYn5sx2JRCKRSF5+AB7uaF8O3a2UAAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAABA0lEQVR4Xu2SvQ4BQRSFr4RCIaJSKyV+CpVH0IkoJB5Ao/IEngGFaCQajVIQErVolHrRqxXCOWYndmbXbqGT/ZIvkT039841IxLxKzV4gBfHkhm/WcCJpS9F2IRj+IRzGDcqRNpwDx9wCOtm7GUA1/AGC1ZGOnAEY3Zgk4crmBF1ujPMunJ+Z866ULgGT0auohq2PrFU4E5U01DYiA0JV2GzDUw639zDArFXqMK7I3+TqXyGBeI3lSvydFw5B5cwZVR8wb2ihn8+L4ENu+Id5gtX3MKyHYC+qGZH2DAjf9xPwobr6ZsNfRIJeIIzmLYyjb6MQHQRp2p7RoWCT4Mnj4j4H15INzZPvLrhKQAAAABJRU5ErkJggg==>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAABB0lEQVR4XmNgGAWUghAgXgrEs9CwJ5KaeizyWIEuA8LA/0CcAcQBQCyCpCYNiL9B8QwGiHq8YBIDxDBGdAkgmAfE84GYD10CG9AE4rdA/BVN3AmIT6KJEQRBDBBXXYXyWYG4DIhXAbEETBGxAOZFkFeEgHgtEO8CYi5kRcQAXiA+zAAxLBqIr0PZ74FYB0kdUQDmRZjLQCAHyn8CxIpQMaIAzIuLgJgZKgYyAGQQSDwCKkYQCALxaQaEF2EAlDymQMUPIInjBbAk8QmI9dHkLIH4JxD/QxPHCUCxBrJ9HQNmzIGSRy8DRN6RAREEGABmKyzgYXgrVB7mYmS5R0AsB5UfBaNg6AIAvxpA9tWZAPEAAAAASUVORK5CYII=>