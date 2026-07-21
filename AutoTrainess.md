## Contents
- 1 Introduction
- 2 AutoTrainess
- 3 Experiments
  - 3.1 Evaluation Setup
  - 3.2 Main Results
  - 3.3 Ablations
  - 3.4 Exploration vs. Exploitation
  - 3.5 Analysis of Agent Behavior
- 4 Related Work
  - Automatic research agents.
  - Benchmarks for research and experimentation agents.
- 5 Conclusion
- References
- Appendix A Case Study on the Effects of Different Skills
  - Data skill helps the agent align training data with benchmark format and distribution.
  - Eval skill helps the agent use evaluation evidence to drive targeted optimization.
- Appendix B Detailed Explanation of the Agent Behavior Taxonomy
  - B.1 Evaluation Strategy
    - E ​ 1 E1 Subset Eval.
    - E ​ 2 E2 Full Eval.
    - E ​ 3 E3 Fine-grained Eval.
  - B.2 Input Format Strategy
    - T ​ 1 T1 Prompt Alignment.
    - T ​ 2 T2 Template Adjustment.
    - T ​ 3 T3 Prompt Style Change.
  - B.3 Output Format Strategy
    - O ​ 1 O1 Direct Answer.
    - O ​ 2 O2 Rationale + Answer.
    - O ​ 3 O3 Targeted Format.
  - B.4 Data Strategy
    - D ​ 1 D1 Benchmark-related Data Source Selection.
    - D ​ 2 D2 Training Dataset Expansion.
    - D ​ 3 D3 Data Difficulty Improvement.
    - D ​ 4 D4 Data Synthesis.
    - D ​ 5 D5 Data Augmentation.
    - D ​ 6 D6 Data Cleanup/Dedup/Rebalance.
  - B.5 Training Strategy
    - U ​ 1 U1 Full SFT.
    - U ​ 2 U2 PEFT.
    - U ​ 3 U3 DPO-style Training.
    - U ​ 4 U4 Self-distillation.
    - U ​ 5 U5 Continual Training.
    - U ​ 6 U6 Train from Base Model.
    - U ​ 7 U7 Annealing Training.
  - B.6 Planning Strategy
    - P ​ 1 P1 Baseline-based Planning.
    - P ​ 2 P2 Error Pattern Diagnosis.
    - P ​ 3 P3 Hypothesis Testing.
    - P ​ 4 P4 Failure Case Diagnosis.
- Appendix C Instructions of AutoTrain Hub Workflow
  - C.1 AGENTS.md
  - C.2 Plan
  - C.3 Data Process
    - C.3.1 Selection
    - C.3.2 Construction
    - C.3.3 Validation
  - C.4 Training
    - C.4.1 SFT
    - C.4.2 RL
    - C.4.3 Shared Instruction
  - C.5 Evaluation
  - C.6 Log
- Appendix D Detailed Results on PostTrainBench

## Abstract

Abstract Training language models (LMs) remains a highly human-intensive process, even as frontier language model agents become increasingly capable at software engineering and other long-horizon tasks. A central challenge is that autonomous post-training is not just a coding problem: it requires the agent to repeatedly plan iterations, construct benchmark-aligned data, run stable training jobs, evaluate checkpoints, and preserve experiment state across many hours of interaction. We present AutoTrainess, a LM agent that exposes these operations as a repository of agent-computer interfaces for planning, data preparation, training, evaluation, and logging.
Rather than leaving the agent to operate in a raw CLI environment with an underspecified action space, AutoTrainess externalizes prior human experience as explicit workflows, rules, and execution constraints that guide the agent toward effective and reliable training behavior.
On PostTrainBench, AutoTrainess consistently outperforms CLI-only baselines, achieving 26.94 average score with GPT-5.4 (Codex) versus 23.21 for CLI-only. It also generalizes across models and harnesses, improving DeepSeek-V4-Flash (OpenCode) from 12.13 to 19.58 1 1 1 Code and data are avaliable at https://github.com/simple-agent-lab/AutoTrainess . .

## 1 Introduction

Figure: Figure 2: AutoTrainess is a LM agent that interacts with training environments through a training-specialized Agent-Computer Interface, named AutoTrainHub.

Recent work has demonstrated the impressive capabilities of frontier LM agents in real-world automation tasks such as software engineering 8; 7 or scientific discovery 12; 24.
Despite this progress, there remains a persistent gap: as language models become increasingly powerful, the process of improving them still relies on extensive human effort.
One possible pathway for LMs’ self-improvement is to treat LM training as a software engineering task, leveraging coding agents to autonomously generate and optimize training code.
However, successful LLM training requires more than strong coding ability: experienced human engineers also rely on substantial accumulated training expertise, empirical intuitions and artifacts (e.g., sophisticated data generation pipelines).
Inspired by this observation, we investigate whether LM agents can similarly benefit from prior experience and artifacts created by human researchers when performing autonomous training tasks.

Consider the simple setting of an agent interacting directly with a command-line interface (CLI) on Linux. We find that even powerful agents struggle to
manage training tasks effectively. For example, when creating custom training datasets, the agent may introduce errors in packing sequences to maximum length or fail to use correct chat templates, resulting in suboptimal training data and frequent dataloader exceptions. These limitations suggest that model self-improvement remains difficult without access to human expertise,
highlighting the necessity of a training-aware agent-computer interface (ACI) 23 that packages human-curated best practices into agent harness.

We introduce AutoTrainess, an LM agent empowered by a training-specialized ACIs that enables autonomous end-to-end LLM post-training. In contrast to CLI-only agent’s highly uncertain action space,
AutoTrainess’s ACI provides a set of semantically meaningful training heuristics supported by standard pipelines for data preparation, model training, and evaluation.
Through AutoTrainess’s rich human prior knowledge on training dynamics and data processing, the agent can diagnose issues and refine training strategies efficiently, enabling stable and iterative model improvement with reduced human supervision.

Using GPT-5.4 (Codex) as backbone, AutoTrainess achieves 26.94 average on PostTrainBench 15, outperforming the CLI-only baseline (23.21). We conduct ablation studies on PostTrainBench Qwen3-4B subset and confirm that the ACIs contributes 3–8 point improvements. To validate the generalization of our ACIs, we show that it is portable to a different LM; AutoTrainess powered by DeepSeek-V4-Flash (OpenCode) still attains 19.58 on average, significantly outperforming the CLI-only baseline of 12.13 while maintaining stable end-to-end training with minimal human oversight.

Our contributions are twofold. First, we identify the fundamental limitations of raw CLI interfaces in LLM training tasks and introduce a training-specialized Agent-Computer Interface (ACI) that embeds rich human expertise directly into the agent-environment interaction. Second, we present AutoTrainess, which achieves strong results on PostTrainBench, enables stable and reliable autonomous end-to-end post-training, and provides a practical pathway toward scalable LLM self-improvement.

## 2 AutoTrainess

We investigate whether autonomous LLM training can benefit from externally provided prior experience, much as human engineers draw upon accumulated workflows and artifacts rather than constructing each training pipeline from scratch.

To this end, we instantiate human experience as a repository of reusable ACIs, named AutoTrainHub, that provide the agent with structured guidance for conducting iterative post-training.
Rather than leaving the agent to solve training as
an unconstrained coding task, AutoTrainHub expose a stage-wise interface over the workspace, specifying what
artifacts should be used, what outputs should be produced, and what operational constraints must be respected.
AutoTrainHub organizes autonomous training into a closed-loop workflow with four modules: data process, training, evaluation, and logging&planning. At a high level, the agent first analyzes evidence from prior experiments to define the goal of the next iteration, then prepares
training data, runs model training, evaluates the resulting checkpoint on the benchmark’s real evaluation pipeline, and finally records the completed iteration in a structured experiment log.

Data processing.
We introduce three explicit actions to construct task-aligned training dataset : data selection, data construction, and data validation.
The data selection action identifies the problems or behaviors implied by previous failures and selects initial source
directions for data construction, such as existing local data, externally collected data, or model-distilled data.
The data construction action supports a bounded set of dataset operations, including extraction, cleaning,
deduplication, rewriting, restructuring, synthesis, distillation, and schema normalization. Before deciding the final
training sample format, the agent must inspect the benchmark’s actual evaluation interface, such as evaluation scripts,
chat templates, or task-context files.
The data validation action first checks whether the constructed examples match the benchmark-facing task interface and
rendered examples. It then filters out low-quality or risky samples, including garbage text, corrupted examples,
duplicates, unrealistic synthetic patterns, and potential data leakage.
The validation stage returns one of three outcomes: approval for training, return to construction, or return to
selection. This explicit return mechanism allows the agent to distinguish between execution errors within a viable data direction and failures caused by an incorrect data optimization direction itself.

Training.
The training interface provides a stable training entry based on LlamaFactory 26.
Rather than allowing the agent to freely choose training frameworks or implement custom loops, the interface fixes LlamaFactory as the training backend and provides dedicated scripts for installation and execution. This reduces engineering variance and makes autonomous training more reproducible.
For supervised fine-tuning, the interface requires full-parameter fine-tuning, a small validation run before scaling up, and export of an evaluation-ready final model. For reinforcement learning, this interface is only applicable when supported by recent evaluation evidence. In such cases, the agent must explicitly specify the reward definition or feedback signal actually used during the run.
In both cases, failures must be debugged within the same LlamaFactory-based workflow rather than bypassed by switching to another framework. This encodes a practical prior commonly used by human practitioners: in iterative post-training, maintaining a stable and comparable training results is often more valuable than maximizing implementation flexibility.

Evaluation.
The evaluation interface runs the trained checkpoint on the benchmark’s real evaluation pipeline and records the evidence needed for the next iteration. Its primary role is to ensure that downstream decisions are based on comparable and sufficiently informative evaluation results.
Specifically, the interface requires the agent to evaluate the final model using the benchmark’s canonical entrypoint, save raw outputs under an evaluation results directory, and produce a concise evaluation summary.
In addition, the agent must generate a compact summary containing 15 randomly selected evaluation examples with score, input, target, and model output.
When compatible evaluation logs are available, a log parser script is used to extract these samples into an inspection artifact.
Finally, the agent is required to summarize the main observed failure modes and classify each as primarily a data problem, a training problem, or an inference or template problem. This structured diagnostic output directly informs the next planning and new decisions.

Logging & Planning.
The logging interface appends one structured entry to the experiment log after each completed iteration. Each entry records the iteration context, motivation, references consulted, starting checkpoint, training data, method, training configuration, evaluation protocol, result, analysis, generated artifacts, and next action.
This persistent log serves as a compact long-horizon memory over the training process. It preserves concrete evidence across iterations, supports reproducibility and retrospective analysis, and provides subsequent agent runs with a structured summary of prior decisions and their outcomes.
The planning interface defines the objective of the next experiment iteration based on empirical evidence from prior runs. Given previous evaluation results, training outcomes, and the current workspace state, the agent is required to identify the main observed problems, decide the primary objective of the current iteration, specify the planned intervention, and define a concrete success criterion.

AutoTrainHub treats human experience not as latent capability inside the base model, but as an explicit external scaffold for autonomous training.
It integrates researcher experience about how to plan iterations, construct benchmark-aligned data, run stable training, perform evidence-grounded evaluation, and preserve experimental memory.
By exposing these prior human experience through explicit ACIs, we transform autonomous LLM training from an open-ended software engineering problem into a structured sequential decision process.
This setup allows us to directly study whether such externalized prior experience improves the effectiveness and efficiency of agentic self-improvement.

## 3 Experiments

**Table 1: Main results on PostTrainBench. Each agent receives 4 base models and 7 benchmarks for each model, access to an H20 GPU, and a 10-hour time limit. Detailed Results are shown at [Appendix D](#A4).**
|  | PostTrainBench |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Harness | Qwen3-1.7B | Qwen3-4B | SmolLM-3B | Gemma-4B | Avg(%). |
| Instruct | 49.41 | 63.75 | 44.81 | 46.58 | 51.14 |
| Base | 6.66 | 14.34 | 4.52 | 4.60 | 7.53 |
| CLI-only |  |  |  |  |  |
| w/ GPT-5.4 (Codex) | 16.90 | 27.09 | 23.96 | 24.88 | 23.21 |
| w/ GPT-5.4 (OpenCode) | 20.01 | 17.01 | 19.51 | 22.32 | 19.71 |
| w/ DeepSeek-V4-Flash (OpenCode) | 8.14 | 15.18 | 14.77 | 10.43 | 12.13 |
| AutoTrainess |  |  |  |  |  |
| w/ GPT-5.4 (Codex) | 25.67 | 32.60 | 25.60 | 23.88 | 26.94 |
| w/ GPT-5.4 (OpenCode) | 22.08 | 25.91 | 24.20 | 21.20 | 23.35 |
| w/ DeepSeek-V4-Flash (OpenCode) | 16.72 | 21.76 | 15.82 | 24.01 | 19.58 |

### 3.1 Evaluation Setup

Datasets. We primarily evaluate on PostTrainBench 15, which pairs each run with one of seven target benchmarks: AIME 2025 15, ArenaHard 9, BFCL 13, GPQA 16, GSM8K 4, HealthBench 1, and HumanEval 3. These benchmarks cover mathematical reasoning, function calling, general knowledge, health, and code generation.

Models. PostTrainBench 15 evaluates post-training over four base models: Qwen3-1.7B, Qwen3-4B, SmolLM3-3B, and Gemma-3-4B. All results, ablations, and analyses in this paper are based on two leading CLI agent scaffolds, Codex and OpenCode. We experimented with a number of closed and open-source agent backbones, including GPT-5.4 and DeepSeek-V4-Flash. We also tried smaller agent models such as Qwen3.5-35B-A3B, but found their performance in this autonomous training setting to be subpar.

Baselines. We compare AutoTrainess to CLI-only harness, where agents can only use the internal tools and system prompt to interact with the autonomous training environment. The CLI-only harness system asks the LM to launch a training process by writing and run code with a shell process in sandbox.

Metrics.We report the aggregated scores of PostTrainBench.
First, for each agent, we average its performance on each benchmark across the four base models (Qwen3-1.7B, Qwen3-4B, SmolLM3-3B, and Gemma-3-4B), yielding the per-benchmark agent score $s_{i}^{\rm agent}$.
Next, we compute the benchmark weights $w_{i}$ as follows:

$$ $w_{i}=\frac{1}{s_{\rm instruct}^{i}-s_{\rm base}^{i}},\quad\hat{w}_{i}=\frac{w_{i}}{\sum_{j}w_{j}}$ $$

where $s_{\rm instruct}^{i}$ and $s_{\rm base}^{i}$ are the scores of the official instruction-tuned model and the base model on benchmark $i$, respectively. This weighting scheme assigns higher weights to harder benchmarks where instruction tuning yields smaller gains.
The final aggregated score for each agent is then defined as:

$$ $\text{Score}_{\rm agent}=\sum_{i}\hat{w}_{i}\cdot s_{i}^{\rm agent}$ $$

This score provides a comprehensive measure of the agent’s post-training effectiveness under the GPU constraint.

### 3.2 Main Results

Across all systems, AutoTrainess w/ GPT-5.4 (Codex) achieves the best performance all-around, getting 26.94 overall score of the PostTrainBench full set and 32.60 of the Qwen3-4B subset. As shown in [Table 1](#S3.T1), compared to CLI-only, AutoTrainess yields 15% improved overall score relatively. An LM-friendly Training ACI is confirmed by AutoTrainess’s 15%-25% relative increase compared to CLI-only, with small decrease (< 5%) on Gemma-4B subset. On the OpenCode scaffold, AutoTrainess shows consistent performance gains, highlighting its robust generalization capability across various CLI agents.

### 3.3 Ablations

We perform several ablations of the AutoTrainess interfaces, specifically with respect to the AutoTrainess w/ GPT-5.4 (Codex) configuration, summarized in [Table 2](#S3.T2). Our further analysis shed light on interesting agent behavior along with the impact of different harness designs.

**Table 2: PostTrainBench (Qwen3-4B) performance under ablations to the interfaces in AutoTrainess w/ GPT5.4 (Codex).**
| Method | Overall |
| --- | --- |
| CLI-only | 26.7 |
| AutoTrainess | 32.6 |
| w/o data processing | 29.1 (-3.5) |
| w/o training | 20.2 (-12.4) |
| w/o evaluation | 24.0 (-8.6) |
| w/o logging&planning | 24.1 (-8.5) |

Figure: Figure 3: Failure rate under interface ablations. Each bar reports the action failure rate, and the second percentage is the difference from the full interfaces.

Data interfaces mainly protect the training input contract.
Removing the data interface produces the largest increase in train action failure rate, from $7.2$% to $12.7$%, a difference of $5.5$ percentage points.
At the same time, the evaluation action failure rate remains close to the full interface ($7.0$%, $-0.6$ percentage points from full).
As shown in [Figure 5](#S3.F5), the data interface leads the agent to engage substantially more in data-centric operations prior to training, including dataset inspection, cleaning, preference-pair construction, and synthetic data generation. This behavior suggests that the agent reasons more explicitly about dataset content, format, metadata, and sampling strategies before passing the data to the training pipeline.
Without this interface, errors about data loading are more likely to be deferred until debug or validation training, where they appear as training action failures rather than evaluation action failures.

Training interfaces have weaker direct effects on train failures but affect downstream evaluation.
The w/o train interface has a train action failure rate similar to the full interface ($6.7$%, $-0.5$ percentage points from full), suggesting that this interface does not strongly reduce per-command training action failures in the aggregate.
However, its evaluation action failure rate rises to $12.0$%, a difference of $4.4$ percentage points from full.
This suggests an indirect role: the train interface helps standardize checkpoint locations, merged-model outputs, and final-model handoff conventions.
When it is removed, training commands do not necessarily fail more often, but their artifacts can become more ambiguous or inconsistently referenced by downstream evaluation commands.
Overall, the ablation results suggest that each interface is most effective at guarding its corresponding boundary: the eval and logging&plan interfaces reduce evaluation action failures, the data interface reduces training-input action failures, and the train interface stabilizes the artifact handoff into evaluation.

Evaluation interfaces provide the strongest protection against evaluation action failures.
Figure [3](#S3.F3) shows that removing the eval interface causes the largest increase in evaluation action failures.
The full interface has an evaluation action failure rate of $7.6$%, while the w/o eval interface increases this rate to $22.8$%, a difference of $15.2$ percentage points.
By contrast, the corresponding train action failure rate remains close to the full interface ($6.6$% vs. $7.2$%; Figure [3](#S3.F3)).
This asymmetry indicates that the eval interface primarily improves the reliability of evaluation orchestration rather than general command execution.
Its contribution is to preserve benchmark-specific invocation conventions, limited-evaluation checks, output-path discipline, and model-serving prerequisites that are easy to violate when evaluation commands are assembled without a dedicated interface.

Logging and planning interfaces stabilize stateful evaluation loops.
The w/o logging&plan interface also substantially increases the evaluation action failure rate, from $7.6$% to $19.6$%, a difference of $12.0$ percentage points.
Unlike the w/o eval setting, this ablation does not remove the ability to call evaluation scripts.
Instead, it weakens the agent’s ability to maintain state across iterations: which model artifact should be evaluated, whether an output directory has been created, which vLLM server is active, and which previous fixes should be reused.
This makes evaluation attempts more brittle, especially for full and limited evaluations that depend on a consistent train-to-eval handoff.
The lower train action failure rate in this condition ($3.2$%, $-4.0$ percentage points from full) should not be interpreted as improved training reliability, since this interface ablation also runs fewer train and evaluation commands.

### 3.4 Exploration vs. Exploitation

Figure: Figure 4: Ablation on data related action counts in PostTrainBench trajectories.

As shown in Figure [5](#S3.F5), the full interface achieves the highest exploration (111 train-to-eval handoffs) while retaining 7 improvements, corresponding to a 6.3% retained-improvement yield. This demonstrates that the complete interface set enables broad search without sacrificing the ability to convert promising trials into durable gains.Removing the data interface maintains relatively high exploration (95 handoffs) but reduces retained improvements to 4 (4.2% yield). This suggests the data tools primarily enhance the quality of explored configurations by supporting standardized dataset construction, cleaning, and formatting, rather than merely increasing trial volume.In contrast, ablating the train interface sharply reduces exploration (58 handoffs) while preserving all 7 improvements, raising the yield to 12.1%. The training interface thus primarily broadens the search frontier by facilitating more training configurations and checkpoint variants.The eval interface plays a key role in selection: its removal lowers both handoffs (70) and retained improvements (4). Similarly, disabling logging&plan causes the most severe collapse (30 handoffs, 2 improvements), underscoring its importance for maintaining iterative state and closed-loop coordination.The CLI-only baseline (86 handoffs, 5 improvements) offers moderate performance but falls short of the full interface on both dimensions. Overall, the ablation results reveal complementary contributions: the train interface drives exploration breadth, data and eval tools improve the conversion of trials into retained gains, and logging & plan sustains the iterative search process.

Figure: Figure 6: Frequency of different agent behaviors in AutoTrainess at different time stages.
Refer to caption: https://ar5iv.labs.arxiv.org/html/2606.31551/assets/behaviour2.png

### 3.5 Analysis of Agent Behavior

We further analyze agent behavior under AutoTrainess using training trajectories produced by GPT-5.4 with Codex. From the concrete actions in these trajectories, we abstract a higher-level taxonomy with 6 major classes and 26 subclasses. We then use an LLM-as-a-judge to annotate the behavior exhibited at each iteration across different model–dataset training runs. Finally, we group iterations by their position in the training timeline to obtain the stage-wise behavior distribution shown in Figure [6](#S3.F6). Based on this, we achieve the following findings:

Agents first adapt to the benchmark before pursuing optimization.
At the beginning of a run, agents primarily calibrate the task interface. In the first two hours, baseline-based planning ($P1$) appears $18$ times, but disappears entirely in the remaining four blocks. Benchmark-interface work follows the same front-loaded pattern. Behaviors like Benchmark prompt alignment ($T1$), Template change ($T2$) and Benchmark-near data selection ($D1$) are more frequently distributed at the beginning, with their frequencies dropping rapidly thereafter. For tasks with time-consuming evaluations, agents often rely on lightweight validation ($E1$) over a small subset, allowing them to obtain feedback quickly and iterate within the limited time budget. These patterns indicate that agents typically begin by running the base model, aligning prompts and templates with the benchmark, and setting up a lightweight validation loop before attempting more specialized interventions. For example, in the GPQA run shown in Figure [7](#S3.F7), the early gains come from adding benchmark-aligned science multiple-choice data and preserving GPQA-style answer formatting.

Figure: (a) Qwen3-4B on GPQA
Refer to caption: https://ar5iv.labs.arxiv.org/html/2606.31551/assets/figures/performance_iteration_gpqa.png

Mid-run strategies shift toward targeted data construction and training.
After the training format stabilizes and the agent obtains error signal, agent’s strategies move toward data construction and optimization. For example, the data synthesis ($D4$), which rises from $19$ occurrences in first two hours to $40$ in the next two hours and remains high at $36$ in the next time block. Agents also start to explore a broader range of training strategies during this phase. DPO-style training ($U3$) 14 is absent in the first two hours, and then appears $7$ and $12$ times in block 2 and 3, respectively. Self-distillation updates $U4$ exhibit a similar trend, suggesting that agents increasingly experiment with more targeted optimization methods once they have identified actionable weaknesses. This pattern is also visible in Figure [7](#S3.F7): in the HumanEval example, the agent first normalizes the task format and then improves further by adding benchmark-guided and synthetic programming tasks.

Later iterations concentrate on remaining failures and targeted fixes.
Late iterations focus on improving the current best model by analyzing its remaining failure cases and adding targeted training data. For example, failure case diagnosis ($P4$) that analyzes the left error samples, increases almost monotonically. full-benchmark evaluation ($E2$) also becomes more prominent later in the run. This indicates that agents are more inclined to conduct full-scale evaluations near the end of a run, in order to obtain a more comprehensive understanding of the model’s shortcomings and avoid being misled by small validation sets. For example, in the ArenaHard trajectory in Figure [7](#S3.F7), later gains come from narrowing the data mixture toward rewrite and style-control examples and then switching from full fine-tuning to LoRA to preserve base writing quality.

Performance gains are primarily driven by benchmark alignment and targeted correction.
Figure [8](#S3.F8) further breaks down which behaviors are more often associated with improvements versus regressions. Among non-planning and non-evaluation strategies, the behaviors most strongly associated with improvements are benchmark-related data ($D1$, $8/26$ improving occurrences), template change ($T2$, $7/31$), self-distillation update ($U4$, $4/19$), data difficulty improvement ($D3$, $4/22$), and benchmark prompt alignment ($T1$, $8/46$). These strategies reduce mismatch between training and evaluation or directly address previously identified weaknesses. Instead, DPO-style training ($U3$) appears in only $1/35$ improving occurrences, making it one of the weakest strategies by this criterion. Annealing training ($U7$) also yields limited gains, with only 5 improving occurrences out of 119, suggesting that agents may still struggle to configure effective annealing data and training hyperparameters.

Agents exhibit training habits distinct from human workflows.
Agents show a strong preference for continuing from the current best checkpoint instead of rebuilding the data pipeline and restarting from the base model. Specifically, continual training ($U5$) from the best model appears $322$ times in the pooled trajectories, whereas retraining from base ($U6$) appears only $133$ times. This suggests that, once a promising checkpoint is found, the agent usually treats it as the anchor for subsequent exploration rather than repeatedly resetting optimization from scratch. This preference is likely tied to the limited time budget of the setting: with only around ten hours available, continuing from the strongest existing checkpoint is a much cheaper way to test new ideas than rerunning a full base-model training path after every data modification. Moreover, agents rarely use data augmentation in the conventional sense. Specifically, explicit augmentation or rewriting behavior ($D5$) appears only $4$ times in total. This is noticeably different from common human practice, where augmentation is often a standard method for improving robustness or generalization of model.

Figure: Figure 8: Statistics of agent behaviors most and least correlated with performance improvement.
Refer to caption: https://ar5iv.labs.arxiv.org/html/2606.31551/assets/figures/good_bad.png

## 4 Related Work

##### Automatic research agents.

Beyond software engineering 8; 19; 5 or computer use 20; 22, recent work has begun to study whether LLM agents can automate research with limited human intervention. The AI Scientist casts scientific discovery as an open-ended loop that proposes ideas, runs experiments, analyzes outcomes, and drafts papers automatically 11. OpenResearcher presents a research assistant for long-horizon literature exploration, evidence collection, and report generation 27, while a later work focuses on synthesizing long-horizon deep-research trajectories for training and evaluation 10. Gottweis et al. present AI co-scientist, a system that supports scientific hypothesis generation and iterative refinement in domain research workflows 6. These works suggest that automated research is a long-horizon, knowledge-intensive problem in which agents must combine domain expertise with sustained coordination across literature understanding, experimentation, and iterative refinement.

##### Benchmarks for research and experimentation agents.

Recently, there has been increasing interest in evaluating whether language agents can conduct long-horizon research work. For example, MLAgentBench investigates the ability of language agents to conduct machine learning experimentation 7. CORE-Bench 17 assesses whether agents can reproduce results from existing research artifacts. PostTrainBench further extends this line of evaluation to end-to-end LLM post-training workflows 15. Beyond these benchmarks, other recent efforts have broadened the evaluation landscape to cover a wider range of scientific and engineering activities, including machine learning engineering, scientific law discovery, and literature discovery 2; 18; 25; 21.

## 5 Conclusion

We presented AutoTrainess, a training-specialized agent-computer interface for autonomous LLM post-training. By externalizing prior human experience as reusable skills for planning&logging, data preparation, training, and evaluation. AutoTrainess turns post-training into a more structured and reliable long-horizon workflow. Experiments on PostTrainBench show that AutoTrainess consistently outperforms CLI-only baselines, while ablations further indicate that data, training, evaluation, and planning interfaces each contribute meaningfully to the overall gains.

## References

- [1]
R. K. Arora, J. Wei, R. S. Hicks, P. Bowman, J. Q. Candela, F. Tsimpourlas, M. Sharman, M. Shah, A. Vallone, A. Beutel, J. Heidecke, and K. Singhal (2025)
HealthBench: Evaluating Large Language Models Towards Improved Human
Health.
CoRR abs/2505.08775.
External Links: [https://doi.org/10.48550/arXiv.2505.08775](https://doi.org/10.48550/arXiv.2505.08775),
[10.48550/ARXIV.2505.08775](https://doi.org/10.48550/ARXIV.2505.08775)
Cited by: [§3.1](#S3.SS1.p1.1).
- [2]
J. S. Chan, N. Chowdhury, O. Jaffe, J. Aung, D. Sherburn, E. Mays, G. Starace, K. Liu, L. Maksin, T. Patwardhan, A. Madry, and L. Weng (2025)
MLE-bench: Evaluating Machine Learning Agents on Machine Learning
Engineering.
In The Thirteenth International Conference on Learning Representations,
ICLR 2025, Singapore, April 24-28, 2025,
OpenReview.net.
External Links: [https://openreview.net/forum?id=6s5uXNWGIh](https://openreview.net/forum?id=6s5uXNWGIh)
Cited by: [§4](#S4.SS0.SSS0.Px2.p1.1).
- [3]
M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. d. O. Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba (2021)
Evaluating Large Language Models Trained on Code.
CoRR abs/2107.03374.
External Links: [https://arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374)
Cited by: [§3.1](#S3.SS1.p1.1).
- [4]
K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman (2021)
Training Verifiers to Solve Math Word Problems.
CoRR abs/2110.14168.
External Links: [https://arxiv.org/abs/2110.14168](https://arxiv.org/abs/2110.14168)
Cited by: [§3.1](#S3.SS1.p1.1).
- [5]
S. Gao, C. Gao, W. Gu, and M. R. Lyu (2024)
Search-Based LLMs for Code Optimization.
CoRR abs/2408.12159.
External Links: [https://doi.org/10.48550/arXiv.2408.12159](https://doi.org/10.48550/arXiv.2408.12159),
[10.48550/ARXIV.2408.12159](https://doi.org/10.48550/ARXIV.2408.12159)
Cited by: [§4](#S4.SS0.SSS0.Px1.p1.1).
- [6]
J. Gottweis, W. Weng, A. N. Daryin, T. Tu, A. Palepu, P. Sirkovic, A. Myaskovsky, F. Weissenberger, K. Rong, R. Tanno, K. Saab, D. Popovici, J. Blum, F. Zhang, K. Chou, A. Hassidim, B. Gokturk, A. Vahdat, P. Kohli, Y. Matias, A. Carroll, K. Kulkarni, N. Tomasev, Y. Guan, V. Dhillon, E. D. Vaishnav, B. Lee, T. R. D. Costa, J. R. Penadés, G. Peltz, Y. Xu, A. Pawlosky, A. Karthikesalingam, and V. Natarajan (2025)
Towards an AI co-scientist.
CoRR abs/2502.18864.
External Links: [https://doi.org/10.48550/arXiv.2502.18864](https://doi.org/10.48550/arXiv.2502.18864),
[10.48550/ARXIV.2502.18864](https://doi.org/10.48550/ARXIV.2502.18864)
Cited by: [§4](#S4.SS0.SSS0.Px1.p1.1).
- [7]
Q. Huang, J. Vora, P. Liang, and J. Leskovec (2024)
MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation.
In Forty-first International Conference on Machine Learning, ICML 2024,
Vienna, Austria, July 21-27, 2024,
(R. Salakhutdinov, Z. Kolter, K. A. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp Eds.), PMLR / OpenReview.net, pp. 20271–20309.
External Links: [https://proceedings.mlr.press/v235/huang24y.html](https://proceedings.mlr.press/v235/huang24y.html)
Cited by: [§1](#S1.p1.1),
[§4](#S4.SS0.SSS0.Px2.p1.1).
- [8]
C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. R. Narasimhan (2024)
SWE-bench: Can Language Models Resolve Real-world Github Issues?.
In The Twelfth International Conference on Learning Representations,
ICLR 2024, Vienna, Austria, May 7-11, 2024,
OpenReview.net.
External Links: [https://openreview.net/forum?id=VTF8yNQM66](https://openreview.net/forum?id=VTF8yNQM66)
Cited by: [§1](#S1.p1.1),
[§4](#S4.SS0.SSS0.Px1.p1.1).
- [9]
T. Li, W. Chiang, E. Frick, L. Dunlap, T. Wu, B. Zhu, J. E. Gonzalez, and I. Stoica (2025)
From Crowdsourced Data to High-quality Benchmarks: Arena-Hard and
Benchbuilder Pipeline.
In Forty-second International Conference on Machine Learning, ICML
2025, Vancouver, BC, Canada, July 13-19, 2025,
(A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaff, and J. Zhu Eds.), PMLR / OpenReview.net.
External Links: [https://proceedings.mlr.press/v267/li25h.html](https://proceedings.mlr.press/v267/li25h.html)
Cited by: [§3.1](#S3.SS1.p1.1).
- [10]
Z. Li, D. Jiang, X. Ma, H. Zhang, P. Nie, Y. Zhang, K. Zou, J. Xie, Y. Zhang, and W. Chen (2026)
OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research
Trajectory Synthesis.
CoRR abs/2603.20278.
External Links: [https://doi.org/10.48550/arXiv.2603.20278](https://doi.org/10.48550/arXiv.2603.20278),
[10.48550/ARXIV.2603.20278](https://doi.org/10.48550/ARXIV.2603.20278)
Cited by: [§4](#S4.SS0.SSS0.Px1.p1.1).
- [11]
C. Lu, C. Lu, R. T. Lange, J. N. Foerster, J. Clune, and D. Ha (2024)
The AI Scientist: Towards Fully Automated Open-Ended Scientific
Discovery.
CoRR abs/2408.06292.
External Links: [https://doi.org/10.48550/arXiv.2408.06292](https://doi.org/10.48550/arXiv.2408.06292),
[10.48550/ARXIV.2408.06292](https://doi.org/10.48550/ARXIV.2408.06292)
Cited by: [§4](#S4.SS0.SSS0.Px1.p1.1).
- [12]
A. Novikov, N. Vu, M. Eisenberger, E. Dupont, P. Huang, A. Z. Wagner, S. Shirobokov, B. Kozlovskii, F. J. R. Ruiz, A. Mehrabian, M. P. Kumar, A. See, S. Chaudhuri, G. Holland, A. Davies, S. Nowozin, P. Kohli, and M. Balog (2025)
AlphaEvolve: A coding agent for scientific and algorithmic discovery.
CoRR abs/2506.13131.
External Links: [https://doi.org/10.48550/arXiv.2506.13131](https://doi.org/10.48550/arXiv.2506.13131),
[10.48550/ARXIV.2506.13131](https://doi.org/10.48550/ARXIV.2506.13131)
Cited by: [§1](#S1.p1.1).
- [13]
S. G. Patil, H. Mao, F. Yan, C. C. Ji, V. Suresh, I. Stoica, and J. E. Gonzalez (2025)
The Berkeley Function Calling Leaderboard (BFCL): From Tool Use
to Agentic Evaluation of Large Language Models.
In Forty-second International Conference on Machine Learning, ICML
2025, Vancouver, BC, Canada, July 13-19, 2025,
(A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaff, and J. Zhu Eds.), PMLR / OpenReview.net.
External Links: [https://proceedings.mlr.press/v267/patil25a.html](https://proceedings.mlr.press/v267/patil25a.html)
Cited by: [§3.1](#S3.SS1.p1.1).
- [14]
R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn (2023)
Direct Preference Optimization: Your Language Model is Secretly a
Reward Model.
In Advances in Neural Information Processing Systems 36: Annual Conference
on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans,
LA, USA, December 10 - 16, 2023,
(A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine Eds.).
External Links: [http://papers.nips.cc/paper\_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html](http://papers.nips.cc/paper\_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)
Cited by: [§3.5](#S3.SS5.p3.1).
- [15]
B. Rank, H. Bhatnagar, A. Prabhu, S. Eisenberg, K. Nguyen, M. Bethge, and M. Andriushchenko (2026)
PostTrainBench: Can LLM Agents Automate LLM Post-Training?.
CoRR abs/2603.08640.
External Links: [https://doi.org/10.48550/arXiv.2603.08640](https://doi.org/10.48550/arXiv.2603.08640),
[10.48550/ARXIV.2603.08640](https://doi.org/10.48550/ARXIV.2603.08640)
Cited by: [§1](#S1.p4.1),
[§3.1](#S3.SS1.p1.1),
[§3.1](#S3.SS1.p2.1),
[§4](#S4.SS0.SSS0.Px2.p1.1).
- [16]
D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman (2023)
GPQA: A Graduate-Level Google-Proof Q&A Benchmark.
CoRR abs/2311.12022.
External Links: [https://doi.org/10.48550/arXiv.2311.12022](https://doi.org/10.48550/arXiv.2311.12022),
[10.48550/ARXIV.2311.12022](https://doi.org/10.48550/ARXIV.2311.12022)
Cited by: [§3.1](#S3.SS1.p1.1).
- [17]
Z. S. Siegel, S. Kapoor, N. Nadgir, B. Stroebl, and A. Narayanan (2024)
CORE-Bench: Fostering the Credibility of Published Research Through
a Computational Reproducibility Agent Benchmark.
Trans. Mach. Learn. Res. 2024.
External Links: [https://openreview.net/forum?id=BsMMc4MEGS](https://openreview.net/forum?id=BsMMc4MEGS)
Cited by: [§4](#S4.SS0.SSS0.Px2.p1.1).
- [18]
G. Starace, O. Jaffe, D. Sherburn, J. Aung, J. S. Chan, L. Maksin, R. Dias, E. Mays, B. Kinsella, W. Thompson, J. Heidecke, A. Glaese, and T. Patwardhan (2025)
PaperBench: Evaluating AI's Ability to Replicate AI Research.
In Forty-second International Conference on Machine Learning, ICML
2025, Vancouver, BC, Canada, July 13-19, 2025,
(A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaff, and J. Zhu Eds.), PMLR / OpenReview.net.
External Links: [https://proceedings.mlr.press/v267/starace25a.html](https://proceedings.mlr.press/v267/starace25a.html)
Cited by: [§4](#S4.SS0.SSS0.Px2.p1.1).
- [19]
C. S. Xia, Y. Deng, S. Dunn, and L. Zhang (2024)
Agentless: Demystifying LLM-based Software Engineering Agents.
CoRR abs/2407.01489.
External Links: [https://doi.org/10.48550/arXiv.2407.01489](https://doi.org/10.48550/arXiv.2407.01489),
[10.48550/ARXIV.2407.01489](https://doi.org/10.48550/ARXIV.2407.01489)
Cited by: [§4](#S4.SS0.SSS0.Px1.p1.1).
- [20]
T. Xie, D. Zhang, J. Chen, X. Li, S. Zhao, R. Cao, T. J. Hua, Z. Cheng, D. Shin, F. Lei, Y. Liu, Y. Xu, S. Zhou, S. Savarese, C. Xiong, V. Zhong, and T. Yu (2024)
OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real
Computer Environments.
CoRR abs/2404.07972.
External Links: [https://doi.org/10.48550/arXiv.2404.07972](https://doi.org/10.48550/arXiv.2404.07972),
[10.48550/ARXIV.2404.07972](https://doi.org/10.48550/ARXIV.2404.07972)
Cited by: [§4](#S4.SS0.SSS0.Px1.p1.1).
- [21]
L. Xiong, K. Luo, Z. Xia, W. Zhang, J. Yao, Z. Liu, J. Shao, J. Chen, H. Qian, X. Yang, Q. Yu, H. Li, C. Yue, X. Du, Y. Wang, Y. Liu, H. Xu, and Z. Dou (2026)
AutoResearchBench: Benchmarking AI Agents on Complex Scientific
Literature Discovery.
CoRR abs/2604.25256.
External Links: [https://doi.org/10.48550/arXiv.2604.25256](https://doi.org/10.48550/arXiv.2604.25256),
[10.48550/ARXIV.2604.25256](https://doi.org/10.48550/ARXIV.2604.25256)
Cited by: [§4](#S4.SS0.SSS0.Px2.p1.1).
- [22]
J. Yang, S. Shao, D. Liu, and J. Shao (2025)
RiOSWorld: Benchmarking the Risk of Multimodal Computer-Use Agents.
CoRR abs/2506.00618.
External Links: [https://doi.org/10.48550/arXiv.2506.00618](https://doi.org/10.48550/arXiv.2506.00618),
[10.48550/ARXIV.2506.00618](https://doi.org/10.48550/ARXIV.2506.00618)
Cited by: [§4](#S4.SS0.SSS0.Px1.p1.1).
- [23]
J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. R. Narasimhan, and O. Press (2024)
SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.
In The Thirty-eighth Annual Conference on Neural Information Processing Systems,
External Links: [https://arxiv.org/abs/2405.15793](https://arxiv.org/abs/2405.15793)
Cited by: [§1](#S1.p2.1).
- [24]
Z. Yu, K. Feng, Y. Zhao, S. He, X. Zhang, and A. Cohan (2025)
AlphaResearch: Accelerating New Algorithm Discovery with Language
Models.
CoRR abs/2511.08522.
External Links: [https://doi.org/10.48550/arXiv.2511.08522](https://doi.org/10.48550/arXiv.2511.08522),
[10.48550/ARXIV.2511.08522](https://doi.org/10.48550/ARXIV.2511.08522)
Cited by: [§1](#S1.p1.1).
- [25]
T. Zheng, K. K. Tam, N. H. K. Nguyen, B. Xu, Z. Wang, J. Cheng, H. T. Tsang, W. Wang, J. Bai, T. Fang, Y. Song, G. Y. Wong, and S. See (2025)
NewtonBench: Benchmarking Generalizable Scientific Law Discovery in
LLM Agents.
CoRR abs/2510.07172.
External Links: [https://doi.org/10.48550/arXiv.2510.07172](https://doi.org/10.48550/arXiv.2510.07172),
[10.48550/ARXIV.2510.07172](https://doi.org/10.48550/ARXIV.2510.07172)
Cited by: [§4](#S4.SS0.SSS0.Px2.p1.1).
- [26]
Y. Zheng, R. Zhang, J. Zhang, Y. Ye, and Z. Luo (2024)
LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models.
In Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 3: System Demonstrations), ACL 2024, Bangkok,
Thailand, August 11-16, 2024,
(Y. Cao, Y. Feng, and D. Xiong Eds.), Association for Computational Linguistics, pp. 400–410.
External Links: [https://doi.org/10.18653/v1/2024.acl-demos.38](https://doi.org/10.18653/v1/2024.acl-demos.38),
[10.18653/V1/2024.ACL-DEMOS.38](https://doi.org/10.18653/V1/2024.ACL-DEMOS.38)
Cited by: [§2](#S2.p4.1).
- [27]
Y. Zheng, S. Sun, L. Qiu, D. Ru, C. Jiayang, X. Li, J. Lin, B. Wang, Y. Luo, R. Pan, Y. Xu, Q. Min, Z. Zhang, Y. Wang, W. Li, and P. Liu (2024)
OpenResearcher: Unleashing AI for Accelerated Scientific Research.
In Proceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing: EMNLP 2024 - System Demonstrations, Miami,
Florida, USA, November 12-16, 2024,
(D. I. H. Farías, T. Hope, and M. Li Eds.), Association for Computational Linguistics, pp. 209–218.
External Links: [https://doi.org/10.18653/v1/2024.emnlp-demo.22](https://doi.org/10.18653/v1/2024.emnlp-demo.22),
[10.18653/V1/2024.EMNLP-DEMO.22](https://doi.org/10.18653/V1/2024.EMNLP-DEMO.22)
Cited by: [§4](#S4.SS0.SSS0.Px1.p1.1).

## Appendix A Case Study on the Effects of Different Skills

We use two representative examples to illustrate the role of different skills in AutoTrainess. The first compares the AutoTrainess against the data-skill ablation on ArenaHard, and the second compares the AutoTrainess against the eval-skill ablation on HealthBench.

##### Data skill helps the agent align training data with benchmark format and distribution.

Figure [9](#A1.F9) shows a clear difference between the full system and the variant without the data skill. With the data skill, AutoTrainess eventually constructs a training mix centered on synthesized long prompt-polish pairs, rewrite anchors, and long roleplay or system-prompt rewrite tasks, which better match the style and distribution of ArenaHard. This route yields a substantially stronger final score. By contrast, the ablation version keeps trying broader or noisier writing mixtures, such as multilingual anchors plus lighter writing data or multilingual roleplay and CJK data, but fails to reshape them into the benchmark-facing supervision format required by the task. As a result, the no-data-skill variant improves only modestly and plateaus at a much lower level. This example suggests that the main benefit of the data skill is not simply giving the agent more data, but teaching it convert available data into a form whose format and distribution are much closer to the benchmark.

##### Eval skill helps the agent use evaluation evidence to drive targeted optimization.

Figure [10](#A1.F10) highlights a different failure mode. With the eval skill, AutoTrainess first improves the core medical-chat SFT mix, then uses evaluation feedback to synthesize and upweight benchmark-shaped examples that specifically target the remaining weaknesses, such as procedural guidance, preventive and travel summaries, multilingual safety, and structured patient-facing responses. This evaluation-guided shift produces another substantial gain late in the run. In contrast, the no-eval-skill ablation does not develop such targeted follow-up data. After an initial lift, it mostly keeps the same clean core fixed and continues by retrying different training parameters such as random seeds and different recipes, rather than constructing new supervision that addresses the concrete problems surfaced by evaluation. Consequently, its later-stage progress is much more limited. This case study suggests that the eval skill is valuable not only for measuring performance, but also for turning evaluation outcomes into actionable diagnoses that guide the next round of optimization.

Figure: Figure 9: Ablation study of data skill on ArenaHard and Qwen3-4B.
Refer to caption: https://ar5iv.labs.arxiv.org/html/2606.31551/assets/figures/performance_iteration_arenahard_main_vs_data_ablation.png

Figure: Figure 10: Ablation study of eval skill on HealthBench and Qwen3-4B.
Refer to caption: https://ar5iv.labs.arxiv.org/html/2606.31551/assets/figures/performance_iteration_healthbench_main_vs_eval_ablation.png

## Appendix B Detailed Explanation of the Agent Behavior Taxonomy

In this section, we briefly explain the meaning of the behavior taxonomy used in Section 3.5.

### B.1 Evaluation Strategy

##### E ​ 1 E1 Subset Eval.

In this iteration, the agent actually runs a sampled subset for validation. This is used for quick comparison, smoke testing, or low-cost iteration.

##### E ​ 2 E2 Full Eval.

In this iteration, the agent actually runs the full benchmark or full official evaluation set.

##### E ​ 3 E3 Fine-grained Eval.

In this iteration, beyond evaluating the final result score, the agent conducts additional statistical analyses from multiple perspectives.

### B.2 Input Format Strategy

##### T ​ 1 T1 Prompt Alignment.

The agent explicitly aligns the prompt to the benchmark prompt contract or user-message style in this iteration.

##### T ​ 2 T2 Template Adjustment.

The agent explicitly changes the model-side template or wrapper in this iteration. This includes chat-template changes, role/message schema changes, or think/non-think template switches.

##### T ​ 3 T3 Prompt Style Change.

The agent explicitly changes prompt wording, system instructions, or answer-behavior instructions without changing the core task family.

### B.3 Output Format Strategy

##### O ​ 1 O1 Direct Answer.

In this iteration, the agent proposes training the model on data containing direct answers only, without explicit reasoning traces.

##### O ​ 2 O2 Rationale + Answer.

In this iteration, the agent proposes training the model on data that includes both rationales and final answers.

##### O ​ 3 O3 Targeted Format.

In this iteration, the agent proposes adopting a benchmark-facing output contract or an exact structured answer schema, such as ‘ANSWER: <number>‘ or ‘ANSWER: <LETTER>‘.

### B.4 Data Strategy

##### D ​ 1 D1 Benchmark-related Data Source Selection.

The agent chooses a data source that is highly similar to the benchmark task distribution. This is often an early move to find the most benchmark-like public data on huggingface.

##### D ​ 2 D2 Training Dataset Expansion.

The agent adds more data from the similar task family primarily to increase volume. The emphasis here is scale rather than special difficulty or narrow targeting.

##### D ​ 3 D3 Data Difficulty Improvement.

The agent deliberately chooses harder, expert-level, or challenge-style data. The main change is increasing task difficulty rather than simply increasing data size.

##### D ​ 4 D4 Data Synthesis.

The agent introduces genuinely new examples that were created rather than taken directly from an existing public dataset. This includes self-generated or programmatically constructed samples.

##### D ​ 5 D5 Data Augmentation.

The agent rewrites or transforms existing examples into new supervision views while keeping them grounded in the same source rows. The main change is creating derived variants rather than introducing a new source.

##### D ​ 6 D6 Data Cleanup/Dedup/Rebalance.

The agent cleans, deduplicates, trims, filters, or rebalances the dataset without introducing new data.

### B.5 Training Strategy

##### U ​ 1 U1 Full SFT.

This iteration runs full-parameter supervised fine-tuning. The core update is standard full-model SFT.

##### U ​ 2 U2 PEFT.

This iteration runs parameter-efficient supervised fine-tuning. Typical examples include LoRA-style updates.

##### U ​ 3 U3 DPO-style Training.

This iteration runs preference-style training such as DPO, ORPO, or other chosen-versus-rejected objectives.

##### U ​ 4 U4 Self-distillation.

This iteration runs distillation-style updates built from accepted or model-generated data.

##### U ​ 5 U5 Continual Training.

The training run in this iteration starts from a previously trained checkpoint.

##### U ​ 6 U6 Train from Base Model.

The training run in this iteration starts from the original base model.

##### U ​ 7 U7 Annealing Training.

The agent uses a deliberately gentler continuation strategy, such as fewer steps or a lower learning rate. The main goal is to reduce update strength while continuing training.

### B.6 Planning Strategy

##### P ​ 1 P1 Baseline-based Planning.

The agent runs a baseline first and uses the observed starting behavior to decide the first optimization direction.

##### P ​ 2 P2 Error Pattern Diagnosis.

The agent diagnoses broad, recurring error patterns from observed failures, such as using the wrong template, producing truncated outputs, or repeatedly violating format constraints.

##### P ​ 3 P3 Hypothesis Testing.

The agent states or implies a concrete hypothesis for why the next change should help. The subsequent iteration is used to test whether the evidence supports that hypothesis.

##### P ​ 4 P4 Failure Case Diagnosis.

The agent focuses on specific remaining failure cases to identify the underlying capability gaps or unresolved weaknesses of the model. Rather than addressing broad error patterns, the agent examines concrete failed samples to determine which abilities are still insufficient.

## Appendix C Instructions of AutoTrain Hub Workflow

### C.1 AGENTS.md

Figure: Figure 11: AGENTS.md for AutoTrainess framework.

### C.2 Plan

Figure: Figure 12: Instruction of Plan skill.

### C.3 Data Process

Figure: Figure 13: Instruction of Data Process skill.

#### C.3.1 Selection

Figure: Figure 14: Instruction of Selection in Data Process skill.

#### C.3.2 Construction

Figure: Figure 15: Instruction of Construction in Data Process skill.

#### C.3.3 Validation

Figure: Figure 16: Instruction of Validation in Data Process skill.

### C.4 Training

Figure: Figure 17: Instruction of Training skill.

#### C.4.1 SFT

Figure: Figure 18: Instruction of SFT in Training skill.

#### C.4.2 RL

Figure: Figure 19: Instruction of RL in Training skill.

#### C.4.3 Shared Instruction

Figure: Figure 20: Shared instruction in Training skill.

### C.5 Evaluation

Figure: Figure 21: Instruction of Evaluation skill.

### C.6 Log

Figure: Figure 22: Instruction of Log skill.

## Appendix D Detailed Results on PostTrainBench

**Table 3: Results of AIME2025 on PostTrainBench.**
|  | AIME2025 |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Harness | Qwen3-1.7B | Qwen3-4B | SmolLM-3B | Gemma-4B | Avg. |
| Instruct | 26.67 | 53.33 | 26.67 | 10.00 | 29.17 |
| Base | 0.00 | 3.33 | 3.33 | 0.00 | 1.67 |
| CLI-only |  |  |  |  |  |
| w/ GPT-5.4 (Codex) | 0.00 | 6.67 | 3.33 | 3.33 | 3.33 |
| w/ GPT-5.4 (OpenCode) | 3.33 | 0.00 | 0.00 | 0.00 | 0.83 |
| w/ DeepSeek-V4-Flash (OpenCode) | 0.00 | 3.33 | 0.00 | 0.00 | 0.83 |
| AutoTrainess |  |  |  |  |  |
| w/ GPT-5.4 (Codex) | 0.00 | 6.67 | 0.00 | 0.00 | 1.67 |
| w/ GPT-5.4 (OpenCode) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| w/ DeepSeek-V4-Flash (OpenCode) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Table 4: Results of ArenaHard Writing on PostTrainBench.**
|  | ArenaHard Writing |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Harness | Qwen3-1.7B | Qwen3-4B | SmolLM-3B | Gemma-4B | Avg. |
| Instruct | 50.00 | 86.84 | 49.20 | 94.80 | 70.21 |
| Base | 0.91 | 3.42 | 0.42 | 0.29 | 1.26 |
| CLI-only |  |  |  |  |  |
| w/ GPT-5.4 (Codex) | 3.89 | 3.36 | 3.61 | 0.29 | 2.79 |
| w/ GPT-5.4 (OpenCode) | 0.59 | 1.50 | 0.21 | 5.37 | 1.92 |
| w/ DeepSeek-V4-Flash (OpenCode) | 1.09 | 2.54 | 0.57 | 0.29 | 1.12 |
| AutoTrainess |  |  |  |  |  |
| w/ GPT-5.4 (Codex) | 0.32 | 1.41 | 0.10 | 16.76 | 4.65 |
| w/ GPT-5.4 (OpenCode) | 0.50 | 4.67 | 1.36 | 3.59 | 2.53 |
| w/ DeepSeek-V4-Flash (OpenCode) | 0.74 | 10.23 | 7.17 | 6.15 | 6.07 |

**Table 5: Results of BFCL on PostTrainBench.**
|  | BFCL |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Harness | Qwen3-1.7B | Qwen3-4B | SmolLM-3B | Gemma-4B | Avg. |
| Instruct | 94.00 | 95.00 | 84.00 | 67.00 | 85.00 |
| Base | 0.00 | 0.00 | 0.00 | 6.00 | 1.50 |
| CLI-only |  |  |  |  |  |
| w/ GPT-5.4 (Codex) | 54.00 | 55.00 | 91.00 | 96.00 | 74.00 |
| w/ GPT-5.4 (OpenCode) | 98.00 | 0.00 | 0.00 | 97.00 | 48.75 |
| w/ DeepSeek-V4-Flash (OpenCode) | 0.00 | 0.00 | 0.00 | 6.00 | 1.50 |
| AutoTrainess |  |  |  |  |  |
| w/ GPT-5.4 (Codex) | 90.00 | 100.00 | 97.00 | 100.00 | 96.75 |
| w/ GPT-5.4 (OpenCode) | 90.00 | 95.00 | 95.00 | 92.00 | 93.00 |
| w/ DeepSeek-V4-Flash (OpenCode) | 0.00 | 0.00 | 0.00 | 70.00 | 17.50 |

**Table 6: Results of GPQA Main on PostTrainBench.**
|  | GPQA Main |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Harness | Qwen3-1.7B | Qwen3-4B | SmolLM-3B | Gemma-4B | Avg. |
| Instruct | 35.49 | 44.64 | 33.26 | 31.47 | 36.22 |
| Base | 14.06 | 13.39 | 4.91 | 1.56 | 8.48 |
| CLI-only |  |  |  |  |  |
| w/ GPT-5.4 (Codex) | 11.16 | 27.68 | 22.54 | 25.22 | 21.65 |
| w/ GPT-5.4 (OpenCode) | 25.67 | 25.00 | 26.12 | 24.33 | 25.28 |
| w/ DeepSeek-V4-Flash (OpenCode) | 14.06 | 13.39 | 28.35 | 1.56 | 14.34 |
| AutoTrainess |  |  |  |  |  |
| w/ GPT-5.4 (Codex) | 29.91 | 35.71 | 28.13 | 22.99 | 29.19 |
| w/ GPT-5.4 (OpenCode) | 29.46 | 32.81 | 26.34 | 23.44 | 28.01 |
| w/ DeepSeek-V4-Flash (OpenCode) | 30.13 | 13.39 | 23.44 | 31.03 | 24.50 |

**Table 7: Results of GSM8K on PostTrainBench.**
|  | GSM8K |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Harness | Qwen3-1.7B | Qwen3-4B | SmolLM-3B | Gemma-4B | Avg. |
| Instruct | 88.48 | 93.78 | 82.18 | 83.55 | 87.00 |
| Base | 12.66 | 41.85 | 21.08 | 6.14 | 20.43 |
| CLI-only |  |  |  |  |  |
| w/ GPT-5.4 (Codex) | 44.20 | 42.38 | 64.67 | 52.99 | 51.06 |
| w/ GPT-5.4 (OpenCode) | 25.93 | 41.85 | 70.36 | 31.69 | 42.46 |
| w/ DeepSeek-V4-Flash (OpenCode) | 4.17 | 47.31 | 54.66 | 34.87 | 35.25 |
| AutoTrainess |  |  |  |  |  |
| w/ GPT-5.4 (Codex) | 54.74 | 75.59 | 57.70 | 40.64 | 57.17 |
| w/ GPT-5.4 (OpenCode) | 54.44 | 45.94 | 50.34 | 14.25 | 41.24 |
| w/ DeepSeek-V4-Flash (OpenCode) | 57.70 | 57.47 | 51.93 | 52.46 | 54.89 |

**Table 8: Results of HealthBench Easy on PostTrainBench.**
|  | HealthBench Easy |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Harness | Qwen3-1.7B | Qwen3-4B | SmolLM-3B | Gemma-4B | Avg. |
| Instruct | 44.92 | 52.72 | 29.58 | 46.06 | 43.32 |
| Base | 7.54 | 13.38 | 0.00 | 17.04 | 9.49 |
| CLI-only |  |  |  |  |  |
| w/ GPT-5.4 (Codex) | 15.06 | 21.09 | 7.34 | 18.09 | 15.40 |
| w/ GPT-5.4 (OpenCode) | 14.09 | 18.80 | 15.38 | 14.13 | 15.60 |
| w/ DeepSeek-V4-Flash (OpenCode) | 2.98 | 7.91 | 6.52 | 20.39 | 9.45 |
| AutoTrainess |  |  |  |  |  |
| w/ GPT-5.4 (Codex) | 14.95 | 18.20 | 18.82 | 12.22 | 16.05 |
| w/ GPT-5.4 (OpenCode) | 6.94 | 13.38 | 18.37 | 17.04 | 13.93 |
| w/ DeepSeek-V4-Flash (OpenCode) | 3.29 | 27.18 | 23.52 | 16.55 | 17.64 |

**Table 9: Results of HumanEval on PostTrainBench.**
|  | HumanEval |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| Harness | Qwen3-1.7B | Qwen3-4B | SmolLM-3B | Gemma-4B | Avg. |
| Instruct | 68.90 | 77.44 | 70.12 | 69.51 | 71.49 |
| Base | 7.93 | 36.59 | 6.10 | 0.61 | 12.81 |
| CLI-only |  |  |  |  |  |
| w/ GPT-5.4 (Codex) | 29.27 | 63.41 | 34.15 | 28.05 | 38.72 |
| w/ GPT-5.4 (OpenCode) | 10.37 | 36.59 | 39.63 | 33.54 | 30.03 |
| w/ DeepSeek-V4-Flash (OpenCode) | 37.20 | 50.00 | 30.49 | 24.39 | 35.52 |
| AutoTrainess |  |  |  |  |  |
| w/ GPT-5.4 (Codex) | 40.85 | 47.56 | 29.88 | 34.76 | 38.26 |
| w/ GPT-5.4 (OpenCode) | 21.95 | 40.24 | 28.05 | 40.24 | 32.62 |
| w/ DeepSeek-V4-Flash (OpenCode) | 36.59 | 70.12 | 6.70 | 31.10 | 36.13 |