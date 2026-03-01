## Experiment Goal

Build a reproducible, resume-ready compression pipeline for Qwen3-14B that applies structured pruning → recovery fine-tuning → quantization, while measuring the accuracy/throughput trade-offs. We compare SFT-based recovery (main) vs knowledge distillation recovery (ablation), and report results before and after final W4A16 quantization to quantify quantization sensitivity.

## Tools
	•	Hugging Face Transformers: model/tokenizer loading, training loop (Trainer), checkpoint I/O
	•	PEFT (LoRA/QLoRA): parameter-efficient recovery fine-tuning
	•	LLM Compressor (llmcompressor): one-shot structured N:M pruning and AWQ/GPTQ quantization recipes
	•	vLLM: fast inference and evaluation runtime (logprob-based multiple-choice scoring)
	•	Hugging Face Datasets: dataset loading and preprocessing
	•	PyTorch: training backend
	•	Matplotlib / Pandas: metrics aggregation and visualization

## Datasets

### Training (Recovery)

Source: HuggingFaceH4/ultrachat_200k
	•	SFT train set (10k)
	•	What it is: 10,000 (user → assistant) instruction-style pairs sampled from UltraChat.
	•	Purpose: Main recovery signal after pruning. We fine-tune (LoRA/QLoRA) to recover instruction-following behavior and downstream robustness.
	•	SFT eval set (200)
	•	What it is: 200 held-out samples from the same sampling pool (disjoint from the 10k train subset).
	•	Purpose: Lightweight training-time monitoring (loss trend / early sanity check). Not used for final reporting.
	•	Calibration set (calib, ~1k)
	•	What it is: ~1,000 prompts (messages) used only for pruning/quantization calibration.
	•	Purpose: Provides representative activations/statistics for compression algorithms (e.g., SparseGPT pruning, AWQ/GPTQ quantization). No gradient updates are performed with this set.

Note: KD (ablation) uses the same UltraChat prompts but replaces the assistant responses with teacher-generated outputs (sequence-level distillation), keeping the overall data distribution comparable to SFT.

### Evaluation (Multiple-Choice Benchmarks)

These datasets are used for pre-quant and post-quant evaluation under the same protocol (so we can attribute changes to pruning/recovery/quantization rather than data differences).
	•	HellaSwag (5,000)
	•	Task: Common-sense completion (4-way multiple choice).
	•	Role in this project: Primary broad-coverage benchmark; sensitive to general language understanding and coherence.
	•	ARC-Challenge (1,172)
	•	Task: Grade-school science questions (multiple choice).
	•	Role: More reasoning-heavy; helps detect recovery/quantization failures that don’t show up on purely “common-sense” tasks.
	•	Winogrande (1,267)
	•	Task: Pronoun/coreference disambiguation (2-way multiple choice).
	•	Role: Measures robustness on a classic linguistic reasoning task; often reveals subtle degradation after compression.
	•	TruthfulQA (MC, 817)
	•	Task: Truthfulness-oriented multiple choice.
	•	Role: Stress test for calibration/robustness; can be sensitive to compression-induced shifts in model preference.

Total evaluation size: 8,256 questions.

## Pruning

### What is pruning?

Pruning reduces model complexity by removing parameters or computations. In LLM compression, the most common form is weight pruning, where some weights are set to zero (or masked), reducing effective capacity and (sometimes) enabling faster inference if the sparsity pattern is supported by kernels/hardware.

Structured vs. unstructured sparsity
	•	Unstructured pruning: zeros are placed anywhere in the weight matrix (arbitrary sparsity pattern).
	•	Pros: flexible sparsity ratios (e.g., 80%, 90%).
	•	Cons: often does not speed up GPU inference without specialized sparse kernels; harder to exploit efficiently.
	•	Structured / semi-structured pruning (N:M): enforces a fixed pattern inside small blocks, e.g. 2:4.
	•	Example (2:4): in every group of 4 weights, exactly 2 are kept and 2 are zeroed (≈50% sparsity).
	•	Pros: much easier for optimized kernels to exploit; reproducible and well-defined.
	•	Cons: sparsity levels are discrete (e.g., 3:4, 2:4, 1:4).

### What is LLM Compressor (llmcompressor)?

LLM Compressor is a compression toolkit (from the vLLM ecosystem) that provides recipe-based, automated pruning and quantization workflows for LLMs. It includes implementations of common compression methods (e.g., SparseGPT for pruning, AWQ/GPTQ for quantization) and produces artifacts that integrate well with deployment stacks such as vLLM.

Our pruning choice and intent

We intentionally adopt a relatively aggressive pruning setup (2:4, 50% sparsity) to clearly observe the degradation → recovery trajectory:
	•	Goal: produce a visible recovery curve (pruned → SFT/KD recovery → quantized).
	•	Not a goal: pushing SOTA accuracy under pruning.

### Why we export back to a dense checkpoint

After pruning, we export the pruned model into a standard “dense” Hugging Face checkpoint (bf16/fp16 weights) before fine-tuning. This avoids tooling incompatibilities (e.g., some training/QLoRA loaders may not support compressed-tensors formats reliably) and makes the recovery stage (SFT/KD) straightforward and reproducible.

### Key parameter choices (pruning)
	•	SEED = 0
Fixes randomness for dataset sampling and improves reproducibility.
	•	NUM_CALIB_SAMPLES = 512
Number of calibration examples used for pruning statistics. 512 is a common default in official workflows, balancing representativeness and runtime.
	•	MAX_SEQ_LEN = 1024
Maximum sequence length used during pruning calibration. We keep it modest to fit comfortably on a 48GB GPU, and increase to 2048 only if memory allows.
	•	SPARSITY = 0.5 and MASK_STRUCTURE = "2:4"
We enforce 2:4 semi-structured pruning, i.e., for every 4 weights, keep 2.
	•	This corresponds to 50% sparsity.
	•	A less aggressive alternative is 3:4 (set MASK_STRUCTURE="3:4" and SPARSITY=0.25).
	•	DAMPENING_FRAC = 0.001, BLOCK_SIZE = 128
Stability/performance-related SparseGPT settings. We keep standard conservative values rather than tuning.
	•	TARGETS = ["Linear"], IGNORE = ["re:.*lm_head"]
We prune Linear layers (the dominant parameter contributor) and exclude lm_head to reduce the risk of damaging the output projection and destabilizing generation.

## Recovery Fine-tuning (SFT + KD Ablation)

After pruning, we perform a recovery stage to regain lost capability before final quantization and deployment evaluation. We include:
	1.	Instruction Fine-tuning (SFT) as the main recovery method
	2.	Knowledge Distillation (KD) as an ablation to compare recovery behavior under the same training budget

### Instruction Fine-tuning (SFT)

Instruction fine-tuning trains the pruned model to produce high-quality assistant responses given user prompts. Concretely, we use standard causal LM teacher-forcing with assistant-only loss masking (we do not train on the user/prompt tokens).

Role in this project: main recovery path, restoring instruction-following behavior and improving downstream robustness after pruning.

### Knowledge Distillation (KD) Ablation

Our KD is implemented as sequence-level distillation:
	•	We use the base (unpruned) model as the teacher.
	•	For each user prompt, the teacher generates a response.
	•	The student (pruned model) is then trained with teacher-forcing to match these teacher-generated responses.

Why not “classic” KL-based KD?
Traditional token-level KD minimizes a KL divergence between teacher and student logit distributions at each position. That requires running the teacher in parallel with the student or caching large logit tensors—both expensive for a 14B model under single-GPU constraints. As a practical compromise, we use teacher-forcing SFT on teacher-generated outputs, which is substantially cheaper and still provides a meaningful distillation signal for recovery.

Role in this project: ablation baseline for recovery—helps isolate whether “learning from teacher outputs” changes post-pruning and post-quantization robustness compared to standard SFT supervision.

### Key training settings
	•	SEED = 0
Ensures reproducible sampling, splitting, and training behavior.
	•	MAX_SEQ_LEN = 1024
Caps training sequence length for stability and throughput on a 48GB GPU.
	•	EVAL_RATIO = 0.02
Holds out 2% of the (10k) training subset for lightweight training-time monitoring.
	•	USE_QLORA = True
Uses QLoRA (4-bit base + LoRA adapters) to reduce memory footprint.
If set to False, we run standard LoRA on bf16 weights.
	•	LR = 1e-4, MAX_STEPS = 600
Conservative recovery budget: enough to observe a clear recovery trend without turning this into a large-scale training run.
	•	BATCH = 1, GRAD_ACCUM = 32
Effective batch size is achieved via gradient accumulation under GPU memory constraints.
	•	LoRA hyperparameters
	•	LORA_R = 16
	•	LORA_ALPHA = 32
	•	LORA_DROPOUT = 0.05
	•	LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
We adapt both attention projections and MLP projections, a common configuration for efficient recovery with low trainable-parameter overhead.

