# CharGPT
Full-Stack End-to-End Character-level Multilingual GPT (Generative Pretrained Transformer) Implementation From Scratch

A complete autoregressive text generation GPT model is built from scratch, including: tokenizer/detokenizer, embeddings, masked multi-head self-attention, feed-forward blocks, transformer stack, and training/inference pipelines, without relying on any pre-existing models.
­
It has saving/loading of training checkpoint, with adjustable hyperparameters, for flexibly experimentation; conducted experiments on multilingual datasets, evaluating generation quality before and after training; optimized for efficient training on GPUs.
­
Extended Andrej Karpathy’s “Let’s Build GPT from Scratch” (2022) with fully annotated educational implementation: intuitive variable naming, inline tensor-shape comments, step-by-step theoretical explanations, unit-style tests of each component, readable training workflow.
