# Union architecture and implementation conventions

This document describes the code in this release. It records the tensor layout
and operations needed to understand or extend the implementation; it does not
assert compatibility with earlier experimental checkpoints.

## Tokens and dataset prompts

Inputs have shape `[B, L, V]`: batch, sequence length, and variables. Patching
runs along time independently for each variable. The final patch is padded
when needed so trailing observations are retained. A shared linear projection
maps each patch to width `d`, producing `[B, V, N, d]`.

The learned positional array is interpolated to the required number of data
patches. For each dataset, a parameter of shape `[1, 1, P, d]` stores its prompt
tokens; it is broadcast over batch and variables. Prompt elements use a normal
initialization with standard deviation 0.02. Dataset names are reversibly
encoded for the parameter dictionary. Register new datasets before constructing
the optimizer.

The complete token order is `[prompt tokens, CLS, data patches, GEN]`. Several
task entries with the same `dataset` value share the prompt. Only prompts used
in a forward pass receive its gradients.

## Frequency-aware interaction

Each block applies channel interaction along `V`, followed by temporal
interaction along the complete token axis. Prompt, CLS, and GEN tokens are part
of that representation in this implementation.

For each axis:

1. Move the axis to the sequence position and reshape the remaining dimensions
   into a batch of feature sequences.
2. Apply `torch.fft.rfft` along the selected axis.
3. Keep all available one-sided frequency bins if there are at most
   `max_freq_modes`; otherwise sample that many distinct bins. The same selected
   indices are used for all slices in that stage and forward call.
4. Pass real and imaginary components through the same attention/normalization/
   feed-forward encoder. Channel and temporal stages have separate encoders.
5. Put processed coefficients back at their original frequency indices and
   set unselected bins to zero. Apply `irfft` with the original axis length.

A learned feature projection and sigmoid gate add the interaction output to
the residual representation. Real/imaginary processing is used; there is no
separate amplitude/phase estimator. Sampling occurs in evaluation as well as
training. The experiment loop replays a split-specific seeded stream inside
`torch.random.fork_rng` and restores the training RNG state afterward. Direct
callers of the core model must manage RNG state themselves.

## Temporal-consistency experts

Each expert stores a learnable square base weight matrix. Bilinear
interpolation resizes it to the current token count `N`. A fixed exponential
kernel multiplies the resized weights elementwise:

```text
K[i,j] = exp(-abs(i-j) / (tau * max(N-1, 1)))
effective_weight = K * interpolate(base_weight, size=(N, N))
```

`tau` is a non-trainable buffer (default 0.2). Kernel entries on the diagonal
are one. Kernel values are recomputed for the current length but are not
optimized by backpropagation; base weights remain trainable. The expert applies
the temporal mapping and a feature MLP.

The router pools variables and tokens into one vector per sample and selects
the highest-scoring `top_k` experts. Selected scores are normalized with
softmax. Each expert processes only samples routed to it, and weighted results
are accumulated back into the batch. A gated residual combines the MoE output
with the representation. Sparse routing reduces which sample/expert pairs are
executed; it does not make the token-square temporal transformation constant
cost in sequence length.

## Shared task outputs

The generative head maps token features to patches with one shared linear
projection. GEN-conditioned data features are interpolated to the required
number of output patches; overlapping patches are averaged and the result is
trimmed to the requested length. Forecasting returns `[B, horizon, V]`;
imputation and anomaly reconstruction return `[B, L, V]`.

For classification, CLS features are averaged over variables and passed to a
shared output projection. The configured maximum class count sets its width;
each task uses its requested prefix of logits. Class labels remain task-local,
so this is a concrete implementation choice rather than a semantic alignment
between class names from different datasets.

The core model can process varying lengths and channel counts. File loaders,
classification arrays, and checkpoint shapes impose their own configuration
constraints. Changing a model's reference length, hidden width, depth, patch
size, or head capacity may make a checkpoint incompatible; use matching
architecture settings for transfer. When adapting to a different input length,
`--reference_seq_len` can retain the source positional/expert base shapes while
the target task's YAML specifies the actual input length. Checkpoint loading
requires `model_config` metadata from this implementation and compares semantic
architecture settings as well as tensor shapes. A bare state dictionary or
older unversioned checkpoint is rejected.
