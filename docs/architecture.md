# Union Architecture Notes

Union follows a unified modeling route for heterogeneous time-series tasks. The model maps all tasks into a shared token space, processes them with the same backbone, then dispatches to lightweight output branches.

## Reference Points

The reference repository, fecam, is useful for engineering choices:

- central `run.py`/experiment entry points,
- YAML-driven task and dataset configs,
- shared backbone across forecasting, classification, imputation, and anomaly detection,
- prompt-like task tokens and reusable data-provider patterns.

Union keeps those practical ideas but centers the paper-specific modules.

## Paper-Derived Components

1. Unified token representation
   - Patchify multivariate time series along the temporal axis.
   - Project patches into a shared hidden space.
   - Add prompt, GEN, and CLS tokens to bridge generative and predictive tasks.

2. Frequency-aware interaction
   - Channel interaction models variable-related feature channels.
   - Temporal interaction uses frequency-domain mixing to expose periodicity.
   - A residual gate controls how strongly frequency-aware updates enter the shared representation.

3. TCI-MoE
   - A router selects top-k experts per sample.
   - Each expert performs interpolation-based sequence transformation.
   - A temporal consistency kernel encourages smooth local mappings.
   - Expert outputs are combined with normalized routing weights.

4. Task heads
   - Generative branch handles forecasting, imputation, and anomaly reconstruction.
   - Classification branch reads the CLS token.
