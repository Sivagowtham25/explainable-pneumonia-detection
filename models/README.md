# Models Directory

This directory stores trained deep learning model artifacts, serialized weights, and model evaluation checkpoints.

## Supported Formats (Planned)
- Keras native archive: `*.keras`
- Legacy HDF5 format: `*.h5`
- SavedModel protobuf directory format: `saved_model/`
- Checkpoint weights: `*.weights.h5`

## Git Tracking Policy
- **Excluded**: All trained model weights, binary checkpoints, and serialized architectures are excluded from version control via `.gitignore` to prevent repository bloat.
- Only configuration files, lightweight metadata (`model_config.json`), or documentation are tracked.
