# SAM3 Model Download Instructions

The SAM3 model file (`model.safetensors` - 3.2GB) is too large for GitHub and is excluded from the repository.

## Download the Model

You have two options:

### Option 1: Download from HuggingFace (Recommended)

1. Install HuggingFace CLI:
   ```bash
   pip install huggingface_hub
   ```

2. Download the model:
   ```bash
   huggingface-cli download facebook/sam3 --local-dir ./SAM3
   ```

### Option 2: Manual Download

1. Go to https://huggingface.co/facebook/sam3
2. Download `model.safetensors` and place it in the `SAM3/` directory
3. Download other required files (config.json, processor_config.json, tokenizer files, etc.)

## Verify Installation

After downloading, verify the model is in place:
```bash
ls -lh SAM3/model.safetensors
# Should show ~3.2GB file
```

The model is required for running SAM3 ball detection tests. See `SAM3/README.md` for usage instructions.
