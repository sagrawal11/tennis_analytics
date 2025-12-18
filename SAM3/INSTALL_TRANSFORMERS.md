# Installing Transformers 5.0 for SAM 3

## Why This Is Needed

SAM 3 requires Transformers version 5.0+, but the latest stable release on PyPI is 4.57.3. We need to install from the GitHub main branch.

## Installation Steps

**Run these commands in your terminal (not through the tool):**

```bash
# 1. Activate your virtual environment
source tennis_env/bin/activate

# 2. Clean up any corrupted installations
pip uninstall -y transformers
rm -rf tennis_env/lib/python3.11/site-packages/*ransform* 2>/dev/null

# 3. Install transformers from GitHub
# This will take 5-10 minutes as it clones and builds the repository
pip install "git+https://github.com/huggingface/transformers.git"

# 4. Verify installation
python -c "from transformers import Sam3Model, Sam3Processor; print('✓ Success!')"
```

## Why Run Manually?

The git clone operation takes several minutes and may timeout when run through automated tools. Running it in your terminal allows you to:
- See real-time progress
- Monitor the download/build process
- Avoid automatic timeouts

## Alternative: Use Pre-built Wheel (If Available)

If the git install is too slow, you can try installing dependencies first, then transformers:

```bash
source tennis_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install "git+https://github.com/huggingface/transformers.git" --no-build-isolation
```

## Troubleshooting

If you encounter errors:

1. **Network issues**: Make sure you have a stable internet connection
2. **Git not found**: Install git: `brew install git` (macOS)
3. **Build errors**: Make sure you have build tools: `xcode-select --install` (macOS)
4. **Permission errors**: Check virtual environment permissions

## After Installation

Once transformers 5.0 is installed, you can test SAM 3:

```bash
python SAM3/test_sam3_ball_detection.py --video old/data/raw/tennis_test5.mp4 --prompt "tennis ball"
```

