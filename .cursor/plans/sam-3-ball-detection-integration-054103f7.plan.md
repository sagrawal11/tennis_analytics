<!-- 054103f7-7bac-4810-a45e-f663799fa472 2e03c600-f979-460a-a715-03ce1d2be83c -->
# SAM 3 Ball Detection Integration Plan

## Current State

- **Ball detection comparison framework**: `tests/ball_model_comparison.py` tests multiple detectors (YOLO, RF-DETR, TrackNet variants)
- **Existing SAM folder**: Contains SAM 3D Body (human pose model) - will be used for player detection separately
- **SAM 3 (segmentation model)**: User will obtain this model separately for ball detection
- **Test videos**: `tennis_test5` and `tennis_test6` already processed with existing detectors
- **Output location**: `outputs/videos/ball_model_trials/`

## Implementation Steps

### 0. Clean Up and Fix SAM 3D Body Scripts (Current Task - Can Start Now)

**Prerequisites:** ✅ SAM 3D Body model already downloaded (checkpoints/sam-3d-body-dinov3/)

**Issues Found:**

1. **SAM/demo.py** and **SAM/test_single_image.py** are wrapper scripts that may have path/dependency issues
2. Scripts use `setup_sam_3d_body()` from `notebook/utils.py` which is correct for HuggingFace
3. Need to verify dependencies, fix import paths, and ensure proper model loading
4. Check if model checkpoint is properly downloaded and accessible

**Fixes Needed:**

- Verify all imports work correctly
- Fix path resolution for sam-3d-body repository
- Ensure HuggingFace authentication is working
- Test model loading from checkpoints
- Create a clean, working test script
- Document proper usage

**Apple Silicon Compatibility:**

- User confirmed: `pip install git+https://github.com/huggingface/transformers torchvision` works on Apple Silicon
- This approach should be used for dependencies that may have compatibility issues
- **Current SAM folder status:**
- `download_model.py` looks good (just downloads from HuggingFace)
- No `requirements.txt` in SAM folder - should create one
- `sam-3d-body/INSTALL.md` has standard install instructions that may need Apple Silicon adjustments
- May need to install `transformers` from GitHub for Apple Silicon compatibility
- Same approach applies to SAM 3 when it's downloaded

**Recommended Changes:**

1. Create `SAM/requirements.txt` with Apple Silicon-compatible installs
2. Update installation instructions to use GitHub source for transformers/torchvision on Apple Silicon
3. Add installation helper script that detects Apple Silicon and uses appropriate install method
4. Document this in SAM/README.md

### 1. Set Up SAM 3 Environment ✅ COMPLETE

- **✅ SAM 3 model downloaded** in `SAM3/` folder:
- Model files: `model.safetensors`, `sam3.pt`
- Config files: `config.json`, `processor_config.json`
- Tokenizer files: `tokenizer.json`, `vocab.json`, etc.
- README.md with usage examples
- **Two API options available:**
- **Option A: Transformers API** (recommended for simplicity)
- `from transformers import Sam3Model, Sam3Processor`
- Load from local: `Sam3Model.from_pretrained("./SAM3")`
- Text prompts: `processor(images=image, text="tennis ball", return_tensors="pt")`
- **Option B: Original sam3 repo API**
- `from sam3.model_builder import build_sam3_image_model`
- `from sam3.model.sam3_image_processor import Sam3Processor`
- Requires installing sam3 package from GitHub
- **Next step**: Install SAM 3 package and test model loading
- **Note**: SAM 3D Body in `SAM-3d-body/` folder will be kept for player detection (separate use case)

### 2. Test SAM 3 Standalone (First Phase - Individual Testing)

**Goal**: Test SAM 3 individually before integrating into comparison framework, similar to how other detectors were tested

**Create Standalone Test Script:**

- **Location**: `SAM/test_sam3_ball_detection.py` (new file)
- **Purpose**: Test SAM 3 ball detection on tennis videos independently
- **Features**:
- Load SAM 3 model from `SAM3/` folder
- Process video frames with text prompt: `"tennis ball"` or `"ball"`
- Generate annotated output videos showing SAM 3 detections
- Test different text prompt variations
- Compare image-based vs video-based approaches

**Test Approach:**

- Process `tennis_test5` and `tennis_test6` videos
- Generate output videos showing:
- SAM 3 masks overlaid on frames
- Bounding boxes
- Confidence scores
- Frame-by-frame detection results
- Test multiple prompt variations:
- `"tennis ball"`
- `"ball"`
- `"yellow ball"`
- `"tennis ball"` with negative prompts (if needed)

**Integration Options to Test:**

**Option A: Image-based (per-frame)**

- Use Transformers API: `Sam3Model`, `Sam3Processor`
- Process each video frame independently
- Simpler, faster to implement

**Option B: Video-based (temporal tracking)**

- Use `Sam3VideoModel` for better temporal consistency
- Track ball across frames automatically
- May provide better results but more complex

**Recommendation**: Start with Option A, then test Option B for comparison

### 3. Integrate SAM 3 into Comparison Framework (Second Phase - After Standalone Testing)

**After standalone testing is complete:**

- **Location**: Add to `tests/ball_model_comparison.py`
- **Implementation**:
- Create `SAM3BallDetector` class matching existing detector interface
- Use `BallDetectionResult` format (center, confidence, source)
- Add to `available_detectors()` function
- Include in ensemble combinations
- Generate comparison videos with SAM 3 alongside other detectors

### 4. Comprehensive Testing & Evaluation (After Integration)

- Run full comparison on `tennis_test5` and `tennis_test6` videos
- Generate all combinations including SAM 3:
- SAM 3 alone
- SAM 3 + YOLO variants
- SAM 3 + RF-DETR
- SAM 3 + TrackNet variants
- SAM 3 + all other combinations
- Compare SAM 3 results visually with existing detectors
- Evaluate:
- Detection accuracy (does it find the ball?)
- Segmentation quality (how precise are the masks?)
- False positives/negatives
- Performance (inference speed)
- Best ensemble combinations

### 6. Potential Optimizations

- Use bounding box prompts from other detectors to guide SAM 3 (hybrid approach)
- Fine-tune text prompt variations ("tennis ball", "ball", "yellow ball")
- Adjust confidence thresholds for SAM 3 outputs

## Files to Modify/Create

1. **`tests/ball_model_comparison.py`**

- Add `SAM3BallDetector` class
- Integrate into `available_detectors()`

2. **`SAM/sam3_ball_detector.py`** (new)

- Standalone SAM 3 ball detection implementation
- Can be reused in main pipeline later

3. **`SAM/requirements.txt`** (new or update)

- Add SAM 3 dependencies: `sam3` package

## Key SAM 3 Features to Leverage

- **Text prompts**: Natural language like "tennis ball"
- **Zero-shot segmentation**: No training needed
- **High-quality masks**: Precise segmentation boundaries
- **Video support**: Temporal tracking capabilities

## Expected Output

- SAM 3 comparison videos in `outputs/videos/ball_model_trials/`
- Standalone SAM 3 detector for future integration into main pipeline
- Performance metrics comparing SAM 3 vs existing detectors

### To-dos

- [ ] Install SAM 3 package and download model checkpoint from HuggingFace
- [ ] Create SAM3BallDetector class in ball_model_comparison.py using text prompts
- [ ] Add SAM3BallDetector to available_detectors() and test ensemble combinations
- [ ] Run comparison on tennis_test5 and tennis_test6 videos
- [ ] Review SAM 3 detection quality, accuracy, and performance compared to existing detectors