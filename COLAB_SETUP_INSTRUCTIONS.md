# Google Colab Setup Instructions for Hero Video Processing

This guide explains how to use the `hero_video_colab.ipynb` notebook to test video processing with a T4 GPU.

## 📋 Prerequisites

1. **Google Colab Account** (free)
2. **Input video file** (`.mp4` format)
3. **Model files** (see list below)

## 🚀 Quick Start

1. **Open the notebook:**
   - Upload `hero_video_colab.ipynb` to Google Colab
   - Or create a new notebook and copy the cells

2. **Enable GPU:**
   - Go to: **Runtime → Change runtime type**
   - Select: **GPU** (T4)
   - Click **Save**

3. **Prepare your folders (on your local machine):**
   - Zip your `SAM-3d-body/` folder → `SAM-3d-body.zip`
   - Zip your `SAM3/` folder → `SAM3.zip`
   - Zip your `hero-video/` folder → `hero-video.zip`
   - Zip your `models/` folder → `models.zip` (if you have one)

4. **Run cells in order:**
   - Step 1: Install dependencies
   - Step 2: Skip cloning (you'll upload instead)
   - Step 3: Set up paths
   - Step 4: **Upload your ZIP files** (SAM-3d-body.zip, SAM3.zip, hero-video.zip, etc.)
   - Step 5: Authenticate with Hugging Face (if needed) OR skip if using local models
   - Step 6: Load models
   - Step 7: Configure parameters
   - Step 8: Process video (uses your existing code!)
   - Step 9: Download result

5. **Download the processed video** at the end

## 📦 Required Model Files

You'll need to upload these files to Colab (or they'll auto-download):

### 1. SAM-3d-body Model ✅ Auto-Downloads
- **Source:** Hugging Face (`facebook/sam-3d-body-dinov3`)
- **Size:** ~2-3 GB
- **Status:** Automatically downloaded by the notebook
- **No action needed** - the notebook handles this

### 2. SAM3 Ball Detector ⚠️ Manual Setup Required
- **Location:** Your `SAM3/` directory
- **What to do:**
  - **Option A (Recommended):** Upload your entire `SAM3/` folder to Colab
    1. Zip your local `SAM3/` folder: `zip -r SAM3.zip SAM3/`
    2. In Colab, use the file upload cell to upload `SAM3.zip`
    3. Unzip it: `!unzip SAM3.zip -d /content/`
    4. The notebook will find it at `/content/SAM3`
  
  - **Option B:** Clone the SAM3 repository (if it's public)
     ```python
     !git clone https://github.com/your-sam3-repo.git SAM3
     ```
  
  - **Option C:** Use Google Drive (for large folders)
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     # Then copy from drive to /content
     !cp -r /content/drive/MyDrive/SAM3 /content/
     ```

**Important:** The notebook looks for `test_sam3_ball_detection.py` inside the SAM3 folder, so make sure your folder structure is:
```
/content/SAM3/
  ├── test_sam3_ball_detection.py
  └── (other SAM3 files)
```

### 3. YOLO Human Detector (Optional but Recommended)
- **File:** `playersnball5.pt`
- **Location:** `old/models/player/playersnball5.pt` in your project
- **Size:** ~50-100 MB
- **What it does:** Helps SAM-3d-body detect humans faster
- **Upload:** Use the file upload cell in the notebook

### 4. Court Detection Model (Optional)
- **File:** `model_tennis_court_det.pt`
- **Location:** `old/models/court/model_tennis_court_det.pt` in your project
- **Size:** ~50-100 MB
- **What it does:** Detects tennis court lines
- **Upload:** Use the file upload cell in the notebook

## 📁 File Structure in Colab

After uploading, your Colab environment should look like:

```
/content/
├── sam-3d-body/          (cloned automatically)
├── SAM3/                 (upload or clone manually)
├── models/               (created by notebook)
│   ├── playersnball5.pt  (upload this)
│   └── model_tennis_court_det.pt  (optional)
├── your_video.mp4        (upload this)
└── output/               (created by notebook)
    └── hero_video_processed.mp4  (final output)
```

## ⚙️ Configuration Options

In the notebook, you can adjust these settings:

```python
config = {
    'frame_skip': 5,              # Process every 5th frame (1 = all frames)
    'fps': 30.0,                  # Output video FPS
    'player_color': '#50C878',    # Emerald green
    'ball_color': '#50C878',      # Emerald green
    'trail_length': 30,           # Ball trajectory trail
    'keypoints_only': False,      # False = full mesh, True = faster
    'process_resolution': 720,     # Downscale to 720px (0 = original)
    'enable_court_detection': False,  # Enable if you have court model
    'use_ensemble_ball': False,    # Use ensemble (slower but more accurate)
}
```

### Performance Tips:

- **Faster processing:** Set `frame_skip: 10` or higher
- **Better quality:** Set `frame_skip: 1` (process all frames)
- **Faster but less detailed:** Set `keypoints_only: True`
- **Full quality:** Set `keypoints_only: False` (full 3D mesh)
- **Lower resolution:** Set `process_resolution: 480` (faster)

## ⏱️ Expected Processing Times

### With T4 GPU:

**1-hour tennis match video:**

| Configuration | Processing Time |
|--------------|-----------------|
| Full mesh, all frames | 2-4 hours |
| Full mesh, every 5th frame | 30-60 minutes |
| Keypoints only, all frames | 30-60 minutes |
| Keypoints only, every 10th frame | 10-20 minutes |

**10-second clip (for testing):**

| Configuration | Processing Time |
|--------------|-----------------|
| Full mesh, all frames | 2-5 minutes |
| Keypoints only, all frames | 30-60 seconds |

## 🔧 Troubleshooting

### Issue: "No module named 'braceexpand'"
**Solution:** ✅ **FIXED** - The notebook now installs `braceexpand` automatically. If you still see this error, run:
```python
!pip install braceexpand
```

### Issue: "No module named 'ultralytics'"
**Solution:** ✅ **FIXED** - The notebook now installs `ultralytics` automatically. If you still see this error, run:
```python
!pip install ultralytics
```

### Issue: "SAM-3d-body not found" or "No module named 'notebook.utils'"
**Solution:** The notebook should auto-clone it. If not:
```python
!git clone https://github.com/facebookresearch/sam-3d-body.git
```
Make sure the path includes both `/content/sam-3d-body` and `/content/sam-3d-body/sam-3d-body` in sys.path.

### Issue: "SAM3 not found" or "No module named 'test_sam3_ball_detection'"
**Solution:** You need to upload your `SAM3/` folder to Colab. The notebook will check these locations:
- `/content/SAM3`
- `/content/sam3`
- `/content/SAM-3`

**To upload:**
1. Zip your local `SAM3/` folder
2. Upload the zip file in Colab
3. Unzip it: `!unzip SAM3.zip -d /content/`

### Issue: "CUDA out of memory"
**Solution:** 
- Reduce `process_resolution` (e.g., 480 instead of 720)
- Increase `frame_skip` (e.g., 10 instead of 5)
- Set `keypoints_only: True`

### Issue: Processing is very slow
**Solution:**
- Make sure GPU is enabled (Runtime → Change runtime type → GPU)
- Check GPU is being used: `print(torch.cuda.is_available())`
- Reduce video resolution or increase frame_skip

### Issue: Models won't download
**Solution:**
- Check internet connection in Colab
- Some models download from Hugging Face (may need account)
- Try running the install cell again

## 📊 What the Notebook Does

1. **Installs dependencies** (OpenCV, PyTorch, etc.)
2. **Clones SAM-3d-body** repository
3. **Loads models** into GPU memory
4. **Processes video** frame-by-frame:
   - Detects players (SAM-3d-body)
   - Detects ball (SAM3)
   - Draws visualizations
5. **Saves output video** with all overlays
6. **Downloads result** to your computer

## 🎯 Testing Strategy

### Step 1: Quick Test (5-10 minutes)
- Use a **10-30 second clip**
- Set `frame_skip: 10`
- Set `keypoints_only: True`
- Verify everything works

### Step 2: Quality Test (30-60 minutes)
- Use a **1-2 minute clip**
- Set `frame_skip: 5`
- Set `keypoints_only: False` (full mesh)
- Check visual quality

### Step 3: Full Processing (2-4 hours)
- Use **full video**
- Set `frame_skip: 5` or `1`
- Set `keypoints_only: False`
- Let it run and measure time

## 💡 Tips for Best Results

1. **Start small:** Test with a short clip first
2. **Monitor progress:** Watch the progress bar
3. **Check GPU usage:** Use `nvidia-smi` in a terminal cell
4. **Save frequently:** Colab sessions can timeout (12 hours max)
5. **Download output:** Don't forget to download before closing!

## 📝 Notes

- **Colab sessions:** Free tier has 12-hour session limit
- **GPU availability:** T4 GPU is free but may have wait times
- **Storage:** Colab provides ~80GB temporary storage
- **Download:** Always download your output before closing!

## 🚀 Next Steps After Testing

Once you know processing times:

1. **Decide on hosting:**
   - If 30-60 min is acceptable → Railway CPU ($100/month)
   - If you need faster → AWS GPU ($200-400/month)

2. **Optimize settings:**
   - Find the best `frame_skip` for quality/speed tradeoff
   - Decide if full mesh or keypoints-only is needed

3. **Plan deployment:**
   - Use similar configuration in production
   - Set up async processing (Celery/Redis)
   - Monitor processing times

---

**Ready to test?** Open `hero_video_colab.ipynb` in Google Colab and follow the cells!
