# MVP Roadmap & Backend Hosting Guide

## 🎯 What's Left for MVP

Based on the current codebase, here's what needs to be completed:

### Critical Path (Must Have for MVP)

1. **Playsight Frame Extraction** ⚠️
   - **Status:** Placeholder only
   - **Location:** `backend/services/playsight.py`
   - **What's needed:** Extract 3-5 frames from Playsight videos for player identification
   - **Options:**
     - Playsight API (if available)
     - Video scraping/downloading
     - Manual frame extraction tool
   - **Complexity:** Medium (depends on Playsight access)

2. **CV Backend Integration** 🔴 **CRITICAL**
   - **Status:** Placeholder only (`TODO` comments)
   - **Location:** `backend/services/cv_integration.py`
   - **What's needed:**
     - Connect to `old/src/core/tennis_CV.py` or create new processing pipeline
     - Download video from Playsight (or use local file)
     - Run CV models (SAM-3d-body, SAM3, YOLO, etc.)
     - Parse JSON output
     - Store results in database
   - **Complexity:** High (core functionality)

3. **Player Tracking** 🔴 **CRITICAL**
   - **Status:** Placeholder only
   - **Location:** `backend/services/player_tracker.py`
   - **What's needed:**
     - Use player identification clicks to track player
     - Color recognition based on clicked coordinates
     - Track player position throughout video
   - **Complexity:** Medium-High

4. **Court Visualization with Real Data** 🔴 **CRITICAL**
   - **Status:** Placeholder court diagram exists
   - **Location:** `frontend/components/court/` (if exists)
   - **What's needed:**
     - Parse shot data from CV output
     - Render shots on court diagram
     - Clickable shots that jump to video timestamp
     - Color coding (green=winner, red=error, blue=in-play)
   - **Complexity:** Medium

5. **Async Video Processing** ⚠️
   - **Status:** Currently synchronous (would timeout)
   - **What's needed:**
     - Task queue (Celery + Redis/RabbitMQ)
     - Background worker processes
     - Progress tracking
     - Error handling
   - **Complexity:** Medium (can use simpler solution for MVP)

### Nice to Have (Post-MVP)

- Statistics visualization with Recharts
- Advanced error handling
- Comprehensive testing
- Match sharing
- Export functionality

---

## 🖥️ Backend Hosting Options

### Option 1: Railway (Recommended for MVP)

**Best for:** Quick deployment, easy setup, good for Python/FastAPI

**Pros:**
- ✅ Very easy setup (connects to GitHub, auto-deploys)
- ✅ Built-in environment variables
- ✅ Automatic HTTPS
- ✅ Good Python support
- ✅ Reasonable pricing
- ✅ Can scale up/down easily

**Cons:**
- ❌ Limited GPU options (may need to use CPU)
- ❌ Less control than AWS/GCP
- ❌ Newer platform (less mature)

**Resource Options:**
- **Starter:** $5/month - 512MB RAM, 1 vCPU
- **Developer:** $20/month - 8GB RAM, 4 vCPU
- **Pro:** $100/month - 32GB RAM, 8 vCPU
- **GPU:** Not available (CPU only)

**Estimated Cost for MVP:**
- **Development:** $20/month (Developer plan)
- **Production:** $100/month (Pro plan) or more if high traffic

**Processing Speed:**
- CPU-only processing will be **slow** (2-4 hours per video)
- But it works and is affordable

---

### Option 2: AWS EC2 (Most Flexible)

**Best for:** Full control, GPU support, production scale

**Pros:**
- ✅ Full control over instance
- ✅ GPU instances available (NVIDIA)
- ✅ Can optimize for ML workloads
- ✅ Pay only for what you use
- ✅ Mature platform with lots of docs

**Cons:**
- ❌ More complex setup
- ❌ Need to manage server yourself
- ❌ More expensive with GPU
- ❌ Need to configure security, networking

**Resource Options:**

**CPU Instances (for MVP):**
- **t3.medium:** $30/month - 2 vCPU, 4GB RAM
- **t3.large:** $60/month - 2 vCPU, 8GB RAM
- **c5.xlarge:** $150/month - 4 vCPU, 8GB RAM

**GPU Instances (for production):**
- **g4dn.xlarge:** $200/month - 1 GPU (NVIDIA T4), 4 vCPU, 16GB RAM
- **g4dn.2xlarge:** $400/month - 1 GPU, 8 vCPU, 32GB RAM
- **g5.xlarge:** $300/month - 1 GPU (NVIDIA A10G), 4 vCPU, 16GB RAM

**Estimated Cost for MVP:**
- **Development:** $60/month (t3.large)
- **Production (CPU):** $150/month (c5.xlarge)
- **Production (GPU):** $200-400/month (g4dn.xlarge or g5.xlarge)

**Processing Speed:**
- **CPU:** 2-4 hours per video
- **GPU:** 30-60 minutes per video (10x faster!)

---

### Option 3: Google Cloud Platform (GCP)

**Best for:** ML workloads, good GPU pricing

**Pros:**
- ✅ Excellent ML/AI support
- ✅ Good GPU pricing
- ✅ Free tier credits ($300)
- ✅ Auto-scaling

**Cons:**
- ❌ More complex than Railway
- ❌ Steeper learning curve
- ❌ Need to manage more

**Resource Options:**

**CPU Instances:**
- **e2-standard-4:** $100/month - 4 vCPU, 16GB RAM
- **e2-highmem-4:** $120/month - 4 vCPU, 32GB RAM

**GPU Instances:**
- **n1-standard-4 + T4 GPU:** $200/month - 4 vCPU, 15GB RAM, 1 GPU
- **n1-standard-8 + T4 GPU:** $350/month - 8 vCPU, 30GB RAM, 1 GPU

**Estimated Cost for MVP:**
- **Development:** $100/month (e2-standard-4)
- **Production (GPU):** $200-350/month

---

### Option 4: DigitalOcean (Simple Alternative)

**Best for:** Simplicity, good for CPU-only workloads

**Pros:**
- ✅ Very simple interface
- ✅ Predictable pricing
- ✅ Good documentation
- ✅ Easy to scale

**Cons:**
- ❌ No GPU options
- ❌ Less powerful than AWS/GCP
- ❌ CPU-only processing

**Resource Options:**
- **Basic:** $12/month - 1 vCPU, 1GB RAM
- **Professional:** $48/month - 4 vCPU, 8GB RAM
- **CPU-Optimized:** $72/month - 4 vCPU, 8GB RAM

**Estimated Cost for MVP:**
- **Development:** $48/month
- **Production:** $72-144/month

---

## 💰 Cost Comparison Summary

| Provider | Plan | CPU | RAM | GPU | Monthly Cost | Processing Time |
|----------|------|-----|-----|-----|--------------|-----------------|
| **Railway** | Developer | 4 | 8GB | ❌ | $20 | 2-4 hours |
| **Railway** | Pro | 8 | 32GB | ❌ | $100 | 1-2 hours |
| **AWS** | t3.large | 2 | 8GB | ❌ | $60 | 3-4 hours |
| **AWS** | c5.xlarge | 4 | 8GB | ❌ | $150 | 2-3 hours |
| **AWS** | g4dn.xlarge | 4 | 16GB | ✅ T4 | $200 | 30-60 min |
| **GCP** | e2-standard-4 | 4 | 16GB | ❌ | $100 | 2-3 hours |
| **GCP** | n1 + T4 | 4 | 15GB | ✅ T4 | $200 | 30-60 min |
| **DigitalOcean** | Professional | 4 | 8GB | ❌ | $48 | 2-4 hours |

---

## 🎯 Recommendations for MVP

### Phase 1: Development & Testing (First 1-3 months)

**Recommended:** Railway Developer Plan ($20/month)

**Why:**
- Easiest to set up and deploy
- Good enough for testing CV processing
- Can process videos (just slower)
- Low cost for development

**What you get:**
- 4 vCPU, 8GB RAM
- Enough to run all ML models (CPU mode)
- Processing time: 2-4 hours per video
- Perfect for testing and iteration

### Phase 2: MVP Launch (First customers)

**Option A: Railway Pro Plan ($100/month)**
- If CPU processing is acceptable (2-4 hours)
- Easier to manage
- Can handle multiple concurrent videos

**Option B: AWS g4dn.xlarge ($200/month)**
- If you need faster processing (30-60 min)
- GPU acceleration
- Better user experience
- More professional

**My Recommendation:** Start with **Railway Pro ($100/month)** for MVP
- Lower cost
- Easier to manage
- Can upgrade to GPU later if needed
- Most users won't mind waiting 2-4 hours

### Phase 3: Scale (After MVP success)

**Recommended:** AWS g4dn.2xlarge or g5.xlarge ($300-400/month)
- GPU for fast processing
- Can handle multiple videos simultaneously
- Professional-grade infrastructure

---

## 🔧 Resource Requirements Breakdown

### Minimum Requirements (CPU-only)

**For MVP:**
- **CPU:** 4+ cores (for parallel processing)
- **RAM:** 8GB+ (models load into memory)
- **Storage:** 50GB+ (temporary video files)
- **Network:** Good bandwidth (download videos)

**What this can do:**
- Process 1 video at a time
- Processing time: 2-4 hours per video
- Can handle 5-10 videos per day

### Recommended Requirements (GPU)

**For Production:**
- **CPU:** 4+ cores
- **RAM:** 16GB+
- **GPU:** NVIDIA T4 or better (8GB+ VRAM)
- **Storage:** 100GB+ SSD
- **Network:** High bandwidth

**What this can do:**
- Process 1-2 videos simultaneously
- Processing time: 30-60 minutes per video
- Can handle 20-40 videos per day

---

## 📊 Processing Time Estimates

### CPU Processing (Railway/AWS CPU instances)

**Per video (1 hour match):**
- Frame extraction: 5 minutes
- Player detection (SAM-3d-body): 60-90 minutes
- Ball detection (SAM3): 30-45 minutes
- Court detection: 10-15 minutes
- Shot classification: 20-30 minutes
- **Total: 2-4 hours**

### GPU Processing (AWS/GCP GPU instances)

**Per video (1 hour match):**
- Frame extraction: 5 minutes
- Player detection (SAM-3d-body): 10-15 minutes (GPU accelerated)
- Ball detection (SAM3): 5-10 minutes (GPU accelerated)
- Court detection: 5 minutes
- Shot classification: 5-10 minutes
- **Total: 30-60 minutes**

**Speedup:** GPU is **4-8x faster** than CPU

---

## 💡 Cost Optimization Strategies

### 1. Use Spot Instances (AWS/GCP)
- **Savings:** 50-70% off regular price
- **Trade-off:** Can be interrupted (not ideal for long processing)
- **Best for:** Non-critical processing

### 2. Process During Off-Peak Hours
- Schedule video processing for nights/weekends
- Use smaller instances during day, scale up at night

### 3. Queue System
- Process videos one at a time (don't need multiple instances)
- Use smaller instance, just takes longer

### 4. Hybrid Approach
- Use CPU for development/testing
- Use GPU only for production/customer videos

---

## 🚀 Deployment Checklist

### For MVP (Railway Recommended)

1. **Sign up for Railway** (railway.app)
2. **Connect GitHub repo**
3. **Set environment variables:**
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `ALLOWED_ORIGINS` (your frontend URL)
4. **Deploy backend:**
   - Railway auto-detects Python
   - Runs `uvicorn main:app`
5. **Install CV models:**
   - Add build script to install models
   - Or use Railway's file system
6. **Test processing:**
   - Upload a test video
   - Monitor processing time
   - Adjust instance size if needed

### For Production (AWS/GCP)

1. **Launch EC2/Compute Engine instance**
2. **Install dependencies:**
   - Python 3.8+
   - CUDA (for GPU)
   - All Python packages
   - CV models
3. **Set up FastAPI:**
   - Use systemd or supervisor to keep it running
   - Configure nginx as reverse proxy
   - Set up SSL certificate
4. **Configure security:**
   - Security groups/firewall rules
   - Only allow HTTPS
   - Restrict SSH access
5. **Set up monitoring:**
   - CloudWatch (AWS) or Cloud Monitoring (GCP)
   - Set up alerts for errors

---

## 📝 Next Steps

### Immediate (This Week)

1. **Choose hosting provider:**
   - **Recommendation:** Start with Railway Developer ($20/month)
   - Test CV processing on CPU
   - See if 2-4 hour processing time is acceptable

2. **Complete Playsight integration:**
   - Research Playsight API access
   - Or implement video downloading/scraping
   - Test frame extraction

3. **Connect CV backend:**
   - Integrate `old/src/core/tennis_CV.py` or create new pipeline
   - Test end-to-end processing
   - Store results in database

### Short Term (Next Month)

4. **Implement player tracking:**
   - Use identification clicks
   - Track player throughout video

5. **Build court visualization:**
   - Parse shot data
   - Render on court diagram
   - Make shots clickable

6. **Set up async processing:**
   - Use Celery + Redis (or simpler queue)
   - Background workers
   - Progress tracking

### Before Launch

7. **Deploy to production:**
   - Upgrade to Railway Pro or AWS GPU instance
   - Set up monitoring
   - Test with real videos
   - Optimize processing time

---

## 🎯 MVP Success Criteria

**You'll know MVP is ready when:**

✅ Users can upload Playsight links
✅ System extracts frames for player identification
✅ CV processing completes successfully
✅ Results appear on court diagram
✅ Shots are clickable and jump to video
✅ Processing completes in reasonable time (< 4 hours)
✅ System handles errors gracefully
✅ Multiple users can use system simultaneously

---

## 💬 Questions to Consider

1. **Is 2-4 hour processing acceptable for MVP?**
   - If yes → Start with Railway CPU ($20-100/month)
   - If no → Need GPU from start ($200+/month)

2. **How many videos per day?**
   - < 10 videos/day → Railway Pro ($100/month) is fine
   - 10-50 videos/day → Need GPU or multiple workers
   - 50+ videos/day → Need dedicated GPU cluster

3. **Budget constraints?**
   - Tight budget → Railway Developer ($20/month), accept slow processing
   - Moderate budget → Railway Pro ($100/month) or AWS CPU ($60-150/month)
   - Good budget → AWS GPU ($200-400/month) for best experience

---

## 📞 Support Resources

- **Railway Docs:** https://docs.railway.app
- **AWS EC2 Pricing:** https://aws.amazon.com/ec2/pricing/
- **GCP Pricing:** https://cloud.google.com/compute/pricing
- **FastAPI Deployment:** https://fastapi.tiangolo.com/deployment/

---

**Bottom Line:** Start with **Railway Developer ($20/month)** for development, then upgrade to **Railway Pro ($100/month)** or **AWS GPU ($200/month)** for MVP launch depending on your processing speed requirements and budget.
