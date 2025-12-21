"""
Ensemble ball detector that combines multiple detection models for maximum quality.
Uses SAM3, RF-DETR, TrackNet, TRACE, and YOLO variants.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class EnsembleBallDetector:
    """Combines multiple ball detection models for robust detection."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize all available ball detectors."""
        self.detectors = []
        self.device = device
        
        # Initialize SAM3 (text-based)
        try:
            from SAM3.test_sam3_ball_detection import SAM3BallDetector
            sam3_path = PROJECT_ROOT / "SAM3"
            if sam3_path.exists():
                self.sam3_detector = SAM3BallDetector(
                    model_path=sam3_path,
                    device=device,
                    use_transformers=True
                )
                self.detectors.append(("SAM3", self._detect_sam3))
                print("✓ SAM3 ball detector loaded")
        except Exception as e:
            print(f"⚠ SAM3 detector not available: {e}")
            self.sam3_detector = None
        
        # Initialize RF-DETR
        try:
            self._init_rfdetr()
        except Exception as e:
            print(f"⚠ RF-DETR detector not available: {e}")
        
        # Initialize TrackNet
        try:
            self._init_tracknet()
        except Exception as e:
            print(f"⚠ TrackNet detector not available: {e}")
        
        # Initialize TRACE TrackNet
        try:
            self._init_trace()
        except Exception as e:
            print(f"⚠ TRACE TrackNet detector not available: {e}")
        
        # Initialize YOLO variants
        try:
            self._init_yolo_variants()
        except Exception as e:
            print(f"⚠ YOLO detectors not available: {e}")
        
        print(f"\n✓ Ensemble detector initialized with {len(self.detectors)} models:")
        for name, _ in self.detectors:
            print(f"  - {name}")
    
    def _init_rfdetr(self):
        """Initialize RF-DETR detector."""
        model_path = PROJECT_ROOT / "old" / "models" / "player" / "playersnball5.pt"
        if not model_path.exists():
            return
        
        rfdetr_root = PROJECT_ROOT / "old" / "external" / "rf-detr"
        if not rfdetr_root.exists():
            return
        
        sys.path.insert(0, str(rfdetr_root))
        from rfdetr import RFDETRNano
        import torch
        
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if "args" not in checkpoint or "model" not in checkpoint:
            return
        
        args = checkpoint["args"]
        class_names = getattr(args, "class_names", ["ball", "player"])
        
        self.rfdetr_model = RFDETRNano(num_classes=len(class_names), pretrain_weights=None)
        self.rfdetr_model.model.model.load_state_dict(checkpoint["model"], strict=False)
        self.rfdetr_model.model.model.eval()
        
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rfdetr_model.model.model.to(device)
        self.rfdetr_device = device
        
        self.rfdetr_ball_class_id = None
        for idx, name in enumerate(class_names):
            if name.lower() == "ball":
                self.rfdetr_ball_class_id = idx
                break
        
        if self.rfdetr_ball_class_id is not None:
            self.detectors.append(("RF-DETR", self._detect_rfdetr))
    
    def _init_tracknet(self):
        """Initialize TrackNet PyTorch detector."""
        model_path = PROJECT_ROOT / "old" / "models" / "ball" / "pretrained_ball_detection.pt"
        if not model_path.exists():
            return
        
        # TrackNet requires frame history - we'll handle this in the detect method
        sys.path.insert(0, str(PROJECT_ROOT / "old" / "external" / "TRACE"))
        from BallTrackNet import BallTrackerNet
        import torch
        
        self.tracknet_model = BallTrackerNet(out_channels=2)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        
        if "model_state" in checkpoint:
            self.tracknet_model.load_state_dict(checkpoint["model_state"])
        elif "model" in checkpoint:
            self.tracknet_model.load_state_dict(checkpoint["model"])
        else:
            self.tracknet_model.load_state_dict(checkpoint)
        
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tracknet_model.to(device)
        self.tracknet_model.eval()
        self.tracknet_device = device
        
        # TrackNet needs 3 frames
        self.tracknet_frame_history = []
        
        self.detectors.append(("TrackNet", self._detect_tracknet))
    
    def _init_trace(self):
        """Initialize TRACE TrackNet detector."""
        trace_root = PROJECT_ROOT / "old" / "external" / "TRACE"
        weights_path = trace_root / "TrackNet" / "Weights.pth"
        
        if not trace_root.exists() or not weights_path.exists():
            return
        
        sys.path.insert(0, str(trace_root))
        from BallDetection import BallDetector as TraceBallDetector
        
        self.trace_detector = TraceBallDetector(str(weights_path))
        self.detectors.append(("TRACE", self._detect_trace))
    
    def _init_yolo_variants(self):
        """Initialize YOLO variants."""
        try:
            from ultralytics import YOLO
        except ImportError:
            return
        
        yolo_variants = [
            ("playersnball4.pt", "YOLO-v4"),
            ("playersnball5.pt", "YOLO-v5"),
        ]
        
        for weight_name, name in yolo_variants:
            model_path = PROJECT_ROOT / "old" / "models" / "player" / weight_name
            if not model_path.exists():
                continue
            
            try:
                model = YOLO(str(model_path))
                ball_class_ids = {
                    idx for idx, class_name in getattr(model, "names", {}).items()
                    if str(class_name).lower() in {"ball", "tennis_ball"}
                }
                if not ball_class_ids:
                    ball_class_ids = {0, 1}
                
                setattr(self, f"yolo_{weight_name.replace('.pt', '')}", model)
                setattr(self, f"yolo_{weight_name.replace('.pt', '')}_ball_ids", ball_class_ids)
                self.detectors.append((name, lambda f, m=model, ids=ball_class_ids: self._detect_yolo(f, m, ids)))
            except Exception as e:
                print(f"⚠ Could not load {name}: {e}")
    
    def _detect_sam3(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], float, np.ndarray]]:
        """Detect ball using SAM3."""
        if self.sam3_detector is None:
            return None
        return self.sam3_detector.detect_ball(frame, text_prompt="tennis ball", threshold=0.2)
    
    def _detect_rfdetr(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], float]]:
        """Detect ball using RF-DETR."""
        if not hasattr(self, 'rfdetr_model'):
            return None
        
        from PIL import Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        detections = self.rfdetr_model.predict(pil_img, threshold=0.2)
        
        if not hasattr(detections, "xyxy") or detections.xyxy is None:
            return None
        
        best = None
        best_conf = 0.0
        for bbox, conf, cls in zip(detections.xyxy, detections.confidence, detections.class_id):
            if int(cls) != self.rfdetr_ball_class_id:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            conf_val = float(conf)
            if conf_val > best_conf:
                best = center
                best_conf = conf_val
        
        if best:
            return (best, best_conf)
        return None
    
    def _detect_tracknet(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], float]]:
        """Detect ball using TrackNet (requires frame history)."""
        if not hasattr(self, 'tracknet_model'):
            return None
        
        # TrackNet needs 3 consecutive frames
        self.tracknet_frame_history.append(frame.copy())
        if len(self.tracknet_frame_history) > 3:
            self.tracknet_frame_history.pop(0)
        
        if len(self.tracknet_frame_history) < 3:
            return None
        
        # Simplified TrackNet detection (full implementation would use combine_three_frames)
        # For now, return None to avoid complex frame combining
        # TODO: Implement proper 3-frame TrackNet detection
        return None
    
    def _detect_trace(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], float]]:
        """Detect ball using TRACE."""
        if not hasattr(self, 'trace_detector'):
            return None
        
        self.trace_detector.detect_ball(frame)
        if len(self.trace_detector.xy_coordinates) == 0:
            return None
        
        x, y = self.trace_detector.xy_coordinates[-1]
        if x is None or y is None:
            return None
        
        return ((int(x), int(y)), 0.6)  # TRACE doesn't provide confidence
    
    def _detect_yolo(self, frame: np.ndarray, model, ball_class_ids: set) -> Optional[Tuple[Tuple[int, int], float]]:
        """Detect ball using YOLO."""
        results = model.predict(frame, verbose=False)
        if not results:
            return None
        
        result = results[0]
        if not hasattr(result, "boxes") or result.boxes is None:
            return None
        
        best = None
        best_conf = 0.0
        for box in result.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            if cls not in ball_class_ids:
                continue
            
            xyxy = box.xyxy.cpu().numpy().astype(int).tolist()[0]
            x1, y1, x2, y2 = xyxy
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if conf > best_conf:
                best = center
                best_conf = conf
        
        if best:
            return (best, best_conf)
        return None
    
    def detect_ball(
        self,
        frame: np.ndarray,
        text_prompt: str = "tennis ball"
    ) -> Optional[Tuple[Tuple[int, int], float, np.ndarray]]:
        """
        Detect ball using ensemble of all available detectors.
        Returns best detection (center, confidence, mask) where mask may be None.
        """
        all_detections = []
        
        # Run all detectors
        for detector_name, detector_func in self.detectors:
            try:
                result = detector_func(frame)
                if result is None:
                    continue
                
                # Handle different return formats
                if len(result) == 3:  # SAM3 format: (center, confidence, mask)
                    center, conf, mask = result
                    all_detections.append({
                        'center': center,
                        'confidence': conf,
                        'mask': mask,
                        'source': detector_name
                    })
                elif len(result) == 2:  # Other formats: (center, confidence)
                    center, conf = result
                    all_detections.append({
                        'center': center,
                        'confidence': conf,
                        'mask': None,
                        'source': detector_name
                    })
            except Exception as e:
                # Silently skip failed detectors
                continue
        
        if not all_detections:
            return None
        
        # Cluster nearby detections (within 50 pixels)
        clusters = []
        for det in all_detections:
            added = False
            for cluster in clusters:
                cluster_center = cluster['center']
                dist = np.sqrt(
                    (det['center'][0] - cluster_center[0])**2 +
                    (det['center'][1] - cluster_center[1])**2
                )
                if dist < 50:  # 50 pixel threshold
                    cluster['detections'].append(det)
                    cluster['total_conf'] += det['confidence']
                    cluster['count'] += 1
                    # Update cluster center as weighted average
                    total_conf = cluster['total_conf']
                    cluster['center'] = (
                        int(sum(d['center'][0] * d['confidence'] for d in cluster['detections']) / total_conf),
                        int(sum(d['center'][1] * d['confidence'] for d in cluster['detections']) / total_conf)
                    )
                    added = True
                    break
            
            if not added:
                clusters.append({
                    'center': det['center'],
                    'detections': [det],
                    'total_conf': det['confidence'],
                    'count': 1
                })
        
        # Find best cluster (highest consensus + confidence)
        best_cluster = None
        best_score = 0.0
        
        for cluster in clusters:
            # Score = consensus_count * average_confidence
            avg_conf = cluster['total_conf'] / cluster['count']
            score = cluster['count'] * avg_conf
            if score > best_score:
                best_score = score
                best_cluster = cluster
        
        if best_cluster is None:
            return None
        
        # Get best detection from best cluster (highest confidence)
        best_det = max(best_cluster['detections'], key=lambda d: d['confidence'])
        
        # Return in SAM3 format: (center, confidence, mask)
        return (
            best_cluster['center'],  # Use cluster center (consensus position)
            best_det['confidence'],  # Use best confidence
            best_det['mask']  # Use mask if available (from SAM3)
        )
