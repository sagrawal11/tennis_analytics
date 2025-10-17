#!/usr/bin/env python3
"""
Tennis Master Controller
Launches both tennis_CV.py and tennis_analytics.py viewers and coordinates data flow.
"""

import subprocess
import threading
import time
import signal
import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
import queue
import socket
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TennisMasterController:
    """Master controller for tennis analysis system"""
    
    def __init__(self, video_path: str, config_path: str = "config.yaml"):
        """Initialize the master controller"""
        self.video_path = video_path
        self.config_path = config_path
        
        # Process management
        self.cv_process: Optional[subprocess.Popen] = None
        self.analytics_process: Optional[subprocess.Popen] = None
        
        # Data communication
        self.data_queue = queue.Queue()
        self.running = False
        
        # Socket for inter-process communication
        self.socket_server = None
        self.socket_port = 12345
        
        # Verify files exist
        self._verify_files()
    
    def _verify_files(self):
        """Verify that all required files exist"""
        # Get the project root directory (two levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        
        required_files = [
            self.video_path,
            project_root / self.config_path,
            project_root / "src" / "core" / "tennis_CV.py",
            project_root / "src" / "core" / "tennis_analytics.py"
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                logger.error(f"Required file not found: {file_path}")
                sys.exit(1)
        
        logger.info("✅ All required files verified")
    
    def start_socket_server(self):
        """Start socket server for inter-process communication"""
        try:
            self.socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket_server.bind(('localhost', self.socket_port))
            self.socket_server.listen(1)
            logger.info(f"🔌 Socket server started on port {self.socket_port}")
        except Exception as e:
            logger.error(f"Failed to start socket server: {e}")
            return False
        return True
    
    def start_cv_viewer(self):
        """Start the CV viewer process"""
        try:
            # Get the project root and path to tennis_CV.py
            project_root = Path(__file__).parent.parent.parent
            cv_script = project_root / "src" / "core" / "tennis_CV.py"
            output_path = project_root / "outputs" / "videos" / "tennis_analysis_output.mp4"
            
            cmd = [
                sys.executable, str(cv_script),
                "--video", self.video_path,
                "--config", str(project_root / self.config_path),
                "--output", str(output_path)
            ]
            
            logger.info("🚀 Starting CV viewer...")
            logger.info(f"Command: {' '.join(cmd)}")
            
            # Set environment variables needed for RF-DETR on macOS
            env = os.environ.copy()
            env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
            # Add additional environment variables for better subprocess compatibility
            env['PYTHONPATH'] = str(project_root) + ':' + env.get('PYTHONPATH', '')
            env['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU usage if CUDA is available
            
            logger.info(f"🔧 Environment variables:")
            logger.info(f"   PYTORCH_ENABLE_MPS_FALLBACK: {env.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
            logger.info(f"   PYTHONPATH: {env.get('PYTHONPATH')}")
            logger.info(f"   Working directory: {str(project_root)}")
            
            # Run CV process without capturing output to avoid subprocess issues
            # This allows OpenCV windows and user input to work properly
            self.cv_process = subprocess.Popen(
                cmd,
                stdout=None,  # Don't capture stdout - let it display normally
                stderr=None,  # Don't capture stderr - let it display normally
                env=env,
                cwd=str(project_root)  # Ensure working directory is correct
            )
            
            logger.info(f"✅ CV viewer started (PID: {self.cv_process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start CV viewer: {e}")
            return False
    
    def start_analytics_viewer(self):
        """Start the analytics viewer process"""
        try:
            # Get the project root and path to tennis_analytics.py
            project_root = Path(__file__).parent.parent.parent
            analytics_script = project_root / "src" / "core" / "tennis_analytics.py"
            csv_path = project_root / "data" / "processed" / "csv" / "tennis_analysis_data.csv"
            output_path = project_root / "outputs" / "videos" / "tennis_analytics_output.mp4"
            
            cmd = [
                sys.executable, str(analytics_script),
                "--csv", str(csv_path),
                "--output", str(output_path)
            ]
            
            logger.info("📊 Starting analytics viewer...")
            logger.info(f"Command: {' '.join(cmd)}")
            
            # Run analytics process without capturing output to avoid subprocess issues
            self.analytics_process = subprocess.Popen(
                cmd,
                stdout=None,  # Don't capture stdout - let it display normally
                stderr=None,  # Don't capture stderr - let it display normally
            )
            
            logger.info(f"✅ Analytics viewer started (PID: {self.analytics_process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start analytics viewer: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor both processes and handle communication"""
        logger.info("👀 Monitoring processes...")
        
        while self.running:
            # Check if processes are still running
            if self.cv_process and self.cv_process.poll() is not None:
                logger.warning("⚠️  CV viewer process terminated")
                self.running = False
                break
            
            if self.analytics_process and self.analytics_process.poll() is not None:
                logger.warning("⚠️  Analytics viewer process terminated")
                self.running = False
                break
            
            # Simple monitoring - data communication is now handled via shared file
            time.sleep(0.1)  # Small delay to prevent busy waiting
    

    
    def handle_shutdown(self, signum=None, frame=None):
        """Handle graceful shutdown"""
        logger.info("🛑 Shutting down tennis master...")
        self.running = False
        
        # Terminate processes
        if self.cv_process:
            logger.info("Terminating CV viewer...")
            self.cv_process.terminate()
            try:
                self.cv_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("CV viewer didn't terminate gracefully, forcing...")
                self.cv_process.kill()
        
        if self.analytics_process:
            logger.info("Terminating analytics viewer...")
            self.analytics_process.terminate()
            try:
                self.analytics_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Analytics viewer didn't terminate gracefully, forcing...")
                self.analytics_process.kill()
        
        # Close socket server
        if self.socket_server:
            self.socket_server.close()
        
        logger.info("✅ Tennis master shutdown complete")
        sys.exit(0)
    
    def _get_csv_line_count(self):
        """Get the current number of lines in the CSV file with file locking"""
        try:
            project_root = Path(__file__).parent.parent.parent
            csv_path = project_root / "data" / "processed" / "csv" / "tennis_analysis_data.csv"
            if csv_path.exists():
                try:
                    import fcntl
                    with open(csv_path, 'r') as f:
                        # Acquire shared lock for reading
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                        lines = f.readlines()
                        # Release lock
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                        return len(lines)
                except ImportError:
                    # fcntl not available on Windows, use regular file operations
                    with open(csv_path, 'r') as f:
                        return len(f.readlines())
                except Exception as e:
                    logger.warning(f"File locking error: {e}")
                    # Fallback to regular file operations
                    with open(csv_path, 'r') as f:
                        return len(f.readlines())
            return 0
        except Exception as e:
            logger.warning(f"Error reading CSV: {e}")
            return 0
    

    
    def run(self):
        """Run the master controller"""
        logger.info("🎾 TENNIS MASTER CONTROLLER STARTING...")
        logger.info(f"📹 Video: {self.video_path}")
        logger.info(f"⚙️  Config: {self.config_path}")
        logger.info("🚀 Launching CV processing first...")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        self.running = True
        
        try:
            # Start CV viewer first
            if not self.start_cv_viewer():
                logger.error("Failed to start CV viewer")
                return False
            
            logger.info("🎬 CV viewer started. Processing video...")
            logger.info("📺 CV window should be open showing real-time processing")
            logger.info("⏳ Waiting for CV processing to complete...")
            
            # Wait for CV process to complete WITHOUT touching the CSV
            # This prevents any interference with the CV process
            logger.info("🔒 CV process is running - CSV file is LOCKED for writing only")
            logger.info("⏳ Waiting for CV process to complete completely...")
            
            start_time = time.time()
            while self.cv_process and self.cv_process.poll() is None:
                time.sleep(1)
                elapsed_time = int(time.time() - start_time)
                
                # Add timeout protection (40 minutes max for full video processing)
                if elapsed_time > 2400:
                    logger.error("⏰ CV process timeout after 20 minutes! Terminating...")
                    self.cv_process.terminate()
                    try:
                        self.cv_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning("CV process didn't terminate gracefully, forcing...")
                        self.cv_process.kill()
                    break
                
                # Show simple progress without CSV access
                if elapsed_time % 10 == 0:
                    logger.info(f"⏳ CV process still running... Time elapsed: {elapsed_time}s")
                    
                    # Simple process health check (no CSV access)
                    if self.cv_process:
                        try:
                            import psutil
                            process = psutil.Process(self.cv_process.pid)
                            cpu_percent = process.cpu_percent()
                            memory_mb = process.memory_info().rss / 1024 / 1024
                            logger.info(f"🔍 Process status: CPU {cpu_percent:.1f}%, Memory {memory_mb:.1f}MB")
                        except ImportError:
                            logger.info(f"🔍 Process status: {self.cv_process.poll()}")
            
            # CV process has completed - now we can safely read the CSV
            if self.cv_process:
                exit_code = self.cv_process.returncode
                logger.info(f"✅ CV processing completed with exit code: {exit_code}")
                
                # Now safely read the final CSV to show results
                project_root = Path(__file__).parent.parent.parent
                csv_path = project_root / "data" / "processed" / "csv" / "tennis_analysis_data.csv"
                if csv_path.exists():
                    try:
                        with open(csv_path, 'r') as f:
                            lines = f.readlines()
                            if len(lines) > 1:
                                logger.info(f"📊 Final CSV results: {len(lines)-1} frames processed successfully!")
                            else:
                                logger.warning(f"⚠️ CSV only has {len(lines)} lines - CV process may have failed")
                    except Exception as e:
                        logger.error(f"❌ Error reading final CSV: {e}")
                else:
                    logger.error("❌ CSV file not found after CV completion")
            
            if self.cv_process:
                exit_code = self.cv_process.returncode
                logger.info(f"✅ CV processing completed with exit code: {exit_code}")
            
            # Check if CSV file was created
            project_root = Path(__file__).parent.parent.parent
            csv_path = project_root / "data" / "processed" / "csv" / "tennis_analysis_data.csv"
            if not csv_path.exists():
                logger.error("❌ CSV file not found. CV processing may have failed.")
                return False
            
            logger.info("📊 CSV data file created successfully!")
            logger.info("🚀 Now launching analytics viewer...")
            
            # Start analytics viewer
            if not self.start_analytics_viewer():
                logger.error("Failed to start analytics viewer")
                return False
            
            logger.info("🎉 Analytics viewer launched successfully!")
            logger.info("📺 Analytics window should now be open")
            logger.info("🎮 Controls:")
            logger.info("   - Press 'q' to quit")
            logger.info("   - Press 'space' to pause/resume")
            logger.info("   - Press 'left/right' arrows to step through frames")
            logger.info("   - Press 'up/down' arrows to change playback speed")
            logger.info("   - Press 't' to toggle trajectories")
            logger.info("   - Press 'a' to toggle analytics panel")
            
            # Monitor analytics process
            while self.analytics_process and self.analytics_process.poll() is None:
                time.sleep(0.1)
            
            logger.info("✅ Analytics viewer completed")
            
        except Exception as e:
            logger.error(f"Error in master controller: {e}")
            self.handle_shutdown()
            return False
        
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Tennis Master Controller')
    parser.add_argument('--video', '-v', type=str, default='tennis_test.mp4',
                       help='Path to input video file')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        logger.error(f"Video file '{args.video}' not found!")
        logger.info("Available video files:")
        for video_file in Path('.').glob('*.mp4'):
            logger.info(f"  - {video_file}")
        return
    
    # Create and run master controller
    controller = TennisMasterController(args.video, args.config)
    success = controller.run()
    
    if not success:
        logger.error("❌ Tennis master controller failed")
        sys.exit(1)
    
    logger.info("✅ Tennis master controller completed successfully")

if __name__ == "__main__":
    main()
