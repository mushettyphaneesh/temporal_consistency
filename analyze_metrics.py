"""
Video Prompt Extraction - Metrics Analysis Script

Analyzes metrics from the video prompt extraction and generation project:
- Video processing statistics
- File sizes and storage usage
- API usage patterns
- Success rates
"""

import os
import json
import cv2
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from video_quality_metrics import VideoQualityMetrics
    QUALITY_METRICS_AVAILABLE = True
except ImportError:
    QUALITY_METRICS_AVAILABLE = False


class MetricsAnalyzer:
    """Analyzes project metrics for video prompt extraction."""
    
    def __init__(self, project_root: str = ".", enable_quality_metrics: bool = False):
        self.project_root = Path(project_root)
        self.uploads_dir = self.project_root / "uploads"
        self.outputs_dir = self.project_root / "outputs"
        self.enable_quality_metrics = enable_quality_metrics and QUALITY_METRICS_AVAILABLE
        
        if self.enable_quality_metrics:
            self.quality_metrics = VideoQualityMetrics(max_frames=30)
        
    def analyze_video_properties(self, video_path: Path) -> Dict:
        """Extract properties from a video file."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {"error": "Cannot open video"}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_seconds": round(duration, 2),
                "resolution": f"{width}x{height}",
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 2) if height > 0 else 0
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB."""
        return file_path.stat().st_size / (1024 * 1024)
    
    def analyze_uploads(self) -> Dict:
        """Analyze uploaded videos."""
        if not self.uploads_dir.exists():
            return {"error": "Uploads directory not found"}
        
        videos = list(self.uploads_dir.glob("*.mp4"))
        
        total_size = 0
        video_stats = []
        
        for video in videos:
            size_mb = self.get_file_size_mb(video)
            total_size += size_mb
            
            props = self.analyze_video_properties(video)
            
            video_stats.append({
                "filename": video.name,
                "size_mb": round(size_mb, 2),
                **props
            })
        
        return {
            "total_videos": len(videos),
            "total_size_mb": round(total_size, 2),
            "average_size_mb": round(total_size / len(videos), 2) if videos else 0,
            "videos": video_stats
        }
    
    def analyze_outputs(self) -> Dict:
        """Analyze generated videos."""
        if not self.outputs_dir.exists():
            return {"error": "Outputs directory not found"}
        
        videos = list(self.outputs_dir.glob("*.mp4"))
        
        total_size = 0
        video_stats = []
        
        for video in videos:
            size_mb = self.get_file_size_mb(video)
            total_size += size_mb
            
            props = self.analyze_video_properties(video)
            
            video_stats.append({
                "filename": video.name,
                "size_mb": round(size_mb, 2),
                **props
            })
        
        return {
            "total_generated": len(videos),
            "total_size_mb": round(total_size, 2),
            "average_size_mb": round(total_size / len(videos), 2) if videos else 0,
            "videos": video_stats
        }
    
    def analyze_storage(self) -> Dict:
        """Analyze overall storage usage."""
        uploads = self.analyze_uploads()
        outputs = self.analyze_outputs()
        
        total_storage = uploads.get("total_size_mb", 0) + outputs.get("total_size_mb", 0)
        
        return {
            "uploads_mb": uploads.get("total_size_mb", 0),
            "outputs_mb": outputs.get("total_size_mb", 0),
            "total_mb": round(total_storage, 2),
            "total_gb": round(total_storage / 1024, 3)
        }
    
    def analyze_performance(self) -> Dict:
        """Analyze performance metrics."""
        uploads = self.analyze_uploads()
        outputs = self.analyze_outputs()
        
        # Calculate average video properties
        input_videos = uploads.get("videos", [])
        output_videos = outputs.get("videos", [])
        
        def get_avg(videos: List[Dict], key: str) -> float:
            valid_values = [v.get(key, 0) for v in videos if key in v and not isinstance(v.get(key), str)]
            return round(sum(valid_values) / len(valid_values), 2) if valid_values else 0
        
        return {
            "input_videos": {
                "count": len(input_videos),
                "avg_duration_sec": get_avg(input_videos, "duration_seconds"),
                "avg_fps": get_avg(input_videos, "fps"),
                "avg_size_mb": uploads.get("average_size_mb", 0)
            },
            "output_videos": {
                "count": len(output_videos),
                "avg_duration_sec": get_avg(output_videos, "duration_seconds"),
                "avg_fps": get_avg(output_videos, "fps"),
                "avg_size_mb": outputs.get("average_size_mb", 0)
            },
            "success_rate": f"{(len(output_videos) / len(input_videos) * 100):.1f}%" if input_videos else "N/A"
        }
    
    def analyze_video_quality(self) -> Dict:
        """Analyze video quality metrics (PSNR, SSIM, warping error)."""
        if not self.enable_quality_metrics:
            return {"enabled": False}
        
        uploads = self.analyze_uploads()
        outputs = self.analyze_outputs()
        
        input_videos = uploads.get("videos", [])
        output_videos = outputs.get("videos", [])
        
        # Analyze temporal consistency for each video
        input_quality = []
        for video in input_videos:
            video_path = self.uploads_dir / video["filename"]
            try:
                result = self.quality_metrics.analyze_single_video(video_path)
                input_quality.append({
                    "filename": video["filename"],
                    "warping_error": result.get("warping_error", {})
                })
            except Exception as e:
                input_quality.append({
                    "filename": video["filename"],
                    "error": str(e)
                })
        
        output_quality = []
        for video in output_videos:
            video_path = self.outputs_dir / video["filename"]
            try:
                result = self.quality_metrics.analyze_single_video(video_path)
                output_quality.append({
                    "filename": video["filename"],
                    "warping_error": result.get("warping_error", {})
                })
            except Exception as e:
                output_quality.append({
                    "filename": video["filename"],
                    "error": str(e)
                })
        
        # Compare first input with first output (if available)
        comparison = None
        if input_videos and output_videos:
            try:
                input_path = self.uploads_dir / input_videos[0]["filename"]
                output_path = self.outputs_dir / output_videos[0]["filename"]
                comparison = self.quality_metrics.compare_videos(input_path, output_path)
            except Exception as e:
                comparison = {"error": str(e)}
        
        return {
            "enabled": True,
            "input_quality": input_quality,
            "output_quality": output_quality,
            "comparison_sample": comparison
        }
    
    def generate_report(self) -> Dict:
        """Generate comprehensive metrics report."""
        report = {
            "report_date": datetime.now().isoformat(),
            "project_root": str(self.project_root.absolute()),
            "uploads": self.analyze_uploads(),
            "outputs": self.analyze_outputs(),
            "storage": self.analyze_storage(),
            "performance": self.analyze_performance()
        }
        
        if self.enable_quality_metrics:
            report["quality_metrics"] = self.analyze_video_quality()
        
        return report
    
    def print_report(self):
        """Print a formatted metrics report."""
        report = self.generate_report()
        
        print("\n" + "=" * 70)
        print("VIDEO PROMPT EXTRACTION - METRICS REPORT")
        print("=" * 70)
        print(f"Generated: {report['report_date']}")
        print(f"Project: {report['project_root']}")
        
        print("\n" + "-" * 70)
        print("📊 STORAGE METRICS")
        print("-" * 70)
        storage = report['storage']
        print(f"  Uploads:         {storage['uploads_mb']:.2f} MB")
        print(f"  Generated:       {storage['outputs_mb']:.2f} MB")
        print(f"  Total Storage:   {storage['total_mb']:.2f} MB ({storage['total_gb']:.3f} GB)")
        
        print("\n" + "-" * 70)
        print("📥 UPLOADED VIDEOS")
        print("-" * 70)
        uploads = report['uploads']
        print(f"  Total Videos:    {uploads['total_videos']}")
        print(f"  Total Size:      {uploads['total_size_mb']:.2f} MB")
        print(f"  Average Size:    {uploads['average_size_mb']:.2f} MB")
        
        if uploads.get('videos'):
            print("\n  Video Details:")
            for video in uploads['videos']:
                print(f"    • {video['filename']}")
                print(f"      Size: {video['size_mb']:.2f} MB | Duration: {video.get('duration_seconds', 'N/A')}s | " +
                      f"Resolution: {video.get('resolution', 'N/A')} | FPS: {video.get('fps', 'N/A')}")
        
        print("\n" + "-" * 70)
        print("📤 GENERATED VIDEOS")
        print("-" * 70)
        outputs = report['outputs']
        print(f"  Total Generated: {outputs['total_generated']}")
        print(f"  Total Size:      {outputs['total_size_mb']:.2f} MB")
        print(f"  Average Size:    {outputs['average_size_mb']:.2f} MB")
        
        if outputs.get('videos'):
            print("\n  Video Details:")
            for video in outputs['videos']:
                print(f"    • {video['filename']}")
                print(f"      Size: {video['size_mb']:.2f} MB | Duration: {video.get('duration_seconds', 'N/A')}s | " +
                      f"Resolution: {video.get('resolution', 'N/A')} | FPS: {video.get('fps', 'N/A')}")
        
        print("\n" + "-" * 70)
        print("⚡ PERFORMANCE METRICS")
        print("-" * 70)
        perf = report['performance']
        print(f"  Success Rate:    {perf['success_rate']}")
        print(f"\n  Input Videos:")
        print(f"    Count:           {perf['input_videos']['count']}")
        print(f"    Avg Duration:    {perf['input_videos']['avg_duration_sec']}s")
        print(f"    Avg Size:        {perf['input_videos']['avg_size_mb']:.2f} MB")
        print(f"\n  Output Videos:")
        print(f"    Count:           {perf['output_videos']['count']}")
        print(f"    Avg Duration:    {perf['output_videos']['avg_duration_sec']}s")
        print(f"    Avg Size:        {perf['output_videos']['avg_size_mb']:.2f} MB")
        
        # Print quality metrics if enabled
        if self.enable_quality_metrics and "quality_metrics" in report:
            quality = report["quality_metrics"]
            if quality.get("enabled"):
                print("\n" + "-" * 70)
                print("🎬 VIDEO QUALITY METRICS")
                print("-" * 70)
                
                # Input video temporal consistency
                print("\n  Input Videos - Temporal Consistency (Warping Error):")
                for vid in quality.get("input_quality", []):
                    if "error" not in vid:
                        we = vid["warping_error"]
                        print(f"    • {vid['filename']}: {we.get('mean', 0):.2f} px/frame")
                
                # Output video temporal consistency
                print("\n  Generated Videos - Temporal Consistency (Warping Error):")
                for vid in quality.get("output_quality", []):
                    if "error" not in vid:
                        we = vid["warping_error"]
                        print(f"    • {vid['filename']}: {we.get('mean', 0):.2f} px/frame")
                
                # Comparison metrics
                if quality.get("comparison_sample"):
                    comp = quality["comparison_sample"]
                    if "error" not in comp:
                        print("\n  Sample Comparison (First Input vs First Output):")
                        print(f"    PSNR:  {comp['psnr']['mean']:.2f} ± {comp['psnr']['std']:.2f} dB")
                        print(f"    SSIM:  {comp['ssim']['mean']:.4f} ± {comp['ssim']['std']:.4f}")
                        print(f"    Frames Analyzed: {comp['frames_analyzed']}")
        
        print("\n" + "=" * 70)
    
    def save_report_json(self, output_path: str = "metrics_report.json"):
        """Save report as JSON file."""
        report = self.generate_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Report saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze metrics for video prompt extraction project")
    parser.add_argument("--project-root", "-p", default=".", help="Project root directory")
    parser.add_argument("--save-json", "-s", action="store_true", help="Save report as JSON")
    parser.add_argument("--output", "-o", default="metrics_report.json", help="JSON output file path")
    parser.add_argument("--quality-metrics", "-q", action="store_true", help="Enable video quality metrics (PSNR, SSIM, warping error)")
    
    args = parser.parse_args()
    
    analyzer = MetricsAnalyzer(args.project_root, enable_quality_metrics=args.quality_metrics)
    
    # Print formatted report
    analyzer.print_report()
    
    # Save JSON if requested
    if args.save_json:
        analyzer.save_report_json(args.output)


if __name__ == "__main__":
    main()
