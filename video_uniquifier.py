"""
Video Uniqueness Generator
Generates multiple variations of a video with imperceptible differences
to bypass content duplication algorithms while maintaining visual quality.
"""

import os
import json
import random
import asyncio
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import hashlib


class VideoUniquifier:
    """Main class for generating unique video variations"""
    
    def __init__(self, input_video: str, output_dir: str = "output", num_variations: int = 10):
        self.input_video = Path(input_video)
        self.output_dir = Path(output_dir)
        self.num_variations = num_variations
        self.variations_log = []
        
        # Validate input
        if not self.input_video.exists():
            raise FileNotFoundError(f"Input video not found: {input_video}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Video info will be loaded asynchronously
        self.video_info = None
        
    async def _get_video_info(self) -> Dict:
        """Extract video metadata using ffprobe"""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-show_format',
            str(self.input_video)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"ffprobe failed: {stderr.decode()}")

            data = json.loads(stdout.decode())
            
            video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
            audio_stream = next((s for s in data['streams'] if s['codec_type'] == 'audio'), None)
            
            return {
                'width': int(video_stream['width']) if video_stream else 1080,
                'height': int(video_stream['height']) if video_stream else 1920,
                'duration': float(data['format'].get('duration', 0)),
                'has_audio': audio_stream is not None,
                'format': data['format']['format_name']
            }
        except Exception as e:
            print(f"Warning: Could not extract video info: {e}")
            return {'width': 1080, 'height': 1920, 'duration': 0, 'has_audio': True}
    
    def _generate_random_params(self) -> Dict:
        """Generate random parameters for video manipulation"""
        
        # Speed adjustment (0.99x to 1.01x)
        speed_factor = random.uniform(0.99, 1.01)
        
        # Color adjustments (subtle changes)
        brightness = random.uniform(-0.02, 0.02)  # -0.02 to +0.02
        contrast = random.uniform(0.98, 1.02)     # 0.98 to 1.02
        saturation = random.uniform(0.98, 1.02)   # 0.98 to 1.02
        gamma = random.uniform(0.98, 1.02)        # 0.98 to 1.02
        
        # Crop parameters (1-3% zoom)
        crop_percent = random.uniform(0.01, 0.03)
        crop_width = int(self.video_info['width'] * (1 - crop_percent))
        crop_height = int(self.video_info['height'] * (1 - crop_percent))
        
        # Ensure even dimensions for video encoding
        crop_width = crop_width - (crop_width % 2)
        crop_height = crop_height - (crop_height % 2)
        
        # Random crop position
        max_x = self.video_info['width'] - crop_width
        max_y = self.video_info['height'] - crop_height
        crop_x = random.randint(0, max(0, max_x))
        crop_y = random.randint(0, max(0, max_y))
        
        # Bitrate variation (±15%)
        base_bitrate = 4000  # 4000k base bitrate
        bitrate = int(base_bitrate * random.uniform(0.85, 1.15))
        
        # Noise level (very subtle)
        noise_strength = random.randint(1, 3)
        
        # Fake metadata
        metadata = self._generate_fake_metadata()
        
        return {
            'speed_factor': speed_factor,
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation,
            'gamma': gamma,
            'crop_x': crop_x,
            'crop_y': crop_y,
            'crop_width': crop_width,
            'crop_height': crop_height,
            'bitrate': bitrate,
            'noise_strength': noise_strength,
            'metadata': metadata
        }
    
    def _generate_fake_metadata(self) -> Dict:
        """Generate fake metadata to mimic different NLE software"""
        
        nle_tools = [
            "Adobe Premiere Pro 2024",
            "DaVinci Resolve 18.6",
            "Final Cut Pro 10.7",
            "iPhone 15 Pro Max",
            "Samsung Galaxy S24 Ultra",
            "Adobe Media Encoder 2024"
        ]
        
        # Random date within the last 30 days
        random_days = random.randint(0, 30)
        creation_date = datetime.now() - timedelta(days=random_days, 
                                                   hours=random.randint(0, 23),
                                                   minutes=random.randint(0, 59))
        
        return {
            'creation_time': creation_date.strftime('%Y-%m-%dT%H:%M:%S'),
            'encoder': random.choice(nle_tools),
            'comment': f'Processed with {random.choice(nle_tools)}'
        }
    
    def _build_ffmpeg_command(self, params: Dict, output_path: str) -> List[str]:
        """Build FFmpeg command with all filters and parameters"""
        
        # Video filter chain
        video_filters = []
        
        # 1. Speed adjustment (setpts)
        pts_factor = 1.0 / params['speed_factor']
        video_filters.append(f"setpts={pts_factor}*PTS")
        
        # 2. Color adjustments (eq filter)
        video_filters.append(
            f"eq=brightness={params['brightness']}:"
            f"contrast={params['contrast']}:"
            f"saturation={params['saturation']}:"
            f"gamma={params['gamma']}"
        )
        
        # 3. Crop with random position
        video_filters.append(
            f"crop={params['crop_width']}:{params['crop_height']}:"
            f"{params['crop_x']}:{params['crop_y']}"
        )
        
        # 4. Scale back to original resolution
        video_filters.append(
            f"scale={self.video_info['width']}:{self.video_info['height']}:"
            f"flags=lanczos"
        )
        
        # 5. Add subtle noise (optional)
        video_filters.append(f"noise=alls={params['noise_strength']}:allf=t")
        
        # Combine video filters
        vf_string = ",".join(video_filters)
        
        # Audio filter (atempo for speed adjustment)
        audio_filters = []
        if self.video_info['has_audio']:
            # atempo must be between 0.5 and 2.0, so we're safe with 0.99-1.01
            audio_filters.append(f"atempo={params['speed_factor']}")
        
        af_string = ",".join(audio_filters) if audio_filters else None
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            '-i', str(self.input_video),
            '-vf', vf_string,
        ]
        
        # Add audio filter if needed
        if af_string:
            cmd.extend(['-af', af_string])
        
        # Encoding parameters
        cmd.extend([
            '-c:v', 'libx264',                    # Video codec
            '-preset', 'medium',                   # Encoding preset
            '-crf', '18',                          # Quality (18 = visually lossless)
            '-b:v', f"{params['bitrate']}k",      # Variable bitrate
            '-maxrate', f"{params['bitrate'] * 1.5}k",
            '-bufsize', f"{params['bitrate'] * 2}k",
            '-pix_fmt', 'yuv420p',                # Pixel format
            '-movflags', '+faststart',             # Web optimization
        ])
        
        # Audio encoding
        if self.video_info['has_audio']:
            cmd.extend([
                '-c:a', 'aac',
                '-b:a', '192k',
                '-ar', '48000'
            ])
        
        # Metadata
        cmd.extend([
            '-metadata', f"creation_time={params['metadata']['creation_time']}",
            '-metadata', f"encoder={params['metadata']['encoder']}",
            '-metadata', f"comment={params['metadata']['comment']}",
            '-map_metadata', '-1',  # Strip original metadata
        ])
        
        # Output
        cmd.extend([
            '-y',  # Overwrite output file
            output_path
        ])
        
        return cmd
    
    async def generate_variations(self):
        """Generate all video variations"""
        
        if self.video_info is None:
            self.video_info = await self._get_video_info()
        
        print(f"🎬 Starting generation of {self.num_variations} variations...")
        print(f"📹 Input: {self.input_video.name}")
        print(f"📊 Resolution: {self.video_info['width']}x{self.video_info['height']}")
        print(f"⏱️  Duration: {self.video_info['duration']:.2f}s")
        print(f"🔊 Audio: {'Yes' if self.video_info['has_audio'] else 'No'}")
        print(f"📁 Output directory: {self.output_dir}\n")
        
        for i in range(1, self.num_variations + 1):
            print(f"[{i}/{self.num_variations}] Generating variation {i}...")
            
            # Generate random parameters
            params = self._generate_random_params()
            
            # Output filename
            output_filename = f"{self.input_video.stem}_variation_{i:03d}.mp4"
            output_path = self.output_dir / output_filename
            
            # Build FFmpeg command
            cmd = self._build_ffmpeg_command(params, str(output_path))
            
            # Execute FFmpeg
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd, output=stdout, stderr=stderr)
                
                # Calculate file hash
                loop = asyncio.get_running_loop()
                file_hash = await loop.run_in_executor(None, self._calculate_file_hash, output_path)
                
                # Log parameters
                variation_log = {
                    'variation_number': i,
                    'filename': output_filename,
                    'file_hash': file_hash,
                    'parameters': {
                        'speed_factor': f"{params['speed_factor']:.4f}x",
                        'brightness': f"{params['brightness']:+.4f}",
                        'contrast': f"{params['contrast']:.4f}",
                        'saturation': f"{params['saturation']:.4f}",
                        'gamma': f"{params['gamma']:.4f}",
                        'crop': f"{params['crop_width']}x{params['crop_height']} at ({params['crop_x']},{params['crop_y']})",
                        'bitrate': f"{params['bitrate']}k",
                        'noise': params['noise_strength'],
                        'metadata_encoder': params['metadata']['encoder']
                    }
                }
                
                self.variations_log.append(variation_log)
                
                print(f"   ✅ Created: {output_filename}")
                print(f"   📊 Hash: {file_hash[:16]}...")
                print(f"   ⚡ Speed: {params['speed_factor']:.4f}x")
                print(f"   🎨 Brightness: {params['brightness']:+.4f}, Contrast: {params['contrast']:.4f}")
                print(f"   ✂️  Crop: {params['crop_width']}x{params['crop_height']}")
                print(f"   💾 Bitrate: {params['bitrate']}k")
                print(f"   🏷️  Encoder: {params['metadata']['encoder']}\n")
                
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Error generating variation {i}: {e}")
                continue
        
        # Save log file
        self._save_log()
        
        print(f"\n✨ Generation complete!")
        print(f"📂 Output: {self.output_dir}")
        print(f"📝 Log: {self.output_dir / 'variations_log.json'}")
    
    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _save_log(self):
        """Save variation parameters to JSON log file"""
        log_path = self.output_dir / 'variations_log.json'
        
        log_data = {
            'input_video': str(self.input_video),
            'timestamp': datetime.now().isoformat(),
            'num_variations': self.num_variations,
            'video_info': self.video_info,
            'variations': self.variations_log
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate unique video variations to bypass duplication detection'
    )
    parser.add_argument(
        'input',
        help='Path to input video file'
    )
    parser.add_argument(
        '-o', '--output',
        default='output',
        help='Output directory (default: output)'
    )
    parser.add_argument(
        '-n', '--num-variations',
        type=int,
        default=10,
        help='Number of variations to generate (default: 10)'
    )
    
    args = parser.parse_args()
    
    try:
        uniquifier = VideoUniquifier(
            input_video=args.input,
            output_dir=args.output,
            num_variations=args.num_variations
        )
        asyncio.run(uniquifier.generate_variations())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
