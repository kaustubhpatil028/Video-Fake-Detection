#!/usr/bin/env python3
"""
Simple Social Media Content Downloader
Usage: python downloader.py <URL>
"""

import os
import sys
import subprocess
from pathlib import Path

class SimpleDownloader:
    def __init__(self):
        """Initialize downloader"""
        self.download_path = Path(__file__).parent / "downloads"
        self.download_path.mkdir(exist_ok=True, parents=True)
        self.check_ytdlp()
    
    def check_ytdlp(self):
        """Check if yt-dlp is installed"""
        try:
            subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ yt-dlp not found. Install with: pip install yt-dlp")
            sys.exit(1)
    
    def detect_platform(self, url):
        """Detect platform from URL"""
        url = url.lower()
        if 'instagram.com' in url:
            return 'instagram'
        elif 'twitter.com' in url or 'x.com' in url:
            return 'twitter'
        elif 'youtube.com' in url or 'youtu.be' in url:
            return 'youtube'
        return 'unknown'
    
    def download(self, url):
        """Download content from URL. Returns output filename if successful, else None."""
        platform = self.detect_platform(url)
        if platform == 'unknown':
            print("❌ Unsupported URL. Only Instagram, YouTube, and Twitter/X are supported.")
            return None
        print(f"🔍 Detected: {platform.upper()}")
        print(f"⬇️ Downloading from: {url}")
        if platform == 'instagram':
            return self._download_instagram(url, self.download_path)
        elif platform == 'youtube':
            return self._download_youtube(url)
        elif platform == 'twitter':
            return self._download_twitter(url)
    
    def _download_instagram(self, url, path):
        """Download Instagram content"""
        downloads_path = self.download_path
        existing_files = list(downloads_path.glob('insta*.mp4'))
        next_num = len(existing_files) + 1
        output_template = str(downloads_path / f'insta{next_num}.mp4')
        methods = [
            ['yt-dlp', '--cookies-from-browser', 'chrome', '-o', output_template, url],
            ['yt-dlp', '--cookies-from-browser', 'firefox', '-o', output_template, url],
            ['yt-dlp', '--user-agent', 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)', '-o', output_template, url],
            ['yt-dlp', '-o', output_template, url]
        ]
        for i, cmd in enumerate(methods, 1):
            print(f"🔄 Trying method {i}...")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print("✅ Download completed!")
                    return output_template
            except Exception:
                continue
        print("❌ All methods failed. Make sure you're logged into Instagram in your browser.")
        return None
    
    def _download_youtube(self, url):
        """Download YouTube content"""
        downloads_path = self.download_path
        existing_files = list(downloads_path.glob('yt*.mp4'))
        next_num = len(existing_files) + 1
        output_template = str(downloads_path / f'yt{next_num}.mp4')
        cmd = [
            'yt-dlp',
            '-f', 'best[ext=mp4]/best',
            '-o', output_template,
            url
        ]
        try:
            result = subprocess.run(cmd, check=True, timeout=300)
            print("✅ Download completed!")
            return output_template
        except subprocess.CalledProcessError:
            print("❌ Download failed")
            return None
        except subprocess.TimeoutExpired:
            print("❌ Download timed out")
            return None
    
    def _download_twitter(self, url):
        """Download Twitter content"""
        downloads_path = self.download_path
        existing_files = list(downloads_path.glob('tw*.mp4'))
        next_num = len(existing_files) + 1
        output_template = str(downloads_path / f'tw{next_num}.mp4')
        methods = [
            ['yt-dlp', '--cookies-from-browser', 'chrome', '-o', output_template, url],
            ['yt-dlp', '--cookies-from-browser', 'firefox', '-o', output_template, url],
            ['yt-dlp', '--extractor-args', 'twitter:api=syndication', '-o', output_template, url],
            ['yt-dlp', '-o', output_template, url]
        ]
        for i, cmd in enumerate(methods, 1):
            print(f"🔄 Trying method {i}...")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print("✅ Download completed!")
                    return output_template
            except Exception:
                continue
        print("❌ All methods failed. Make sure you're logged into Twitter/X in your browser.")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python downloader.py <URL>")
        print("Example: python downloader.py https://www.instagram.com/reel/ABC123/")
        sys.exit(1)
    
    url = sys.argv[1]
    downloader = SimpleDownloader()
    
    filename = downloader.download(url)
    
    if filename:
        print(f"📁 File saved to: {filename}")
    else:
        print("❌ Download failed")
        sys.exit(1)

if __name__ == "__main__":
    main()