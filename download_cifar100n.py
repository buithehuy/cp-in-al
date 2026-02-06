"""Helper script to download CIFAR-100N noisy labels."""
import os
import urllib.request

# Mirror URLs
URLS = [
    "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-100_human.pt",
    "https://huggingface.co/datasets/nateraw/cifar-100n/resolve/main/CIFAR-100_human.pt",
    "http://www.yliuu.com/web-cifarN/files/CIFAR-100_human.pt",
]

OUTPUT_PATH = "data/CIFAR-100_human.pt"

def download_cifar100n():
    """Download CIFAR-100N noisy labels with fallback mirrors."""
    # Create directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Check if already exists
    if os.path.exists(OUTPUT_PATH):
        file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)  # MB
        print(f"✓ File already exists: {OUTPUT_PATH} ({file_size:.2f} MB)")
        return True
    
    print("=" * 70)
    print("Downloading CIFAR-100N Noisy Labels")
    print("=" * 70)
    
    # Try each mirror
    for i, url in enumerate(URLS):
        print(f"\n[{i+1}/{len(URLS)}] Trying: {url}")
        try:
            def progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                print(f"\r  Progress: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB)", end='')
            
            urllib.request.urlretrieve(url, OUTPUT_PATH, reporthook=progress)
            print()  # New line after progress
            
            # Verify file
            file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
            print(f"  ✓ Download complete! Size: {file_size:.2f} MB")
            print(f"  ✓ Saved to: {OUTPUT_PATH}")
            return True
            
        except Exception as e:
            print(f"\n  ✗ Failed: {e}")
            # Remove partial download
            if os.path.exists(OUTPUT_PATH):
                os.remove(OUTPUT_PATH)
            continue
    
    # All mirrors failed
    print("\n" + "=" * 70)
    print("ERROR: All download mirrors failed!")
    print("=" * 70)
    print("\nManual download options:")
    print()
    print("1. Using wget:")
    print(f"   wget {URLS[0]} -O {OUTPUT_PATH}")
    print()
    print("2. Using curl:")
    print(f"   curl -L {URLS[0]} -o {OUTPUT_PATH}")
    print()
    print("3. Clone repository:")
    print("   git clone https://github.com/UCSC-REAL/cifar-10-100n.git")
    print(f"   cp cifar-10-100n/data/CIFAR-100_human.pt {OUTPUT_PATH}")
    print("=" * 70)
    return False

if __name__ == "__main__":
    success = download_cifar100n()
    if success:
        print("\n" + "=" * 70)
        print("SUCCESS! You can now use CIFAR-100N:")
        print("  python src/train.py data=cifar100n data.noise_type=noisy")
        print("=" * 70)
    else:
        exit(1)
