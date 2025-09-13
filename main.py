import os
import aiohttp
import asyncio
from PIL import Image
from io import BytesIO
from tqdm.asyncio import tqdm
import shutil
import random
import string

global FAILED

# ===== CONFIG =====
IMAGE_DIR = "images"
FAILED = 0
COVER_DIR = os.path.join(IMAGE_DIR, "cover")
STEGO_DIR = os.path.join(IMAGE_DIR, "stego")
DATASET_DIR = "dataset"
NUM_IMAGES = 10000  # number of cover images to download
MAX_CONCURRENT = 16  # parallel downloads
MESSAGE_LENGTH = 20  # length of random message in each stego image

# Example URLs (using picsum.photos for random images)
IMAGE_URLS = [
    f"https://picsum.photos/128/128?random={i}" for i in range(NUM_IMAGES)
]

# ===== FUNCTIONS =====
def random_message(length=MESSAGE_LENGTH):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

async def download_image(session, url, save_dir):
    filename = os.path.join(save_dir, url.split('=')[-1] + ".png")
    if os.path.exists(filename):
        return filename
    try:
        async with session.get(url, timeout=10) as resp:
            data = await resp.read()
            img = Image.open(BytesIO(data))
            img.save(filename)
            return filename
    except Exception as e:
        print(f"Failed {url}: {e}")
        FAILED == FAILED+1
        return None

async def download_all_images(urls, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_image(session, url, save_dir) for url in urls]
        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await f
            if result:
                results.append(result)
        return results

def embed_lsb(image_path, message):
    """Embed a random message using LSB steganography."""
    img = Image.open(image_path).convert("RGB")
    binary_message = ''.join(format(ord(c), '08b') for c in message)
    binary_message += '1111111111111110'  # delimiter
    data_index = 0
    pixels = img.load()

    for y in range(img.height):
        for x in range(img.width):
            if data_index >= len(binary_message):
                break
            r, g, b = pixels[x, y]
            r = (r & ~1) | int(binary_message[data_index])
            data_index += 1
            if data_index < len(binary_message):
                g = (g & ~1) | int(binary_message[data_index])
                data_index += 1
            if data_index < len(binary_message):
                b = (b & ~1) | int(binary_message[data_index])
                data_index += 1
            pixels[x, y] = (r, g, b)
        if data_index >= len(binary_message):
            break

    stego_path = os.path.join(STEGO_DIR, os.path.basename(image_path))
    img.save(stego_path)
    return stego_path

def split_dataset():
    os.makedirs(os.path.join(DATASET_DIR, "train", "cover"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "test", "cover"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "train", "stego"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "test", "stego"), exist_ok=True)

    cover_images = os.listdir(COVER_DIR)
    stego_images = os.listdir(STEGO_DIR)
    random.shuffle(cover_images)
    random.shuffle(stego_images)

    split_index = int(len(cover_images) * 0.8)

    for i, fname in enumerate(cover_images):
        dest = "train" if i < split_index else "test"
        shutil.copy(os.path.join(COVER_DIR, fname), os.path.join(DATASET_DIR, dest, "cover", fname))

    for i, fname in enumerate(stego_images):
        dest = "train" if i < split_index else "test"
        shutil.copy(os.path.join(STEGO_DIR, fname), os.path.join(DATASET_DIR, dest, "stego", fname))

# ===== MAIN SCRIPT =====
os.makedirs(COVER_DIR, exist_ok=True)
os.makedirs(STEGO_DIR, exist_ok=True)
for i in range(50):
    print("Downloading cover images asynchronously (skipping existing)...")
    asyncio.run(download_all_images(IMAGE_URLS, COVER_DIR))

print("Generating stego images with random messages...")
cover_paths = [os.path.join(COVER_DIR, f) for f in os.listdir(COVER_DIR)]
for path in tqdm(cover_paths):
    embed_lsb(path, random_message(MESSAGE_LENGTH))

print("Splitting dataset into train/test...")
split_dataset()
print("Dataset split completed!")

print(f"Train cover images: {len(os.listdir(os.path.join(DATASET_DIR, 'train', 'cover')))}")
print(f"Test cover images: {len(os.listdir(os.path.join(DATASET_DIR, 'test', 'cover')))}")
print(f"Train stego images: {len(os.listdir(os.path.join(DATASET_DIR, 'train', 'stego')))}")
print(f"Test stego images: {len(os.listdir(os.path.join(DATASET_DIR, 'test', 'stego')))}")
print("Pipeline completed! Dataset ready for ML.")
print("Failed downloads:",FAILED)
