#!/usr/bin/env python3
"""
Benchmark Dataset Downloader
Downloads classic image restoration benchmark datasets and creates
synthetically degraded versions for evaluation.
"""
import os
import urllib.request
import cv2
import numpy as np

BENCHMARK_DIR = "benchmark"
ORIGINAL_DIR = os.path.join(BENCHMARK_DIR, "original")
DEGRADED_DIR = os.path.join(BENCHMARK_DIR, "degraded")
RESTORED_DIR = os.path.join(BENCHMARK_DIR, "restored")

os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(DEGRADED_DIR, exist_ok=True)
os.makedirs(RESTORED_DIR, exist_ok=True)


def generate_gradient_image(size=(512, 512)):
    img = np.zeros((*size, 3), dtype=np.uint8)
    for i in range(size[0]):
        r = int(255 * (i / size[0]))
        g = int(255 * (1 - i / size[0]))
        b = int(128 + 127 * np.sin(i * np.pi / size[0]))
        img[i, :] = [b, g, r]
    return img


def generate_checkerboard(size=(512, 512), square_size=32):
    img = np.zeros((*size, 3), dtype=np.uint8)
    for i in range(0, size[0], square_size):
        for j in range(0, size[1], square_size):
            color = 255 if ((i // square_size) + (j // square_size)) % 2 == 0 else 50
            img[i:i+square_size, j:j+square_size] = [color, color, color]
    return img


def generate_circles_image(size=(512, 512)):
    img = np.ones((*size, 3), dtype=np.uint8) * 240
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255)]
    for i, (cx, cy, r) in enumerate([(150, 150, 80), (350, 200, 60), (250, 350, 100), (400, 400, 50), (100, 400, 70)]):
        cv2.circle(img, (cx, cy), r, colors[i % len(colors)], -1)
    cv2.line(img, (50, 50), (450, 450), (50, 50, 50), 3)
    cv2.rectangle(img, (200, 100), (300, 400), (100, 100, 100), 2)
    return img


def generate_text_scene(size=(512, 512)):
    img = np.ones((*size, 3), dtype=np.uint8) * 220
    texts = [
        ("BENCHMARK", (50, 100), 2.0, (50, 50, 50)),
        ("Image Restoration", (50, 200), 1.2, (100, 100, 100)),
        ("Test Suite", (50, 300), 1.5, (80, 80, 80)),
        ("0123456789", (50, 400), 1.0, (60, 60, 60)),
    ]
    for text, pos, scale, color in texts:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    return img


def generate_landscape(size=(512, 512)):
    img = np.zeros((*size, 3), dtype=np.uint8)
    for i in range(size[0] // 2):
        img[i, :] = [int(200 - i * 0.5), int(220 - i * 0.3), int(255 - i * 0.2)]
    for i in range(size[0] // 2, size[0]):
        img[i, :] = [int(50 + (i - size[0]//2) * 0.3), int(100 + (i - size[0]//2) * 0.4), int(30)]
    for x in range(size[1]):
        peak = int(size[0] * 0.3 + 50 * np.sin(x * np.pi / 200))
        for y in range(peak, size[0] // 2):
            img[y, x] = [100, 100, 120]
    cv2.circle(img, (400, 100), 40, (200, 220, 255), -1)
    return img


def generate_portrait(size=(512, 512)):
    img = np.ones((*size, 3), dtype=np.uint8) * 200
    cv2.ellipse(img, (256, 256), (120, 150), 0, 0, 360, (210, 180, 160), -1)
    cv2.circle(img, (210, 230), 20, (255, 255, 255), -1)
    cv2.circle(img, (302, 230), 20, (255, 255, 255), -1)
    cv2.circle(img, (210, 230), 8, (50, 50, 50), -1)
    cv2.circle(img, (302, 230), 8, (50, 50, 50), -1)
    cv2.line(img, (256, 250), (246, 300), (180, 150, 130), 3)
    cv2.line(img, (246, 300), (266, 300), (180, 150, 130), 3)
    cv2.ellipse(img, (256, 340), (40, 20), 0, 0, 180, (150, 80, 80), 3)
    for i in range(180):
        x = 136 + i
        y = int(150 + 30 * np.sin(i * np.pi / 90))
        cv2.line(img, (x, 106), (x, y), (80, 60, 40), 2)
    return img


def generate_architecture(size=(512, 512)):
    img = np.ones((*size, 3), dtype=np.uint8) * 230
    cv2.rectangle(img, (150, 100), (350, 450), (180, 180, 200), -1)
    for row in range(3):
        for col in range(3):
            x1 = 170 + col * 60
            y1 = 130 + row * 100
            cv2.rectangle(img, (x1, y1), (x1 + 40, y1 + 60), (100, 150, 200), -1)
            cv2.rectangle(img, (x1, y1), (x1 + 40, y1 + 60), (50, 50, 50), 2)
    cv2.rectangle(img, (220, 350), (280, 450), (120, 80, 60), -1)
    pts = np.array([[130, 100], [256, 30], [370, 100]], np.int32)
    cv2.fillPoly(img, [pts], (150, 50, 50))
    return img


def generate_nature(size=(512, 512)):
    img = np.zeros((*size, 3), dtype=np.uint8)
    for i in range(size[0]):
        green = int(80 + 40 * np.sin(i * np.pi / 100))
        img[i, :] = [30, green, 30]
    for _ in range(15):
        cx = np.random.randint(50, size[1] - 50)
        cy = np.random.randint(50, size[0] - 50)
        color = [int(c) for c in np.random.randint(100, 255, 3)]
        cv2.circle(img, (cx, cy), np.random.randint(10, 25), color, -1)
    cv2.rectangle(img, (400, 300), (420, 450), (60, 40, 20), -1)
    cv2.circle(img, (410, 250), 80, (40, 120, 40), -1)
    return img


def generate_urban(size=(512, 512)):
    img = np.ones((*size, 3), dtype=np.uint8) * 180
    cv2.rectangle(img, (200, 0), (312, 512), (80, 80, 80), -1)
    for y in range(0, 512, 60):
        cv2.rectangle(img, (248, y), (264, y + 30), (255, 255, 255), -1)
    for i in range(4):
        h = np.random.randint(150, 350)
        cv2.rectangle(img, (20 + i * 45, 512 - h), (55 + i * 45, 512), (150 + i * 20, 150, 150), -1)
    for i in range(4):
        h = np.random.randint(150, 350)
        cv2.rectangle(img, (332 + i * 45, 512 - h), (367 + i * 45, 512), (150, 150 + i * 20, 150), -1)
    return img


def generate_abstract(size=(512, 512)):
    img = np.ones((*size, 3), dtype=np.uint8) * 255
    np.random.seed(123)
    for _ in range(50):
        x1, y1 = np.random.randint(0, size[1]), np.random.randint(0, size[0])
        x2, y2 = np.random.randint(0, size[1]), np.random.randint(0, size[0])
        color = [int(c) for c in np.random.randint(0, 255, 3)]
        cv2.line(img, (x1, y1), (x2, y2), color, np.random.randint(1, 5))
    for _ in range(20):
        cx, cy = np.random.randint(50, size[1] - 50), np.random.randint(50, size[0] - 50)
        r = np.random.randint(10, 50)
        color = [int(c) for c in np.random.randint(0, 255, 3)]
        cv2.circle(img, (cx, cy), r, color, -1)
    return img


degradation_params = [
    {"blur": 7, "noise": 20, "gamma": 0.6, "downscale": 1.0},
    {"blur": 5, "noise": 15, "gamma": 0.7, "downscale": 1.0},
    {"blur": 9, "noise": 30, "gamma": 0.5, "downscale": 0.5},
    {"blur": 3, "noise": 10, "gamma": 0.8, "downscale": 1.0},
    {"blur": 11, "noise": 25, "gamma": 0.4, "downscale": 0.5},
    {"blur": 5, "noise": 35, "gamma": 0.6, "downscale": 1.0},
    {"blur": 7, "noise": 12, "gamma": 0.7, "downscale": 0.5},
    {"blur": 9, "noise": 18, "gamma": 0.5, "downscale": 1.0},
    {"blur": 3, "noise": 22, "gamma": 0.8, "downscale": 0.5},
    {"blur": 11, "noise": 28, "gamma": 0.4, "downscale": 1.0},
]


def apply_degradation(img, seed=0):
    params = degradation_params[seed % len(degradation_params)]
    degraded = img.copy().astype(np.float32)

    if params["blur"] > 0:
        k = params["blur"]
        if k % 2 == 0:
            k += 1
        degraded = cv2.GaussianBlur(degraded.astype(np.uint8), (k, k), 0).astype(np.float32)

    if params["noise"] > 0:
        noise = np.random.normal(0, params["noise"], degraded.shape)
        degraded = np.clip(degraded + noise, 0, 255)

    if params["gamma"] != 1.0:
        gamma_inv = 1.0 / params["gamma"]
        table = np.array([((i / 255.0) ** gamma_inv) * 255 for i in range(256)]).astype(np.uint8)
        degraded = cv2.LUT(degraded.astype(np.uint8), table).astype(np.float32)

    if params["downscale"] < 1.0:
        h, w = degraded.shape[:2]
        small = cv2.resize(degraded.astype(np.uint8), (int(w * params["downscale"]), int(h * params["downscale"])))
        degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    return np.clip(degraded, 0, 255).astype(np.uint8)


def generate_synthetic_benchmark(num_images=10):
    print("   Generating synthetic benchmark images...")

    np.random.seed(42)

    templates = [
        ("gradient", generate_gradient_image),
        ("checkerboard", generate_checkerboard),
        ("circles", generate_circles_image),
        ("text_scene", generate_text_scene),
        ("landscape", generate_landscape),
        ("portrait", generate_portrait),
        ("architecture", generate_architecture),
        ("nature", generate_nature),
        ("urban", generate_urban),
        ("abstract", generate_abstract),
    ]

    for i, (name, generator) in enumerate(templates[:num_images]):
        original = generator()
        orig_path = os.path.join(ORIGINAL_DIR, "{name}_{i:02d}.png".format(name=name, i=i))
        cv2.imwrite(orig_path, original)

        degraded = apply_degradation(original, seed=i)
        deg_path = os.path.join(DEGRADED_DIR, "{name}_{i:02d}.png".format(name=name, i=i))
        cv2.imwrite(deg_path, degraded)

        p = degradation_params[i]
        print("      {name}_{i:02d}.png — blur={blur}, noise={noise}, gamma={gamma}".format(
            name=name, i=i, blur=p["blur"], noise=p["noise"], gamma=p["gamma"]))

    return num_images


def try_download_real_dataset():
    print("   Attempting to download real benchmark images...")

    test_images = {
        "baboon.png": "https://homepages.cae.wisc.edu/~ece533/images/baboon.png",
        "barbara.png": "https://homepages.cae.wisc.edu/~ece533/images/barbara.png",
        "boat.png": "https://homepages.cae.wisc.edu/~ece533/images/boat.png",
        "goldhill.png": "https://homepages.cae.wisc.edu/~ece533/images/goldhill.png",
        "lena.png": "https://homepages.cae.wisc.edu/~ece533/images/lena.png",
        "peppers.png": "https://homepages.cae.wisc.edu/~ece533/images/peppers.png",
        "tulips.png": "https://homepages.cae.wisc.edu/~ece533/images/tulips.png",
        "airplane.png": "https://homepages.cae.wisc.edu/~ece533/images/airplane.png",
    }

    downloaded = 0
    for filename, url in test_images.items():
        try:
            filepath = os.path.join(ORIGINAL_DIR, filename)
            urllib.request.urlretrieve(url, filepath)
            img = cv2.imread(filepath)
            if img is not None:
                degraded = apply_degradation(img, seed=downloaded)
                cv2.imwrite(os.path.join(DEGRADED_DIR, filename), degraded)
                downloaded += 1
                print("      Downloaded {filename}".format(filename=filename))
            else:
                os.remove(filepath)
        except Exception as e:
            fp = os.path.join(ORIGINAL_DIR, filename)
            if os.path.exists(fp):
                os.remove(fp)
            print("      Failed to download {filename}: {e}".format(filename=filename, e=e))

    return downloaded


def main():
    print("=" * 60)
    print("Benchmark Dataset Setup")
    print("=" * 60)

    for d in [ORIGINAL_DIR, DEGRADED_DIR, RESTORED_DIR]:
        if os.path.exists(d):
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))

    real_count = try_download_real_dataset()

    synthetic_count = 0
    if real_count < 5:
        synthetic_count = generate_synthetic_benchmark(num_images=10)

    total = len(os.listdir(ORIGINAL_DIR))
    print("\nBenchmark dataset ready!")
    print("   Original images: {total}".format(total=total))
    print("   Degraded images: {total_deg}".format(total_deg=len(os.listdir(DEGRADED_DIR))))
    print("   Real downloaded: {real_count}".format(real_count=real_count))
    print("   Synthetic generated: {synthetic_count}".format(synthetic_count=synthetic_count))
    print("\nLocation: {path}".format(path=os.path.abspath(BENCHMARK_DIR)))


if __name__ == "__main__":
    main()
