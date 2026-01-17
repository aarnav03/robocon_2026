import os
import cv2
import numpy as np

INPUT_DIR = "img"
OUTPUT_DIR = "imgr"

#bg col to replace
BG_BGR = np.array([232, 231, 228], dtype=np.uint8)

# replace with
REPLACE_BGR = np.array([0, 0, 0], dtype=np.uint8)


# tolerance
TOL = 25


def replace_bg(img):
    img_int = img.astype(np.int16)
    bg_int = BG_BGR.astype(np.int16)

    mask = np.all(np.abs(img_int - bg_int) <= TOL, axis=2)

    out = img.copy()
    out[mask] = REPLACE_BGR

    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img = cv2.imread(os.path.join(INPUT_DIR, fname))
        if img is None:
            continue

        result = replace_bg(img)

        cv2.imwrite(os.path.join(OUTPUT_DIR, fname), result)
        print("processed:", fname)

    print("\ndone")


if __name__ == "__main__":
    main()

