import os
from PIL import Image, ImageDraw
import random

SHAPES = ['circle', 'square', 'triangle']
IMG_SIZE = 64
NUM_IMAGES = 300

os.makedirs("grayscale_shapes/train", exist_ok=True)
os.makedirs("grayscale_shapes/test", exist_ok=True)

def draw_shape_only(shape):
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), "white")  # L = grayscale
    draw = ImageDraw.Draw(img)

    if shape == "circle":
        draw.ellipse((16, 16, 48, 48), fill=0)
    elif shape == "square":
        draw.rectangle((16, 16, 48, 48), fill=0)
    elif shape == "triangle":
        draw.polygon([(32, 10), (10, 54), (54, 54)], fill=0)

    return img

for i in range(NUM_IMAGES):
    shape = random.choice(SHAPES)
    img = draw_shape_only(shape)
    img.save(f"grayscale_shapes/train/{shape}_{i}.png")

# Copy 60 test images
for i in range(60):
    shape = random.choice(SHAPES)
    img = draw_shape_only(shape)
    img.save(f"grayscale_shapes/test/{shape}_{i}.png")

print("✅ Grayscale shape dataset created.")
