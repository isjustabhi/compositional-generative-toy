import os
import random
from PIL import Image, ImageDraw

# Settings
SHAPES = ['circle', 'square', 'triangle']
COLORS = ['red', 'green', 'blue']
IMG_SIZE = 64
TRAIN_SAMPLES = 300
TEST_SAMPLES = 60

# For generalization test: leave out these combos in training
HELD_OUT_COMBOS = [('triangle', 'green'), ('circle', 'blue')]

# Color map
COLOR_RGB = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255)
}

def draw_shape(draw, shape, color):
    if shape == 'circle':
        draw.ellipse((16, 16, 48, 48), fill=color)
    elif shape == 'square':
        draw.rectangle((16, 16, 48, 48), fill=color)
    elif shape == 'triangle':
        draw.polygon([(32, 10), (10, 54), (54, 54)], fill=color)

def generate_image(shape, color):
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), 'white')
    draw = ImageDraw.Draw(img)
    draw_shape(draw, shape, COLOR_RGB[color])
    return img

def save_dataset(path, num_samples, exclude_combos=[]):
    os.makedirs(path, exist_ok=True)
    meta = []

    while len(meta) < num_samples:
        shape = random.choice(SHAPES)
        color = random.choice(COLORS)
        if (shape, color) in exclude_combos:
            continue
        img = generate_image(shape, color)
        filename = f"{shape}_{color}_{len(meta)}.png"
        img.save(os.path.join(path, filename))
        meta.append((shape, color, filename))

    return meta

# Create folders
os.makedirs('toy_dataset', exist_ok=True)

# Generate training data (excluding held-out combos)
train_meta = save_dataset('toy_dataset/train', TRAIN_SAMPLES, exclude_combos=HELD_OUT_COMBOS)

# Generate test data (including held-out combos)
test_meta = save_dataset('toy_dataset/test', TEST_SAMPLES, exclude_combos=[])  # test can include all

print("Dataset generated.")
print(f"Training samples: {len(train_meta)}")
print(f"Testing samples: {len(test_meta)}")
print("Held out from training:", HELD_OUT_COMBOS)
