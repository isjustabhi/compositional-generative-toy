# Compositional Generative Modeling (Toy Implementation)

This project demonstrates the key idea from the paper  
**"Compositional Generative Modeling: A Single Model is Not All You Need"**  
by training:

- A **Shape VAE** on grayscale 2D shapes
- A **Color Mapper CNN** to colorize them
- A **Compositional Generator** that mixes shape + color

### Folder Overview:
- `generate_shapes_dataset.py`: creates the toy dataset
- `train_shape_vae.py`: trains the shape generator
- `train_color_mapper.py`: trains the color mapper (with edge-aware loss)
- `compose_generate.py`: composes grayscale + color into new samples
- `view_results.ipynb`: visualizes progress over epochs

### Example Output:
(attach one or two `output_color_epoch*.png` images here)

---
This reproduces the compositional generalization discussed in the paper, showing improved performance over a monolithic VAE.
