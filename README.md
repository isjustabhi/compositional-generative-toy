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
<img width="266" height="134" alt="image" src="https://github.com/user-attachments/assets/bf61c9c8-38d0-4776-b312-6376787d5d8b" />
<img width="266" height="134" alt="image" src="https://github.com/user-attachments/assets/5f53aeea-42b4-4705-84bd-ea0e717b2c26" />

---
This reproduces the compositional generalization discussed in the paper, showing improved performance over a monolithic VAE.
