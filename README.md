# Conditional GAN Demo (MNIST) - DCGAN-ish + Gradio



This project trains a **conditional GAN** on MNIST so you can generate **specific digits (0-9)** on demand.


## QuickStart



```bash

python -m venv .venv

# Windows: .venv\Scripts\Activate

# macOS/Linus: source .venv/bin/activate

pip install -r requirements.txt

```

## Train



```bash

python -m src.train --config configs/default.yaml

```



This writes:

* `outputs/checkpoints/cgan_mnist_dcgan.pt`
* `outputs/samples/epoch_XXX.png`



## Run the Gradio app



```bash

python app/app.py

```



## What's inside

* DCGAN-ish Generator/Discriminator
* Label conditioning via embeddings
* Checkpointing + sampling grids each epoch
* Gradio UI for generation + latent interpolation



---

