# 💜 Violet Vision — Cup vs Bottle

A free-to-deploy Streamlit application using the **same DINOv2 ViT-S/14 embedding model and cosine k-nearest neighbors** from the Assignment 2 experiments.

## Free deployment
1. Create a PUBLIC GitHub repository and upload the seven files in this directory (not the enclosing ZIP).
2. Open https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repo and `main` branch, then select `app.py` as main file.
4. Deploy and wait for DINOv2 model weights to download. Open the generated `*.streamlit.app` URL and test with a new cup or bottle photo.

If the free service reaches its memory cap, deployment may fail. The Colab notebook remains the alternative way to demo the full model with a share link while Colab is active.

## Local use
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Data provenance:** images.cv public cup and water bottle downloads, not self-collected. Only the 240 selected images are in dataset.zip. Prediction scores are k-NN weights, not calibrated confidence probabilities.
