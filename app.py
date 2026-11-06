import json
import time
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Violet Vision | AI Classifier", page_icon="💜", layout="wide")
st.markdown('''<style>
.stApp {background: radial-gradient(circle at top right,#271144 0%,#110b22 55%,#0b0818 100%); color:#efe6ff}
[data-testid="stHeader"]{background:transparent}
h1,h2,h3,p,label {color:#f1e7ff!important}
[data-testid="stFileUploader"]{border:1px dashed #985bfa;border-radius:18px;background:#241539;padding:14px}
.stButton>button{background:linear-gradient(90deg,#7c3aed,#b56dff);color:white;border:none;border-radius:12px;font-weight:700;width:100%;padding:12px}
[data-testid="stMetric"]{background:#25153f;border:1px solid #6d3d9e;border-radius:14px;padding:12px}
.violet-card{background:#25163d;border:1px solid #654092;border-radius:18px;padding:20px;margin-bottom:20px}
</style>''', unsafe_allow_html=True)

st.markdown('<div style="color:#bba0ff;font-weight:700;letter-spacing:2px">COMPUTER VISION · ASSIGNMENT 2</div>', unsafe_allow_html=True)
st.title('💜 Violet Vision')
st.markdown('### Cup vs Bottle — Explainable AI Classifier')
st.markdown('Upload a photograph or use your camera. Our **DINOv2 ViT-S/14** feature extractor and cosine k-NN classifier predict the object and show the three most similar training pictures.')

@st.cache_resource(show_spinner="Loading DINOv2 and nearest-neighbor classifier for the first visit...")
def load_model():
    import torch
    from torchvision import transforms
    from sklearn.neighbors import KNeighborsClassifier
    if not (ROOT / 'dataset').exists():
        with zipfile.ZipFile(ROOT / 'dataset.zip') as z:
            z.extractall(ROOT)
    paths = json.loads((ROOT / 'reference_paths.json').read_text(encoding='utf-8'))
    ref_paths = [ROOT / p for p in paths]
    ref_data = np.load(ROOT / 'reference_features.npz')
    feats, labels = ref_data['features'],ref_data['labels']
    k=int((ROOT/'selected_k.txt').read_text())
    knn=KNeighborsClassifier(n_neighbors=k,metric='cosine',algorithm='brute',weights='distance').fit(feats,labels)
    device=torch.device('cpu')
    torch.set_num_threads(2)
    model=torch.hub.load('facebookresearch/dinov2','dinov2_vits14',trust_repo=True).to(device).eval()
    tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    return torch,model,knn,tf,feats,labels,ref_paths,k

left,right=st.columns([1,1],gap='large')
with left:
    st.markdown('#### Upload your picture')
    upload=st.file_uploader('JPG, PNG or WEBP',type=['jpg','jpeg','png','webp'])
    camera=st.camera_input('Or take a picture with your camera')
    source=camera if camera is not None else upload
    if source:
        photo=Image.open(source).convert('RGB')
        st.image(photo,caption='Your image',use_container_width=True)
    run=st.button('✨ Analyze photo',disabled=(source is None))
with right:
    st.markdown('#### AI prediction')
    if run:
        try:
            start=time.perf_counter()
            torch,model,knn,tf,feats,labels,ref_paths,k=load_model()
            with torch.inference_mode():
                x=tf(photo).unsqueeze(0)
                emb=model(x)
                if isinstance(emb,dict): emb=emb.get('x_norm_clstoken',next(iter(emb.values())))
                emb=torch.nn.functional.normalize(emb.float(),dim=1).numpy()
            prob=knn.predict_proba(emb)[0]
            scores={['BOTTLE','CUP'][int(label)]:float(val) for label,val in zip(knn.classes_,prob)}
            winner=max(scores,key=scores.get)
            elapsed=(time.perf_counter()-start)*1000
            st.success(f'Prediction: {winner}')
            for name in ['CUP','BOTTLE']:
                st.write(f'{name}: **{scores.get(name,0):.1%}**')
                st.progress(scores.get(name,0))
            st.metric('End-to-end inference',f'{elapsed:.0f} ms')
            st.caption(f'CPU · k={k} · Scores are k-NN voting weights, not calibrated probabilities.')
            st.markdown('#### Most similar reference images')
            sim=feats@emb[0]
            neighbors=np.argsort(-sim)[:3]
            cols=st.columns(3)
            for col,idx in zip(cols,neighbors):
                with col:
                    st.image(str(ref_paths[int(idx)]),use_container_width=True)
                    st.caption(f'{["Bottle","Cup"][int(labels[idx])]} · similarity {sim[idx]:.2f}')
        except Exception as ex:
            st.error(f'Could not load the model or analyze the photo: {ex}')
            st.info('If the free host exceeds memory limits, run the original Colab Gradio app instead.')
    else:
        st.info('Upload an image and press Analyze photo to get a prediction.')
st.divider()
st.caption('Astana IT University · Computer Vision Assignment 2 · Source images: images.cv (public dataset). No synthetic media. This app classifies whole images; it does not detect bounding boxes. Upload only photographs you have permission to use.')
