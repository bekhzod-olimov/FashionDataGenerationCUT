import streamlit as st
from PIL import Image
import io
from options.test_options import TestOptions
from models import create_model
from data.base_dataset import get_transform
from PIL import Image
import torch
import io
import util.util as util

class GANModelWrapper:
    def __init__(self):
        # self.opt = TestOptions().parse(save=False)
        self.opt = TestOptions().parse()
        self.opt.num_threads = 0
        self.opt.batch_size = 1
        self.opt.serial_batches = True
        self.opt.no_flip = True
        self.opt.display_id = -1
        self.opt.phase = 'test'
        self.opt.isTrain = False
        self.opt.checkpoints_dir = "model_files"        

        # Initialize model
        self.model = create_model(self.opt)
        self.model.setup(self.opt)
        self.model.eval()

        # Use transform defined in data pipeline
        self.transform = get_transform(self.opt)

    def infer(self, image: Image.Image) -> Image.Image:
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0)  # [1,C,H,W]
        data = {'A': input_tensor, 'A_paths': ''}

        # Run inference
        self.model.set_input(data)
        self.model.test()
        visuals = self.model.get_current_visuals()
        output_tensor = visuals['fake_B'].detach().cpu()

        # Convert to PIL
        output_image = util.tensor2im(output_tensor[0])
        return Image.fromarray(output_image)

# Title
st.set_page_config(page_title="GAN Image Generator", layout="centered")
st.title("🎨 Image-to-Image GAN Demo")

# Load model once
@st.cache_resource
def load_model(): return GANModelWrapper()

model = load_model()

# Upload input image
uploaded_file = st.file_uploader("Upload an input image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Input Image", use_column_width=True)

    if st.button("Generate"):
        with st.spinner("Generating image..."):
            output_image = model.infer(input_image)
            st.image(output_image, caption="Generated Image", use_column_width=True)

        # Download button
        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        st.download_button("Download Result", buf.getvalue(), file_name="generated.png", mime="image/png")