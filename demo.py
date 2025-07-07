import streamlit as st
from PIL import Image
import io
from options.test_options import TestOptions
from models import create_model
from data.base_dataset import get_transform
from PIL import Image
import io

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
        # self.opt.checkpoints_dir = "model_files"

        # Initialize model
        self.model = create_model(self.opt)
        self.model.setup(self.opt)
        self.model.eval()

        # Use transform defined in data pipeline
        self.transform = get_transform(self.opt)

    def infer(self, image_A: Image.Image, image_B: Image.Image) -> Image.Image:
        # Preprocess both images
        input_tensor_A = self.transform(image_A).unsqueeze(0)  # [1,C,H,W]
        input_tensor_B = self.transform(image_B).unsqueeze(0)  # [1,C,H,W]        
        data = {
            'A': input_tensor_A,
            'B': input_tensor_B,
            'A_paths': '',
            'B_paths': ''
        }

        # Run inference
        self.model.set_input(data)
        self.model.test()
        visuals = self.model.get_current_visuals()
        output_tensor = visuals['fake_B'].detach().cpu()
        # Convert to PIL
        # print(f"output_tensor.shape -> {output_tensor.shape}")        
        output_image = (output_tensor[0] * 255).detach().cpu().permute(1,2,0).numpy().astype("uint8")
        return Image.fromarray(output_image)


st.set_page_config(page_title="GAN Image Generator", layout="centered")
st.title("🎨 Image-to-Image GAN Demo")

@st.cache_resource
def load_model(): return GANModelWrapper()

model = load_model()

uploaded_file_A = st.file_uploader("Upload an input image (A)", type=["png", "jpg", "jpeg"])
uploaded_file_B = st.file_uploader("Upload a style image (B)", type=["png", "jpg", "jpeg"])

if uploaded_file_A and uploaded_file_B:
    input_image_A = Image.open(uploaded_file_A).convert("RGB")
    input_image_B = Image.open(uploaded_file_B).convert("RGB")
    st.image(input_image_A, caption="Input Image (A)", use_column_width=True)
    st.image(input_image_B, caption="Style Image (B)", use_column_width=True)

    if st.button("Generate"):
        with st.spinner("Generating image..."):
            output_image = model.infer(input_image_A, input_image_B)
            st.image(output_image, caption="Generated Image", use_column_width=True)

        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        st.download_button("Download Result", buf.getvalue(), file_name="generated.png", mime="image/png")