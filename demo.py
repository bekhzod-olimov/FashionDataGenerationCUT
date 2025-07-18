import io, os, streamlit as st
from PIL import Image
from models import create_model
from util.util import list_sample_images
from data.base_dataset import get_transform
from options.test_options import TestOptions

class GANModelWrapper:
    def __init__(self):        
        self.opt = TestOptions().parse()
        self.opt.num_threads = 0
        self.opt.batch_size = 1
        self.opt.serial_batches = True
        self.opt.no_flip = True
        self.opt.display_id = -1
        self.opt.phase = 'test'
        self.opt.isTrain = False        

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

# --- Sample image selection ---
image_A_samples, image_B_samples = list_sample_images('imgs')

st.sidebar.header("Or select a sample image:")
placeholder = "Click here to choose from the list"
sample_A = st.sidebar.selectbox("Sample Input Image (A)", [placeholder] + image_A_samples)
sample_B = st.sidebar.selectbox("Sample Style Image (B)", [placeholder] + image_B_samples)

# --- File uploaders ---
uploaded_file_A = st.file_uploader("Upload an input image (A)", type=["png", "jpg", "jpeg"])
uploaded_file_B = st.file_uploader("Upload a style image (B)", type=["png", "jpg", "jpeg"])

# --- Image selection logic ---
input_image_A, input_image_B = None, None

if sample_A != placeholder:
    input_image_A = Image.open(os.path.join('imgs', sample_A)).convert("RGB")
elif uploaded_file_A:
    input_image_A = Image.open(uploaded_file_A).convert("RGB")

if sample_B != placeholder:
    input_image_B = Image.open(os.path.join('imgs', sample_B)).convert("RGB")
elif uploaded_file_B:
    input_image_B = Image.open(uploaded_file_B).convert("RGB")

# --- Display selected images ---
if input_image_A:
    st.image(input_image_A, caption="Input Image (A)", use_container_width=True)
if input_image_B:
    st.image(input_image_B, caption="Style Image (B)", use_container_width=True)

# --- Generate and download ---
if input_image_A and input_image_B:
    if st.button("Generate"):
        with st.spinner("Generating image..."):
            output_image = model.infer(input_image_A, input_image_B)
            st.image(output_image, caption="Generated Image", use_container_width=True)

        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        st.download_button("Download Result", buf.getvalue(), file_name="generated.png", mime="image/png")