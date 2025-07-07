import fal
from pydantic import BaseModel, Field
from fal.toolkit import Image, download_file
from pathlib import Path

MODEL_DIR = Path("/data/models/flux-dev-1")
TEMP_DIR = Path("/data/temp")

class TextRequest(BaseModel):
    prompt: str = Field(
        description="The text prompt to be processed. to create a new image.",
        examples=["A beautiful sunset over the mountains"]
    )
    guidance: float = Field(
        default=3.5,
        description="The guidance scale for the image generation. Higher values lead to more adherence to the prompt.",
        examples=[3.5, 7.0]
    )

class ImageRequest(BaseModel):
    image_url: str
    prompt: str = Field(
        description="The text prompt to be processed. to create a new image.",
        examples=["Remove the noise from the image"]
    )
    guidance: float = Field(
        default=3.5,
        description="The guidance scale for the image generation. Higher values lead to more adherence to the prompt.",
        examples=[3.5, 7.0]
    )


class ImageResponse(BaseModel):
    image: Image


class ImageGenApp(fal.App, keep_alive=300, name="fal-image-gen-app"):
    machine_type = "GPU-H100"
    requirements = [
        "hf-transfer==0.1.9",
        "transformers[sentencepiece]==4.51.0",
        "numpy",
        "accelerate==1.6.0",
        "git+https://github.com/dorukbulut/flux-nag-pag.git",
        "huggingface-hub==0.33.2",
        "torchvision==0.22.1",
    ]


    def download_and_save_model(self):
        from huggingface_hub import snapshot_download
        import os

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-dev",
            local_dir=MODEL_DIR,
            token=os.getenv("HF_TOKEN"),
        )
        
    def setup(self):
        import os
        if not MODEL_DIR.exists():            
            self.download_and_save_model()

        os.environ["FLUX_MODEL"] = str(MODEL_DIR / "flux1-dev.safetensors")
        os.environ["FLUX_AE"] = str(MODEL_DIR / "ae.safetensors")
        os.environ['FLUX_T5_TOKENIZER'] = str(MODEL_DIR / "tokenizer_2")
        os.environ['FLUX_T5_MODEL'] = str(MODEL_DIR / "text_encoder_2")
        os.environ['FLUX_CLIP_TOKENIZER'] = str(MODEL_DIR / "tokenizer")
        os.environ['FLUX_CLIP_MODEL'] = str(MODEL_DIR / "text_encoder")
        
        
        from flux.pipeline import FluxInference
        self.inference_engine = FluxInference(model_name="flux-dev", device="cuda")
        
        self.warmup()

    def warmup(self):
        pass

    @fal.endpoint("/flux/dev/text-to-image/")
    def text_to_image(self, request: TextRequest) -> ImageResponse:
        result = self.inference_engine.text_to_image(
            prompt=request.prompt,
            width=1024,
            height=768,
            num_steps=25,
            guidance=request.guidance,
            seed=42
        )
        image = Image.from_pil(result)
        return ImageResponse(image=image)
    

    @fal.endpoint("/flux/dev/image-to-image/")
    def image_to_image(self, request: ImageRequest) -> ImageResponse:
        file_path = download_file(request.image_url, TEMP_DIR)
        req_image = Image.from_path(file_path).to_pil()
        result = self.inference_engine.image_to_image(
            prompt=request.prompt,
            init_image=req_image,
            strength=0.7,
            num_steps=50,
            guidance=request.guidance,
            seed=42
        )
        res_image = Image.from_pil(result)

        # delete the temporary file
        if file_path.exists():
            file_path.unlink()

        return ImageResponse(image=res_image)
