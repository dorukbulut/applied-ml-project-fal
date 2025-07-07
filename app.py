import fal
from pydantic import BaseModel
from typing import Union
from fal.toolkit import Image, download_file
from pathlib import Path

MODEL_DIR = Path("/data/models/flux-dev-1")
TEMP_DIR = Path("/data/temp")

class TextRequest(BaseModel):
    prompt: str
    guidance: float | None = 2.5
    pag_weight: float | None = 0.01
    tau: float | None = 1.2
    width: int | None = 1024
    height: int | None = 512
    num_steps: int | None = 25
    seed: int | None = 42



class ImageRequest(BaseModel):
    image_url: str
    prompt: str
    guidance: float | None = 2.5
    pag_weight: float | None = 0.01
    tau: float | None = 1.2
    width: int | None = 1024
    height: int | None = 512
    num_steps: int | None = 25
    seed: int | None = 42
    strength: float | None = 0.7


class ImageResponse(BaseModel):
    image: Image

class ErrorResponse(BaseModel):
    error: str


class ImageGenApp(fal.App, keep_alive=300, name="fal-image-gen-app"):
    machine_type = "GPU-H100"
    requirements = [
        "hf-transfer==0.1.9",
        "transformers[sentencepiece]==4.51.0",
        "accelerate==1.6.0",
        "git+https://github.com/dorukbulut/flux-nag-pag.git",
        "huggingface-hub==0.33.2",
        "numpy==2.3.1",
        "torchvision==0.22.1",
    ]

    def _validate_common_fields(self, request) -> Union[None, ErrorResponse]:
        validations = [
            (request.width <= 0 or request.height <= 0, "Width and height must be positive integers."),
            (request.num_steps <= 0, "Number of steps must be a positive integer."),
            (request.guidance < 0, "Guidance must be a non-negative number."),
            (request.pag_weight < 0, "PAG weight must be a non-negative number."),
            (request.tau <= 0, "Tau must be a positive number.")
        ]
        
        for condition, error_msg in validations:
            if condition:
                return ErrorResponse(error=error_msg)
        
        return None

    def validate_text_to_image_request(self, request: TextRequest) -> Union[None, ErrorResponse]:
        return self._validate_common_fields(request)

    def validate_image_to_image_request(self, request: ImageRequest) -> Union[None, ErrorResponse]:
        if request.strength < 0 or request.strength > 1:
            return ErrorResponse(error="Strength must be between 0 and 1.")
        return self._validate_common_fields(request)


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

    def set_env_vars(self):
        import os
        os.environ["FLUX_MODEL"] = str(MODEL_DIR / "flux1-dev.safetensors")
        os.environ["FLUX_AE"] = str(MODEL_DIR / "ae.safetensors")
        os.environ['FLUX_T5_TOKENIZER'] = str(MODEL_DIR / "tokenizer_2")
        os.environ['FLUX_T5_MODEL'] = str(MODEL_DIR / "text_encoder_2")
        os.environ['FLUX_CLIP_TOKENIZER'] = str(MODEL_DIR / "tokenizer")
        os.environ['FLUX_CLIP_MODEL'] = str(MODEL_DIR / "text_encoder")
        
    def setup(self):
        from flux.pipeline import FluxInference
        
        if not MODEL_DIR.exists():
            try:
                self.download_and_save_model()
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")
        
        self.set_env_vars()

        try:
            self.inference_engine = FluxInference(model_name="flux-dev", device="cuda")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize inference engine: {e}")
        
        self.warmup()

    def warmup(self):
        pass

    @fal.endpoint("/flux/dev/text-to-image/")
    def text_to_image(self, request: TextRequest) -> Union[ImageResponse, ErrorResponse]:
        
        err = self.validate_text_to_image_request(request)
        if err:
            return err
        
        result = self.inference_engine.text_to_image(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            num_steps=request.num_steps,
            guidance=request.guidance,
            pag_weight=request.pag_weight,
            tau=request.tau,
            seed=request.seed
        )
        image = Image.from_pil(result)
        return ImageResponse(image=image)
    

    @fal.endpoint("/flux/dev/image-to-image/")
    def image_to_image(self, request: ImageRequest) -> Union[ImageResponse, ErrorResponse]:
        err = self.validate_image_to_image_request(request)
        if err:
            return err
        try:
            file_path = download_file(request.image_url, TEMP_DIR)
        except Exception as e:
            return ErrorResponse(error=f"Failed to download image: {e}. Please check the URL.")
        
        req_image = Image.from_path(file_path).to_pil()
        result = self.inference_engine.image_to_image(
            prompt=request.prompt,
            init_image=req_image,
            strength=request.strength,
            num_steps=request.num_steps,
            guidance=request.guidance,
            seed=request.seed,
            pag_weight=request.pag_weight,
            tau=request.tau,
            width=request.width,
            height=request.height
        )
        res_image = Image.from_pil(result)

        # delete the temporary file
        if file_path.exists():
            file_path.unlink()

        return ImageResponse(image=res_image)
