import fal
import os
from pathlib import Path
from typing import Union, Optional
from pydantic import BaseModel, Field
from fal.toolkit import Image, download_file


class Config:
    MODEL_DIR = Path("/data/models/flux-dev-1")
    TEMP_DIR = Path("/data/temp")
    MAX_WIDTH = 2048
    MAX_HEIGHT = 2048
    MAX_PROMPT_LENGTH = 1000
    MAX_FILE_SIZE_MB = 50
    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}


class TextRequest(BaseModel):
    prompt: str = Field(max_length=Config.MAX_PROMPT_LENGTH)
    guidance: float = Field(default=2.5, ge=0.0)
    pag_weight: float = Field(default=0.01, ge=0.0)
    tau: float = Field(default=1.2, ge=0.0)
    width: int = Field(default=1024, ge=64, le=Config.MAX_WIDTH)
    height: int = Field(default=512, ge=64, le=Config.MAX_HEIGHT)
    num_steps: int = Field(default=25, ge=0)
    seed: Optional[int] = Field(default=42, ge=0)


class ImageRequest(TextRequest):
    image_url: str = Field(pattern=r"^https?://")
    prompt: str = Field(max_length=Config.MAX_PROMPT_LENGTH)
    strength: float = Field(default=0.7, ge=0.0)
    guidance: float = Field(default=2.5, ge=0.0)
    pag_weight: float = Field(default=0.01, ge=0.0)
    tau: float = Field(default=1.2, ge=0.0)
    width: int = Field(default=1024, ge=64, le=Config.MAX_WIDTH)
    height: int = Field(default=512, ge=64, le=Config.MAX_HEIGHT)
    num_steps: int = Field(default=25, ge=0)
    seed: Optional[int] = Field(default=42, ge=0)


class ImageResponse(BaseModel):
    image: Image
    generation_time: float
    seed_used: int


class ErrorResponse(BaseModel):
    error: str
    error_code: str
    details: Optional[str] = None


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

    def _create_error_response(
        self, error_msg: str, error_code: str, details: str = None
    ) -> ErrorResponse:
        return ErrorResponse(error=error_msg, error_code=error_code, details=details)

    def _validate_image_file(self, file_path: Path) -> Optional[ErrorResponse]:
        if not file_path.exists():
            return self._create_error_response("File not found", "FILE_NOT_FOUND")

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > Config.MAX_FILE_SIZE_MB:
            return self._create_error_response(
                f"File too large: {file_size_mb:.1f}MB (max: {Config.MAX_FILE_SIZE_MB}MB)",
                "FILE_TOO_LARGE",
            )

        if file_path.suffix.lower() not in Config.SUPPORTED_FORMATS:
            return self._create_error_response(
                f"Unsupported format: {file_path.suffix}. Supported: {Config.SUPPORTED_FORMATS}",
                "UNSUPPORTED_FORMAT",
            )

        return None

    def download_and_save_model(self):
        from huggingface_hub import snapshot_download

        try:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

            snapshot_download(
                repo_id="black-forest-labs/FLUX.1-dev",
                local_dir=Config.MODEL_DIR,
                token=os.getenv("HF_TOKEN"),
            )

        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")

    def set_env_vars(self):
        env_vars = {
            "FLUX_MODEL": str(Config.MODEL_DIR / "flux1-dev.safetensors"),
            "FLUX_AE": str(Config.MODEL_DIR / "ae.safetensors"),
            "FLUX_T5_TOKENIZER": str(Config.MODEL_DIR / "tokenizer_2"),
            "FLUX_T5_MODEL": str(Config.MODEL_DIR / "text_encoder_2"),
            "FLUX_CLIP_TOKENIZER": str(Config.MODEL_DIR / "tokenizer"),
            "FLUX_CLIP_MODEL": str(Config.MODEL_DIR / "text_encoder"),
        }

        for key, value in env_vars.items():
            os.environ[key] = value

    def setup(self):
        try:
            Config.TEMP_DIR.mkdir(parents=True, exist_ok=True)

            if not Config.MODEL_DIR.exists():
                self.download_and_save_model()

            self.set_env_vars()

            from flux.pipeline import FluxInference

            self.inference_engine = FluxInference(model_name="flux-dev", device="cuda")

            self.warmup()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize application: {e}")

    def warmup(self):
        try:
            import numpy as np
            from PIL import Image as PILImage

            _ = self.inference_engine.text_to_image(
                prompt="test",
                width=512,
                height=512,
                num_steps=1,
                guidance=1.0,
                pag_weight=0.0,
                tau=1.0,
                seed=42,
            )

            noise_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            noise_image = PILImage.fromarray(noise_array, "RGB")

            _ = self.inference_engine.image_to_image(
                prompt="test",
                init_image=noise_image,
                strength=0.5,
                width=512,
                height=512,
                num_steps=1,
                guidance=1.0,
                pag_weight=0.0,
                tau=1.0,
                seed=42,
            )

        except Exception as e:
            raise RuntimeError(f"Warmup failed: {e}")

    @fal.endpoint("/flux/dev/text-to-image/")
    def text_to_image(
        self, request: TextRequest
    ) -> Union[ImageResponse, ErrorResponse]:
        try:
            import time

            start_time = time.time()

            result = self.inference_engine.text_to_image(
                prompt=request.prompt,
                width=request.width,
                height=request.height,
                num_steps=request.num_steps,
                guidance=request.guidance,
                pag_weight=request.pag_weight,
                tau=request.tau,
                seed=request.seed,
            )

            generation_time = time.time() - start_time

            image = Image.from_pil(result)

            return ImageResponse(
                image=image, generation_time=generation_time, seed_used=request.seed
            )

        except Exception as e:
            return self._create_error_response(
                "Image generation failed", "GENERATION_ERROR", str(e)
            )

    @fal.endpoint("/flux/dev/image-to-image/")
    def image_to_image(
        self, request: ImageRequest
    ) -> Union[ImageResponse, ErrorResponse]:
        file_path = None
        try:
            import time

            start_time = time.time()

            file_path = download_file(request.image_url, Config.TEMP_DIR)

            validation_error = self._validate_image_file(file_path)

            if validation_error:
                return validation_error

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
                height=request.height,
            )

            generation_time = time.time() - start_time
            res_image = Image.from_pil(result)

            return ImageResponse(
                image=res_image, generation_time=generation_time, seed_used=request.seed
            )

        except Exception as e:
            return self._create_error_response(
                "Image generation failed", "GENERATION_ERROR", str(e)
            )

        finally:
            if file_path and file_path.exists():
                file_path.unlink()
