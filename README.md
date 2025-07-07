# ImageGenApp

A serverless image generation application powered by FLUX.1-dev with enhanced PAG (Perturbed Attention Guidance) and NAG (Negative Attention Guidance) hybrid attention mechanisms.

## 🚀 Features

- **Text-to-Image Generation**
- **Image-to-Image Transformation**
- **Enhanced FLUX Model**
- **Serverless Architecture**

## 📋 Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager
- [direnv](https://direnv.net/) for environment management
- FAL account and CLI

## 🛠️ Setup

### 1. Environment Configuration

Set up your environment using direnv:

```bash
# Copy the example environment file
cp .envrc.example .envrc

# Edit .envrc with your configuration
# Then allow direnv to load the environment
direnv allow
```

### 2. Install Dependencies

```bash
# Install dependencies using uv
uv sync
```

### 3. Authentication

Authenticate with the FAL platform:

```bash
fal auth login
```

## 🚀 Usage

### Local Development

Run the serverless app locally for testing:

```bash
fal run app.py::ImageGenApp
```

### Deployment

Deploy the application to FAL's serverless infrastructure:

```bash
fal deploy --auth shared
```

## 🔐 Secrets Management

Configure required secrets using FAL's secret management:

```bash
# Set HuggingFace token for model access
fal secrets set HF_TOKEN your_huggingface_token_here

# Add any additional secrets as needed
fal secrets set SECRET_NAME secret_value
```

### Required Secrets

- `HF_TOKEN`: HuggingFace access token for downloading FLUX.1-dev model weights

## 📚 API Endpoints

### Text-to-Image
```
POST /flux/dev/text-to-image/
```

### Image-to-Image
```
POST /flux/dev/image-to-image/
```

## 🔗 Related Links
- [FAL Platform Documentation](https://docs.fal.ai/)
- [FLUX Model Hub](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [Modified FLUX Implementation](https://github.com/dorukbulut/flux-nag-pag)