import argparse
import torch
import numpy as np

from PIL import Image
from sam.build_sam import sam_model_registry
from load_model import load_quantized_model, warmup_model
from transformers import BitsAndBytesConfig
from sam.transforms import ResizeLongestSide

parser = argparse.ArgumentParser()
parser.add_argument("--b16", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--b8", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--b4", default=False, action=argparse.BooleanOptionalAction)


def run_16bit():
    torch.cuda.reset_peak_memory_stats()
    transform = ResizeLongestSide(1024)
    model, preprocess = sam_model_registry["vit_h"]()
    model.to("cuda")
    warmup_model(model, (1, 3, 1024, 1024))

    image = np.asarray(Image.open("images/dog.jpg"))
    image = transform.apply_image(image)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = preprocess(image)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        features = model(image.half().to(torch.device("cuda", 0)))
        end.record()
        torch.cuda.synchronize()
    print(f"Time taken: {start.elapsed_time(end)} ms")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB")


def run_8bit():
    torch.cuda.reset_peak_memory_stats()
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    transform = ResizeLongestSide(1024)
    model, preprocess = load_quantized_model("vit_h", "vit_h.pth", quantization_config)

    image = np.asarray(Image.open("images/dog.jpg"))
    image = transform.apply_image(image)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = preprocess(image)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        features = model(image.half().to(torch.device("cuda", 0)))
        end.record()
        torch.cuda.synchronize()
    print(f"Time taken: {start.elapsed_time(end)} ms")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB")


def run_4bit():
    torch.cuda.reset_peak_memory_stats()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    transform = ResizeLongestSide(1024)
    model, preprocess = load_quantized_model("vit_h", "vit_h.pth", quantization_config)

    image = np.asarray(Image.open("images/dog.jpg"))
    image = transform.apply_image(image)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = preprocess(image)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        features = model(image.half().to(torch.device("cuda", 0)))
        end.record()
        torch.cuda.synchronize()
    print(f"Time taken: {start.elapsed_time(end)} ms")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.b16:
        run_16bit()
    elif args.b8:
        run_8bit()
    elif args.b4:
        run_4bit()
    else:
        raise ValueError("Please specify the quantization type")
