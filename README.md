## Quantized SAM

This repository contains code that allows you to run the SAM backbone in 8bit and 4bit precision. To get started, download the SAM backbone model here (not the full SAM model, just the backbone), following the installation instructions and run:

```
python bnb_examples.py --b8
```

### Installation

To install, first install the [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) library by following their installation instructions. Then run `pip install -r requirements.txt`. Make sure you install the `transformers` and `accelerate` library from github.

### Numbers

In the table below you can see the latency and memory allocation statistics from nvidia-smi for different quantized types. All numbers below are from running on an RTX A5000.

| QType | Latency (ms) | Max Memory Allocation (MB) |
| --- | --- | --- |
| 32 bit | 561 | 5721 |
| 16 bit | 200 | 5345 |
| 8 bit | 294 | 4776 |
| 4 bit | 255 | 4484 |

Running the Automatic Mask Generator code on an image for 32 bit, 8 bit and 4 bit images gives very similar looking segmentations:

![](assets/images/image_32bit.png)
![](assets/images/image_8bit.png)
![](assets/images/image_4bit.png)
