# GFPGAN-ncnn-vulkan 🚀

[![Windows Dev Build](https://github.com/onuralpszr/GFPGAN-ncnn-vulkan/actions/workflows/build_dev_windows.yml/badge.svg)](https://github.com/onuralpszr/GFPGAN-ncnn-vulkan/actions/workflows/build_dev_windows.yml)
[![Ubuntu Dev Build](https://github.com/onuralpszr/GFPGAN-ncnn-vulkan/actions/workflows/build_dev_ubuntu.yml/badge.svg)](https://github.com/onuralpszr/GFPGAN-ncnn-vulkan/actions/workflows/build_dev_ubuntu.yml)
[![Fedora Dev Build (RPM Vulkan)](https://github.com/onuralpszr/GFPGAN-ncnn-vulkan/actions/workflows/build_dev_fedora_rpm_vulkan.yml/badge.svg)](https://github.com/onuralpszr/GFPGAN-ncnn-vulkan/actions/workflows/build_dev_fedora_rpm_vulkan.yml)
[![Fedora Dev Build (Lunar Vulkan SDK)](https://github.com/onuralpszr/GFPGAN-ncnn-vulkan/actions/workflows/build_dev_fedora_lunar_vulkan.yml/badge.svg)](https://github.com/onuralpszr/GFPGAN-ncnn-vulkan/actions/workflows/build_dev_fedora_lunar_vulkan.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/onuralpszr/GFPGAN-ncnn-vulkan/main.svg)](https://results.pre-commit.ci/latest/github/onuralpszr/GFPGAN-ncnn-vulkan/main)
![GitHub](https://img.shields.io/github/license/onuralpszr/GFPGAN-ncnn-vulkan?color=red)
[![Open issue](https://img.shields.io/github/issues/onuralpszr/GFPGAN-ncnn-vulkan)](https://github.com/onuralpszr/GFPGAN-ncnn-vulkan/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/onuralpszr/GFPGAN-ncnn-vulkan)](https://github.com/onuralpszr/GFPGAN-ncnn-vulkan/issues)
![GitHub pull requests](https://img.shields.io/github/issues-pr-raw/onuralpszr/GFPGAN-ncnn-vulkan)
![cpp](https://img.shields.io/badge/C++20-Project-blue.svg?style=flat&logo=c%2B%2B)
[![Github All Releases](https://img.shields.io/github/downloads/onuralpszr/GFPGAN-ncnn-vulkan/total.svg)](https://github.com/onuralpszr/GFPGAN-ncnn-vulkan/releases)

Ncnn with Vulkan implementation of **GFPGAN aims at developing Practical Algorithms for Real-world Face Restoration**

This repository contains the code and pre-trained models for a real-world face restoration algorithm based on the [GFPGAN](https://github.com/TencentARC/GFPGAN) method and optimized for mobile devices using the [NCNN](https://github.com/Tencent/ncnn) framework with a Vulkan backend.

The goal of this project is to develop practical algorithms that can restore the appearance of damaged or low-quality face images, such as those obtained from security cameras, old photographs, or social media profiles. The proposed approach combines the power of deep learning with the speed and efficiency of hardware acceleration, making it suitable for real-time applications on smartphones, drones, or robots.

## Clone Project and Get Submodules

Make sure submodules are initialized and updated

```console
git clone https://github.com/onuralpszr/GFPGAN-ncnn-vulkan.git
git submodule update --init --recursive
```

## Clone project with Submodules

```sh
git clone --recursive https://github.com/onuralpszr/GFPGAN-ncnn-vulkan.git
```

## Project Prerequisites ⚙️

- CMake version 3.20 or later
- C++17 or above with filesystem support
- Clang-Tidy for code analysis (optional)
- Threads library
- Vulkan SDK
- glslangValidator executable
- OpenCV library
- OpenMP library
- ncnn library
- libwebp library

## Building 🛠️

Configure and build

```sh
mkdir -p build && cd build
cmake ..
cmake --build . --parallel $(($(nproc) - 1))
```

## Usage 🖥️

```console
Usage: GFPGAN-ncnn-vulkan -i infile -o outfile [options]...
  -h                   show this help
  -i input-path        input image path (jpg/png/webp) or directory
  -o output-path       output image path (jpg/png/webp) or directory
  -m model-path        folder path to the pre-trained models (default=./gfpgan-models)
  -f output format       output image format (jpg/png/webp, default=ext/png)
*Unmodifiable Options*
 -s scale               upscale ratio (default=2)
 -t tile-size           tile size (default = 400)
 -n model name     GFPGANCleanv1-NoCE-C2 supports only one type of model
```

### Default output paths (when `-o` is omitted)

| Input | Output |
| --- | --- |
| Single file, e.g. `0001.jpg` | Same folder, named `0001-output.jpg` (extension follows `-f`, or the input's own extension, or `png` as a last resort) |
| Folder, e.g. `./image` (mixed jpg/png/webp files) | A new subfolder `./image/image-output/` is created; every image directly inside `./image` is restored and saved there under its **original filename** (extension changed only if `-f` is given) |

Folder mode is not recursive — only files directly inside the given folder are processed, and the newly created `<foldername>-output` subfolder is automatically skipped.

Examples:

```console
GFPGAN-ncnn-vulkan /?
GFPGAN-ncnn-vulkan -i 0001.jpg
GFPGAN-ncnn-vulkan -i 0001.jpg -o restored.png
GFPGAN-ncnn-vulkan -i 0001.jpg -f webp
GFPGAN-ncnn-vulkan -i ./image
GFPGAN-ncnn-vulkan -i ./image -f jpg
GFPGAN-ncnn-vulkan -i ./image -o ./restored -m D:\models\gfpgan-models
```

`-f` forces the output image format regardless of the extension in `-o` or the input file (e.g. `-i ./image -f jpg` saves every restored image as `.jpg` inside `./image/image-output`).

The model files (`encoder.param`, `encoder.bin`, `style.bin`, `yolov5-blazeface.param`, `yolov5-blazeface.bin`, `real_esrgan.param`, `real_esrgan.bin`) must be placed in a folder named **`gfpgan-models`** next to the executable, or pointed to explicitly with `-m`.

The *Unmodifiable Options* section is informational only — `-s`, `-t`, and `-n` are not accepted as command-line flags. Scale (2x) and tile size (400) are fixed in the current implementation, and only the bundled GFPGANCleanv1-NoCE-C2 model is supported.

## :construction: Model support :construction:

1. GFPGANCleanv1-NoCE-C2

### TODO: :bookmark_tabs:

- [x] Support ncnn-vulkan
- [ ] Convert pth->onnx->ncnn
- [ ] Model with colorization

### References

1. <https://github.com/xinntao/Real-ESRGAN>
2. <https://github.com/TencentARC/GFPGAN>
3. <https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan>
4. <https://github.com/Tencent/ncnn>
5. <https://github.com/Tencent/ncnn/tree/master/tools/pnnx>
6. <https://github.com/pnnx/pnnx>
7. <https://github.com/deepcam-cn/yolov5-face>
8. <https://github.com/derronqi/yolov7-face>
9. <https://github.com/derronqi/yolov8-face>
10. <https://github.com/FeiGeChuanShu/GFPGAN-ncnn>
11. <https://github.com/ultralytics/ultralytics>

## Download Model files (GFPGAN-ncnn model files)

### Models-v0.0.1

<https://github.com/onuralpszr/GFPGAN-ncnn-vulkan/releases/download/v0.0.1-models/GFPGAN-ncnn-models.zip>
