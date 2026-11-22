# ArkMage
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)

A simple Python package for hassle-free image processing. (Note: The internal function names and code are written in French).

## Installation

1.  Clone the repository :

    `git clone https://github.com/Arkstone27/ArkMage.git`
    `cd ArkMage`

2.  Load the repo :

    `pip install -e .`

## Quick Start
```
from ArkMage.image import ouvrir_image, Pixels
from ArkMage.commun import flou_classique

# Ask for an image
pixels = ouvrir_image()

# Apply a box blur on the image
pixels_flou = flou_classique(pixels)

# Show the image & ask to save it
pixels_flou.show()
pixels.sauvegarde()
```

## Structure
```
ArkMage/
│
├── image.py         # Main Pixels class to store image (open, show, save, ...)
├── stats_data.py    # Tools in Pixels to store statistics effectively (variance, histogram, ...)
├── scaling.py       # Scaling (winsorization, z-score, normalization, ...)
├── couleurs.py      # Colors (saturation, luminosity, HSV, ...)
│
├── rotation.py      # Rotation (90°, 180°, 240°, any°, ...)
├── bordure.py       # Edge (own algorithm, sobel operator, ...)
├── resize.py        # Resizing (simple, mean, bilinear, ...)
├── quantization.py  # Color Quantization (own algorithm based on mean shift clustering)
├── commun.py        # Others (grayscale, blurs, ...)
│
└── __init__.py      
```

## Information
This package supports jxl image format and uses `numba.njit` and `numba.prange` to accelerate image processing. I'm currently working on implementing CUDA (using `numba`) to speed up some of the slower process.

## Documentation

_Coming..._




