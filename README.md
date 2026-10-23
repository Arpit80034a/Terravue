# TerraVue
### From Monocular Vision to Geospatial 3D Understanding

**TeraVue** is a computer vision and remote-sensing project focused on transforming aerial and satellite RGB imagery into depth-informed elevation maps and 3D representations of the Earth's surface. It aims to bridge the gap between relative monocular depth estimation and metrically meaningful terrain and building-height information.

By combining deep learning, semantic understanding, geospatial processing, and 3D reconstruction, TeraVue aims to support applications in urban planning, disaster management, infrastructure monitoring, and environmental analysis.

> **Project status:** Research and development. Model integration, metric calibration, and validation depend on the selected datasets and implementation stage.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Workflow](#project-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Input and Output](#input-and-output)
- [Applications](#applications)
- [Challenges and Limitations](#challenges-and-limitations)
- [Roadmap](#roadmap)
- [Research Foundation](#research-foundation)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Traditional elevation mapping often relies on LiDAR, stereo imagery, photogrammetry, or existing Digital Surface Models (DSMs). Although these techniques can provide accurate geometric information, they may require specialized sensors, multiple images, expensive data acquisition, or substantial processing resources.

TeraVue explores an alternative approach using single-view RGB imagery and monocular depth estimation models to infer scene geometry.

The proposed pipeline combines:

1. **Monocular depth estimation** to infer relative scene depth from RGB imagery.
2. **Metric calibration** to estimate physically meaningful height or elevation values using suitable reference information.
3. **Semantic understanding** to distinguish buildings, vegetation, roads, and other surface objects.
4. **Geospatial processing** to preserve coordinate reference systems, spatial resolution, and raster alignment.
5. **3D reconstruction and visualization** to produce interpretable terrain and surface representations.

The long-term goal is to create a modular pipeline that accepts suitable aerial or satellite imagery and produces geospatially useful, depth-informed elevation products.

## Problem Statement

A conventional RGB image contains appearance information but does not directly provide the absolute height of objects above the ground.

Monocular depth estimation models can infer relative depth, but their predictions generally do not provide reliable absolute distances or elevations without additional calibration.

This creates several challenges:

- Converting relative depth predictions into metric height estimates.
- Distinguishing object height from camera-to-surface distance.
- Handling variations in camera altitude, orientation, and image resolution.
- Separating buildings and vegetation from terrain.
- Maintaining geospatial accuracy when processing GeoTIFF imagery.
- Evaluating the generated elevation products against trusted reference data.

TeraVue investigates how depth estimation, semantic information, calibration, and geospatial processing can work together to address these challenges.

## Objectives

- Develop an image-to-elevation processing pipeline for aerial and satellite imagery.
- Investigate Depth Anything V2 and other suitable monocular depth estimation models.
- Explore relative-depth-to-absolute-height conversion using reference data and calibration methods.
- Integrate semantic segmentation to improve the interpretation of buildings, vegetation, roads, and terrain.
- Support georeferenced imagery through appropriate geospatial processing tools.
- Generate depth maps, height rasters, and 3D surface visualizations.
- Evaluate output quality using suitable ground-truth elevation data and quantitative metrics.

## Key Features

| Feature | Description |
|---|---|
| RGB image processing | Load and preprocess aerial or satellite images. |
| Monocular depth estimation | Generate relative depth predictions using deep learning. |
| Metric calibration | Investigate methods for estimating physically meaningful heights. |
| Semantic segmentation | Identify surface classes relevant to geometric interpretation. |
| Geospatial raster handling | Preserve CRS, resolution, bounds, and spatial alignment where available. |
| Elevation map generation | Produce height or surface-elevation rasters when sufficient calibration information exists. |
| 3D reconstruction | Convert suitable raster outputs into visualizable 3D surfaces. |
| Evaluation | Compare predictions with reference elevation data where available. |

*Feature availability depends on the current implementation. The table describes the project's target capabilities, not a claim that every module is complete.*

## System Architecture

The proposed architecture follows a modular processing pipeline.

```text
             INPUT IMAGERY
          RGB / GeoTIFF Images
                   |
                   v
          IMAGE PREPROCESSING
       Resize, Normalize, Validate
                   |
                   v
        MONOCULAR DEPTH ESTIMATION
             Depth Anything V2
                   |
                   v
          RELATIVE DEPTH MAP
                   |
          +--------+---------+
          |                  |
          v                  v
   METRIC CALIBRATION   SEMANTIC SEGMENTATION
   Reference Data      Buildings / Vegetation
   DEM / GCPs          Roads / Terrain
          |                  |
          +--------+---------+
                   |
                   v
         HEIGHT ESTIMATION
      Calibration and Fusion
                   |
                   v
         GEOSPATIAL PROCESSING
       Raster Alignment / CRS
                   |
                   v
        ELEVATION / HEIGHT MAP
                   |
                   v
          3D RECONSTRUCTION
                   |
                   v
       VISUALIZATION AND EXPORT
```

### Core Modules

**1. Image Processing**

Loads images, validates input dimensions, prepares model inputs, and reads geospatial metadata when available.

**2. Depth Estimation**

Uses a monocular depth model to infer relative scene geometry from RGB imagery.

**3. Metric Calibration**

Investigates how relative depth can be mapped to metric quantities using reference elevation data, known control points, sensor metadata, or other justified constraints.

**4. Semantic Fusion**

Uses semantic information to help interpret depth discontinuities and distinguish different surface classes.

**5. Geospatial Processing**

Handles raster alignment, coordinate reference systems, pixel resolution, and georeferenced output generation.

**6. 3D Visualization**

Converts appropriate raster products into 3D surfaces for inspection, analysis, and visualization.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| Monocular depth | Depth Anything V2 |
| Semantic segmentation | SegFormer / CMX (under evaluation) |
| Height estimation research | RDAH-Net and related calibration approaches |
| Image processing | OpenCV, Pillow, NumPy |
| Geospatial processing | Rasterio, GDAL, PyProj |
| Data analysis | pandas, Matplotlib |
| 3D processing | Open3D |
| Visualization | Matplotlib and potential web-based 3D viewers |
| Development | Jupyter Notebook, VS Code, Git |

The final stack may change as the research and implementation evolve. Specific model integrations should be documented once their code, checkpoints, and dependencies are available.

## Project Workflow

### Step 1: Input acquisition

Provide an RGB image or a georeferenced GeoTIFF. For geospatial processing, retain the original raster metadata.

### Step 2: Depth prediction

Run the selected monocular depth model to generate a relative depth map.

### Step 3: Metric calibration

Use suitable reference information to estimate the relationship between model predictions and physical measurements. A low-resolution DEM, surveyed ground-control points, or other reliable constraints may be useful, depending on the method.

### Step 4: Semantic interpretation

Predict relevant classes such as buildings, vegetation, roads, and bare ground. Use these predictions to support height estimation where the chosen fusion method permits.

### Step 5: Elevation product generation

Generate the appropriate output raster. Depending on the available information, this could be a relative depth map, estimated object-height map, or georeferenced surface-elevation product.

### Step 6: Validation

Compare outputs against trusted reference data where available. Evaluate geometric accuracy, spatial alignment, and performance across different scene types.

### Step 7: 3D visualization

Visualize the resulting surface using suitable 3D tools.

## Installation

### Prerequisites

- Python 3.10 or another Python version supported by the selected model dependencies.
- Git.
- A compatible PyTorch environment.
- A CUDA-compatible GPU for accelerated inference, if available.
- Sufficient memory and storage for the selected model checkpoints and datasets.

Exact version requirements should be pinned after the implementation environment is finalized.

### 1. Clone the repository

Replace the placeholder URL with the actual repository URL.

```bash
git clone <YOUR_REPOSITORY_URL>
cd Teravue
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Once the project dependency file is available:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

A compatible PyTorch installation should be configured according to the official installation instructions: https://pytorch.org/get-started/locally/

Geospatial libraries such as GDAL may require platform-specific installation steps.

### 4. Configure model checkpoints

Download the required pretrained weights from their official sources and place them in the configured model directory. Follow the applicable model licenses and checkpoint usage requirements.

Do not commit large model checkpoints, private datasets, or credentials to Git.

## Usage

The following illustrates the intended workflow. Actual commands depend on the implemented entry point and configuration.

### Generate a depth map

```bash
python infer_depth.py \
    --input data/input/image.jpg \
    --output outputs/depth_map.png
```

### Generate a calibrated height map

```bash
python estimate_height.py \
    --depth outputs/depth_map.png \
    --reference data/reference_dem.tif \
    --output outputs/height_map.tif
```

These commands are **illustrative interfaces**, not guaranteed executable commands. They should be updated to match the actual scripts and calibration method in the repository.

For an initial notebook-based implementation, the same workflow can be organized into separate notebooks for preprocessing, depth inference, calibration, evaluation, and visualization.

## Input and Output

### Supported input types

- RGB images in common formats such as JPEG and PNG.
- Georeferenced GeoTIFF imagery.
- Reference DEMs or DSMs, when supported by the calibration workflow.
- Ground-control points or other reference measurements, where applicable.

### Expected outputs

| Output | Purpose |
|---|---|
| Relative depth map | Represents model-predicted relative depth. |
| Calibrated height map | Estimates object height above a defined ground reference. |
| Surface elevation raster | Represents estimated elevation relative to a specified vertical datum, when supported. |
| Semantic mask | Identifies relevant scene classes. |
| 3D surface model | Visualizes the generated height or elevation data. |
| Evaluation report | Records quantitative results against reference data. |

**Important distinction:** A depth map, a digital surface model (DSM), and a normalized digital surface model (nDSM) are not interchangeable.

- **Depth map:** Represents camera-relative or model-relative scene depth.
- **DSM:** Represents elevations of visible surfaces, including buildings and vegetation.
- **nDSM:** Represents object heights above an appropriate ground-surface model.

Converting one representation into another requires suitable geometry, calibration, and reference information. Georeferencing an image alone does not guarantee that its inferred heights are metrically accurate.

## Applications

### Urban planning

Investigate building-height estimation, urban morphology, and three-dimensional representations of built-up areas.

### Disaster management

Explore how elevation and surface information may support flood-risk analysis, damage assessment, and infrastructure monitoring when combined with validated geospatial datasets.

### Environmental monitoring

Support research into vegetation structure, land-cover interpretation, and surface changes.

### Infrastructure analysis

Investigate the extraction of building and infrastructure geometry from aerial imagery.

### Geospatial research

Evaluate the potential of monocular computer vision to complement conventional elevation-mapping workflows.

These are potential applications. Operational use requires domain-specific validation and suitable accuracy guarantees.

## Challenges and Limitations

- **Scale ambiguity:** Monocular predictions do not inherently establish absolute metric scale.
- **Camera geometry:** Nadir, oblique, and perspective imagery may require different processing assumptions.
- **Reference data:** Reliable calibration and validation require suitable ground measurements or elevation products.
- **Occlusions:** A single image cannot directly reveal hidden surfaces.
- **Domain shift:** Models trained on terrestrial scenes may perform differently on satellite or aerial imagery.
- **Spatial resolution:** Fine image detail does not automatically translate into equally accurate elevation detail.
- **Vertical accuracy:** Errors in reference data, alignment, calibration, and model predictions can propagate into final outputs.
- **Generalization:** Performance must be evaluated across different locations, sensor types, resolutions, and surface conditions.

TeraVue should therefore be treated as a research pipeline for estimated elevation products rather than a guaranteed replacement for LiDAR or validated photogrammetric DSM generation.

## Roadmap

- [ ] Establish the initial repository structure and reproducible environment.
- [ ] Implement RGB and GeoTIFF preprocessing.
- [ ] Integrate monocular depth inference.
- [ ] Develop and evaluate metric calibration methods.
- [ ] Integrate semantic segmentation.
- [ ] Implement geospatial raster alignment and export.
- [ ] Generate 3D surface visualizations.
- [ ] Build a reproducible evaluation pipeline using reference elevation data.
- [ ] Benchmark accuracy, runtime, and GPU memory usage.
- [ ] Develop an end-to-end interface for processing images.

Roadmap items should be marked complete as their implementations are verified.

## Research Foundation

TeraVue builds on research directions in monocular depth estimation, relative-depth-to-height estimation, semantic segmentation, and geospatial reconstruction.

Relevant starting points include:

- **Depth Anything V2:** https://github.com/DepthAnything/Depth-Anything-V2
- **SegFormer:** https://github.com/NVlabs/SegFormer
- **Open3D:** https://www.open3d.org/
- **Rasterio:** https://rasterio.readthedocs.io/
- **GDAL:** https://gdal.org/
- **PyProj:** https://pyproj4.github.io/pyproj/

Additional research models and datasets should be incorporated with their original publications, implementation links, licensing information, and evaluation protocols.

## Contributing

Contributions are welcome as the project develops.

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Document dependencies, configuration, and evaluation results.
5. Submit a pull request describing the changes.

For research contributions, include relevant references and clearly distinguish measured results from preliminary observations.

## License

The project license has not yet been specified. Add a `LICENSE` file before distributing the repository publicly, and verify the licenses of all incorporated models, datasets, and third-party components.

---

**TeraVue — Exploring the transition from visual imagery to measurable 3D understanding.**
