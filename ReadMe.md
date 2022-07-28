# Implicit High Resolution Representation of 3D Shapes

This is the repository for the semester project "Implicit High Resolution Representation of 3D Shapes" done by Fedor Sergeev at EPFL CVLab in collaboration with Neural Concept.

## Abstract

Deep implicit functions provide a powerful and flexible framework for encoding 3D shapes. Many previous works focused on training a deep neural network to represent multiple shapes, using an additional latent vector as input to differentiate between them. This approach produces representations of sufficient quality on average. However, in individual cases, the learned shapes often lack high resolution details. This limits the use of deep implicit functions for engineering and design, where precise shape encoding is crucial. In this work, we propose a simple setting for training a deep neural implicit to represent a single complex 3D shape and maintain its high-resolution features. On the particular shape we used, the reconstructed shape is visually more accurate then those obtained in previous works. Furthermore, we compare various approaches to designing the network architecture, sampling the implicit function values, picking the loss function, and identify the important parameters that affect the final quality.

## Requirements

```
igl==2.2.1
numpy==1.20.3
scikit_image==0.18.3
scipy==1.5.3
skimage==0.0
torch==1.12.0+cu113
trimesh==3.12.9
```

Using an Nvidia GPU is recommended for reasonable training times.

## File Organization

- `data/` - folder containing the input shapes represented as meshes stored in STL format
    - `Fennec_Fox.stl` - Fennec Fox shape from the Thingi10k dataset [ [link]](https://ten-thousand-models.appspot.com/detail.html?file_id=65414)
    - `block-2x4.stl` - Trilego shape from the Thingi10k dataset [[link]](https://ten-thousand-models.appspot.com/detail.html?file_id=44633)
    - `mesh.obj` - A car shape from the [ShapeNet](https://shapenet.org/) dataset
- `src/`: Folder contains source code 
    - `chamfer.py` - computation of chamfer distance
    - `emd.py` - computation of earth mover's distance
    - `generator_sdf.py` - DeepSDF model for a single shape 
    - `load_sdf.py` - loading SDF samples from file 
    - `mesh.py` - computing SDF from a mesh, and reconstructing mesh from SDF values
    - `sample_sdf.py` - various SDF sampling procedures
- `report.pdf` - report that describes the study and presents the results
- `sdf_generator_and_warper.ipynb.ipynb` - notebook for running and visualizing the experiments

## Citing

```
@misc{misc,
  author = {Fedor Sergeev, Nicolas Talabot, Jonathan Donier and Pascal Fua},
  title = {Implicit High Resolution Representation of 3D Shapes},
  submissiondate = {2022/06/10},
  year = {2022},
  urldate = {2022-06-10},
  url = {https://github.com/TheodorSergeev/implicit-high-res-templates/blob/main/report.pdf},
  note = {Semester project report},
  institution = {CVLab, EPFL},
}
```