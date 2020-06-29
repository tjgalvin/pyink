# pyink

A useful set of classes and utilities to produce self-organising maps with PINK, annotate these neurons and perform object collation with the outputs. 

The process has three main steps:
- create a SOM using `PINK` that accurately describes the predominate features within an image set,
- annotate the SOM in following some user defined classification scheme, and
- transfer these labels from the SOM and it's neurons to real on-sky catalogue data by using the mapping and spatial transform solutions returned by `PINK`.

A more thorough description is provided by Galvin et al. (submitted), who use image data at radio and infrared wavelengths to demonstrate multi-wavelength host identification is possible within an *unsupervised* framework. 

Throughout we use `PINK v2` while developing this module. Note that the binary files produced by `PINK` and the corresponding classes are not directly interchangeable between `PINK` version 1 and 2 releases. 

## Dependencies
- numpy
- scipy
- astropy
- scikit-image
- tqdm

## Development

Collaborators and pull requests are welcome. Please do not hesitate to ask questions or get involved. 

Throughout the code base type annotations, doc-strings and the `black` code formatter have been used, with `mypy` as a linter.  