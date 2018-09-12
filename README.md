
# Classification and Bounding Box Regression Nets

This repository contains material related to the paper "Multi-View Classification and 3D Bounding Box Regression Networks" by Pramerdorfer et al., ICPR 2018. If you use this material, please cite that paper.

## Dataset

Use the following links to download the synthetic depth maps and metadata used for training and testing: [part1](https://cloud.cogvis.at/s/zgGeFrPfCDDQNE5), [part2](https://cloud.cogvis.at/s/ct23n8ZXnSeSLPR), [part3](https://cloud.cogvis.at/s/RBKcgZeCWTWLX3w), [part4](https://cloud.cogvis.at/s/icxDFwEcRpKdHCr).

## Models

`load.py` is a PyTorch script that loads a trained network.  
Conversion functions from depth maps to views are not included.

## Results

Qualitative regression results on test data:

![results](results.png)

## License

[zlib](https://opensource.org/licenses/Zlib)
