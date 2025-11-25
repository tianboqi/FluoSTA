# FluoSTA

Structure Tensor Analysis (axon bundle tracking) on fluorescence images from cleared brain

## General workflow

`sta_flow.py` calculates the eigenvector field of the local Hessian matrix

`sta.track.py` tracks from the seed along the flow vector field

`sta_filter.py` applies postprocessing to filter the streamlines by angle, length, coherence, intensity, etc

---
Tianbo Qi, Scripps Research, 2025

Part of the code was adapted from [MIRACL](https://github.com/AICONSlab/MIRACL)
