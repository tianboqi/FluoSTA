# FluoSTA
Structure Tensor Analysis (axon bundle tracking) on fluorescence images from cleared brain

## General wordflow
`sta_flow.py` calculates the eigenvector field of the local Hessian matrix
`sta.track.py` tracks from the seed along the flow vector field
`sta_filter.py` applies postprocessing to filter the streamlines by angle, length, coherence, intensity, etc
