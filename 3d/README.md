# LPBF Inc625 Displacement Dataset

### Description of Dataset:

This dataset contains 24,880 NetFabb simulation results as zipped numpy archive files (.npz). 
These are vertex displacements for LPBF of Inc625 (40 um) with an AM250.
The shapes originate from the [Fusion360 Gallery Segmentation dataset](https://www.research.autodesk.com/publications/fusion-360-gallery/).  

Each .npz file contains:
- 'verts': vertex coordinates
- 'elems': the 8 vertices at the corner of each element
- 'disp': simulated xyz-displacements of each vertex at the final time step

### Loading the Data into Python:

To load a geometry in Python, use the following code snippet:
```python
filename = 'data/16550_e88d6986_0.npz'    # (for example)
data = np.load(filename)
verts, elems, disp = data['verts'], data['elems'], data['disp']
```

### Dataset Generation Procedure:

1. Extract parts from Fusion360 Gallery Segmentation Dataset
2. Load part into Autodesk NetFabb Simulation
   - Keep native orientation
   - Center bottom at (x,y) = (0,0) extending into positive z-direction
   - Scale part 10x, such that part fits in 2cmx2cmx2cm bounding box
3. Run LPBF simulation
   - Machine = AM250
   - Material = Inc625 (40 um)
4. Export ASCII results
   - Node/element geometry stored in: '/results/mechanical_{max}.geo'
   - Nodal displacements stored in: '/results/mechanical00_{max}.dis.ens'
   - These raw output files are omitted from this dataset
5. Extract vertices, elements, and displacements; save to .npz file
   - The 'data/' folder contains these .npz files
