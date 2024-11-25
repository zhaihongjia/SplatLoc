### NOTES

We provide two visualization demo for rendering localization process with / without 2D-3D matches.

First, you need update the 2D image, 3D mesh, estimated/GT pose, 2D-3D match path in the scripts.

Then, you can run following codes for visualization via Open3D:
```
python render_localization_with_matches.py
# or 
python render_localization.py
```

- press 'a' to update the mesh and pose, once
- press 'b' to update process automatically
- [in render_localization_with_matches.py] if you want to select the viewpoint and fix it by yourself, (before you press 'b') you can drag the view in open3D, and press 'c' to fix the view, and then press 'b' to update it.
- [in render_localization.py] set current viewpoint as fixed viewpoint: press 'd', set our pre-defined viewpoint:  press 'c'

