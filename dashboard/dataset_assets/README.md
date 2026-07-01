# SAFE -> UniScene dataset viewer assets

- `lib/` : Three.js + OrbitControls.
- `<case>/scene_NN.png`, `ego_NN.png` : per-timestep UniScene BEV (scene + ego V1).
- `<case>/data.json` : per-frame 3D boxes [x,y,z,l,w,h,yaw,is_ego], drivable cells, objects.

Open `../dashboard_dataset.html` (this folder must sit beside it). No server/network.
