import trimesh
mesh = trimesh.load('/media/cvlab/data/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb')
mesh.export('converted_model.obj')
