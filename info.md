copy pretrain weights from 'https://drive.google.com/uc?id=1vzBZ2rNCQWBQpYvC6hpyJfR3iK1O_FEG' which I got if from Colab: https://colab.research.google.com/drive/1Dp6VsZp2ozKuu9ccMmsDjyij_vXfCYb3?usp=sharing#scrollTo=3WWX0dF9pH-W

The weights they specified in the info it's not compateble with the model ()
```
self.encoder.load_state_dict(checkpoint['encoder'], strict=True)
KeyError: 'encoder'
```

# Install
- When install, using  ~/miniconda3/envs/motiondiffuse/bin/pip install
- Pytorch3D needs to install from source, conda install doesn't work. [Instruction](pip install -r requirements.txt)
```
~/miniconda3/envs/motiondiffuse/bin/pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```






run visualize
```
cd /home/epinyoan/git/MotionDiffuse/text2motion
conda activate motiondiffuse
python -u ./tools/visualization.py --opt_path checkpoints/t2m/t2m_motiondiffuse/opt.txt --text "a man backflip" --motion_length 2     --result_path "test_sample.gif"
```