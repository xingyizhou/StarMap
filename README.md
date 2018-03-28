# StarMap for Category-Agnostic Keypoint and Viewpoint Estimation

PyTorch implementation for **category-agnostic** keypoint and viewpoint estimation.

<img src='Framework.jpg' align="center" width="700px">


> Xingyi Zhou, Arjun Karpur, Linjie Luo, Qixing Huang,     
> **StarMap for Category-Agnostic Keypoint and Viewpoint Estimation**      
> [arXiv:1803.09331](https://arxiv.org/abs/1803.09331)

Supplementary material with more qualitative results and higer resulution can be found [here](https://drive.google.com/file/d/1IEcHBdQ8u2HTKiNz88ItJWDRQUOJnf60/view?usp=sharing).


Contact: [zhouxy2017@gmail.com](mailto:zhouxy2017@gmail.com)

## Requirements
- Python with h5py, opencv
- [PyTorch](http://pytorch.org/)


## Demo
- Download our pre-trained [model](https://drive.google.com/file/d/1bwCeC4F0OLFYceiaAuUGB6pU8OOZor1k/view?usp=sharing) and move it to `models`.
- Run 
```
python demo.py -demo /path/to/image [-loadModel /path/to/model/] [-GPU 0]
```
The demo code runs in CPU by default. 

We provide example images in `images/`. 
The results are shown with predicted canonical view (triangle), the predicted 3D keypoints (cross), and the rotated keypoints with the estimated viewpoint (star). 

## Training & Benchmark Evaluation
 Coming soon.


## Citation
    @InProceedings{zhou2018starmap,
    author = {Zhou, Xingyi and Karpur, Arjun and Luo, Linjie and Huang, Qixing},
    title = {StarMap for Category-Agnostic Keypoint and Viewpoint Estimation},
    journal={arXiv preprint arXiv:1803.09331},
    year={2018}
    }
