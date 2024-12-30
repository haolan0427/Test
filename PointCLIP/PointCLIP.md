# PointCLIP: Point Cloud Understanding by CLIP

#### `关键：`

##### 	`找到2D和3D间的桥梁：`

`将CLIP用于3D点云理解（有zero-shot能力，但是performance不好）`

`将3D点云做投影，multi-view depth maps`

##### 	`Inter-view Adapter：`

`从zero-shot改变为few-shot，performance显著提高`

`实际上就是一个three-layer Multi-layer Perceptron`

##### 	`Multi-knowledge Ensembling`

`将3D点云数据（fine-tuning）和2D image-text pairs数据（Pre-training）做为complimentary knowledge`

#### `Architecture：`

<img src=".\Architecture.jpg" style="zoom:50%;" align="left"/>