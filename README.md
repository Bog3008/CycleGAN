# My CycleGAN
It is my implementation of CycleGAN.  
The original [paper](https://arxiv.org/abs/1703.10593)

![logo](images/main_pic.png)

## Horse2Zebra  
First i tryed [horese2zebra dataset](https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset)

On the left is the original picture, on the right it is changed, but it is obvious)      
Horses to zebras translation:  
![hz1](images/HZ/h2z_1.png) ![hz2](images/HZ/h2z_2.png) ![hz3](images/HZ/h2z_3.png)  
Zebras to horses translation:  
![zh1](images/HZ/z2h_1.png) ![hz2](images/HZ/z2h_2.png) ![hz3](images/HZ/z2h_3.png)


###  Couple of notes
* NN(neural network) doesn't know that we only want to translate zebras, so NN also often changes the color of the grass from green to yellow. This is because zebras live in Africa in dry savannas where the grass is yellow and most of the photos with zebras are taken there  
* NN does not work well when a person is riding a horse, since zebras are not ridden, therefore there are no photos of a person riding a zebra in train dataset  
* NN doesn't work well with boundaries between multiple horses

<todo paste here some examples>
.

## Summer2Winter  
Then i took [this dataset](https://www.kaggle.com/datasets/balraj98/summer2winter-yosemite)

Summer to winter translation:  
The third images i found interesting, NN made snowdrop flower
![sw1](images/SW/s2w_1.png) ![sw2](images/SW/s2w_2.png) ![sw3](images/SW/s2w_3.png)  
Winter to summer translation:  
![sw1](images/SW/w2s_1.png) ![sw2](images/SW/w2s_2.png) ![sw3](images/SW/w2s_3.png)  

* I got moderately good results
* The translation from winter to summer does not work in the best way. Most likely there was not enough epochs
* Source photos taken in winter had a "blue" filter, and summer ones had a "contrast" filter. Often NN has applied these filters to the images

## Training
* As in the original article, I number of epochs = 200
* I used LR sheduler with warmup on 30 epoch


## Installation and run
Install the dependencies  
```bash
pip install -r requirements.txt
```  

Clone the project

```bash
  git clone https://github.com/Bog3008/CycleGAN
```

Go to the project directory

```bash
  cd CycleGAN
```

### Start the server(web app)

```bash
  & <your python interpreter path> app.py <your path to the folder with model weights>
```
* "your python interpreter path" - For example, for me, it looks like this: "D:/Anaconda/python.exe"
* "your path to the folder with model weights"  - This folder should contain the following files: 'gen_zebras', 'gen_horse', 'gen_winter', 'gen_summer'  - weights for generators where the second word in name means the target domain transformation. For example, for me, it looks like this: "D:\CYCLEWEIGHTS".

Then open http://localhost:9000/  
Select the transformation you want  
Upload image  
Click on the "Apply Transformation" button  
Wait a little bit and then you'll see the result

### Run train

```bash
  & <your python interpreter path> train.py <folder train A> <folder train B> <folder test A> <folder test B> 
```

This is how it looks for me:  
```bash
D:/Anaconda/python.exe d:/VS_code_proj/CycleGAN/CycleGAN/train.py data\train_summer data\train_winter data\test_summer data\test_winter
```
data\\... - local path (this folders are in the project folder)

### Run inference

```bash
  & <your python interpreter path> inference.py <image path> <path to the model> <path for generated image to save> 
```
This is how it looks for me:  
```bash
D:/Anaconda/python.exe d:/VS_code_proj/CycleGAN/CycleGAN/inference.py D:\VS_code_proj\CycleGAN\CycleGAN\saved_test_images\fake_summer_iter_0.png saved_models\256\2023_12_01_05h33m\gen_W D:\
```
saved_models\\... - local path (this folder are in the project folder)

## Pretrained models
The wieghts for model are available:  
[Generator to winter](https://drive.google.com/file/d/14LsiKrQivNwW8laLPWNXFDJYRYoXDED0/view?usp=sharing)  
[Generator to summer](https://drive.google.com/file/d/1s3lyme5RhvCck9FJ5UtfuQAYgOe4Ym5M/view?usp=sharing) 

 &ndash; Where are the wieghts for zebras and horses ?  
 &ndash; I accidentally deleted it...

## Metrics

<table>
  <tr>
    <th>Model Type</th>
    <th>IS</th>
    <th>FID</th>
  </tr>
  <tr>
    <td>To summer</td>
    <td>3.5054</td>
    <td>0.1137</td>
  </tr>
  <tr>
    <td>To winter</td>
    <td>2.7693</td>
    <td>0.0969</td>
  </tr>
</table>

## Run metrics
Open the project folder 
```bash
cd CycleGAN
```  

Run calc_metrics.py

```bash
  & <your python interpreter path> calc_metrics.py <path to generated images folder> <your path to real images folder> 
```
Images folders must contain only images. The structure should look this:

Generated_Images/  
├── img1.png   
├── img2.png  
├── img3.png 

Real_Images/  
├── img_r1.jpg   
├── img2_r.jpg  
├── img3_r.jpg 