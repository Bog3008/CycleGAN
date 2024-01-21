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

Clone the project

```bash
  git clone https://github.com/Bog3008/CycleGAN
```

Go to the project directory

```bash
  cd CycleGAN
```

Start the server

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
