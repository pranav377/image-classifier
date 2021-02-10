# ImageClassifier

ImageClassifier is a python package for creating Image classifiers with the help of CNNs.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ImageClassifier.

```bash
pip install image-classifier
```
## Data format

First you need to create or download the images and save them in a folder in their corresponding label folders.

example folder structure:
```bash
data-->
    apple-->
        (images related to apple)
    banana-->
        (images realated to banana)
```
![alt text](https://drive.google.com/file/d/1RVZXutlC3WwjSJYja_yzqOupS1auh_Sl/view "example_data")
here "data" is the main folder, "apple" and "banana" are the label folders which have their corresponding images related to them.

[NOTE]: You can create as many label folders as you want but they should have their corresponding images.

## Usage 

To create a Image classifier :-

### Creating the data and model.

```python
from ImageClassifier import CreateDataAndModel

main = CreateDataAndModel(file_path="./data/", model_file_name_to_save="model.hdf5", init_lr=0.0001, epochs=100, batch_size=32)

# Here "./data/" is the data folder where we have the label folder- "apple" and "banana" (Remember to include a "/" after the folder like "./data/" ). Basically, you have to give the data folder path where you have the label folders, it can also be like "./seg_pred/seg_pred/".   

main.create() # This will create the data and the model for you.
```

Running the image classifier to predict a single image.

```python
from ImageClassifier import Run

r = Run(model_file_name="model.hdf5") # The model file in your current directory

img_to_predict = "./example_dir/example.jpg"

pred = r.run(img_to_predict)

print(pred)

```
Output

```bash
apple
```

Running the image classifier to predict multiple images.

```python

import os

r = Run("model.hdf5")

file_path = "./seg_pred/seg_pred/"

for file in os.listdir(file_path):
  pred = r.run(file_path + file)
  print(pred)

```
Output
```bash
apple
banana
banana
apple
banana
```


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

See the source code here [/image-classifier/ImageClassifier/](https://github.com/pranav377/image-classifier/tree/main/ImageClassifier)

## Data types

You can use any type of images for creating data like- jpeg, png etc. Even mixed!!!

## License

[MIT License](https://github.com/pranav377/image-classifier/blob/main/LICENSE)

