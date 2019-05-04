# iphoneOrNot

You should to create model to detect iphone (all versions) on a picture.
If picture contains two or more iphones you should return only one probability for all picture. Picture is a typical for internet shop.
Solution should contains all ML stages (you can skip collect data stage) and
pretrained model in git or link to another data storage.
Also you should to provide example running inference.


## Data:
Collect data is a part of task.


## Restrictions:
* inference work on CPU
* neural net frameworks - keras, pytorch
* python3
* work without internet


## Performance measure:
Area under precision and recall on hidden data.
Who will commit solution which better then random and don't copy  pasted get 20 scores.
Another scores will depend on you rating based on hidden data.



## Hard deadline:
June 1st


## Interface to detection

python predict.py --model path_to_model --input path_to_input_data --output path_to_results

input data - folder with images
output data - csv file with two columns: image_name,iphone_probability