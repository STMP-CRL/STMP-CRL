# SpatioTemporal-View Member Preference Contrastive Representation Learning for Group Recommendation

This is our implementation for the paper:

## Environment Settings
- torch=1.10.0
- python=3.7
- tqdm=4.64.0
- numpy=1.21.6
- scipy=1.7.3
- scikit-learn=0.24.2

## Example to run the codes.

Run STMP-CRL:

```
python main.py
```

After training process, the value of HR and NDCG in the test dataset will be printed in command window and save in 'res/'.

## Parameter Tuning

we put all the papameters in the `config.py`

## Dataset

We provide one processed dataset: meetupCA. 

groupMember.txt:
* Member file.
* Each Line is a group: groupID userID1,userID2,...,userIDn

group(user)RatingTrain.txt:

* Train file.
* Each Line is a training instance: groupID(userID) itemID timestamp is/notinteract 

group(user)RatingTest.txt:

* group(user) Test file (positive instances).
* Each Line is a testing instance: groupID(userID) itemID timestamp is/notinteract 

group(user)RatingNegative.txt:

* group(user) Test file (negative instances).
* Each line corresponds to the line of test.rating, containing 100 negative samples.
* Each line is in the format: (groupID(userID),itemID) negativeItemID1 negativeItemID2 ... negativeItemIDn
