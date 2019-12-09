
## BOVW(Bag of Visual word) in Colab


-------------------------------------
### Paper 

[Beyond Bags of Features: Spatial Pyramid Matching
for Recognizing Natural Scene Categories](https://inc.ucsd.edu/~marni/Igert/Lazebnik_06.pdf)


-------------------------------------
### Report

#### BOVW

| Level | codebook_size | step_size | img_size | background | histogram_intersection | scaler |accuracy |
|:--------: |:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 0 | 200 | 8 | 256x256 | O | - | O | 0.41607 |
| 0 | 200 | 8 | 256x256 | X | - | O | 0.43735 |
| 2 | 600 | 8 | 256x256 | O | X | O | 0.50236 |
| 2 | 600 | 8 | 256x256 | O | O | X | 0.57505 |
| 2 | 600 | 8 | 256x256 | O | O | O | 0.58510 |
| 2 | 600 | 8 | 256x256 | X | O | X | 0.60933 |
| 2 | 600 | 8 | 256x256 | X | O | O | 0.62056 |

[Base(41.6%)]()</br>
[Best(62%)](https://github.com/socome/BOVW_2019/blob/master/BOVW_Caltech101_62.ipynb)

#### VLAD

| Level | codebook_size | step_size | img_size | background | accuracy |
|:--------: |:--------:|:--------:|:--------:|:--------:|:--------:|
| 0 | 600 | 8 | 256x256 | X | 0.63593 |

[Base(63%)](https://github.com/socome/BOVW_2019/blob/master/VLAD_Caltech101_63.ipynb)

-------------------------------------
### Reference

##### [Histogram_intersection](https://github.com/wihoho/Image-Recognition/blob/5dc8834dd204e36172815345f0abe5640a4a37ef/recognition/classification.py#L10)</br>
##### [VLAD](https://github.com/jorjasso/VLAD)
-------------------------------------
