
## BOVW(Bag of Visual word) in Colab


-------------------------------------
### Paper 
-------------------------------------

[Beyond Bags of Features: Spatial Pyramid Matching
for Recognizing Natural Scene Categories](https://inc.ucsd.edu/~marni/Igert/Lazebnik_06.pdf)


-------------------------------------
### Report
-------------------------------------

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

#### VLAD

| Level | codebook_size | step_size | img_size | background | histogram_intersection | accuracy |
|:--------: |:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 0 | 600 | 8 | 256x256 | X | - | 0.63593 |


-------------------------------------
### Reference
-------------------------------------
[Histogram_intersection](https://github.com/wihoho/Image-Recognition/blob/5dc8834dd204e36172815345f0abe5640a4a37ef/recognition/classification.py#L10)</br>
[VLAD](https://github.com/jorjasso/VLAD)
