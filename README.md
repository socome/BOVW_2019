## BOVW(Bag of Visual word) in Colab

### Detail

#### Data Loader 1 (from caltech)

```
!wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
!tar -zxvf 101_ObjectCategories.tar.gz
```

#### Data Loader 2 (from kaggle)

```
! kaggle competitions download -c 2019-ml-finalproject
! unzip 2019-ml-finalproject.zip
```

#### SPM(Spatial Pyramid Matching)

```
total_train_feature = list()

for i in tqdm(range(len(train_des))) :

  ## level 0
  level_0 = vq(train_des[i],codebook)
  level_0 = np.histogram(level_0[0],bins=list(range(code_book_size+1)))[0]

  des_map = train_des[i].reshape(des_size,des_size,128)

  ## level 1
  level_1_1 = vq(des_map[:p2,:p2,:].reshape(-1,128),codebook)
  level_1_1_train_feature = np.histogram(level_1_1[0],bins=list(range(code_book_size+1)))
  level_1_1_train_feature = level_1_1_train_feature[0].reshape(code_book_size,1)

  level_1_2 = vq(des_map[p2:,:p2,:].reshape(-1,128),codebook)
  level_1_2_train_feature = np.histogram(level_1_2[0],bins=list(range(code_book_size+1)))
  level_1_2_train_feature = level_1_2_train_feature[0].reshape(code_book_size,1)

  level_1_3 = vq(des_map[:p2,p2:,:].reshape(-1,128),codebook)
  level_1_3_train_feature = np.histogram(level_1_3[0],bins=list(range(code_book_size+1)))
  level_1_3_train_feature = level_1_3_train_feature[0].reshape(code_book_size,1)

  level_1_4 = vq(des_map[p2:,p2:,:].reshape(-1,128),codebook)
  level_1_4_train_feature = np.histogram(level_1_4[0],bins=list(range(code_book_size+1)))
  level_1_4_train_feature = level_1_4_train_feature[0].reshape(code_book_size,1)

  level_1 = np.concatenate((level_1_1_train_feature,level_1_2_train_feature,level_1_3_train_feature,level_1_4_train_feature),axis=1).flatten()

  ## level 2
  level_2_1 = vq(des_map[:p1,:p1,:].reshape(-1,128),codebook)
  level_2_1_train_feature = np.histogram(level_2_1[0],bins=list(range(code_book_size+1)))
  level_2_1_train_feature = level_2_1_train_feature[0].reshape(code_book_size,1)

  level_2_2 = vq(des_map[p1:p2,:p1,:].reshape(-1,128),codebook)
  level_2_2_train_feature = np.histogram(level_2_2[0],bins=list(range(code_book_size+1)))
  level_2_2_train_feature = level_2_2_train_feature[0].reshape(code_book_size,1)

  level_2_3 = vq(des_map[p2:p3,p1:,:].reshape(-1,128),codebook)
  level_2_3_train_feature = np.histogram(level_2_3[0],bins=list(range(code_book_size+1)))
  level_2_3_train_feature = level_2_3_train_feature[0].reshape(code_book_size,1)

  level_2_4 = vq(des_map[p3:,p1:,:].reshape(-1,128),codebook)
  level_2_4_train_feature = np.histogram(level_2_4[0],bins=list(range(code_book_size+1)))
  level_2_4_train_feature = level_2_4_train_feature[0].reshape(code_book_size,1)

  level_2_5 = vq(des_map[:p1,p1:p2,:].reshape(-1,128),codebook)
  level_2_5_train_feature = np.histogram(level_2_5[0],bins=list(range(code_book_size+1)))
  level_2_5_train_feature = level_2_5_train_feature[0].reshape(code_book_size,1)

  level_2_6 = vq(des_map[p1:p2,p1:p2,:].reshape(-1,128),codebook)
  level_2_6_train_feature = np.histogram(level_2_6[0],bins=list(range(code_book_size+1)))
  level_2_6_train_feature = level_2_6_train_feature[0].reshape(code_book_size,1)

  level_2_7 = vq(des_map[p2:p3,p1:p2,:].reshape(-1,128),codebook)
  level_2_7_train_feature = np.histogram(level_2_7[0],bins=list(range(code_book_size+1)))
  level_2_7_train_feature = level_2_7_train_feature[0].reshape(code_book_size,1)

  level_2_8 = vq(des_map[p3:,p1:p2,:].reshape(-1,128),codebook)
  level_2_8_train_feature = np.histogram(level_2_8[0],bins=list(range(code_book_size+1)))
  level_2_8_train_feature = level_2_8_train_feature[0].reshape(code_book_size,1)

  level_2_9 = vq(des_map[:p1,p2:p3,:].reshape(-1,128),codebook)
  level_2_9_train_feature = np.histogram(level_2_9[0],bins=list(range(code_book_size+1)))
  level_2_9_train_feature = level_2_9_train_feature[0].reshape(code_book_size,1)

  level_2_10 = vq(des_map[p1:p2,p2:p3,:].reshape(-1,128),codebook)
  level_2_10_train_feature = np.histogram(level_2_10[0],bins=list(range(code_book_size+1)))
  level_2_10_train_feature = level_2_10_train_feature[0].reshape(code_book_size,1)

  level_2_11 = vq(des_map[p2:p3,p2:p3,:].reshape(-1,128),codebook)
  level_2_11_train_feature = np.histogram(level_2_11[0],bins=list(range(code_book_size+1)))
  level_2_11_train_feature = level_2_11_train_feature[0].reshape(code_book_size,1)

  level_2_12 = vq(des_map[p3:,p2:p3,:].reshape(-1,128),codebook)
  level_2_12_train_feature = np.histogram(level_2_12[0],bins=list(range(code_book_size+1)))
  level_2_12_train_feature = level_2_12_train_feature[0].reshape(code_book_size,1)

  level_2_13 = vq(des_map[:p1,p3:,:].reshape(-1,128),codebook)
  level_2_13_train_feature = np.histogram(level_2_13[0],bins=list(range(code_book_size+1)))
  level_2_13_train_feature = level_2_13_train_feature[0].reshape(code_book_size,1)

  level_2_14 = vq(des_map[p1:p2,p3:,:].reshape(-1,128),codebook)
  level_2_14_train_feature = np.histogram(level_2_14[0],bins=list(range(code_book_size+1)))
  level_2_14_train_feature = level_2_14_train_feature[0].reshape(code_book_size,1)

  level_2_15 = vq(des_map[p2:p3,p3:,:].reshape(-1,128),codebook)
  level_2_15_train_feature = np.histogram(level_2_15[0],bins=list(range(code_book_size+1)))
  level_2_15_train_feature = level_2_15_train_feature[0].reshape(code_book_size,1)

  level_2_16 = vq(des_map[p3:,p3:,:].reshape(-1,128),codebook)
  level_2_16_train_feature = np.histogram(level_2_16[0],bins=list(range(code_book_size+1)))
  level_2_16_train_feature = level_2_16_train_feature[0].reshape(code_book_size,1)

  level_2 = np.concatenate((level_2_1_train_feature,level_2_2_train_feature,level_2_3_train_feature,level_2_4_train_feature,
                            level_2_5_train_feature,level_2_6_train_feature,level_2_7_train_feature,level_2_8_train_feature,
                            level_2_9_train_feature,level_2_10_train_feature,level_2_11_train_feature,level_2_12_train_feature,
                            level_2_13_train_feature,level_2_14_train_feature,level_2_15_train_feature,level_2_16_train_feature),axis=1).flatten()

  total_train_feature.append(np.concatenate((0.25*level_0,0.25*level_1,0.5*level_2)))
  # total_train_feature.append(level_0)

total_train_feature = np.array(total_train_feature)
```


#### SPM(Spatial Pyramid Matching) Kernel
```
def histogramIntersection(M, N):
    m = M.shape[0]
    n = N.shape[0]
    result = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            temp = np.sum(np.minimum(M[i], N[j]))
            result[i][j] = temp
    return result
    

gramMatrix = histogramIntersection(total_train_feature, total_train_feature)
clf = svm.SVC(kernel='precomputed')
clf.fit(gramMatrix, train_labels)
predictMatrix = histogramIntersection(total_test_feature, total_train_feature)
SVMResults = clf.predict(predictMatrix)

```


#### 

-------------------------------------
### Paper 

[Beyond Bags of Features: Spatial Pyramid Matching
for Recognizing Natural Scene Categories](https://inc.ucsd.edu/~marni/Igert/Lazebnik_06.pdf)


-------------------------------------
### Report

#### BOVW

| Level | codebook_size | step_size | img_size | background | histogram_intersection | scaler |accuracy |
|:--------: |:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| 0 | 200 | 8 | 256x256 | O | - | O | [0.41607](https://github.com/socome/BOVW_2019/blob/master/BOVW_Caltech101_base.ipynb) |
| 0 | 200 | 8 | 256x256 | X | - | O | 0.43735 |
| 2 | 600 | 8 | 256x256 | O | X | O | 0.50236 |
| 2 | 600 | 8 | 256x256 | O | O | X | 0.57505 |
| 2 | 600 | 8 | 256x256 | O | O | O | 0.58510 |
| 2 | 600 | 8 | 256x256 | X | O | X | 0.60933 |
| 2 | 600 | 8 | 256x256 | X | O | O | [0.62056](https://github.com/socome/BOVW_2019/blob/master/BOVW_Caltech101_62.ipynb) |


#### VLAD

| Level | codebook_size | step_size | img_size | background | accuracy |
|:--------: |:--------:|:--------:|:--------:|:--------:|:--------:|
| 0 | 600 | 8 | 256x256 | X | [0.63593](https://github.com/socome/BOVW_2019/blob/master/VLAD_Caltech101_63.ipynb) |

-------------------------------------
### Reference

##### [Histogram_intersection](https://github.com/wihoho/Image-Recognition/blob/5dc8834dd204e36172815345f0abe5640a4a37ef/recognition/classification.py#L10)</br>
##### [VLAD](https://github.com/jorjasso/VLAD)
-------------------------------------
