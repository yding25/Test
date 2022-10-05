def test(accuracy, precision, recall):
    N = 3460
    TP_FP = 1639
    TN_FN = 1821    
    TP = int(TP_FP * precision)    
    FP = TP_FP - TP 
    FN = int(TP / recall - TP)
    TN = N - (TP + FP + FN)
    beta = 1.0
    print('TP:{} TN:{} FP:{} FN:{} f-beta-score:{}'.format(TP, TN, FP, FN, ((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)))

# ------------------------------------------------------------------
# ANN_100
accuracy = 0.8549
precision = 0.8464
recall = 0.8434
print('ANN_100')
test(accuracy, precision, recall)

# ANN_83
accuracy = 0.8533
precision = 0.8652
recall = 0.8235
print('ANN_83')
test(accuracy, precision, recall)

# ANN_66
accuracy = 0.8519
precision = 0.8502
recall = 0.8264
print('ANN_66')
test(accuracy, precision, recall)

# ANN_50
accuracy = 0.8416
precision = 0.8238
recall = 0.8378
print('ANN_50')
test(accuracy, precision, recall)

# ANN_33
accuracy = 0.8382
precision = 0.8215
recall = 0.8321
print('ANN_33')
test(accuracy, precision, recall)

# ANN_16
accuracy = 0.8147
precision = 0.7999
recall = 0.8011
print('ANN_16')
test(accuracy, precision, recall)

print('-'*30)

# ------------------------------------------------------------------

# ANN_100 (new)
accuracy = 0.8350
precision = 0.8407
recall = 0.7959
print('ANN_100')
test(accuracy, precision, recall)

# ANN_83
accuracy = 0.8301
precision = 0.8328
recall = 0.7939
print('ANN_83')
test(accuracy, precision, recall)

# ANN_66
accuracy = 0.8226
precision = 0.8138
recall = 0.8019
print('ANN_66')
test(accuracy, precision, recall)

# ANN_50
accuracy = 0.8179
precision = 0.8077
recall = 0.7984
print('ANN_50')
test(accuracy, precision, recall)

# ANN_33
accuracy = 0.8107
precision = 0.7964
recall = 0.7964
print('ANN_33')
test(accuracy, precision, recall)

# ANN_16
accuracy = 0.7889
precision = 0.7675
recall = 0.7833
print('ANN_16')
test(accuracy, precision, recall)

print('-'*30)

# ------------------------------------------------------------------
# SVM_100
accuracy = 0.8356
precision = 0.8236
recall = 0.8296
print('SVM_100')
test(accuracy, precision, recall)

# SVM_83
accuracy = 0.8384
precision = 0.8307
recall = 0.8187
print('SVM_83')
test(accuracy, precision, recall)

# SVM_66
accuracy = 0.8296
precision = 0.8156
recall = 0.8177
print('SVM_66')
test(accuracy, precision, recall)

# SVM_50
accuracy = 0.8250
precision = 0.8160
recall = 0.8042
print('SVM_50')
test(accuracy, precision, recall)

# SVM_33
accuracy = 0.8118
precision = 0.8066
recall = 0.7802
print('SVM_33')
test(accuracy, precision, recall)

# SVM_16
accuracy = 0.7958
precision = 0.7968
recall = 0.7515
print('SVM_16')
test(accuracy, precision, recall)

print('-'*30)

# ------------------------------------------------------------------
# SVM_100
accuracy = 0.8137
precision = 0.8060
recall = 0.7893
print('SVM_100')
test(accuracy, precision, recall)

# # SVM_83
accuracy = 0.8125
precision = 0.8058
recall = 0.7863
print('SVM_83')
test(accuracy, precision, recall)

# SVM_66
accuracy = 0.8076
precision = 0.7990
recall = 0.7833
print('SVM_66')
test(accuracy, precision, recall)

# SVM_50
accuracy = 0.7992
precision = 0.7949
recall = 0.7657
print('SVM_50')
test(accuracy, precision, recall)

# SVM_33
accuracy = 0.7878
precision = 0.7831
recall = 0.7516
print('SVM_33')
test(accuracy, precision, recall)

# SVM_16
accuracy = 0.7775
precision = 0.7754
recall = 0.7340
print('SVM_16')
test(accuracy, precision, recall)

print('-'*30)