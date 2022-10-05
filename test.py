import numpy as np
# ------------------------------
# ANN:
# ------------------------------
accuracy=0.8431
precision=0.8321
recall=0.8377

TP = round((1.0 - accuracy) * 3460. / (1.0 / precision + 1.0 / recall - 2.0), 0)
TP_TN = round(accuracy * 3460., 0)
TN = round(TP_TN - TP, 0)
FP = round(TP * (1.0 / precision - 1.0), 0)
# FN = TP * (1.0 / recall - 1.0)
FN = 3460. - round((TP + TN + FP),0)

print('-'*30)
print('TP: {} TN: {} FN: {} FP: {}\n'.format(TP, TN, FN, FP))

# ------------------------------
# ANN_70:
# ------------------------------
accuracy=0.8283
precision=0.8204
recall=0.8164

TP = round((1.0 - accuracy) * 3460. / (1.0 / precision + 1.0 / recall - 2.0), 0)
TP_TN = round(accuracy * 3460., 0)
TN = round(TP_TN - TP, 0)
FP = round(TP * (1.0 / precision - 1.0), 0)
# FN = TP * (1.0 / recall - 1.0)
FN = 3460. - round((TP + TN + FP),0)

print('-'*30)
print('TP: {} TN: {} FN: {} FP: {}\n'.format(TP, TN, FN, FP))


# ------------------------------
# ANN_40:
# ------------------------------
accuracy=0.7928
precision=0.7662
recall=0.8096

TP = round((1.0 - accuracy) * 3460. / (1.0 / precision + 1.0 / recall - 2.0), 0)
TP_TN = round(accuracy * 3460., 0)
TN = round(TP_TN - TP, 0)
FP = round(TP * (1.0 / precision - 1.0), 0)
# FN = TP * (1.0 / recall - 1.0)
FN = 3460. - round((TP + TN + FP),0)

print('-'*30)
print('TP: {} TN: {} FN: {} FP: {}\n'.format(TP, TN, FN, FP))