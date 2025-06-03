from medpy import metric

pred = [10,224,224]
gt = [10,224,224]
asd = metric.binary.assd(pred, gt)
print(asd.shape)