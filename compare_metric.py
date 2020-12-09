import pprint as pp
import matplotlib.pyplot as plt

logs = open("history/evaluation/log_00_1.txt", "r")
data = logs.readlines()

train_acc = []
val_acc = []
train_loss = []
val_loss = []

isTraining = True
isVal = True

l1 = 0
l2 = 0

for row in data[3:]:
    values = row.split(" ")
    if 'train_acc' in row:
        train_acc.append(float(values[3].replace("\n","")))

    elif 'valid_acc' in row:
        val_acc.append(float(values[3].replace("\n","")))

    elif 'train' in row and 'acc' not in row:
        train_loss.append(float(values[5]))

    elif 'valid' in row and 'acc' not in row:
        val_loss.append(float(values[5]))

acc_1 = train_acc
val_acc_1 = val_acc
# pp.pprint(train_acc[:10])
# pp.pprint(val_acc[:10])
# pp.pprint(train_loss[:10])
# pp.pprint(val_loss[:10])
print("Max Train Acc 00 : " + str(max(acc_1)))
print("Max Val Acc 00   : " + str(max(val_acc_1)))

logs = open("history/evaluation/log_05_1.txt", "r")
data = logs.readlines()

train_acc = []
val_acc = []
train_loss = []
val_loss = []

isTraining = True
isVal = True

l1 = 0
l2 = 0

for row in data[3:]:
    values = row.split(" ")
    if 'train_acc' in row:
        train_acc.append(float(values[3].replace("\n","")))

    elif 'valid_acc' in row:
        val_acc.append(float(values[3].replace("\n","")))

    elif 'train' in row and 'acc' not in row:
        train_loss.append(float(values[5]))

    elif 'valid' in row and 'acc' not in row:
        val_loss.append(float(values[5]))


acc_2 = train_acc
val_acc_2 = val_acc


print("Max Train Acc 05 : " + str(max(acc_2)))
print("Max Val Acc 05   : " + str(max(val_acc_2)))

plt.plot(acc_1, label="0.00")
plt.plot(acc_2, label="0.05")
plt.ylabel('Train Accuracy')
plt.legend()
plt.savefig('train_acc.png', dpi=600)
plt.clf()

plt.plot(val_acc_1, label="0.00")
plt.plot(val_acc_2, label="0.05")
plt.ylabel('Val Accuracy')
plt.legend()
plt.savefig('val_acc.png', dpi=600)
plt.clf()
