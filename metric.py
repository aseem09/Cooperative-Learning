import pprint as pp
import matplotlib.pyplot as plt

logs = open("log.txt", "r")
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

pp.pprint(train_acc[:10])
pp.pprint(val_acc[:10])
pp.pprint(train_loss[:10])
pp.pprint(val_loss[:10])

plt.plot(train_acc)
plt.ylabel('Train Accuracy')
plt.savefig('train_acc.png', dpi=100)
plt.clf()

plt.plot(val_acc)
plt.ylabel('Val Accuracy')
plt.savefig('val_acc.png', dpi=100)
plt.clf()

plt.plot(train_loss)
plt.ylabel('Train Loss')
plt.savefig('train_loss.png', dpi=100)
plt.clf()

plt.plot(val_loss)
plt.ylabel('Val Loss')
plt.savefig('val_loss.png', dpi=100)
plt.clf()