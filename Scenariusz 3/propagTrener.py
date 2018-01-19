from pybrain3.supervised.trainers import BackpropTrainer
from literki import daneWejsciowe
import literki
import siec

litery = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "U", "M", "L", "O", "P", "R", "T", "W", "S"]
inp = daneWejsciowe['input']
trener = BackpropTrainer(siec.siec, dataset=literki.daneWejsciowe, learningrate=0.1)
trener.trainEpochs(1000)

for i in range(20):
    print(litery[i])
    temp = siec.siec.activate(inp[i])
    for j in range(20):
        print(temp[j])
print("\n")