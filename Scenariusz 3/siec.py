from pybrain3 import *

siec = FeedForwardNetwork()

wejscie = LinearLayer(35)
ukryty = SigmoidLayer(30)
wyjscie = LinearLayer(20)

bias = BiasUnit()

siec.addInputModule(wejscie)
siec.addModule(bias)
siec.addModule(ukryty)
siec.addOutputModule(wyjscie)

bias_ukryty = FullConnection(bias, ukryty)
wejscie_ukryty = FullConnection(wejscie, ukryty)
ukryty_wyjscie = FullConnection(ukryty, wyjscie)

siec.addConnection(bias_ukryty)
siec.addConnection(wejscie_ukryty)
siec.addConnection(ukryty_wyjscie)

siec.sortModules()

