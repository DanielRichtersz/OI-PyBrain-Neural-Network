#----------------------------------------------
#   Feed Forward Networks
#----------------------------------------------
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

n = FeedForwardNetwork()

#   Construct input, hidden and output layers
inLayer = LinearLayer(2)
hiddenlayer = SigmoidLayer(3)
outLayer = LinearLayer(1)

#   Add layers to the network
n.addInputModule(inLayer)
n.addModule(hiddenlayer)
n.addOutputModule(outLayer)

#   Add full connection between the neurons of each layer
in_to_hidden = FullConnection(inLayer, hiddenlayer)
hidden_to_out = FullConnection(hiddenlayer, outLayer)

#   Final step: Necessary for sorting the modules, and other internal initialization
n.sortModules()

#----------------------------------------------
#   Examining a Network
#----------------------------------------------

n.activate([1, 2])