# import class
from functions import pumpy

# Initialise chain
chain = pumpy.Chain("COM10", baudrate=19200)

# Initialise PHD 2000 with address = 2
pump = pumpy.Pump2000(chain, address=0)

pump.cvolume()
pump.setdiameter(3.26)
pump.setsyringevolume(500, "u")
pump.setinfusionrate(100, "u/m")
pump.setwithdrawrate(100, "u/m")
pump.settargetvolume(100, "u")
pump.infuse()
pump.waituntilfinished()
