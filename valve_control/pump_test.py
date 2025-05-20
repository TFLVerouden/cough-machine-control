# import class
from functions import pumpy
# from archive import pumpy_orig as pumpy

# Initialise chain
chain = pumpy.Chain("COM10", baudrate=19200)

# Initialise PHD 2000 with address = 0
# pump = pumpy.Pump2000(chain, address=0)
pump = pumpy.Pump(chain, address=0, name='PHD2000')

# # pump.cvolume()
# pump.clear_accumulated_volume()
# # pump.setdiameter(3.26)
# # pump.setdiameter(10000)
# pump.setdiameter(7.28)
# # pump.setinfusionrate(100, "u/m")
# # pump.setwithdrawrate(100, "u/m")
# pump.set_rate(100, "u/m")
# # pump.settargetvolume(100, "u")
# pump.settargetvolume(0.01)
# # pump.infuse()
pump.infuse()
# pump.waituntilfinished()
pump.waituntilfinished()
