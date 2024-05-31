import numpy as np
import pandas as pd


vui1 = np.loadtxt("../CollectedData/new_data/ThaiVui.txt")
vui2 = np.loadtxt("../CollectedData/new_data/ThaiVui2.txt")
vui3 = np.loadtxt("../CollectedData/new_data/QuangVui3.txt")
vui4 = np.loadtxt("../CollectedData/new_data/SonVui4.txt")
vui5 = np.loadtxt("../CollectedData/new_data/ThanhfVui2.txt")
vui6 = np.loadtxt("../CollectedData/new_data/ThanhfVui.txt")

vui = np.concatenate((vui1, vui2, vui3, vui4, vui5))
print(vui.shape)

buon1 = np.loadtxt("../CollectedData/new_data/BachBuon.txt")
buon2 = np.loadtxt("../CollectedData/new_data/ThaiBuon.txt")
buon3 = np.loadtxt("../CollectedData/new_data/SonBuon3.txt")
buon4 = np.loadtxt("../CollectedData/new_data/SonBuon4.txt")
buon5 = np.loadtxt("../CollectedData/new_data/ThanhfBuon.txt")


buon = np.concatenate((buon1, buon2, buon3, buon4, buon5))

calm1 = np.loadtxt("../CollectedData/new_data/ThaiCalm.txt")
calm2 = np.loadtxt("../CollectedData/new_data/BachCalm.txt")
calm3 = np.loadtxt("../CollectedData/new_data/QuangCalm3.txt")
calm4 = np.loadtxt("../CollectedData/new_data/ThaiCalm2.txt")
calm5 = np.loadtxt("../CollectedData/new_data/ThanhfCalm.txt")


calm = np.concatenate((calm1, calm2, calm3, calm4, calm5))


