from Modules.Settings.settings import *


def probabilities(pref=P_REFLECTION, petch=P_ETCH, pdep=P_DEPOSIT, padsorb=P_ADSORB):
    # assert pref.shape
    # if (pref).equal(P_REFLECTION):
    #     print('Using default Reflection Probabilities', pref)
    # if (petch).equal(P_ETCH):
    #     print('Using default Etching Probabilities', petch)
    # if (pdep).equal(P_DEPOSIT):
    #     print('Using default Deposition Probabilities', pdep)
    # if (padsorb).equal(P_ADSORB):
    #     print('Using default Adsorption Probabilities', padsorb)

    return pref, petch, pdep, padsorb