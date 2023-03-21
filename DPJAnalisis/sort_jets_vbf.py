from pandas import cut
import uproot
import awkward as aw
import yaml
from tabulate import tabulate
import math
from IPython.display import Javascript as js, clear_output
from scipy.stats import distributions
import numpy as np
def sort_jets(proccess: str, config: dict, show_stat: bool = True) -> aw.Array:
    """
    Sorts the jets information in cal_jets by pt, from high to low
    Returns a dictionary with the sorted information by event
    """

    if show_stat:
        print("-----------------------------------------------------------------------------------------------------------")
        print(f"                                   The input process is {proccess}")
        print("-----------------------------------------------------------------------------------------------------------")
    # Get config file

    cuts_config = config['cuts_config']
    branches = config['ttree']['recover_branches']
    files = config['ttree']['files']

    #set up the root folders
    entryfile: uproot.ReadOnlyDirectory = uproot.open(files[proccess])
    minitree = entryfile['miniT']
    jet_branches = ["jet_cal_pt", "jet_cal_isSTDOR", "jet_cal_eta", "jet_cal_phi", "jet_cal_e"]
    # print("before array cache")
    events_array = minitree.arrays(branches+jet_branches, array_cache="200MB")    
    # print("after array")
    nevents = len(events_array)
    # new branches
    jets_pt_sorted = []
    jets_eta_sorted = []
    jets_phi_sorted = []
    jets_e_sorted = []

    for entry in range(nevents):
    # for entry in range(5):
        if entry > 0 and entry%10000==0:
            print("Processed {} of {} entries".format(entry,len(events_array)))
        event = events_array[entry]
        njet30 = event["njet30"]

        # jet30 inexes in jet_cal_pt
        jet30_idx = []
        counter = 0
        m_jets_pt = []
        m_jets_eta = []
        m_jets_phi = []
        m_jets_e = []
        for i in range(len(event["jet_cal_pt"])):
            # if jet_pt>30GeV
            if bool(event["jet_cal_isSTDOR"][i])==1 and event["jet_cal_pt"][i]>30000:
                jet30_idx.append(i)
                counter+=1
                m_jets_pt.append(event["jet_cal_pt"][i])
                m_jets_eta.append(event["jet_cal_eta"][i])
                m_jets_phi.append(event["jet_cal_phi"][i])
                m_jets_e.append(event["jet_cal_e"][i])
        if counter != njet30:
            print(f"Algo paso aqui, njet30: {njet30}, manual counter: {counter}")
        # I dont think this line is necessary but i can't remember why I put it there
        jet30_idx = np.array(jet30_idx)

        #sort jets lists
        #sorted indexes of jets in m_jets lists
        sorted_jet30_idx = np.argsort(-1*np.array(m_jets_pt[:njet30]))
        #sorted indexes of jets in jet_cal lists
        sorted_jets_idx = [] 
        m_jets_pt_sorted = []
        m_jets_eta_sorted = []
        m_jets_phi_sorted = []
        m_jets_e_sorted = []
        for i in range(njet30):
            sorted_jets_idx.append(jet30_idx[sorted_jet30_idx[i]])
            m_jets_pt_sorted.append(m_jets_pt[sorted_jet30_idx[i]]) 
            m_jets_eta_sorted.append(m_jets_eta[sorted_jet30_idx[i]])
            m_jets_phi_sorted.append(m_jets_phi[sorted_jet30_idx[i]])
            m_jets_e_sorted.append(m_jets_e[sorted_jet30_idx[i]])
        
        jets_pt_sorted.append(m_jets_pt_sorted)
        jets_eta_sorted.append(m_jets_eta_sorted)
        jets_phi_sorted.append(m_jets_phi_sorted)
        jets_e_sorted.append(m_jets_e_sorted)

        r = {"jets_pt_sorted": jets_pt_sorted,
            "jets_eta_sorted": jets_eta_sorted,
            "jets_phi_sorted": jets_phi_sorted,
            "jets_e_sorted": jets_e_sorted
        }

    return r, aw.Array(aw.flatten(events_array,axis=0))
    
# if __name__=='__main__':
    # with open('config.yaml') as conf_file:
    #     config = yaml.load(conf_file, Loader=yaml.Loader) 
    # array = sort_jets("VBF_500757", config)
    # print(array["weight"])
    # print(array["jets_e_sorted"])
