import uproot
import awkward as aw
import numpy as np
import vector

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
    if "GGF" in proccess:
        branches = config['ttree']['recover_branches_ggf']
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
    jet1_pt = []
    jet1_eta = []
    jet1_phi = []
    jet1_e = []
    jet2_pt= []
    jet2_eta = []
    jet2_phi = []
    jet2_e = []
    dphijj = []
    signetajj = []
    mjj = []
    detajj = []
    njet30_branch = []
    scale1fb = []
    intLumi_branch = []


    for entry in range(nevents):
    # for entry in range(5):
        if entry > 0 and entry%10000==0:
            print("Processed {} of {} entries".format(entry,len(events_array)))
        event = events_array[entry]

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
        if "VBF" in proccess:
            njet30 = event["njet30"]
            if counter != njet30:
                print(f"Algo paso aqui, njet30: {njet30}, counter: {counter}")
        # I dont think this line is necessary but i can't remember why I put it there
        jet30_idx = np.array(jet30_idx)

        #sort jets lists
        #sorted indexes of jets in m_jets lists
        sorted_jet30_idx = np.argsort(-1*np.array(m_jets_pt[:counter]))
        #sorted indexes of jets in jet_cal lists
        sorted_jets_idx = [] 
        m_jets_pt_sorted = []
        m_jets_eta_sorted = []
        m_jets_phi_sorted = []
        m_jets_e_sorted = []
        for i in range(counter):
            sorted_jets_idx.append(jet30_idx[sorted_jet30_idx[i]])
            m_jets_pt_sorted.append(m_jets_pt[sorted_jet30_idx[i]]) 
            m_jets_eta_sorted.append(m_jets_eta[sorted_jet30_idx[i]])
            m_jets_phi_sorted.append(m_jets_phi[sorted_jet30_idx[i]])
            m_jets_e_sorted.append(m_jets_e[sorted_jet30_idx[i]])
        
        jets_pt_sorted.append(m_jets_pt_sorted)
        jets_eta_sorted.append(m_jets_eta_sorted)
        jets_phi_sorted.append(m_jets_phi_sorted)
        jets_e_sorted.append(m_jets_e_sorted)

        if "GGF" in proccess:
            m_jet1_pt = -999
            m_jet1_eta = -999
            m_jet1_phi = -999
            m_jet1_e = -999
            m_jet2_pt= -999
            m_jet2_eta = -999
            m_jet2_phi = -999
            m_jet2_e = -999
            m_dphijj = -999
            m_signetajj = -999
            m_mjj = -999
            m_detajj = -999
            sumOfWeights = event["sumWeightPRW"]
            xs = event["amiXsection"]
            weight = event["weight"]
            if event["dsid"] in range(508885,508903): xs = 48.61*0.1 # cross section * branching ratio for ggF
            m_scale1fb = xs*1000.*weight*event["filterEff"]*event["kFactor"]/sumOfWeights

            #ggF ranges for RunNumber, currently using
            # mc16a: 2015+2016
            if (event["RunNumber"] in range (267069, 284669)) or (event["RunNumber"] in range(296938,311563)): intLumi = 36.1 
            # mc16d: 2017
            elif event["RunNumber"] in range (324317, 341650): intLumi = 44.3 
            # mc16e: 2018
            elif event["RunNumber"] in range (348154, 364486): intLumi = 58.45
            else: 
                if event["weight"]!=0:
                    print(f"Run number not in ggF ranges. RunNumber: {event['RunNumber']}")
                    print(f"weight: {event['weight']}")

            if counter>0:
                p4_j1 = vector.obj(pt=m_jets_pt_sorted[0], phi=m_jets_phi_sorted[0], eta=m_jets_eta_sorted[0], E=m_jets_e_sorted[0])
                m_jet1_pt = p4_j1.pt
                m_jet1_eta = p4_j1.eta
                m_jet1_phi = p4_j1.phi
                m_jet1_e = p4_j1.E
                
            if counter>1:
                p4_j2 = vector.obj(pt=m_jets_pt_sorted[1], phi=m_jets_phi_sorted[1], eta=m_jets_eta_sorted[1], E=m_jets_e_sorted[1])
                m_jet2_pt = p4_j2.pt
                m_jet2_eta = p4_j2.eta
                m_jet2_phi = p4_j2.phi
                m_jet2_e = p4_j2.E
                m_dphijj = p4_j1.deltaphi(p4_j2)
                m_signetajj = 1
                m_mjj = (p4_j1+p4_j2).M
                m_detajj = abs(p4_j1.eta-p4_j2.eta)
                if p4_j1.eta*p4_j2.eta < 0: m_signetajj = -1 
            jet1_pt.append(m_jet1_pt)
            jet1_eta.append(m_jet1_eta)
            jet1_phi.append(m_jet1_phi)
            jet1_e.append(m_jet1_e)
            jet2_pt.append(m_jet2_phi)
            jet2_eta.append(m_jet2_eta)
            jet2_phi.append(m_jet2_phi)
            jet2_e.append(m_jet2_e)
            dphijj.append(m_dphijj)
            signetajj.append(m_signetajj)
            mjj.append(m_mjj)
            detajj.append(m_detajj)
            njet30_branch.append(counter)
            scale1fb.append(m_scale1fb)
            intLumi_branch.append(intLumi)
            
    if "GGF" in proccess:
        r = {"jets_pt_sorted": jets_pt_sorted,
            "jets_eta_sorted": jets_eta_sorted,
            "jets_phi_sorted": jets_phi_sorted,
            "jets_e_sorted": jets_e_sorted,
            "jet1_pt": jet1_pt,
            "jet1_eta": jet1_eta,
            "jet1_phi": jet1_phi,
            "jet1_e": jet2_e,
            "jet2_pt": jet2_pt,
            "jet2_eta": jet2_eta,
            "jet2_phi": jet2_phi,
            "jet2_e": jet2_e,
            "dphijj": dphijj,
            "signetajj": signetajj,
            "mjj": mjj,
            "detajj": detajj,
            "njet30": njet30_branch,
            "scale1fb": scale1fb,
            "intLumi": intLumi_branch
        }
    else:
        r = {"jets_pt_sorted": jets_pt_sorted,
            "jets_eta_sorted": jets_eta_sorted,
            "jets_phi_sorted": jets_phi_sorted,
            "jets_e_sorted": jets_e_sorted
        }

    return r
    
# if __name__=='__main__':
    # with open('config.yaml') as conf_file:
    #     config = yaml.load(conf_file, Loader=yaml.Loader) 
    # array = sort_jets("VBF_500757", config)
    # print(array["weight"])
    # print(array["jets_e_sorted"])
