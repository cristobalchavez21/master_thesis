from pandas import cut
import uproot
import awkward as aw
import yaml
from tabulate import tabulate
import math
from IPython.display import Javascript as js, clear_output
from scipy.stats import distributions
import numpy as np

#Function that speaks. Used to know when cells are done running
def speak(text):
    
    # Escape single quotes
    text = text.replace("'", r"\'")
    display(js(f'''
    if(window.speechSynthesis) {{
        var synth = window.speechSynthesis;
        synth.speak(new window.SpeechSynthesisUtterance('{text}'));
    }}
    '''))
    # Clear the JS so that the notebook doesn't speak again when reopened/refreshed
    clear_output(False)
def ks_weighted(data1, data2, wei1, wei2, alternative='two-sided'):
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1)/sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2)/sum(wei2)])
    cdf1we = cwei1[np.searchsorted(data1, data, side='right')]
    cdf2we = cwei2[np.searchsorted(data2, data, side='right')]
    d = np.max(np.abs(cdf1we - cdf2we))
    # calculate p-value
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    m, n = sorted([float(n1), float(n2)], reverse=True)
    en = m * n / (m + n)
    if alternative == 'two-sided':
        prob = distributions.kstwo.sf(d, np.round(en))
    else:
        z = np.sqrt(en) * d
        # Use Hodges' suggested approximation Eqn 5.3
        # Requires m to be the larger of (n1, n2)
        expt = -2 * z**2 - 2 * z * (m + 2*n)/np.sqrt(m*n*(m+n))/3.0
        prob = np.exp(expt)
    return d, prob

def set_stats(stat: dict, cut: str, array: aw.Array, do_stat: bool):
    """
    Calculate the stats an returned it in the stat dict
    """
    if do_stat:
        event_sum = aw.count(array['weight'])
        weight_sum = aw.sum(array['weight'])
        scale1fb_times_intlumi = aw.sum(array['scale1fb'] * array['intLumi'])
        stat[cut] = (event_sum, weight_sum, scale1fb_times_intlumi)
        return stat
    else:
        return stat


def get_minit_from_procces_file(proccess: str, config: dict, show_stat: bool = True, do_cut: int = 12, with_VBF_cut: bool=True) -> aw.Array:
    """
    Gets the miniT array from the config file and returns and akward array with the branches in recover_branches
    thats in the config.yaml file

    proccess: string with the name of the proccess to be read
    config: dict from config.yaml
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
        if do_cut>1:
            do_cut = 1
    files = config['ttree']['files']
    print(f"file: {files[proccess]}")
    #set up the root folders
    entryfile: uproot.ReadOnlyDirectory = uproot.open(files[proccess])
    minitree = entryfile['miniT']
    print("file opened")

    events_array = minitree.arrays(branches, array_cache="1GB")
    print("array done")
    stats = {}
    # the stats are {"cut": (# of events, sum weight, sum of scale1fb*intLumi)}    
    set_stats(stats,"Baseline pre-selection", events_array, do_stat=show_stat)

    # CUT 1 VBF filter
    if do_cut >= 1:
        if with_VBF_cut:
            array_masked = aw.mask(events_array,events_array['mjj'] >= cuts_config['min_mjj'])
            # array_masked = aw.mask(array_masked, array_masked['njet30'] >= cuts_config['min_njet30'])
            # array_masked = aw.mask(array_masked, array_masked['detajj'] >= cuts_config['min_detajj'])
            set_stats(stats, "VBF filter cut", array_masked, do_stat=show_stat)
        else:
            array_masked = events_array
    
    # CUT 2 Jet pre-selection
    if do_cut >= 2:
        array_masked = aw.mask(array_masked,array_masked['MET'] > cuts_config['min_MET'])
        set_stats(stats, "MET cut", array_masked, do_stat=show_stat)
    
    # CUT 3 MET trigger
    if do_cut >= 3:
        array_masked = aw.mask(array_masked,array_masked['metTrig'] == True)
        set_stats(stats, "MET trigger pre-selection", array_masked, do_stat=show_stat)

    # CUT 4 number LJ-muons 
    if do_cut >= 4:
        array_masked = aw.mask(array_masked,array_masked['nLJmus20'] == 0)
        set_stats(stats, "Number of LJ-muons", array_masked, do_stat=show_stat)
    
    # CUT 5 min number of LJ-jets
    if do_cut >= 5:
        array_masked = aw.mask(array_masked,array_masked['nLJjets20'] >= cuts_config['min_nLJjets20'])
        set_stats(stats, "Min. number of LJ-jets", array_masked, do_stat=show_stat)
    
    # CUT 6 BIB tagger for LJjet1
    if do_cut >= 6:
        array_masked = aw.mask(array_masked,array_masked['LJjet1_BIBtagger'] >= cuts_config['min_LJjet1_BIBtagger'])
        set_stats(stats, "Min. BIB tagger LJ-leadi jet", array_masked, do_stat=show_stat)
    
    # CUT 7 Prompt leptons and electrons veto
    if do_cut >= 7:
        array_masked = aw.mask(array_masked,array_masked['neleSignal'] == cuts_config['min_neleSignal'])
        array_masked = aw.mask(array_masked,array_masked['nmuSignal'] == cuts_config['min_nmuSignal'])
        set_stats(stats, "Prompt leptons and electrons veto", array_masked, do_stat=show_stat)
    
    # CUT 8 Number of b-jets
    if do_cut >= 8:
        array_masked = aw.mask(array_masked,array_masked['hasBjet'] == False)
        set_stats(stats, "Number of b-jets", array_masked, do_stat=show_stat)
    
    # CUT 9 Min. angle between jet-MET
    if do_cut >= 9:
        array_masked = aw.mask(array_masked,array_masked['min_dphi_jetmet'] >= cuts_config['min_dphijetmet'])
        set_stats(stats, "Min. angle between jet-MET", array_masked, do_stat=show_stat)
    
    # CUT 10 Max. jvt tagger LJ-leading
    if do_cut >= 10:
        array_masked = aw.mask(array_masked,array_masked['LJjet1_jvt'] <= cuts_config['max_LJjet1_jvt'])
        set_stats(stats, "Max. jvt tagger LJ-leading", array_masked, do_stat=show_stat)

    # CUT 11 Min. gapRatio LJ-leadin jet
    if do_cut >= 11:
        array_masked = aw.mask(array_masked,array_masked['LJjet1_gapRatio'] >= cuts_config['min_LJjet1_gapRatio'])
        set_stats(stats, "Min. gapRatio LJ-leadin jet", array_masked, do_stat=show_stat)
    
    # CUT 12 Min. DPJtagg and deltaphijj
    if do_cut >= 12:
        array_masked = aw.mask(array_masked,array_masked['LJjet1_DPJtagger'] >= cuts_config['min_LJjet1_DPJtagger'])
        array_masked = aw.mask(array_masked,array_masked['dphijj'] <= cuts_config['max_dphijj'])
        array_masked = aw.mask(array_masked,array_masked['dphijj'] >= -cuts_config['max_dphijj'])
        set_stats(stats, "Min. DPJtagg and deltaphijj", array_masked, do_stat=show_stat)
    
    
    # print stats
    if show_stat:
        stats_arr = [[k, v[0], v[1], v[2]] for k,v in stats.items()]
        print(tabulate(stats_arr, headers=["Cut", "Events", "Sum of weights", "Sum of intLumi*scale1fb"], tablefmt="orgtbl"))
    
    if do_cut==0:
        return events_array

    return aw.Array(aw.flatten(array_masked,axis=0))
    
    
def save_array_to_file(array: aw.Array, filename: str, config: dict, extra_branch: list = []):
    """
    Saves a akward array to a root file with the branches given as recover_branches in the config file

    array: aw.Array
    filename: string
    config: dict from config.yaml
    """
    #Save the new array into a root file
    with uproot.recreate(f"output/{filename}") as outfile:
        branches = config['ttree']['recover_branches']
        if "GGF" in filename:
            branches = config['ttree']['recover_branches_ggf']
        dic_tree = {}
        dic_jets = {}
        for br in branches + extra_branch:
            if "sorted" in br:
                dic_jets[br.replace("jets_","")] = array[br]
            else:
                dic_tree[br] = array[br]
        dic_tree["jets"] = aw.zip(dic_jets)

        outfile['miniT'] = dic_tree

def get_cutted_files(file: str, config: dict,):
    """
    Returns a list of cutted files

    file: string representing wich sample
    config: dict from config.yaml
    """
    
    branches = config['ttree']['recover_branches']

    #set up the root folders
    entryfile: uproot.ReadOnlyDirectory = uproot.open(file)
    minitree = entryfile['miniT']


    return minitree.arrays(branches)

def significance(sgn: float,bkg: float) -> float:
    return math.sqrt(2 * abs( (sgn+bkg) * math.log(1 + ( sgn / bkg )) - sgn))

if __name__ == "__main__":
    with open('config.yaml') as conf_file:
        config = yaml.load(conf_file, Loader=yaml.Loader) 
    for procees in config['ttree']['files'].keys():
        array = get_minit_from_procces_file(procees, config)
        save_array_to_file(array, f"{procees}_all_cuts.root", config)
