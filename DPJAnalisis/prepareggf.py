#
# Script to add jet30 variables to ggF samples
# Runs for list of files with same did and 
# outputs a single file
# The script was based on vbfdresser_mp.py
# 

from ROOT import TFile, TTree, TH1D, TLorentzVector, TChain
import sys, os, math
import numpy as np
from array import array

def prepareggf(files, outputname, tree):
    print("Processing", files[0], tree, outputname)

    ttree = TChain(tree)
    for file in files:
        ttree.Add(file)
  
    if not ttree:
        print("no valid input tree, exitting")
        return 
    
    tfile_out = TFile(outputname, "recreate")
   
    ttree_out = ttree.CloneTree(0)
    ttree_out.SetDirectory(tfile_out)

    #declare variables
    m_scale1fb = array("f", [0])
    m_intLumi = array("f", [0])

    m_jet1_pt = array("f",[0])
    m_jet1_eta = array("f",[0])
    m_jet1_phi = array("f",[0])
    m_jet1_e = array("f",[0])
    m_jet2_pt = array("f",[0])
    m_jet2_eta = array("f",[0])
    m_jet2_phi = array("f",[0])
    m_jet2_e = array("f",[0])
    m_mjj = array("f",[0])
    m_detajj = array("f",[0])

    # list of jets with pt>30 GeV
    m_jets_pt = array("f", 100*[0.])
    m_jets_eta = array("f", 100*[0])
    m_jets_phi = array("f", 100*[0])
    m_jets_e = array("f", 100*[0])

    # list of jets30 sorted by pt
    m_jets_pt_sorted = array("f", 100*[0])
    m_jets_eta_sorted = array("f", 100*[0])
    m_jets_phi_sorted = array("f", 100*[0])
    m_jets_e_sorted = array("f", 100*[0])
    
    # number of jets with pt>30 GeV
    m_njet30 = array("i", [0])
    
    # m_hasBjet = array("i",[0])

    # leading di-jet information
    m_signetajj = array("f",[0])
    m_dphijj = array("f",[0])



    # Declare the branches of these new variables
    scale1fb_branch = ttree_out.Branch( 'scale1fb', m_scale1fb, 'scale1fb/F' )
    intLumi_branch = ttree_out.Branch( 'intLumi', m_intLumi, 'intLumi/F' )

    jet1_pt_branch = ttree_out.Branch( 'jet1_pt', m_jet1_pt, 'jet1_pt/F' )
    jet1_eta_branch = ttree_out.Branch( 'jet1_eta', m_jet1_eta, 'jet1_eta/F' )
    jet1_phi_branch = ttree_out.Branch( 'jet1_phi', m_jet1_phi, 'jet1_phi/F' )
    jet1_e_branch = ttree_out.Branch( 'jet1_e', m_jet1_e, 'jet1_e/F' )

    njet30_branch = ttree_out.Branch('njet30', m_njet30, 'njet30/I')

    jet2_pt_branch = ttree_out.Branch( 'jet2_pt', m_jet2_pt, 'jet2_pt/F' )
    jet2_eta_branch = ttree_out.Branch( 'jet2_eta', m_jet2_eta, 'jet2_eta/F' )
    jet2_phi_branch = ttree_out.Branch( 'jet2_phi', m_jet2_phi, 'jet2_phi/F' )
    jet2_e_branch = ttree_out.Branch( 'jet2_e', m_jet2_e, 'jet2_e/F' )

    m_jj_branch = ttree_out.Branch('mjj', m_mjj, 'mjj/F')
    m_detajj_branch = ttree_out.Branch('detajj', m_detajj, 'detajj/F')

    # these branches are arrays of variable lenghts (lenght njet30)
    jets_pt_branch = ttree_out.Branch( 'jets_pt', m_jets_pt, 'jets_pt[njet30]/F' )
    jets_eta_branch = ttree_out.Branch( 'jets_eta', m_jets_eta, 'jets_eta[njet30]/F' )
    jets_phi_branch = ttree_out.Branch( 'jets_phi', m_jets_phi, 'jets_phi[njet30]/F' )
    jets_e_branch = ttree_out.Branch( 'jets_e', m_jets_e, 'jets_e[njet30]/F' )

    jets_pt_sorted_branch = ttree_out.Branch( 'jets_pt_sorted', m_jets_pt_sorted, 'jets_pt_sorted[njet30]/F' )
    jets_eta_sorted_branch = ttree_out.Branch( 'jets_eta_sorted', m_jets_eta_sorted, 'jets_eta_sorted[njet30]/F' )
    jets_phi_sorted_branch = ttree_out.Branch( 'jets_phi_sorted', m_jets_phi_sorted, 'jets_phi_sorted[njet30]/F' )
    jets_e_sorted_branch = ttree_out.Branch( 'jets_e_sorted', m_jets_e_sorted, 'jets_e_sorted[njet30]/F' )
    
    # hasBjet_branch = ttree_out.Branch( 'hasBjet', m_hasBjet, 'hasBjet/I' )

    # di-jet kinematics
    signetajj_branch = ttree_out.Branch( 'signetajj', m_signetajj, 'signetajj/F' )
    dphijj_branch = ttree_out.Branch( 'dphijj', m_dphijj, 'dphijj/F' )
   

    # special for the dijet samples
    # https://twiki.cern.ch/twiki/bin/view/AtlasProtected/JetEtMissMCSamples
    # BinContent(1): nubmer of total events, use this for the JZW slicing, DSID  361020 (JZ0W) - 361032 (JZ12W)
    # BinContent(2): sum of weights, use this for JZWithSW slicing 
    # sumOfWeights_nEvents = tfile.numEvents.GetBinContent(1)
    # this is the default sum of weight 
    # sumOfWeights = tfile.numEvents.GetBinContent(2)
    for entry in range(ttree.GetEntries()):

        ttree.GetEntry(entry)
        sumOfWeights = ttree.sumWeightPRW
    
        if entry > 0 and entry%10000==0:
            print("Processed {} of {} entries".format(entry,ttree.GetEntries()))

        ttree_out.GetEntry(entry)
        
        xs = ttree.amiXsection
        weight = ttree.weight
        if ttree.dsid in range(500757, 500764):     xs = 3.78*0.1 

        if ttree.dsid in range(508885,508903): xs = 48.61*0.1 # cross section * branching ratio for ggF
        m_scale1fb[0] = 1

        if ttree.isMC:
            m_scale1fb[0] = xs*1000.*weight*ttree.filterEff*ttree.kFactor/sumOfWeights
        
        # assign integrated luminosity according to luminosity for MC
        # this variable will be used to scale the MC predictions together with scale1fb
        intLumi = 1
        if ttree.isMC:
            #VBF ranges for RunNumber, not using
            # mc16a: 2015+2016
            if ttree.RunNumber in range (266904, 311482): kkkk=0 #intLumi = 36.1 
            # mc16d: 2017
            elif ttree.RunNumber in range (324320, 341649): kkkk=0 #intLumi = 44.3 
            # mc16e: 2018
            elif ttree.RunNumber in range (348197, 364485): kkkk=0 #intLumi = 58.45
            else: 
                if ttree.weight!=0:
                    print(f"Run number not in VBF ranges. RunNumber: {ttree.RunNumber}")
                    print(f"weight: {ttree.weight}")

            #ggF ranges for RunNumber, currently using
            # mc16a: 2015+2016
            if (ttree.RunNumber in range (267069, 284669)) or (ttree.RunNumber in range(296938,311563)): intLumi = 36.1 
            # mc16d: 2017
            elif ttree.RunNumber in range (324317, 341650): intLumi = 44.3 
            # mc16e: 2018
            elif ttree.RunNumber in range (348154, 364486): intLumi = 58.45
            else: 
                if ttree.weight!=0:
                    print(f"Run number not in ggF ranges. RunNumber: {ttree.RunNumber}")
                    print(f"weight: {ttree.weight}")


            # Special MC weight treatment if needed
            # e.g. if mc16d is missing, one can add the missing lumi for mc16d to the mc16a sample 
            # if ttree.dsid == 500757 and ttree.RunNumber in range (266904, 311481): intLumi = 36.1 + 44.3 
            #if ttree.dsid == 363237 and ttree.RunNumber in range (266904, 311481): intLumi = 36.1 + 44.3 
            #if ttree.dsid == 312461 and ttree.RunNumber in range (324320, 341649): intLumi =  44.3  + 58.45
        m_intLumi[0] = intLumi    

        # Set initial values of the output branches
        m_jet1_pt[0] = -999
        m_jet1_eta[0] = -999
        m_jet1_phi[0] = -999
        m_jet1_e[0] = -999
        m_jet2_pt[0] = -999
        m_jet2_eta[0] = -999
        m_jet2_phi[0] = -999
        m_jet2_e[0] = -999
        m_mjj[0] = -999
        m_detajj[0] = -999

        # dijet 
        m_signetajj[0] = -999  
        m_dphijj[0] = -999

       
        # # b-jets
        # m_hasBjet[0] =  0 

        njet30 = 0

        # jet30 inexes in jet_cal_pt
        jet30_idx = []

        for i in range(ttree.jet_cal_pt.size()):
            # if jet_pt>30GeV
            if bool(ttree.jet_cal_isSTDOR.at(i))==1 and ttree.jet_cal_pt.at(i)>30000:
                njet30+=1
                jet30_idx.append(i)
                m_jets_pt[njet30-1] = ttree.jet_cal_pt.at(i)
                m_jets_eta[njet30-1] = ttree.jet_cal_eta.at(i)
                m_jets_phi[njet30-1] = ttree.jet_cal_phi.at(i)
                m_jets_e[njet30-1] = ttree.jet_cal_e.at(i)

        # I dont think this line is necessary but i can't remember why I put it there
        jet30_idx = np.array(jet30_idx)

        #sort jets lists
        #sorted indexes of jets in m_jets lists
        sorted_jet30_idx = np.argsort(-1*np.array(m_jets_pt[:njet30]))
        #sorted indexes of jets in jet_cal lists
        sorted_jets_idx = [] 
        for i in range(njet30):
            sorted_jets_idx.append(jet30_idx[sorted_jet30_idx[i]])
            m_jets_pt_sorted[i] = m_jets_pt[sorted_jet30_idx[i]]
            m_jets_eta_sorted[i] = m_jets_eta[sorted_jet30_idx[i]]
            m_jets_phi_sorted[i] = m_jets_phi[sorted_jet30_idx[i]]
            m_jets_e_sorted[i] = m_jets_e[sorted_jet30_idx[i]]

        m_njet30[0] = njet30

        p4_j1 = TLorentzVector(0, 0, 0, 0)
        p4_j2 = TLorentzVector(0, 0, 0, 0)
        if njet30>0:
            idx_jet1 = int(sorted_jets_idx[0])
            p4_j1.SetPtEtaPhiE(ttree.jet_cal_pt.at(idx_jet1), ttree.jet_cal_eta.at(idx_jet1), ttree.jet_cal_phi.at(idx_jet1), ttree.jet_cal_e.at(idx_jet1))
            m_jet1_pt[0] = p4_j1.Pt()
            m_jet1_eta[0] = p4_j1.Eta()
            m_jet1_phi[0] = p4_j1.Phi()
            m_jet1_e[0] = p4_j1.E()
            # could add mass for single jet uncommenting the line below
            # m_mjj[0] = p4_j1.M()
        if njet30>1:
            idx_jet2 = int(sorted_jets_idx[1])
            p4_j2.SetPtEtaPhiE(ttree.jet_cal_pt.at(idx_jet2), ttree.jet_cal_eta.at(idx_jet2), ttree.jet_cal_phi.at(idx_jet2), ttree.jet_cal_e.at(idx_jet2))
            m_jet2_pt[0] = p4_j2.Pt()
            m_jet2_eta[0] = p4_j2.Eta()
            m_jet2_phi[0] = p4_j2.Phi()
            m_jet2_e[0] = p4_j2.E()
            m_dphijj[0] = p4_j1.DeltaPhi(p4_j2)
            m_signetajj[0] = 1
            m_mjj[0] = (p4_j1+p4_j2).M()
            m_detajj[0] = abs(p4_j1.Eta()-p4_j2.Eta())
            if p4_j1.Eta()*p4_j2.Eta() < 0: m_signetajj[0] = -1 
        
        # fill the output tree
        ttree_out.Fill()

    print ("")
    tfile_out.Write()
    tfile_out.Close()

if __name__=='__main__':
    #Define here the list of files you want to merge 
    files = ["test1.root", "test1.root"]
    outputname = "test_merge.root"
    tree = "miniT"
    prepareggf(files, outputname, tree)        
    print("done! :)")