import glob
import numpy as np
from scipy import sparse
from scipy import io

def run_one_prr(allReports, allReactions, modelIdx=3, save=True):
    modelOutcome = 0.0
    
    if len(modelIdx) == 1:
        score_files = glob.glob('scores_lrc_*__'+str(modelIdx[0])+'.npy')
        print ('Found '+str(len(score_files))+' score files.')

        scores_keep = np.zeros(shape=(np.load(score_files[0]).shape[0]))
        sf = 0

        for score_file in score_files:
            log_file = score_file.replace('scores','log')
            log_file = np.load(log_file,encoding='latin1').item()
            if log_file['auc'] > 0.5:
                sf += 1
                scores_keep = scores_keep + np.load(score_file)
        scores_keep = scores_keep/float(sf)

        modelOutcome = sparse.csc_matrix.todense(allReports[:,modelIdx[0]])

    else:
        modelString = ''
        for idx in modelIdx:
            modelString = modelString + '_' + str(idx)

        print (modelString)
        scores_keep = np.load('scores_lrc_' + modelString + '.npy')

        modelOutcome = allReports[:,modelIdx]
        modelOutcome = np.sum(modelOutcome, axis=1)
        modelOutcome[np.where(modelOutcome == len(modelIdx))[0]] = 1

    A = np.zeros(shape=(1,allReactions.shape[1]))
    AplusB = 0

    C = np.zeros(shape=(1,allReactions.shape[1]))
    CplusD = 0

    for i in np.arange(0.0,1.0,0.2):
        low = i
        high = i+0.20

        lo_ind = np.where(scores_keep > low)
        hi_ind = np.where(scores_keep <= high)

        inds = set(lo_ind[0]) & set(hi_ind[0])

        AplusB_inds = np.where(modelOutcome == 1)[0]
        AplusB_inds = set(AplusB_inds) & set(inds)

        A = A + np.sum(allReactions[list(AplusB_inds)],axis=0)
        AplusB = AplusB + len(AplusB_inds)

        CplusD_inds = np.where(modelOutcome == 0)[0]
        CplusD_inds = set(CplusD_inds) & set(inds)

        if len(CplusD_inds) == 0:
            continue

        CplusD_inds = np.random.choice(list(CplusD_inds),10*len(AplusB_inds))

        C = C + np.sum(allReactions[list(CplusD_inds)],axis=0)
        CplusD = CplusD + len(CplusD_inds)
        
    PRRs = (A/AplusB) / (C/CplusD)
    PRR_s = np.sqrt((1/A) + (1/C) -(1/AplusB)-(1/CplusD))
    
    
    if len(modelIdx) == 1:
        np.save('PRR__'+str(modelIdx[0])+'.npy',PRRs)
        np.save('PRRs__'+str(modelIdx[0])+'.npy',PRR_s)
    else:
        modelString = ''
        for idx in modelIdx:
            modelString += '_' + str(idx)
        if save==True:
            np.save('PRR_'+modelString+'.npy',PRRs)
            np.save('PRRs_'+modelString+'.npy',PRR_s)

    return (PRRs, PRR_s)



        

    

    

def main():
    for reportblock in range(0,50):
        #if args.verbose:
            #print("Report Block: {0} out of 49.".format(reportblock))
        thisReportBlock = np.load("/data/nsides_scores/data/AEOLUS_all_reports_IN_"+str(reportblock)+".npy").item()
        if reportblock == 0:
            allReports = thisReportBlock
        else:
            allReports = sparse.vstack((allReports,thisReportBlock))
        print ("Processed ",reportblock)
    
    allReports = allReports.tocsc()

    allReactions = io.mmread('/data/nsides_scores/data/AEOLUS_all_reports_IN_alloutcomes.mtx')
    allReactions = allReactions.tocsc()
    allReports = allReports[:allReactions.shape[0]]
    
    #run_one_prr(allReports,allReactions,modelIdx=[1460,4353])
    #run_one_prr(allReports,allReactions,modelIdx=[3])

    all_model_indices = glob.glob('scores_lrc_1__*.npy')
    all_model_indices = [int(x.split('_')[4].strip('.npy')) for x in all_model_indices]

    for x in all_model_indices:
        try:
            run_one_prr(allReports, allReactions, modelIdx=[x], save=True)
        except:
            print ("Failed on model ",x)


if __name__=='__main__':
    main()
    
