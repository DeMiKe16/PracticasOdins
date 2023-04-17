import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

n_parties = 50
dir_base = os.getcwd()
rondas = 15
nodesMal = []
acc_media_total = np.zeros(rondas)
recall_media_total = np.zeros(rondas)
precision_media_total = np.zeros(rondas)
f1score_media_total = np.zeros(rondas)
fpr_media_total = np.zeros(rondas)
mcc_media_total = np.zeros(rondas)
cks_media_total = np.zeros(rondas)

rango = range(1, n_parties+1)
#rango = [1,430,431,432,433]
for i in rango:
    party = pd.read_csv(dir_base + "/metricas_parties/history_" + str(i) + ".csv")
    acc = np.array(party["Accuracy"])
    recall = np.array(party["Recall"])
    precision = np.array(party["Precision"])
    f1_score = np.array(party["F1_score"])
    fpr = np.array(party["FPR"].tail(20))
    mcc = np.array(party['Matthew_Correlation_coefficient'])
    cks = np.array(party["Cohen_Kappa_Score"])

    acc_media_total = np.add(acc_media_total, acc)
    recall_media_total = np.add(recall_media_total, recall)
    precision_media_total = np.add(precision_media_total, precision)
    f1score_media_total = np.add(f1score_media_total, f1_score)
    fpr_media_total = np.add(fpr_media_total, fpr)
    mcc_media_total = np.add(mcc_media_total, mcc)
    cks_media_total = np.add(cks_media_total, cks)


acc_media_total = acc_media_total/n_parties
recall_media_total = recall_media_total/n_parties
precision_media_total = precision_media_total/n_parties
f1score_media_total = f1score_media_total/n_parties
fpr_media_total = fpr_media_total/n_parties
mcc_media_total = mcc_media_total/n_parties
cks_media_total = cks_media_total/n_parties

print("Accuracy total")
print(acc_media_total[-1])

acc_media_total = np.zeros(rondas)
recall_media_total = np.zeros(rondas)
precision_media_total = np.zeros(rondas)
f1score_media_total = np.zeros(rondas)
fpr_media_total = np.zeros(rondas)
mcc_media_total = np.zeros(rondas)
cks_media_total = np.zeros(rondas)


ran = list(set(range(1,n_parties +1)) - set(nodesMal))

for i in ran:
    party = pd.read_csv(dir_base + "/metricas_parties/history_" + str(i) + ".csv")
    acc = np.array(party["Accuracy"])
    recall = np.array(party["Recall"])
    precision = np.array(party["Precision"])
    f1_score = np.array(party["F1_score"])
    fpr = np.array(party["FPR"].tail(20))
    mcc = np.array(party['Matthew_Correlation_coefficient'])
    cks = np.array(party["Cohen_Kappa_Score"])

    acc_media_total = np.add(acc_media_total, acc)
    recall_media_total = np.add(recall_media_total, recall)
    precision_media_total = np.add(precision_media_total, precision)
    f1score_media_total = np.add(f1score_media_total, f1_score)
    fpr_media_total = np.add(fpr_media_total, fpr)
    mcc_media_total = np.add(mcc_media_total, mcc)
    cks_media_total = np.add(cks_media_total, cks)

n_parties=n_parties-len(nodesMal)
acc_media_total = acc_media_total / n_parties
recall_media_total = recall_media_total / n_parties
precision_media_total = precision_media_total / n_parties
f1score_media_total = f1score_media_total / n_parties
fpr_media_total = fpr_media_total / n_parties
mcc_media_total = mcc_media_total / n_parties
cks_media_total = cks_media_total / n_parties

print("Accuracy solo a nodos buenos")
print(acc_media_total[-1])



metricas_total = [acc_media_total, recall_media_total,precision_media_total,f1score_media_total,fpr_media_total,mcc_media_total,cks_media_total]
metricas_total = np.transpose(metricas_total)
metricas_total = pd.DataFrame(metricas_total, columns=['Accuracy', 'Recall', 'Precision', 'F1_score', 'FPR', 'Matthew_Correlation_coefficient',
                    'Cohen_Kappa_Score'])
#metricas_total.to_csv(dir_base+'/metricas_parties/Metricas_medias_ST_51-449p_20r_5e_fp098.csv')


