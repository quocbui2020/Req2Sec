"""RECONSTRUCTION, NOT ORIGINAL. Seven Title+Description baselines.
Uses the original scripts' pickle and LP workbook interfaces.
"""
import argparse
import ast
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, jaccard_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score

p=argparse.ArgumentParser()
p.add_argument('--processed',default='processed_data_all.pkl')
p.add_argument('--mapping',default='label_powerset_filtered_for_smote_with_updated_data.xlsx')
p.add_argument('--out',default='vuln_model_outputs_combined_reconstructed')
p.add_argument('--skip-cv',action='store_true')
a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
with open(a.processed,'rb') as f: data=pickle.load(f)
X_train,y_train=data['train_resampled_combined']; X_test,y_test=data['test_tfidf_combined']
mapdf=pd.read_excel(a.mapping)
required={'Label_Powerset_ID','CWE_Label'}
if not required.issubset(mapdf.columns): raise ValueError(f'Mapping missing columns {required-set(mapdf.columns)}')
def parse(v):
    if isinstance(v,(list,tuple,set)):return list(v)
    if pd.isna(v):return []
    try:
        r=ast.literal_eval(str(v)); return list(r) if isinstance(r,(list,tuple,set)) else [str(r)]
    except (ValueError,SyntaxError): return [s.strip() for s in str(v).split('|') if s.strip()]
lookup=dict(zip(mapdf.Label_Powerset_ID,mapdf.CWE_Label.map(parse)))
true_sets=[lookup.get(i,[]) for i in y_test]
mlb=MultiLabelBinarizer().fit([lookup.get(i,[]) for i in y_train]+true_sets)
Y_test=mlb.transform(true_sets)
models={
'Decision Tree':DecisionTreeClassifier(random_state=42),
'Random Forest':RandomForestClassifier(random_state=42),
'Naive Bayes':GaussianNB(),
'Artificial Neural Network (ANN)':MLPClassifier(random_state=42),
'Support Vector Machine (SVM)':LinearSVC(random_state=42,max_iter=100000),
'Logistic Regression (LR)':LogisticRegression(random_state=42,max_iter=1000),
'K-Nearest Neighbors (KNN)':KNeighborsClassifier(),
}
rows=[]
for name,model in models.items():
    print('Training:',name,flush=True)
    train=X_train;test=X_test
    if name=='Naive Bayes':
        train=train.toarray() if hasattr(train,'toarray') else np.asarray(train)
        test=test.toarray() if hasattr(test,'toarray') else np.asarray(test)
    if name in ('Artificial Neural Network (ANN)','Support Vector Machine (SVM)','Logistic Regression (LR)'):
        scaler=StandardScaler(with_mean=False)
        train=scaler.fit_transform(train);test=scaler.transform(test)
    if not a.skip_cv:
        counts=pd.Series(y_train).value_counts()
        if counts.min()>=5:
            scores=cross_val_score(model,train,y_train,cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring='accuracy')
            print('CV accuracy:',scores.mean(),scores.std())
        else:print('CV skipped: at least one class has fewer than five samples')
    model.fit(train,y_train)
    pred=model.predict(test)
    Y_pred=mlb.transform([lookup.get(i,[]) for i in pred])
    report=classification_report(Y_test,Y_pred,target_names=[str(x) for x in mlb.classes_],zero_division=0,output_dict=True)
    pd.DataFrame(report).T.to_csv(out/f'report_{name}.csv')
    pd.DataFrame(confusion_matrix(y_test,pred)).to_csv(out/f'conf_matrix_lp_{name}.csv',index=False)
    metrics={'model':name,'lp_accuracy':accuracy_score(y_test,pred),'subset_accuracy':accuracy_score(Y_test,Y_pred),
             'hamming_loss':hamming_loss(Y_test,Y_pred),'macro_jaccard':jaccard_score(Y_test,Y_pred,average='macro',zero_division=0),
             'macro_precision':report['macro avg']['precision'],'macro_recall':report['macro avg']['recall'],
             'macro_f1':report['macro avg']['f1-score'],'micro_f1':report['micro avg']['f1-score'],
             'weighted_f1':report['weighted avg']['f1-score']}
    pd.Series(metrics).to_csv(out/f'metrics_{name}.csv')
    rows.append(metrics)
pd.DataFrame(rows).to_csv(out/'all_baseline_summary.csv',index=False)
print('Wrote results to',out)
