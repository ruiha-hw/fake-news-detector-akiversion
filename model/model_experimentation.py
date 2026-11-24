# model_experimentation.py
# Compares title, article, and combined TF-IDF Logistic Regression candidates.
from __future__ import annotations
import json, platform, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final
import joblib, numpy as np, pandas as pd, scipy, sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from model.dataset_preparation import LABEL_COLUMN, MANIFEST_PATH, METADATA_PATH, RECORD_ID_COLUMN, SPLIT_COLUMN, TRAIN_SPLIT, VALIDATION_SPLIT, calculate_file_hash, create_record_id, validate_manifest
from model.preprocessing import IDENTITY_KEY_COLUMN, TITLE_COLUMN, TEXT_COLUMN, FeatureMode, create_model_text, preprocess_welfake
PROJECT_ROOT=Path(__file__).resolve().parents[1]
SOURCE_DATASET=PROJECT_ROOT/'data'/'raw'/'WELFake_Dataset.csv'
EXPERIMENT_DIRECTORY=PROJECT_ROOT/'model_artifacts'/'experiments'
RESULTS_PATH=EXPERIMENT_DIRECTORY/'validation_experiment_results.json'
SELECTED_PIPELINE_PATH=EXPERIMENT_DIRECTORY/'selected_validation_pipeline.joblib'
RANDOM_STATE=205; FEATURE_MODES=('title','text','combined'); SELECTION_METRIC='macro_f1'
@dataclass(frozen=True)
class ExperimentConfiguration:
 max_features:int=100_000; ngram_min:int=1; ngram_max:int=2; min_df:int=2; max_df:float=.98; sublinear_tf:bool=True; strip_accents:str='unicode'; solver:str='liblinear'; classifier_c:float=1.; max_iterations:int=1000
class ExperimentError(ValueError): pass
def build_pipeline(c):
 return Pipeline([('tfidf',TfidfVectorizer(lowercase=True,strip_accents=c.strip_accents,ngram_range=(c.ngram_min,c.ngram_max),min_df=c.min_df,max_df=c.max_df,sublinear_tf=c.sublinear_tf,max_features=c.max_features,dtype=np.float32)),('classifier',LogisticRegression(C=c.classifier_c,solver=c.solver,max_iter=c.max_iterations,random_state=RANDOM_STATE))])
def evaluate_predictions(y,p):
 macro=precision_recall_fscore_support(y,p,labels=[0,1],average='macro',zero_division=0); weighted=precision_recall_fscore_support(y,p,labels=[0,1],average='weighted',zero_division=0); per=precision_recall_fscore_support(y,p,labels=[0,1],average=None,zero_division=0)
 return {'accuracy':round(float(accuracy_score(y,p)),8),'macro_precision':round(float(macro[0]),8),'macro_recall':round(float(macro[1]),8),'macro_f1':round(float(macro[2]),8),'weighted_precision':round(float(weighted[0]),8),'weighted_recall':round(float(weighted[1]),8),'weighted_f1':round(float(weighted[2]),8),'confusion_matrix':confusion_matrix(y,p,labels=[0,1]).tolist(),'per_class':{'0_fake':{'precision':float(per[0][0]),'recall':float(per[1][0]),'f1':float(per[2][0]),'support':int(per[3][0])},'1_real':{'precision':float(per[0][1]),'recall':float(per[1][1]),'f1':float(per[2][1]),'support':int(per[3][1])}}}
def attach_manifest(cleaned,manifest):
 p=cleaned.copy(); p[RECORD_ID_COLUMN]=p[IDENTITY_KEY_COLUMN].map(create_record_id); j=p.merge(manifest[[RECORD_ID_COLUMN,LABEL_COLUMN,SPLIT_COLUMN]],on=RECORD_ID_COLUMN,how='inner',validate='one_to_one',suffixes=('','_manifest'))
 if len(j)!=len(cleaned): raise ExperimentError('Manifest join row mismatch.')
 if not j[LABEL_COLUMN].equals(j[LABEL_COLUMN+'_manifest'].astype('int8')): raise ExperimentError('Manifest labels do not match cleaned dataset labels.')
 return j.drop(columns=[LABEL_COLUMN+'_manifest'])
def load_experiment_data():
 m=json.loads(METADATA_PATH.read_text(encoding='utf-8')); actual=calculate_file_hash(MANIFEST_PATH,'sha256')
 if actual!=m['manifest_sha256']: raise ExperimentError('Split manifest SHA-256 mismatch.')
 manifest=pd.read_csv(MANIFEST_PATH); validate_manifest(manifest,int(m['manifest_rows'])); source=pd.read_csv(SOURCE_DATASET,low_memory=False); cleaned,s=preprocess_welfake(source)
 if s.output_rows!=int(m['manifest_rows']): raise ExperimentError('Preprocessing output mismatch.')
 return attach_manifest(cleaned,manifest),m
def make_feature_series(frame,mode): return pd.Series([create_model_text(a,b,mode) for a,b in zip(frame[TITLE_COLUMN],frame[TEXT_COLUMN],strict=True)],index=frame.index,dtype='object')
def run_candidate(data,mode,c):
 train=data[data[SPLIT_COLUMN].eq(TRAIN_SPLIT)]; val=data[data[SPLIT_COLUMN].eq(VALIDATION_SPLIT)]; pipe=build_pipeline(c); start=time.perf_counter(); pipe.fit(make_feature_series(train,mode),train[LABEL_COLUMN]); fit=time.perf_counter()-start; start=time.perf_counter(); pred=pipe.predict(make_feature_series(val,mode)); pt=time.perf_counter()-start; metrics=evaluate_predictions(val[LABEL_COLUMN],pred)
 return pipe,{'feature_mode':mode,'training_rows':len(train),'validation_rows':len(val),'feature_count':len(pipe.named_steps['tfidf'].get_feature_names_out()),'classifier_iterations':[int(x) for x in pipe.named_steps['classifier'].n_iter_.tolist()],'fit_seconds':round(fit,4),'validation_prediction_seconds':round(pt,4),'metrics':metrics}
def select_best_result(results):
 if not results: raise ExperimentError('No experiment results were supplied for selection.')
 pri={'title':0,'text':1,'combined':2}; return max(results,key=lambda r:(r['metrics']['macro_f1'],r['metrics']['accuracy'],r['metrics']['weighted_f1'],pri[r['feature_mode']]))
def run_experiments(data,c):
 pipes={}; results=[]
 for mode in FEATURE_MODES:
  print(f'Training validation candidate: {mode}'); pipe,result=run_candidate(data,mode,c); pipes[mode]=pipe; results.append(result); print(f"  macro_f1={result['metrics']['macro_f1']:.6f} accuracy={result['metrics']['accuracy']:.6f}")
 selected=select_best_result(results); mode=selected['feature_mode']; report={'status':'validation_only','test_split_evaluated':False,'selection_metric':SELECTION_METRIC,'selected_feature_mode':mode,'configuration':asdict(c),'candidates':results,'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'joblib':joblib.__version__,'pandas':pd.__version__}}
 return pipes[mode],report
def main():
 print('Loading locked training and validation assignments...'); data,metadata=load_experiment_data(); pipe,report=run_experiments(data,ExperimentConfiguration()); report['dataset_preparation']=metadata; EXPERIMENT_DIRECTORY.mkdir(parents=True,exist_ok=True); joblib.dump(pipe,SELECTED_PIPELINE_PATH,compress=3); report['selected_pipeline_sha256']=calculate_file_hash(SELECTED_PIPELINE_PATH,'sha256'); report['selected_pipeline_path']=str(SELECTED_PIPELINE_PATH); RESULTS_PATH.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,indent=2)); print('The held-out test split has not been evaluated.')
if __name__=='__main__': main()
