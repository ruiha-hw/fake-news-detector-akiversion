import numpy as np, pandas as pd, pytest
from model.dataset_preparation import LABEL_COLUMN,SPLIT_COLUMN,TRAIN_SPLIT,VALIDATION_SPLIT,create_split_manifest
from model.model_experimentation import ExperimentConfiguration,ExperimentError,attach_manifest,build_pipeline,evaluate_predictions,make_feature_series,run_candidate,select_best_result
from model.preprocessing import TEXT_COLUMN,TITLE_COLUMN,preprocess_welfake
def source(n=20):
 rows=[]
 for i in range(n): rows += [{TITLE_COLUMN:f'fake {i}',TEXT_COLUMN:f'hoax fabricated {i}',LABEL_COLUMN:0},{TITLE_COLUMN:f'real {i}',TEXT_COLUMN:f'official confirmed {i}',LABEL_COLUMN:1}]
 return pd.DataFrame(rows)
def joined():
 c,_=preprocess_welfake(source()); return attach_manifest(c,create_split_manifest(c))
def test_pipeline(): assert list(build_pipeline(ExperimentConfiguration(max_features=50)).named_steps)==['tfidf','classifier']
def test_metrics():
 m=evaluate_predictions(pd.Series([0,1]),np.array([0,1])); assert m['accuracy']==1 and m['confusion_matrix']==[[1,0],[0,1]]
def test_join(): assert len(joined())==40
@pytest.mark.parametrize('mode',['title','text','combined'])
def test_features(mode): assert make_feature_series(joined(),mode).str.len().gt(0).all()
def test_candidate():
 d=joined(); _,r=run_candidate(d,'combined',ExperimentConfiguration(max_features=100,min_df=1,max_df=1.,max_iterations=100)); assert r['training_rows']==d[SPLIT_COLUMN].eq(TRAIN_SPLIT).sum() and r['validation_rows']==d[SPLIT_COLUMN].eq(VALIDATION_SPLIT).sum()
def test_select():
 r=[{'feature_mode':'title','metrics':{'macro_f1':.8,'accuracy':.9,'weighted_f1':.8}},{'feature_mode':'combined','metrics':{'macro_f1':.9,'accuracy':.8,'weighted_f1':.8}}]; assert select_best_result(r)['feature_mode']=='combined'
def test_empty():
 with pytest.raises(ExperimentError): select_best_result([])
