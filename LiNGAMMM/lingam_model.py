import warnings
import numpy as np
import pandas as pd
import graphviz
import lingam
from sklearn.utils import check_array
from lingam.utils import make_prior_knowledge, make_dot
from sklearn.linear_model import Ridge

def causal_prediction(
  self
  ,x_train
  ,y_train
  ,x_val
  ,y_val
  ,x_test
  ,y_test
  ,lambda_scaled
  ,prior_knowledge=None
  ):
    # Generate causal model by LiNGAM model
    all_X = pd.concat([x_train,x_val,x_test])
    all_y = pd.concat([y_train,y_val,y_test])
    all_df = pd.concat([all_y,all_X],axis=1)
    lg_model = lingam_model(all_df,prior_knowledge)
    
    y_tr_trans, x_tr_trans = causal_transform(x_train, y_train, lg_model)
    y_val_trans, x_val_trans = causal_transform(x_val, y_val, lg_model)
    y_test_trans, x_test_trans = causal_transform(x_test, y_test, lg_model)
    
    mod = Ridge(random_state=0,alpha=lambda_scaled)
    mod.fit(x_tr_trans,y_tr_trans)
    
    y_tr_pred = mod.predict(x_tr_trans)
    df_int = mod.intercept_
    # sklearnの既存関数を使う？(`r2_score`)
    rsq_train = get_rsq_py(
      true = y_tr_trans
      ,predicted = y_tr_pred
      ,p = x_tr_trans.shape[1]
      ,df_int = df_int
      )
  
    if x_val_trans is not None:
      y_val_pred = mod.predict(x_val_trans)
      # sklearnの既存関数を使う？(`r2_score`)
      rsq_val = get_rsq_py(
        true = y_val_trans
        ,predicted = y_val_pred
        ,p = x_val_trans.shape[1]
        ,df_int = df_int
        ,n_train = len(y_tr_trans)
        )
      
      y_test_pred = mod.predict(x_test_trans)
      # sklearnの既存関数を使う？(`r2_score`)
      rsq_test = get_rsq_py(
        true = y_test_trans
        ,predicted = y_test_pred
        ,p = x_test_trans.shape[1]
        ,df_int = df_int
        ,n_train = len(y_tr_trans)
        )
        
      y_pred = pd.concat([y_tr_pred, y_val_pred, y_test_pred])
    else:
      rsq_val = rsq_test = np.nan
      y_pred = y_tr_pred
  
    # Calculate all NRMSE
    nrmse_train = np.sqrt(np.mean((y_tr_trans - y_tr_pred)**2)) / (max(y_tr_trans) - min(y_tr_trans))
    if x_val_trans is not None:
      nrmse_val = np.sqrt(np.mean(sum((y_val_trans - y_val_pred)**2))) / (max(y_val_trans) - min(y_val_trans))
      nrmse_test = np.sqrt(np.mean(sum((y_test_trans - y_test_pred)**2))) / (max(y_test_trans) - min(y_test_trans))
    else:
      nrmse_val = nrmse_test = y_val_pred = y_test_pred = np.nan
    
    self.rsq_train = rsq_train
    self.rsq_val = rsq_val
    self.rsq_test = rsq_test
    self.nrmse_train = nrmse_train
    self.nrmse_val = nrmse_val
    self.nrmse_test = nrmse_test
    self.coefs = mod.coef_
    self.mod = mod
    self.y_train_pred = y_tr_pred
    self.y_val_pred = y_val_pred
    self.y_test_pred = y_test_pred
    self.y_pred = y_pred
    self.lg_mod = lg_model
    self.df_int = df_int

def lingam_model(input_df,prior_knowledge=None):
  if prior_knowledge is not None:
    model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
  else:
    model = lingam.DirectLiNGAM()
  model.fit(input_df)
  return model

def causal_transform(
  X
  ,y
  ,lg_model
  ):
    ce = lingam.CausalEffect(lg_model)
    input_df = pd.concat([y,X],axis=1)
    mat_ = check_array(input_df)
    ce._check_init_params()
    
    all_effected_list=[]
    for each_row in mat_:
      En = each_row - np.dot(ce._B, each_row)
      effects = np.zeros(len(ce._causal_order))
      for i in ce._causal_order:
        effects[i] = np.dot(ce._B[i, :], effects) + En[i]
      all_effected_list.append(effects)
    
    return pd.DataFrame(all_effected_list)[[0]], pd.DataFrame(all_effected_list).iloc[:,1:]

def get_rsq_py(true, predicted, p = None, df_int = None, n_train = None):
  sse = sum((predicted - true)**2)
  sst = sum((true - np.mean(true))**2)
  rsq = 1 - sse / sst
  rsq_out = rsq
  if p is not None & df_int is not None:
    if n_train is not None:
      n = n_train
    else:
      n = len(true)
    rdf = n-p-1
    rsq_adj = 1 - (1 - rsq) * ((n - df_int) / rdf)
    rsq_out = rsq_adj
    
  return rsq_out


def make_prior_knowledge_graph(labels,prior_knowledge_matrix):
  d = graphviz.Digraph(engine='dot')
  
  for label in labels:
      d.node(label, label)

  dirs = np.where(prior_knowledge_matrix > 0)
  for to, from_ in zip(dirs[0], dirs[1]):
      d.edge(labels[from_], labels[to])

  dirs = np.where(prior_knowledge_matrix < 0)
  for to, from_ in zip(dirs[0], dirs[1]):
      if to != from_:
          d.edge(labels[from_], labels[to], style='dashed')
  return d
