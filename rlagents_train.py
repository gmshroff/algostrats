#!/usr/bin/env python
# coding: utf-8

# In[ ]:


script=True
DATAPATH='~/DataLocal/algo_fin_new/five_min_data/'


# Need to import algorithms from stable-baselines3

# In[ ]:


from stable_baselines3 import PPO,A2C,DQN
from stable_baselines3.common.vec_env import StackedObservations
from stable_baselines3.common.monitor import Monitor as Mon


# In[ ]:


import warnings
warnings.simplefilter("ignore")


# In[ ]:


import import_ipynb
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from datetime import datetime as dt
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pickle
from threading import Thread
import threading
from IPython import display
import time,getopt,sys,os


# In[ ]:


from feeds import BackFeed,DataFeed
from featfuncs import feat_aug,add_addl_features_feed,add_ta_features_feed,add_sym_feature_feed
from featfuncs import add_global_indices_feed


# In[ ]:


from feed_env import Episode
import aspectlib


# ### Trading strategies as agents
# 
# RL(++)StratAgents imported from ruleagents<br>

# In[ ]:


from rlagents import RLStratAgentDyn


# ### Strategy Development: Training RLStratAgent using BackTestWorld

# In[ ]:


from backtest import Backtest
from feeds import BackFeed,DataFeed
from validation import Validate


# <b>Configuration for training RL model<b>

# In[ ]:


algorithm=PPO
synthetic=False #use synthetic data
nd,nw=4,5 #for BackFeed


# In[ ]:


if script:
    try:
        opts,args = getopt.getopt(sys.argv[1:],"hl:f:d:m:s:w:t:p:",["load=","feed=","datafile=","modelname=","synthetic","weeks","training_steps","deploy"])
    except getopt.GetoptError:
        print('rlagents_train.py -l <load:True/False> -f <scan:back/data> -d <datafile> -m <modelname> -s <synthetic> -w <weeks> -t <training_steps> -p <deploy>')
        sys.exit(2)
    load,feed,date,modelname=False,'back','01-Jan-2000','RLA1.pth'
    training_steps=50000 # if less then n_steps then n_steps is used
    deploy=True
    date=datetime.today().strftime('%d-%b-%Y')
    for opt, arg in opts:
        if opt == "-h":
            print('rlagents_train.py -l <load:True/False> -f <scan:back/data> -d <datafile> -m <modelname> -s <synthetic> -w <weeks> -t <training_steps> -p <deploy>')
            sys.exit()
        elif opt in ("-l", "--load"):
            load = (lambda x: True if x=='True' else False)(arg)
        elif opt in ("-f", "--feed"):
            feed = (lambda x: 'data' if x=='data' else 'back')(arg)
        elif opt in ("-d", "--datafile"):
            datafile = arg.split('/')[-1]
        elif opt in ("-m", "--modelname"):
            modelname = arg
        elif opt in ("-s", "--synthetic"):
            synthetic = (lambda x: True if x=='True' else False)(arg)
        elif opt in ("-w", "--weeks"):
            nw = int(arg)
        elif opt in ("-t", "--training_steps"):
            training_steps=int(arg)
        elif opt in ("-p", "--deploy"):
            deploy = (lambda x: True if x=='True' else False)(arg)
    if len(opts)==0: 
        print('rlagents_train.py -l <load:True/False> -f <scan:back/data> -d <datafile> -m <modelname> -s <synthetic> -w <weeks> -t <training_steps> -p <deploy>')
        sys.exit()
    print(f"load:{load},feed:{feed},datafile:{datafile},modelname:{modelname},synthetic:{synthetic},weeks:{nw},training_steps:{training_steps},deploy:{deploy}")
    loadfeed=load
    if feed=='data': datafeed=True
    else: datafeed=False


# In[ ]:


if not script:
    loadfeed=True
    datafeed=False
    datafile='augdata_16-Dec-2022_5m.csv'
    modelname='RLA1.pth' # replace with modelname if model to be saved to saved_models
    date=datetime.today().strftime('%d-%b-%Y')
    training_steps=20000 # if less then n_steps then n_steps is used
    deploy=True


# In[ ]:


n_steps=2048 # reduce for debugging only else 2048


# In[ ]:


def stringify(x):
    return pd.to_datetime(x['Datetime']).strftime('%d-%b-%Y')


# In[ ]:


import pickle
if not loadfeed and not datafeed:
    data=pd.read_csv('./capvol100.csv')
    tickers=list(data.iloc[0:10]['ticker'].values)
    print('Creating feed')
    feed=BackFeed(tickers=tickers,nd=nd,nw=nw,interval='5m',synthetic=synthetic)
    print('Processing feed')
    add_addl_features_feed(feed,tickers=feed.tickers)
    add_sym_feature_feed(feed,tickers=feed.tickers)
    add_global_indices_feed(feed)
    with open('../../temp_data/btfeed.pickle','wb') as f: pickle.dump(feed,f)
elif loadfeed and not datafeed:
    with open('../../temp_data/btfeed.pickle','rb') as f: feed=pickle.load(f)


# In[ ]:


if not loadfeed and datafeed:
    #DATAFILE=DATAPATH+'augdata_'+date+'_5m.csv'
    DATAFILE=DATAPATH+datafile
    print('Reading datafile')
    df=pd.read_csv(DATAFILE)
    if 'Date' not in df.columns: 
        print('Adding Date')
        df['Date']=df.apply(stringify,axis=1)
    print('Creating feed')
    feed=DataFeed(tickers=list(df.ticker.unique()[0:10]),dfgiven=True,df=df)
    print('Processing feed')
    add_addl_features_feed(feed,tickers=feed.tickers)
    add_sym_feature_feed(feed,tickers=feed.tickers)
    add_global_indices_feed(feed)
    with open('../../temp_data/btdatafeed.pickle','wb') as f: pickle.dump(feed,f)
elif loadfeed and datafeed:
    with open('../../temp_data/btdatafeed.pickle','rb') as f: feed=pickle.load(f)


# In[ ]:


agent=RLStratAgentDyn(algorithm,monclass=Mon,soclass=StackedObservations,n_steps=n_steps,verbose=1,win=5,
                   metarl=True)
agent.use_memory=True #depends on whether RL algorithm uses memory for state computation
agent.debug=False


# In[ ]:


if modelname and os.path.exists('./saved_models/'+modelname): 
    agent.load_model(filepath='./saved_models/'+modelname)


# In[ ]:


@aspectlib.Aspect
def my_decorator(*args, **kwargs):
    # print("Got called with args: %s kwargs: %s" % (args, kwargs))
    # result = yield
    # print(" ... and the result is: %s" % (result,))
    state,rew,done,exit_type = yield
    # args[0].policy.logL+=[(state.keys(),rew,done,exit_type)]
    args[0].policy.reward((rew,done,{'exit_type':exit_type}))
    return state,rew,done,exit_type


# In[ ]:


aspectlib.weave(Episode, my_decorator, methods='env_step')


# In[ ]:


bt=Backtest(feed,tickers=feed.tickers,add_features=False,target=5,stop=5,txcost=0.001,
            loc_exit=True,scan=True,topk=3,deploy=deploy,save_dfs=False,
            save_func=None)


# In[ ]:


def run_btworld():
    global bt,feed,agent
    while agent.training:
        bt.run_all(tickers=feed.tickers,model=agent,verbose=False)


# In[ ]:


agent.start(training_steps=training_steps)


# In[ ]:


btworldthread=Thread(target=run_btworld,name='btworld')
btworldthread.start()


# In[ ]:


def check_bt_training_status():
    threadL=[thread.name for thread in threading.enumerate()]
    # print(threadL)
    if 'monitor' not in threadL and 'btworld' not in threadL:
        print(f'Training Over after {agent.model.num_timesteps} steps')
        return False
    else:
        print(f'Model Training for {agent.model.num_timesteps} steps')
        return True


# In[ ]:


while check_bt_training_status():
    time.sleep(2)


# In[ ]:


# Save learned model
if modelname: torch.save(agent.model.policy.state_dict(),'./saved_models/'+modelname)


# ## Training Curves

# In[ ]:


# import pandas as pd
# df=pd.read_csv('/tmp/aiagents.monitor.csv',comment='#')


# In[ ]:


# import plotly.express as px
# px.line(df['r'].rolling(window=200).mean().values).show()


# ## Testing

# In[ ]:


# data=pd.read_csv('./capvol100.csv')
# tickers=list(data.iloc[10:20]['ticker'].values)
# feed=BackFeed(tickers=tickers,nd=nd,nw=nw,interval='5m',synthetic=False)
# add_addl_features_feed(feed,tickers=feed.tickers)
# add_sym_feature_feed(feed,tickers=feed.tickers)
# add_global_indices_feed(feed)


# In[ ]:


# agent=RLStratAgentDyn(algorithm,monclass=Mon,soclass=StackedObservations,n_steps=2048,verbose=1,win=5,
#                    metarl=True)
# agent.use_memory=True #depends on whether RL algorithm uses memory for state computation
# agent.debug=False


# In[ ]:


# agent.training=False


# In[ ]:


# bt=Backtest(feed,tickers=feed.tickers,add_features=True,target=5,stop=5,txcost=0.001,
#             loc_exit=True,scan=True,topk=3,deploy=True,save_dfs=False,
#             save_func=None)
# bt.run_all(tickers=feed.tickers,model=agent,verbose=False)


# In[ ]:


# bt.results


# In[ ]:


# bt.returns


# Debugging

# In[ ]:


# df1=pd.read_csv(DATAPATH+'augdata_01-Dec-2021_5m.csv')


# In[ ]:


# df1.apply(stringify,axis=1)


# In[ ]:





# In[ ]:




