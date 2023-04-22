#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install import_ipynb --quiet
# !pip install anvil-uplink --quiet
# !pip install yfinance --quiet
# !pip install pandas_ta --quiet
# !pip install ipynb --quiet
# !pip install rpyc --quiet
# !pip install stable-baselines3 --quiet
# !pip install aspectlib


# In[ ]:


# !git clone https://github.com/gmshroff/algostrats.git


# In[ ]:


# %cd algostrats


# Uncomment below if on Colab and using datasets from Kaggle

# In[ ]:


# from google.colab import files
# uploaded=files.upload()


# In[ ]:


# !mkdir /root/.kaggle
# !mv ./kaggle.json /root/.kaggle/.
# !chmod 600 /root/.kaggle/kaggle.json


# In[ ]:


# %mkdir data
# %cd data
# !kaggle datasets download -d gmshroff/marketdatafivemin
# %cd ../algostrats


# In[ ]:


colab=False
script=True
if not colab: 
    DATAPATH='~/DataLocal/algo_fin_new/five_min_data/'
elif colab: DATAPATH='../data/'


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
simple='sinewave' #False,True or 'sinewave'
nd,nw=5,2 #for BackFeed


# In[ ]:


if script:
    try:
        opts,args = getopt.getopt(sys.argv[1:],"hl:f:d:m:s:w:t:p:u:z:",["load=","feed=","datafile=","modelname=","synthetic","weeks","training_steps","deploy","use_alt_data","win"])
    except getopt.GetoptError:
        print('rlagents_train.py -l <load:True/False> -f <scan:back/data> -d <datafile> -m <modelname> -s <synthetic> -w <weeks> -t <training_steps> -p <deploy>')
        sys.exit(2)
    load,feed,modelname=False,'back',''
    training_steps=50000 # if less then n_steps then n_steps is used
    deploy=True
    date=datetime.today().strftime('%d-%b-%Y')
    use_alt_data=True
    datafile='alldata.csv'
    win=5
    for opt, arg in opts:
        if opt == "-h":
            print('rlagents_train.py -l <load:True/False> -f <scan:back/data> -d <datafile> -m <modelname> -s <synthetic> -w <weeks> -t <training_steps> -p <deploy> -u <use_alt_data>')
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
        elif opt in ("-u", "--use_alt_data"):
            use_alt_data = (lambda x: True if x=='True' else False)(arg)
        elif opt in ("-z", "--win"):
            win=int(arg)
    if len(opts)==0: 
        print('rlagents_train.py -l <load:True/False> -f <scan:back/data> -d <datafile> -m <modelname> -s <synthetic> -w <weeks> -t <training_steps> -p <deploy> -u <use_alt_data>')
        sys.exit()
    print(f"load:{load},feed:{feed},datafile:{datafile},modelname:{modelname},synthetic:{synthetic},weeks:{nw},training_steps:{training_steps},deploy:{deploy},use_alt_data:{use_alt_data},win:{win}")
    loadfeed=load
    if feed=='data': datafeed=True
    else: datafeed=False


# In[ ]:


if not script:
    loadfeed=True
    datafeed=True
    datafile='alldata.csv'
    modelname=''
    win=10
    date=datetime.today().strftime('%d-%b-%Y')
    training_steps=500000 # if less then n_steps then n_steps is used
    deploy=True
    use_alt_data=False


# In[ ]:


n_steps=2048 # reduce for debugging only else 2048


# In[ ]:


def stringify(x):
    return pd.to_datetime(x['Datetime']).strftime('%d-%b-%Y')


# In[ ]:


import pickle
if not loadfeed and not datafeed:
    data=pd.read_csv('./capvolfiltered.csv')
    tickers=list(data['ticker'].values)
    print('Creating feed')
    feed=BackFeed(tickers=tickers,nd=nd,nw=nw,interval='5m',synthetic=synthetic,simple=simple)
    print('Processing feed')
    add_addl_features_feed(feed,tickers=feed.tickers)
    add_sym_feature_feed(feed,tickers=feed.tickers)
    if not synthetic: add_global_indices_feed(feed)
    if not colab: 
        with open('../../temp_data/btfeed.pickle','wb') as f: pickle.dump(feed,f)
    elif colab: 
        with open('/tmp/btfeed.pickle','wb') as f: pickle.dump(feed,f)
elif loadfeed and not datafeed:
    if not colab: 
        with open('../../temp_data/btfeed.pickle','rb') as f: feed=pickle.load(f)
    elif colab: 
        with open('/tmp/btfeed.pickle','rb') as f: feed=pickle.load(f)


# In[ ]:


if not loadfeed and datafeed:
    DATAFILE=DATAPATH+datafile
    print('Reading datafile')
    df=pd.read_csv(DATAFILE)
    if 'Date' not in df.columns: 
        print('Adding Date')
        df['Date']=df.apply(stringify,axis=1)
    print('Creating feed')
    data=pd.read_csv('./capvolfiltered.csv')
    tickers=[t for t in list(df['ticker'].unique()) if t in list(data['ticker'].values)]
    feed=DataFeed(tickers=tickers,dfgiven=True,df=df)
    print('Processing feed')
    add_addl_features_feed(feed,tickers=feed.tickers)
    add_sym_feature_feed(feed,tickers=feed.tickers)
    add_global_indices_feed(feed)
    if not colab: 
        with open('../../temp_data/btdatafeed.pickle','wb') as f: pickle.dump(feed,f)
    elif colab: 
        with open('/tmp/btdatafeed.pickle','wb') as f: pickle.dump(feed,f)
elif loadfeed and datafeed:
    if not colab: 
        with open('../../temp_data/btdatafeed.pickle','rb') as f: feed=pickle.load(f)
    elif colab:
        with open('/tmp/btdatafeed.pickle','rb') as f: feed=pickle.load(f)


# In[ ]:


# pd.options.mode.use_inf_as_na = True
for t in feed.ndata:
    for d in feed.ndata[t]:
        if feed.ndata[t][d].isnull().values.any(): 
            feed.ndata[t][d]=feed.ndata[t][d].fillna(1)
            # print(t,d)
        if feed.ndata[t][d].isin([-np.inf,np.inf]).values.any():
            feed.ndata[t][d]=feed.ndata[t][d].replace([np.inf, -np.inf],1)
            # print(t,d)


# In[ ]:


from math import isnan
for d in feed.gdata:
    for g in feed.gdata[d]:
         for k in g: 
                if isnan(g[k]): g[k]=1


# In[ ]:


def get_alt_data_live():
    aD={'gdata':feed.gdata}
    return aD


# In[ ]:


# feed.data[list(feed.data.keys())[0]]


# In[ ]:


agent=RLStratAgentDyn(algorithm,monclass=Mon,soclass=StackedObservations,verbose=1,
                   metarl=True,myargs=(n_steps,use_alt_data,win))
agent.use_memory=True #depends on whether RL algorithm uses memory for state computation
agent.debug=False
if use_alt_data: agent.set_alt_data(alt_data_func=get_alt_data_live)


# In[ ]:


if modelname and os.path.exists('./saved_models/'+modelname):
    print('Loading model')
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
            loc_exit=True,scan=True,topk=10,deploy=deploy,save_dfs=False,
            save_func=None)


# In[ ]:


agent.data_cols=agent.data_cols+['Date']
agent.error=False


# In[ ]:


def run_btworld():
    global bt,feed,agent
    while agent.training and not agent.error:
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
modelname='RLC2W10.pth'
if modelname: torch.save(agent.model.policy.state_dict(),'./saved_models/'+modelname)


# ## Training Curves

# In[ ]:


if not script:
    import pandas as pd
    df=pd.read_csv('/tmp/aiagents.monitor.csv',comment='#')


# In[ ]:


if not script:
    import plotly.express as px
    px.line(df['r'].rolling(window=2000).mean().values).show()


# In[ ]:


### Test on training data 


# In[ ]:


# def compute_avg_episode_rew(bt):
#     i,tot=0,0
#     for t in bt.results:
#         for d in bt.results[t]:
#             for r in bt.results[t][d]['rew']:
#                 tot+=r[1]
#                 i+=1
#     return tot/i,i,tot


# In[ ]:


# bt_new=Backtest(feed,tickers=feed.tickers,add_features=False,target=5,stop=5,txcost=0.001,
#             loc_exit=True,scan=True,topk=5,deploy=deploy,save_dfs=False,
#             save_func=None)


# In[ ]:


# bt_new.run_all(tickers=feed.tickers,model=agent,verbose=False)


# In[ ]:


# sum([bt_new.results[t][d]['tot'] for t in bt_new.results for d in bt_new.results[t]])


# In[ ]:


# avg,n_ep,tot=compute_avg_episode_rew(bt_new) # Avg episode reward (not day)
# print(f"Average reward {avg} across {n_ep} episodes; total {tot}")


# In[ ]:


# bt.run_all(tickers=feed.tickers,model=agent,verbose=False)


# In[ ]:


# sum([bt.results[t][d]['tot'] for t in bt.results for d in bt.results[t]])


# In[ ]:


# avg,n_ep,tot=compute_avg_episode_rew(bt) # Avg episode reward (not day)
# print(f"Average reward {avg} across {n_ep} episodes; total {tot}")


# In[ ]:




