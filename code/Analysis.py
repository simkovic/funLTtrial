import numpy as np
from scipy import stats
import os,pickle
from scipy.special import digamma
DPATH=os.getcwd()+os.path.sep+'data'+os.path.sep
FPATH=os.getcwd()+os.path.sep+'figs'+os.path.sep
from scipy.stats import scoreatpercentile as sap
import pylab as plt
from matusplotlib import errorbar,figure,subplot,ndarray2latextable,subplotAnnotate

SEED=6
DPI=400

def printCI(w,var=None,decimals=3):
    sfmt=' {:.{:d}f} [{:.{:d}f},{:.{:d}f}]'
    def _print(b):
        d=np.round([np.median(b), sap(b,2.5),sap(b,97.5)],decimals).tolist()
        print(sfmt.format(d[0],decimals,d[1],decimals,d[2],decimals))
        #print var+' %.3f, CI %.3f, %.3f'%tuple(d) 
    if var is None: d=w;var='var'
    else: d=w[var]
    if d.ndim==2:
        for i in range(d.shape[1]):
            _print(d[:,i])
    elif d.ndim==1: _print(d)

LBLS=[]
for j in range(3):
    cid=list('333333333')
    cid[j]='1'
    cid[j+4]='5'
    LBLS.append(int(''.join(cid)))
for j in [1,5,7]:
    LBLS.append(133353330+j)
for j in [1,5,7]:
    LBLS.append(313335330+j)
for j in [1,5,7]:
    LBLS.append(331333530+j)    

def loadData():
    '''loads data and performs integrity checks
        output: info - metadata
            Dres - LTs
    '''
    deg2cm=np.pi/180*70
    cpath='/../output/pseud/'
    opath=cpath+'funLT/'
    fn='vpinfo.res'
    info=np.int32(np.loadtxt(cpath+fn,delimiter=','))
    info=info[info[:,7]!=-1,:]
    print('Check if all data files present')
    for i in range(info.shape[0]):
        fnn='funLTVp%dc%dM.'%(info[i,0],info[i,1])
        for suf in ['res','log']:
            if not os.path.isfile(opath+fnn+suf):
                print(opath+fnn+suf+' is missing')

    print('Checking for surplus files missing from vpinfo')
    fnsall=os.listdir(opath)

    fns=list(filter(lambda x: x[-4:]=='.res',fnsall))
    fns=fns+list(filter(lambda x: x[-4:]=='.log',fnsall))
    fns=np.sort(fns)
    vpns=info[:,[0,1]]
    for fn in fns:
        vp=int(fn.rsplit('.')[0].rsplit('c')[0].rsplit('Vp')[1])
        m=int(fn.rsplit('.')[0].rsplit('c')[1][:-1])
        temp=(np.logical_and(vpns[:,0]==vp, vpns[:,1]==m)).nonzero()[0]
        if not temp.size:
            #os.remove(opath+fn)
            print(fn+' surplus file')
    print('Checking format of .res files')        
    Dres=[];nrtrs=[]
    for i in range(info.shape[0]):
        fnn='funLTVp%dc%dM.res'%(info[i,0],info[i,1])
        try: d=np.loadtxt(opath+fnn,delimiter=',')
        except: 
            print(opath+fnn+' could not read')
            continue
        Dres.append(d)
        nrtr=(len(d)-1)/2
        if d[-1]==-1: 
            nrtr-=1
            suf=' 40s'
        else: suf=''
        if nrtr<8: nrtrs.append(nrtr)
        #if nrtr!=8: print('%d saw %d trials'%(info[i,0],nrtr)+suf)
        assert(d[0]==info[i,7])
    #plt.hist(nrtrs,bins=np.arange(0,9)-0.5)

    print('Check format of .log files')
    Dlog=[]
    Det=[]
    for i in range(info.shape[0]):
        fnn='funLTVp%dc%dM.log'%(info[i,0],info[i,1])
        f=open(opath+fnn,'r')
        lines=f.readlines()
        f.close()
        la=-1;ea=-1;trialon=False;left=True;saw=False
        d=[[],[],[],[]]
        Det.append([[],[]])
        for line in lines:
            if line[:2]=='##':continue
            line=line.rstrip('\n')
            words=line.rsplit(';')
            if words[3]=='MSG' and words[4][:7]=='trialon':
                d[0].append(np.int64(words[1]))
                trialon=True
            if words[3]=='MSG' and words[4][:3]=='saw':
                if ea!=-1: d[1].append(ea)
                ea=-1;saw=True
            if words[3]=='MSG' and words[4][:4]=='left':
                la=np.int64(words[1]);left=True
            if words[3]=='MSG' and words[4][:4]=='ente':
                ea=np.int64(words[1]);left=False
            if words[3]=='MSG' and words[4][:8]=='waitGaze':
                d[1].append(np.nan)
            if words[3]=='MSG' and words[4][:8]=='trialoff':
                if la!=-1: d[2].append(la)
                la=-1;trialon=False;saw=False
                d[3].append(np.int64(words[1]))
            if words[3]!='MSG' and trialon:
                if not left: Det[-1][0].append(np.float64(words[:7]))
                else: Det[-1][1].append(np.float64(words[:7]))
        if not np.all(np.array(list(map(len,d)))==(Dres[i].size-1)/2):
            print('%d %d nr trials not consistent'%(info[i,0],info[i,1]))
            print(np.array(list(map(len,d))))
            print((Dres[i].size-1)/2)
        
        d=np.array(d)/1000000
        Dlog.append(np.array([d[1,:]-d[0,:],d[2,:]-d[1,:]]))
        for t in range(Dlog[i].shape[1]):
            a=np.isclose(Dlog[i][1,t],
                (Dres[i][2::2][t]-Dres[i][1::2][t]),atol=0.017)
            if not a  and not Dres[i][1::2][t]==-1: 
                print('%d %d trial %d duration intervals not consistent %.2f vs %.2f'%(info[i,0],info[i,1],t,Dlog[i][1,t],(Dres[i][2::2][t]-Dres[i][1::2][t])))
            b=np.isclose(Dlog[i][0,t],Dres[i][1::2][t],atol=0.017)
            if not b  and not Dres[i][1::2][t]==-1: 
                print('%d %d trial %d start intervals not consistent %.2f vs %.2f'%(info[i,0],info[i,1],t,Dlog[i][0,t],Dres[i][1::2][t]))
    # readded 8445,8610,8583,8234
    return info,Dres
    
def plotSample(info,Dres):
    '''plot sample description'''
    import pylab as plt
    #assert(np.all(info[:,4]>=30*3))
    assert(np.all(info[:,4]<=30*12))
    tc=np.int32(list(map(lambda x: (x[1:]>-1).sum()/2,Dres)))
    plt.hist(info[:,4],bins=np.linspace(30*3,30*11,15))
    plt.gca().set_xticks(np.linspace(30*3,30*11,8))
    plt.xlabel('age in days');
    plt.figure(figsize=[12,6])
    for i in range(3):
        ax=plt.subplot(1,3,i+1)
        sel=info[:,1]==[4,7,10][i]
        found=np.zeros(sel.sum())
        temp=[]
        for j in range(len(LBLS)):
            condd=np.logical_and(info[sel,7]==LBLS[j],tc[sel]>3)
            found[info[sel,7]==LBLS[j]]=1
            temp.append(np.sum(condd))
            ax.barh(j,temp[-1],color='k')
            plt.text(np.sum(condd),j,'%d'%np.sum(condd))
        #print(info[sel,0][~np.bool8(found)])
        ax.set_yticks(range(len(LBLS)))
        if not i: ax.set_yticklabels(LBLS)
        else: ax.set_yticklabels([])
        plt.title(['4M','7M','10M'][i])
        plt.xlim([0,14])
        
def preprocessData(info,Dres,plot=False):
    '''puts data in format suitable for pystan'''
    if plot: plt.figure(figsize=[16,3*len(LBLS)])
    gm=np.nan*np.ones((len(LBLS),3,20,10))
    yLT=np.nan*np.ones((info.shape[0],10))
    xSC=np.zeros(info.shape[0],dtype=np.int32)#stimulus change
    xD=np.nan*np.ones(info.shape[0],dtype=np.int32)
    # saliency dimension 0-light, 1-color, 2-orient
    xA=info[:,4]
    hs=np.zeros((len(LBLS),3),dtype=np.int32)
    for i in range(len(LBLS)):
        for j in range(3):
            if plot: ax=plt.subplot(len(LBLS),3,i*3+j+1)
            for k in range(len(Dres)):
                if info[k,1]!=[4,7,10][j] or info[k,7]!=LBLS[i]: continue
                temp=Dres[k][2::2]-Dres[k][1::2]
                temp[temp==0]=np.nan
                gm[i,j,hs[i,j],:temp.size]=temp
                yLT[k,:temp.size]=temp
                #temp=Dlog[k][1,:]
                if plot: plt.plot(range(1,11),gm[i,j,hs[i,j],:],alpha=0.25)
                hs[i,j]+=1
                xSC[k]=int(str(LBLS[i])[-1])
                xD[k]=str(LBLS[i]).find('1')
            if plot: 
                plt.xlim([1,11])
                ax.set_xticks(range(1,11))
                plt.ylim([0,30])
                if j>0: ax.set_yticklabels([])
                #if i<2: ax.set_xticklabels([])
                if not j:plt.ylabel(LBLS[i])
                gml=np.exp(np.nanmean(np.log(gm[i,j,:,:]),axis=0))
                if np.ndim(gml)>0: plt.plot(range(1,11),gml,'b')
    yLT[np.isnan(yLT)]=0 
    xC=np.ones(xA.size)*np.nan
    xC[xA<165]=0;xC[np.logical_and(165<=xA,xA<255)]=1;xC[xA>=255]=2
    xAll=np.array([xSC,xD,xA,xC]).T 
    #participant exclusion
    sel=[]
    for i in range(xAll.shape[0]): sel.append(yLT[i,int(xAll[i,0])+1]>0)
    sel=np.array(sel)
    print('perc. of excluded participants = %.02f'%((~sel).mean()*100))
    return yLT[sel,:], xAll[sel,:]         

TN0="""functions{
real ltnormal_rng(real mu, real sigma) {
  real p_lb = normal_cdf(0, mu, sigma);
  real u = uniform_rng(p_lb, 1);
  real y = mu + sigma * Phi(u);
  for (i in 1:1000){
    if (y>=0) break;
    y= mu + sigma * Phi(u);}
   return y;}}"""        
     
DATABLOCK="""    
    data {
        int<lower=0> N; //nr subjects
        real<lower=0> y[N,10];
        int<lower=1,upper=10> xS[N];
        int<lower=1,upper=3> xC[N];
        int<lower=0,upper=4> xD[N];
        real xA[N];
    }parameters {"""
PPARBLOCK="""
        real z0[N];
        real zd[N];
        real zh[N];
        real<lower=0.01,upper=20> sigma[N];
        real<lower=0.01,upper=10> zhs;
        real<lower=0.01,upper=10> zds;
        real<lower=0.01,upper=10> z0s;
        real<lower=-100,upper=100> zhc[2];
        real<lower=-100,upper=100> zhd[3];
        real<lower=-100,upper=100> zdd[3];
        real<lower=-100,upper=100> zdt[3];
        real<lower=-100,upper=100> zdc[2];
        real<lower=-100,upper=100> z0d[3];
        real<lower=-100,upper=100> z0c[2];
    } transformed parameters {
        real<lower=-100,upper=100> zdtT[4];
        real<lower=-100,upper=100> zdcT[3];
        real<lower=-100,upper=100> z0cT[3];
        real<lower=-100,upper=100> zhcT[3];
        zdtT[1]=0;zdtT[2]=zdt[1];zdtT[3]=zdt[2];zdtT[4]=zdt[3];
        zdcT[1]=0;zdcT[2]=zdc[1];zdcT[3]=zdc[2];
        z0cT[1]=0;z0cT[2]=z0c[1];z0cT[3]=z0c[2];
        zhcT[1]=0;zhcT[2]=zhc[1];zhcT[3]=zhc[2];
    }model {"""

NPARBLOCK="""
        real<lower=-100,upper=100> z0[N];
        real<lower=-100,upper=100> zd[N];
        real<lower=-100,upper=100> zh[N];
        real<lower=0.01,upper=20> sigma[N];
    }model {""" 
PPRIORS="""
        zh[n]~normal(zhcT[xC[n]]+zhd[xD[n]],zhs);
        zd[n]~normal(zdtT[xS[n]/2]+zdcT[xC[n]] +zdd[xD[n]],zds); 
        z0[n]~normal(z0d[xD[n]]+z0cT[xC[n]],z0s);""" 
             
TMODBLOCK="""
    for (n in 1:N){{
        sigma[n]~cauchy(0,3);{priors}
        for (t in 1:10)
            if (y[n,t]>0)
                """        
MMODBLOCK="""
    for (n in 1:N){{
        sigma[n]~cauchy(0,3);
        {z0}
        for (t in 2:10)
            if (y[n,t]>0 && y[n,t-1]>0)
                """
UBLOCK='''
        real z0[N];
        real zd[N];
        real zh[N];
        real<lower=0.01,upper=20> sigma[N];
        real<lower=0.01,upper=10> zhs;
        real<lower=0.01,upper=10> zds;
        real<lower=0.01,upper=10> z0s;
        real<lower=-100,upper=100> zhc[2];
        real<lower=-100,upper=100> zhd[3];
        real<lower=-100,upper=100> zdm[3,4];
        real<lower=-100,upper=100> z0d[3];
        real<lower=-100,upper=100> z0c[2];
    } transformed parameters {
        real<lower=-100,upper=100> zhcT[3];
        real<lower=-100,upper=100> z0cT[3];
        zhcT[1]=0;zhcT[2]=zhc[1];zhcT[3]=zhc[2];
        z0cT[1]=0;z0cT[2]=z0c[1];z0cT[3]=z0c[2];
    }model {
    for (n in 1:N){
        sigma[n]~cauchy(0,3);
        zh[n]~normal(zhcT[xC[n]]+zhd[xD[n]],zhs);
        zd[n]~normal(zdm[xD[n],xS[n]/2],zds); 
        z0[n]~normal(z0d[xD[n]]+z0cT[xC[n]],z0s);
        for (t in 1:10)
            if (y[n,t]>0)'''                           
    
def smCodeGenerator(nm):
    ''' returns Stan code string for the model specified by nm
        nm - string of length 3
            The prefix indicates the truncated Normal (N), EN (E), 
            Lognormal (L), Weibull (W) and Gamma (G) distribution. 
            The infix indicates a trend (T), autoregressive (A) or 
            quadratic (Q) model. The suffix indicates a no-pooling (N) 
            or partial-pooling (P) version of the model.'''
    distr={'N':'normal','X':'normal','L':'lognormal',
        'W':'weibull','G':'gamma'}[nm[0]]
    
    out=DATABLOCK
    if nm[1]=='Q':
        if nm[2]=='N': out+='\n\treal<lower=-30,upper=30> zt[N];'
        elif nm[2]=='P': out+='real<lower=-30,upper=30> zt;\n'#real<lower=-10,upper=20> ztm;\nreal<lower=0,upper=10> zts;\n'
    
    if nm[2]=='N': out+=NPARBLOCK
    elif nm[2]=='P': out+=PPARBLOCK
    elif nm[2]=='U':out+=UBLOCK
    if nm[1]=='T' or nm[1]=='Q':
        if nm[2]=='N':out+=TMODBLOCK.format(priors='')
        elif nm[2]=='P':out+=TMODBLOCK.format(priors=PPRIORS)
    elif nm[1]=='M':
        mu='z0[n]'
        if nm[0] in 'XG': mu='exp('+mu+')'
        elif nm[0]=='W': mu='exp(-('+mu+'))'
        if nm[0] in 'WG':z0= f'y[n,1]~{distr}(sigma[n], {mu});'
        else: z0= f'y[n,1]~{distr}({mu},sigma[n]);'
        if nm[2]=='N':out+=MMODBLOCK.format(z0='\n\t'+z0)
        elif nm[2]=='P':out+=MMODBLOCK.format(z0=PPRIORS+'\n\t'+z0)

    if nm[1]=='T': mu='z0[n]+(t-1)*zh[n]+(xS[n]<=t)*zd[n]'
    elif nm[1]=='Q':
        mu='z0[n]+zh[n]*square(t-1-zt'+['[n]',
            ''][int(nm[2]=='P')]+')+(xS[n]<=t)*zd[n]'
    elif nm[1]=='M':
        mu='log(y[n,t-1])+zh[n]+(xS[n]==t)*zd[n]'
        if nm[0]=='N':mu='y[n,t-1]+zh[n]+(xS[n]==t)*zd[n]'
    if nm[0] in 'XG': mu='exp('+mu+')'
    elif nm[0]=='W': mu='exp(-('+mu+'))'
    if nm[0] in 'GW': out+=f'y[n,t]~{distr}(sigma[n],{mu});'
    else: out+=f'y[n,t]~{distr}({mu},sigma[n]);'
    out+='\n}}'
    return out  
  
  
def fit2dict(fit,w0=None):
    '''translates Stanfit data class to Python dictionary''' 
    w=fit.extract()
    w['lp__']=w['lp__'][:,np.newaxis]
    if not w0 is None:
        for k in w.keys():
            w[k]=np.concatenate([w0[k],w[k]],axis=1)
    temp=fit.summary()
    sumr=temp['summary']
    w['nms']=temp['summary_rownames']
    if not w0 is None: assert(np.all(w['nms']==w0['nms']))
    if w0 is None: w['rhat']=sumr[np.newaxis,:,-1]
    else: w['rhat']=np.concatenate([w0['rhat'],sumr[np.newaxis,:,-1]],axis=0)
    return w
    
        
def fitmodel(nm,compileModel=True,omit=[]):
    ''' fit model specified by nm
        nm - string of length 3
            first character describes probability distribution
            second characted describes type temporal structure
            third character describes the type of pooling
        compileModel - if False, previously compiled model will be loaded
        omit - string of integers indicating censored trials
        
        results are saved to DPATH 
    '''
    import pystan
    yLT=np.load(DPATH+'yLT.npy')
    xAll=np.load(DPATH+'xAll.npy')
    if compileModel:
        smc=smCodeGenerator(nm)
        print(smc)
        sm = pystan.StanModel(model_code=smc)
        #with open(DPATH+f'{nm}.sm', 'wb') as f: pickle.dump(sm, f)  
    nmo=nm+['','o'][len(omit)]
    for om in omit: yLT[:,om]=0
    if nm[2]=='N': 
        a2=np.atleast_2d;a1=np.atleast_1d
        #with open(DPATH+f'{nm}.sm', 'rb') as f: sm=pickle.load(f)
        for n in range(yLT.shape[0]):
            temp={'y':a2(yLT[n,:]),'N':1,'xS':a1(np.int32(xAll[n,0])+1),
                'xC':a1(np.int32(xAll[n,3])+1),'xD':a1(np.int32(xAll[n,1])+1),
                'xA':a1((xAll[n,2]-7*30)/30)}
            try:
                fit=sm.sampling(data=temp,chains=6,n_jobs=6,
                        seed=SEED,thin=10,iter=10000,warmup=6000)
                if n==0: w=fit2dict(fit)
                else: w=fit2dict(fit,w0=w) 
            except RuntimeError:
                print('n=',n)
                if n==0: print('not implemented')
                else: w['rhat']=np.vstack([w['rhat'],np.ones(w['rhat'].shape[1])*np.nan])
        np.save(DPATH+f'{nmo}sf',w)
    else: 
        temp={'y':yLT,'N':yLT.shape[0],'xS':np.int32(xAll[:,0])+1,
                'xC':np.int32(xAll[:,3])+1,'xD':np.int32(xAll[:,1])+1,
                'xA':(xAll[:,2]-7*30)/30}
        #with open(DPATH+f'{nm}.sm', 'rb') as f: sm=pickle.load(f)
        fit=sm.sampling(data=temp,chains=6,n_jobs=6,
                seed=SEED,thin=10,iter=10000,warmup=6000)
        #with open(DPATH+f'{nm}.stanfit','wb') as f: pickle.dump(fit,f,protocol=-1)
        np.save(DPATH+f'{nmo}sf',fit2dict(fit))
        sumr=fit.summary()
        print(np.nanmax(sumr['summary'][:-1,-1]))


def compareModels(suf,N=1000,tt=-1):
    ''' compute models comaprison and save to DPATH
        suf - string of length 2
            first character describes temporal structure
            second character describes the type of pooling
        N - number of generated samples used to compute the absolute error
        tt- index of selected trial on which comparison is computed
            default =-1 ie all trials are used'''
    na=np.newaxis
    yLT=np.load(DPATH+'yLT.npy')
    yLT[yLT==0]=np.nan
    xAll=np.load(DPATH+'xAll.npy')
    out=[];out2=[];out3=[]
    for j in range(5):
        nm='NXLWG'[j]+suf
        out.append(np.nan*np.zeros(N))
        out2.append(np.nan);out3.append(np.nan*np.zeros((101,yLT.shape[0],10)))
        try:
            w=np.load(DPATH+f'{nm}sf.npy',allow_pickle=True).tolist()
        except:continue
        if suf[1]=='N':
            sel1=~np.isnan(w['rhat'][:,0])
            #print('sel1', sel1.shape,sel1.sum())
            rh=w['rhat'][sel1,:]
            sel2=(rh>1.1).sum(1)==0
            #print(w['rhat'].shape)
            #print('sel2', sel2.shape,sel2.sum())
            y=yLT[sel1,:][sel2,:]
            x=xAll[sel1,:][sel2,:]
            pars=['z0','zh','zd','sigma']
            if suf[0]=='Q':pars.append('zt')
            for k in pars: w[k]=w[k][:,sel2]
            #print(nm,y.shape[0]/yLT.shape[0]*100,np.median(np.median(w['zt'],0)))
        else: 
            print(w['rhat'].shape)
            print(nm,'worst rhat',np.nanmax(w['rhat'][0,:-1]))
            if np.nansum(w['rhat'][0,:-1]>1.1)!=0:continue
            y=yLT;x=xAll
        mu=[]
        for t in range(1,11):
            if suf[0] in 'TQ':
                if suf[0]=='Q': temp=np.square(t-1-np.median(w['zt'],0))
                else: temp=t-1
                mu.append(np.median(w['z0'],0)+temp*np.median(w['zh'],0)+np.int32(x[:,0]+1<=t)*np.median(w['zd'],0))
            elif suf[0]=='M': 
                if t==1: mu.append(np.median(w['z0'],0))
                else: 
                    temp=[np.log(y[:,t-2]),y[:,t-2]][int(nm[0]=='N')]
                    mu.append(temp+np.median(w['zh'],0)+np.int32(x[:,0]+1==t)*np.median(w['zd'],0))
                    mu[-1][np.isnan(y[:,t-1])]=0
        mu=np.array(mu).T
        sigma=np.median(w['sigma'],0)[:,na]
        
        if nm[0]=='N':v=stats.truncnorm(-mu/sigma,np.inf,mu,sigma)
        elif nm[0]=='X':v=stats.truncnorm(-np.exp(mu)/sigma,np.inf,np.exp(mu),sigma)
        elif nm[0]=='L':v=stats.lognorm(sigma,scale=np.exp(mu))
        elif nm[0]=='W':v=stats.weibull_min(sigma,scale=np.exp(-mu))
        elif nm[0]=='G':v=stats.gamma(sigma,scale=np.exp(-mu))
        try: yhat=v.rvs(size=(N,mu.shape[0],mu.shape[1]))
        except: yhat=np.nan*np.zeros((N,1,1))
        eta=y[na,:,:]-yhat
        if tt!=-1: eta=eta[:,:,tt][:,:,na]
        out[-1]=np.nanmean(np.abs(eta),axis=(1,2))
        try: p=v.pdf(y)
        except:p=np.nan
        if np.any(p==0):
            print('inf likelihood nr',(p==0).sum())
            p[p==0]=np.nan
        LL=np.log(p)
        
        if tt!=-1 and LL.ndim>1: LL=LL[:,tt]
        print('LL nan # ',np.isnan(LL).sum())
        out2[-1]=np.nansum(LL)
        if suf[1]!='N':out3[-1]=v.pdf(np.linspace(0,20,101)[:,na,na])
        else:out3[-1][:,sel1,:][:,sel2,:]=v.pdf(np.linspace(0,20,101)[:,na,na]) 
        
    np.save(DPATH+f'cae{suf}.npy',out)  
    np.save(DPATH+f'll{suf}.npy',out2) 
    np.save(DPATH+f'mp{suf}.npy',out3)

def plotComparison(pred=False):
    ''' plots model comparison in terms of data log-likelihood and 
        mean absolute error in seconds
        pred - if pred is true plots predictive performance
    '''
    pred=int(pred)
    if pred:
        nms=['NTPo','ETPo','LTPo','WTPo','GTPo','NTNo','LTNo','WTNo','GTNo']
        ylims=[[-700,-1200],[3.5,6]] 
        figure(size=3,aspect=0.6)
    else:
        nms=['NTP','ETP','LTP','WTP','GTP','NTN','LTN','WTN','GTN','EMP','WMP','GMP','NMN','LMN','WMN','GMN','NQP','LQP']
        ylims=[[-5400,-6800],[3.5,10.5],[3.6,4.5]]
        figure(size=3,aspect=1)
    CLR={'P':'k','N':'grey'}
    for k in range(len(nms)):
        out=np.load(DPATH+f'll{nms[k][1:]}.npy')
        i='NELWG'.index(nms[k][0])
        if pred: subplot(1,2,1)
        else: subplot(3,1,1)
        
        plt.plot(k,out[i],'+',color=CLR[nms[k][2]])
        out=np.load(DPATH+f'cae{nms[k][1:]}.npy') 
        if pred: subplot(1,2,2)
        else: subplot(3,1,2)
        print(nms[k],errorbar(np.array(out[i,:],ndmin=2).T,x=[k],clr=CLR[nms[k][2]]))
        if not pred: 
            subplot(3,1,3)
            errorbar(np.array(out[i,:],ndmin=2).T,x=[k],clr=CLR[nms[k][2]])
    for h in range([3,2][pred]):
        if pred: ax=subplot(1,2,h+1)
        else: ax=subplot(3,1,h+1)
        ofs=[0.1,0.9]
        ax.set_xticks(range(len(nms)))
        plt.xlim([-0.5,len(nms)-0.5])
        nms=list(map(lambda x: x.replace('M','A'),nms))
        ax.set_xticklabels([nms,list(map(lambda x: x[:-1],nms))][pred])
        if len(ylims):plt.ylim(ylims[h])
        plt.ylabel(['Log-likelihood','Mean absolute'+['\n',' '][pred] +'error in sec.'][int(h>0)])
        subplotAnnotate()
    plt.savefig(FPATH+['lik','pred'][pred]+'.png',bbox_inches='tight',dpi=DPI)

def plotDistribution():
    ''' plots the pdf for each infant with the median model parameters'''
    figure(size=3,aspect=1)
    sufs=['TP','MP','QP']
    for h in range(len(sufs)):
        S=np.load(f'data/mp{sufs[h]}.npy',allow_pickle=True)
        i=0;j=0
        for i in range(5):
            if np.isnan(S[i,0,0,0]):continue
            subplot(3,5,i+1+5*min(1,h))
            for k in range(301):

                plt.plot(np.linspace(0,20,101),S[i,:,k,j],'k',alpha=0.02)
            plt.ylim([0,0.4])
            plt.title('NELWG'[i]+sufs[h][0])
            if i: plt.gca().set_yticklabels([])
            if h==0: plt.gca().set_xticklabels([])
            plt.grid(False)
    plt.savefig(FPATH+'distr.png',bbox_inches='tight',dpi=DPI)

def plotMC(suf,ylims=[]):
    '''earlier routing to plot model comparison'''
    figure(size=2,aspect=0.5)
    subplot(1,2,2)
    out=np.load(DPATH+f'cae{suf}.npy')  
    print(errorbar(out.T))
    plt.gca().set_xticks(range(6))
    plt.gca().set_xticklabels(['N','EN','LN','W','G'])
    plt.xlim([-1,5])
    if len(ylims): plt.ylim(ylims[1])
    plt.title('Mean Absolute Error');  
    subplot(1,2,1)
    out=np.load(DPATH+f'll{suf}.npy')  
    print(out)   
    plt.plot(out,'+')
    plt.gca().set_xticks(range(6))
    plt.gca().set_xticklabels(['N','EN','LN','W','G'])
    plt.xlim([-1,5])
    plt.title('Log-Likelihood');  
    #plt.gca().invert_yaxis()
    if len(ylims):plt.ylim(ylims[0])
    plt.savefig(FPATH+f'mc{suf}.png',bbox_inches='tight',dpi=DPI)


def unpooledAnova():
    ''' print latex table with results of an omnibus ANOVA that 
        was applied to the parameters from the no-pooling models'''
    import pandas as pd
    import statsmodels.api as st
    import statsmodels.formula.api as sf
    out=[];#lbls={'N':'N','X':'EN','L':'LN','W':'W','G':'G'}
    for suf in ['TN','MN']:
        for nm in 'NLWG':
            w=np.load(DPATH+f'{nm}{suf}sf.npy',allow_pickle=True).tolist()
            if suf[1]=='N':sel=(w['rhat']>1.1).sum(1)==0
            else: sel=np.bool8(np.ones(w['rhat'].shape[0]))
            z0=np.median(w['z0'],0)[sel,np.newaxis]
            zd=np.median(w['zd'],0)[sel,np.newaxis]
            zh=np.median(w['zh'],0)[sel,np.newaxis]
            xAll=np.load(DPATH+'xAll.npy')[sel,:]
            df=pd.DataFrame(np.concatenate([z0,zd,zh,xAll],axis=1),
                columns=['z0','zd','zh','t','d','age','c'])
            out.append([nm])
            for f in ['z0 ~ C(d)*C(c)','zh ~ C(d)*C(c)','zd ~ C(d)*C(c)*C(t)']: 
                model=sf.ols(f,df).fit()
                res=st.stats.anova_lm(model, typ=2)
                if f[1]=='d': sig=res.to_numpy()[:-1,-1][[0,1,3,2,4,5,6]]<0.05
                else: sig=res.to_numpy()[:-1,-1]<0.05
                out[-1].extend(np.array(['','$\star$'])[np.int32(sig)])
    col1=['','$\\alpha$','','','$\\beta$','','','$\\gamma$','','','','','','']
    col2=['']+3*['$S$','$A$', '$S \\times A$'] \
        +['$T$', '$T \\times S$','$T \\times A$', '$T \\times S \\times A$']
    tab=np.array([col1,col2]+out)
    ndarray2latextable(tab.T,hline=[0,3,6])

def plotME(suf,avg=False,mask=[]):
    ''' Estimate the main effect of age, stimulus type and number of 
        habituation trials on the parameters of the set of models 
        specified by suf 
        suf - string of length 2
            first character describes temporal structure
            second character describes the type of pooling
        avg - if true use average over infant-level estimates instead
            of population-level estimates
        mask - integer list of columns in the figure to hide  
    '''
    from matusplotlib import errorbar,plotCIttest1,figure,subplot
    import pylab as plt
    xAll=np.load(DPATH+'xAll.npy')
    #figure(size=3,aspect=1)
    plt.figure(figsize=(10,10))

    for j in range(5):
        if j in set(mask):continue
        nm='NXLWG'[j]+suf
        if suf[1]=='N' and nm[0]=='X':continue 
        try: w=np.load(DPATH+f'{nm}sf.npy',allow_pickle=True).tolist()
        except: continue
        w['rhat']=w['rhat'][~np.isnan(w['rhat'][:,0])]
        if suf[1]=='N':sel=(w['rhat']>1.1).sum(1)==0
        else: sel=np.bool8(np.ones(w['rhat'].shape[0]))
        if suf[0]=='Q':print(nm,errorbar(w['zt']-1,plot=False))
        for i in range(4):
            #if suf[1]=='N': ax=subplot(4,4,i*4+[0,None,1,2,3][j]+1)

            ax=subplot(4,5,i*5+j+1)
            
            if not i: plt.title(['Truncated Normal','EN','Lognormal','Weibull','Gamma'][j])
            
            #if suf[0]=='Q':sign=1
            if nm[0]=='W' or nm[0]=='G':sign=-1
            else: sign=1
            if avg:
                z=np.median(w[['z0','zh','zd','zd'][i]],0)[sel,...]
                if i<2: b=z[np.logical_and(xAll[sel,1]==0,xAll[sel,3]==0)].mean()
                else: b=z[np.logical_and(xAll[sel,0]==1,np.logical_and(xAll[sel,1]==0,xAll[sel,3]==0))].mean()
                for h in range(5): 
                    if i<3:
                        if h<3:tmp=sign*(z[xAll[sel,1]==2-h]-z[xAll[sel,1]==0].mean()+b)
                        else: tmp=sign*(z[xAll[sel,3]==5-h]-z[xAll[sel,3]==0].mean()+b)
                    else:tmp=sign*(z[xAll[sel,0]==2*h+1]-z[xAll[sel,0]==1].mean()+b)
                    if i>0 and nm[0]!='N' and nm[1]!='Q': tmp=(np.exp(tmp)-1)*100
                    plotCIttest1(tmp,x=[h,7-h][int(i<3 and h>2)],clr='k')
            else: 
                k=['z0','zh','zd','zd'][i]
                
                tmp=w[k+'d'][:,0][:,np.newaxis]
                if i==3: y=np.concatenate([tmp,w[k+'t']+tmp],axis=1)
                else: y=np.concatenate([w[k+'d'][:,::-1],w[k+'c']+tmp],axis=1)
                if i>0 and nm[0]!='N' : errorbar((np.exp(sign*y)-1)*100,clr='k')
                else: errorbar(sign*y,clr='k')
            
            if nm[1:]=='TP' and nm[0]!='N':
                if i==1: plt.ylim([-16,2])
                elif i==2: plt.ylim([-40,120])
                elif i==3: plt.ylim([0,160])
            if nm[1:]=='TN' and nm[0]!='N':
                if i==1: plt.ylim([-8,8])
                elif i==2: plt.ylim([0,250])
                elif i==3: plt.ylim([0,350])
            #if j>1 and i>0:plt.gca().set_yticklabels([])
            ax.set_xticks(range([5,4][i==3]))
            ax.set_xticklabels([['4O','4C','4L','7L','10L'],['4O','4C','4L','7L','10L'],['4O1','4C1','4L1','7L1','10L1'],['4L1','4L3','4L5','4L7']][i]);
            if not j:plt.ylabel(['offset','slope','dishabituation'][min(i,2)])
    avg=int(avg)
    plt.savefig(FPATH+f'me{suf}.png',bbox_inches='tight',dpi=DPI)

    
def parCor():
    ''' prints the latex table with correlation between the 
        median parameter estimates of infants'''
    from matusplotlib import ndarray2latextable

    out=[[' ',' ','$r_{\\alpha,\\sigma}$','$r_{\\beta,\\sigma}$',
        '$r_{\\gamma,\\sigma}$','$r_{\\alpha,\\beta}$','$r_{\\alpha,\\gamma}$','$r_{\\beta,\\gamma}$']]
    for suf in ['TP','TN','MN']:
        for nm in 'NXLWG':
            if (suf=='TN' or suf=='MN') and nm=='X':continue
            try: w=np.load(DPATH+f'{nm}{suf}sf.npy',allow_pickle=True).tolist()
            except: continue
            #w['rhat']=w['rhat'][~np.isnan(w['rhat'][:,0])]
            #if suf[1]=='N':sel=(w['rhat']>1.1).sum(1)==0
            #else: sel=np.bool8(np.ones(w['rhat'].shape[0]))
            z0=np.median(w['z0'],0)
            zd=np.median(w['zd'],0)
            zh=np.median(w['zh'],0)
            sig=np.median(w['sigma'],0)
            if nm=='W' or nm=='G': sig= -sig
            out.append([nm,suf])
            for i in range(4):
                for j in range(i+1,4):
                    r=np.corrcoef([sig,z0,zh,zd][i],[sig,z0,zh,zd][j])
                    out[-1].append(np.round(r[0,1],2))
    ndarray2latextable(np.array(out,dtype=object),decim=2)
def plotDishabX(nm='L'):
    ''' plot estimates of the interaction of stimulus type and 
        number of habituation trials on the dishabituation parameter 
        of the trend model with partial pooling
        nm - specifies the type of distribution'''
    from matusplotlib import errorbar,figure
    import pylab as plt
    figure(size=3,aspect=0.6)
    w=np.load(DPATH+f'{nm}TUsf.npy',allow_pickle=True).tolist()
    print(nm,np.nanmax(w['rhat'][0,:-1]))
    zd=w['zdm']
    if nm=='W' or nm=='G':sgn=-1
    else: sgn=1
    clrs=['k','grey','lightgrey']#plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(3): errorbar(100*sgn*zd[:,i,:],x=np.arange(4)+i*4,clr=clrs[i])
    plt.xlim([-1,12])
    #plt.title(nm+' %.3f'%np.nanmax(sumr['summary'][:-1,-1]))
    plt.grid(True,axis='y')
    plt.ylabel('Multiplicative look duration change on dishabituation')
    plt.gca().set_xticks(range(12))
    plt.gca().set_xticklabels(['B1','B3','B5','B7','C1','C3', 'C5','C7','O1','O3','O5','O7'])
    plt.savefig(FPATH+f'dishabX{nm}.png',bbox_inches='tight',dpi=DPI)
    print(errorbar(100*sgn*(zd[:,0,2]+zd[:,0,3]-zd[:,0,0]-zd[:,0,1]+
        zd[:,1,2]+zd[:,1,3]-zd[:,1,0]-zd[:,1,1])/4))

def printConverged():
    ''' prints latex table with the number of infants for which 
        the estimation of the no-pooling model failed to converge'''
    for k in range(4):
        suf=['N','P'][int(k/2)]+['','o'][k%2]
        D=np.zeros((3,5))*np.nan
        for i in range(3):
            for j in range(5):
                nm='NXLWG'[j]+['T','Q','M'][i]+suf
                try:w=np.load(DPATH+f'{nm}sf.npy',allow_pickle=True).tolist()
                except:continue
                if int(k/2): 
                    rs=w['rhat'][0,:-1][~np.isnan(w['rhat'][0,:-1])]
                    #print(nm,(rs<1.1).sum(),rs.size);bla
                    D[i,j]=(rs<1.1).sum()==rs.size
                else: D[i,j]=w['rhat'].shape[0]-np.sum((w['rhat'][:,:-1]<1.1).sum(1)==w['rhat'].shape[1]-1)
        from matusplotlib import ndarray2latextable
        print(suf)
        D[np.isnan(D)]=-1
        print(ndarray2latextable(D,decim=0) )   

if __name__=='__main__':
    plotME('TN',avg=True)
    plotME('MN',avg=True)
    plotME('QP',avg=False,mask=[1,3,4])
    plotME('TP',avg=False);bla
    # get data
    info,Dres=loadData()
    yLT,xAll=preprocessData(info,Dres) 
    np.save('data/yLT',yLT)
    np.save('data/xAll',xAll) 
    # fit models
    for suf in ['TN','TP','MP','MN','QP']:
        for nm in 'NXLWG':
            fitmodel(nm+suf,compileModel=True)  
    for nm in 'NXLWG':
        fitmodel(nm+'TN',compileModel=False,omit=[4]) 
        fitmodel(nm+'TP',compileModel=False,omit=[4])   
        fitmodel(nm+'MN',compileModel=False,omit=[4]) 
        fitmodel(nm+'MP',compileModel=False,omit=[4])
    for nm in 'NXLWG':
        fitmodel(nm+'TU',compileModel=True)  
    for suf in ['TN','TP','TPo','TNo','MN','MP','QP']:
        if len(suf)==3: compareModels(suf,tt=4)
        else: compareModels(suf)
        
    # generated figures and tables from the manuscript
    unpooledAnova()
    plotME('TN',avg=True)
    plotME('TP',avg=False)
    plotME('MN',avg=True)
    plotME('QP',avg=False,mask=[1,3,4])
    
    plotComparison()
    plotComparison(pred=True) 
    parCor()
    plotDishabX(nm='L')
    printConverged()
    
    # misc routines, figures not reported
    plotDistribution()
    plotSample(info,Dres)
    
    plotModelComparison('TP',ylims=[[-5700,-6100],[3.7,4.3]])
    plotModelComparison('TN',[[-5400,-6000],[3.7,5]]) 
    plotModelComparison('TPo',ylims=[[-700,-900],[3.5,5]])
    plotModelComparison('TNo',ylims=[[-700,-900],[3.5,5]])
    plotModelComparison('MN')
    plotModelComparison('MP')

