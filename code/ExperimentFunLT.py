# -*- coding: utf-8 -*-
from psychopy import visual, core, gui,parallel,event,prefs
from psychopy.misc import pix2deg, deg2pix,cm2deg
from ExperimentManager import *
from Constants import *
import os, time
import numpy as np
np.set_printoptions(suppress=True)
# color stack
COLORS=['r','g','b','y','k','c']

def createRectangleMask(H2Wratio,N=16): 
    ''' the mask has NxN resolut ion
        the size of the rectangle is Nx floor(H2Wration*N)
    '''
    import numpy as np
    out=-np.ones((N,N))
    for j in range(N):
        if np.abs((N-1)/2.-j)<N*H2Wratio/2.:
            out[:,j]=1 
    return out

class Q():
    expName='funLT'
    monitor=asusMG279
    refreshRate=120 # [hz]
    screen=0
    fullscr= True
    winPos=(0,0)# in pixels
    bckgCLR= [-1,-1,-1]
    itemCLR= 1
    itemSize=1.8 # in degrees of visial angle
    stimOffset=(0,33.5-22/2.-33.5/2.) #stimulus offset with respect to 
    #the screen center in cm,transformed to deg below
    nrFactorLevels= 5
    rectH2Wratio=0.6
    nrTrials=8
    endTrialAfter=1 # seconds
    startTrialAfter=0.2# seconds
    interTrialInterval=1# seconds
    waitGazeMax=40#seconds max time to wait until infant saw trial
    aoiBoxSizeCm=(30,30)
    ###############################################
    # transformations
    ###############################################
    refreshRate=float(refreshRate)
    itemMask=createRectangleMask(rectH2Wratio)
    from os import getcwd as __getcwd
    __path = __getcwd()
    __path = __path.rstrip('code')
    from os.path import sep as __sep
    inputPath=__path+"input"+__sep
    outputPath=__path+"output"+__sep
    i2idist=2*itemSize
    xyGaussNoiseSD=0.1*itemSize
    stimOffset=[cm2deg(stimOffset[0],monitor),
            cm2deg(stimOffset[1],monitor)]
    stimOffset=np.array(stimOffset)
    aoiBoxSizeDeg=cm2deg(np.array(aoiBoxSizeCm),monitor)
    
def isOnAOI(x,y):
    if np.isnan(x)or np.isnan(y): return False
    aoi=Q.aoiBoxSizeDeg/2.
    return -aoi[0]<x and aoi[0]>x and -aoi[1]<y and aoi[1]>y
    
def initElem(arrayRowSize,win,offset=0): 
    elem=visual.ElementArrayStim(win,fieldShape='sqr',
                nElements=arrayRowSize**2, sizes=Q.itemSize,interpolate=False,
                elementMask=Q.itemMask ,elementTex=None)
    n=arrayRowSize
    xys=np.meshgrid(range(n),range(n))
    center=n/2.-0.5
    xys=Q.i2idist*(np.array([xys[0].flatten(),xys[1].flatten()]).T-center)
    #print xys, center
    elem.setXYs(xys-offset)
    return elem
    
    
def createStimulus(level,win,offset=0):
    ''' elem - psychopy.visual.ElementArrayStim instance
        dim - perceptual dimension
        level - from 1 to 5 
    '''
    #TODO make indexing independent of nr of levels
    stimRGB=np.load('stimulusRGBsDKLlogLum.npy')[1:,3:6]
    stimRGB=(stimRGB-0.5)*2
    elem=initElem(level[SIZE],win,offset)
    mapp=[5,6,2,7,8]
    if level[COLOR]==3: elem.setColors(stimRGB[level[BRIGHT]-1])#grey  
    elif level[BRIGHT]==3: elem.setColors(stimRGB[mapp[level[COLOR]-1]]) 
    else: raise NotImplementedError('brightness and color interactions not implemented') 
    elem.setOris(22.5*(level[ORIENT]-3))
    return elem

def stimulusColorPreview(scale=4):
    '''displays the color and light patches
        linear luminance on left, loglinear on right'''
    wind=QinitDisplay(Q)
    for p in [-3,3]:
        stimRGB=np.load('stimulusRGBs%s.npy'%['DKL','DKLlogLum'][int(p>0)])[:,3:6]
        stimRGB=(stimRGB-0.5)*2
        #stimRGB=stimRGB[range(0,8)+range(9,11),:]
        elem=visual.ElementArrayStim(wind,nElements=9, sizes=[scale,scale/2.],
                interpolate=False,elementMask='sqr' ,elementTex=None)
        pos2=np.array([np.linspace(-2,2,5)[[0,1,3,4]], np.zeros(4)])
        pos1=np.array([np.zeros(5),np.linspace(-2,2,5)])
        pos=np.concatenate([pos1,pos2],axis=1).T
        pos[:,X]+=p
        elem.setXYs(pos*scale)
        elem.setColors(stimRGB[1:,:])
        elem.draw()
    wind.flip()
    while True:
        for key in event.getKeys():
            if key in ['escape']:
                wind.close()
                core.quit()
                break

def showDimensionsNlevels():
    '''shows stimuli for dimensions luminance, color and orientation'''
    win=QinitDisplay(Q)
    Q.itemSize=3
    scale= 5
    for i in range(3):
        for j in range(Q.nrFactorLevels):
            offset=np.array([(j-2)*scale,(i-1)*scale],ndmin=2)  
            print(offset, i)
            stim=np.ones(4,dtype=np.int32)*3
            stim[i]=j+1
            stim[3]=1
            elem=createStimulus(stim,win,offset)
            elem.draw()
    win.flip()
    event.clearEvents()
    while True:
        for key in event.getKeys():
            if key in ['escape']:
                core.quit()

class Experiment():
    def __init__(self):
        self.f=-1
        self.EM=ExperimentManager(ecallback=self.controlCallback,
            fcallback=self.getFrameIndex,Qexp=Q,loglevel=2)
        # check if eye tracker is properly configured
        assert(self.EM.ET.geometryName=='Asus FunLT')
        assert(self.EM.ET.refreshRate==60)
        self.schedulefn=Q.expName+self.EM.cohort+'.sched'
        schedule=np.loadtxt(Q.inputPath+self.schedulefn)
        temp=list(map(int,str(int(schedule[0]))))
        self.expinfo=np.array(temp,ndmin=2)
        self.win=self.EM.getExpWin() 
        # init vars
        self.jumpToNextBlock=False
        self.onAOISince=-1 # duration of looking on screen
        self.offAOISince=-1 # duration of looking away from screen
        self.ttf=-1
        self.sawTrial=False
        self.k=0
        self.EM.start()

    def getFrameIndex(self):
        return self.f
    def controlCallback(self,command):
        if command==1 or command==-1: 
            self.jumpToNextBlock=True

    def checkGaze(self):
        sd=self.EM.ET.getLatestGaze()
        if not sd is None:
            if (not (sd[LDIAM]>0 and isOnAOI(sd[LX],sd[LY])) and not
                (sd[RDIAM]>0 and isOnAOI(sd[RX],sd[RY]))):
                self.onAOISince=-1
                if self.offAOISince==-1: 
                    self.offAOISince=core.getTime()
                    #print 'left AOI'
                    self.EM.writeLog('left AOI',self.f)
            else:
                self.offAOISince=-1
                if self.onAOISince==-1: 
                    #print 'entered AOI'
                    self.EM.writeLog('entered AOI',self.f)
                    self.onAOISince=core.getTime()
                
    def trialFinished(self):
        if self.jumpToNextBlock: return True
        self.ttf=core.getTime()
        if self.onAOISince>0 and (core.getTime()-self.onAOISince)>Q.startTrialAfter:
            if self.sawTrial==False:
                self.EM.writeShortLog('{},'.format(self.onAOISince-self.t0))
                self.EM.writeLog('saw Trial',self.f)
                print('saw Trial')
                self.sawTrial=True
        out= self.sawTrial and self.offAOISince>0 and (core.getTime()-self.offAOISince)>Q.endTrialAfter

        return out
    def trialStarted(self):
        if self.jumpToNextBlock: return True
        return (core.getTime()-self.ttf)>Q.interTrialInterval
    def showTrial(self):        
        print('block: ',self.block,'trial:',self.trial)
        if self.trial>=self.expinfo[self.block,8]: stim=self.testStim
        else: stim = self.habStim
        oldxys=np.copy(stim.xys)
        newxys=oldxys+Q.xyGaussNoiseSD**2*np.random.multivariate_normal([0,0],
                [[1,0],[0,1]],size=stim.nElements)
        stim.setXYs(newxys)
        stim.draw()
        self.win.flip()
        event.clearEvents()
        self.t0=core.getTime()
        self.EM.writeLog('trialon '+str(newxys.tolist()))
        self.EM.startSound()
        self.f=0
        while not self.trialFinished() and not self.jumpToNextBlock:
            self.checkGaze()
            stim.draw()
            self.win.flip()
            if core.getTime()-self.t0>Q.waitGazeMax and not self.sawTrial:
                self.EM.writeLog('waitGazeMax exceeded')
                print( 'waitGazeMax exceeded, terminating experiment')
                self.jumpToNextBlock=True
            self.f+=1
        self.EM.writeLog('trialoff')
        self.f=-1
        self.EM.stopSound()
        stim.setXYs(oldxys)
        if not self.sawTrial:s='-1,-1'
        else:
            s='{}'.format(self.offAOISince-self.t0)
            if (self.trial+1)<Q.nrTrials: s+=','
        self.EM.writeShortLog(s)
        self.sawTrial=False
        self.onAOISince=-1 
        self.offAOISince=-1 
        
    def showIntertrial(self):
        self.win.flip()
        event.clearEvents()
        self.t0=core.getTime()
        while not self.trialStarted() and not self.jumpToNextBlock: 
            self.win.flip()

            
    def showBlock(self):
        self.trial=-1
        for trial in range(Q.nrTrials):
            self.trial=trial
            self.showTrial()
            if self.jumpToNextBlock: break
            self.showIntertrial()
            if self.jumpToNextBlock: break
        self.win.flip()
        


    def run(self):
        ei=self.expinfo
        self.block=0
        while self.block<ei.shape[0]:
            self.jumpToNextBlock=False
            b=self.block
            self.habStim=createStimulus(ei[b,:4],
                self.win,offset=Q.stimOffset)
            self.testStim=createStimulus(ei[b,4:8],
                self.win,offset=Q.stimOffset)
            self.showTest=False
            self.EM.writeShortLog(('{}'*ei.shape[1]+',').format(*(ei[b,:])))
            self.EM.writeLog(('Experiment Schedule '+'{}'*ei.shape[1]).format(*(ei[b,:])))
            self.EM.waitForKey()
            self.showBlock()
            self.EM.writeShortLog('\n')
            self.block+=1
        self.EM.terminate()
        
if __name__ == '__main__':
    exp=Experiment()
    exp.run()
    #stimulusColorPreview()
    #showDimensionsNlevels()


