import numpy as np
import scipy.stats as sp
from scipy.optimize import minimize
import scipy as sc
from scipy import signal as sig
import scipy.special as sc
import powerlaw
import pickle
import glob, os.path
import networkx as nx
import pandas as pd

def autocorr(timeseries: list[float],
             stepnumber: int
             )-> list[list[int],list[float]]:
    """Computes autocorrelation function using a multistep regression estimation.

    Args:
        timeseries (list[float]): timeseries to compute autocorrelation of.
        stepnumber (int): number of regression steps to take.

    Returns:
        steps (list[int]): list of integer steps [1, 2, ..., stepnumber]
        rk (list[float]): multistep regression estimate (autocorrelation function)
    """
    rk=[]
    steps=np.arange(1,stepnumber,1)
    for k in steps:
        res=sp.linregress(timeseries[:-k], timeseries[k:])
        rk.append(res[0])
    return [steps,rk]

# Function 2: Simulation
def full_sim4(potential,N,time_max,w,ext_cur,k,ninhb,J,weakening,weakeninginhb,jwidth,style,ratee,flag,trialnumber):
    fullsizes=[]
    fullnumber=[]
    fullstimes=[]
    fulldurations=[]
    fullstresses=[]
    for triall in range(trialnumber):
        size=np.zeros(time_max)
        stress=np.zeros(time_max)
        durs=np.zeros(time_max)
        number=np.zeros(time_max)
        indds=np.arange(0,N,1)
        if style ==1:
            vthresh=np.ones(N)
            # arr_pot=parabdist(N,w)
            arr_pot=np.random.uniform(-w/2,w/2,N)
            delta_p=vthresh-arr_pot
            weak=np.ones(N)*weakening
            weak[:ninhb-1]=weakeninginhb
            if np.sum(potential)==0:
                rand=np.random.uniform(0,1,N)
                potential=arr_pot+rand*(delta_p)
            data=[]
            jsums=np.zeros(N)
            potrelfull=np.zeros(time_max)
            ininputs=np.zeros(time_max)
            exinputs=np.zeros(time_max)
            time=0
            conn=np.zeros(shape=(N,N))
            for i in range(N):
                for j in np.arange(i,N,1):
                    conn[i,j]=np.exp(np.random.normal(np.log10(J),jwidth,1))
                    conn[j,i]=conn[i,j]
                    if j==i:
                        conn[i,j]=0
            if flag==1:
                conn[ninhb:N,0:ninhb]=-conn[ninhb:N,0:ninhb]/ratee
            for i in range(N):
                for j in range(N):
                    jsums[i]=jsums[i]+abs(conn[i,j])
            while time < time_max:
                vthresh = vthresh * 0 + 1
                potential=potential+ext_cur
                rate=[]
                spikes = potential >= vthresh
                if np.sum(spikes)==0:
                    time+=1
                while np.sum(spikes) > 0:
                    delta_pot = (vthresh - arr_pot)
                    potential[spikes] -= delta_pot[spikes]
                    indss=np.where(spikes)
                    inds=indss[0]
                    fullpotrel=np.zeros(shape=(N,1))
                    for i in inds:
                        for j in range(N):
                            if i==j:
                                continue
                            potrelfull[time]+=(conn[j,i]/(jsums[i]+N*k))
                            potrelfull[time]=potrelfull[time]/2
                            fullpotrel[j]+=(conn[j,i]/(jsums[i]+N*k))*delta_pot[i]
                    for i in range (N):
                        potential[i]+=fullpotrel[i]
                    mask = (vthresh == 1.0) * spikes
                    ininputs[time]=fullpotrel[0]
                    exinputs[time]=fullpotrel[-1]
                    vthresh[mask] -= (weak[mask] * delta_pot[mask])
                    rate.append(inds)
                    spikes=potential>=vthresh
                    time += 1
                    if time >= time_max:
                        break
                if time>=time_max:
                    break
                data.append(rate)
        else:
            ftimes=[]
            fneurs=[]
            indds=np.arange(0,N,1)
            vthresh=np.ones(N)
            np.random.seed()
            arr_pot=np.random.uniform(-w/2,w/2,N)
            delta_p=vthresh-arr_pot
            weakf=np.ones(N)*weakening
            weakf[0:ninhb-1]=weakeninginhb
            if np.sum(potential)==0:
                np.random.seed()
                rand=np.random.uniform(0,1,N)
                potential=arr_pot+rand*(delta_p)
            data=[]
            jsums=np.zeros(N)
            time=0
            ininputs=np.zeros(time_max)
            exinputs=np.zeros(time_max)
            singpot=np.zeros(time_max)
            spikingr=np.zeros(time_max)
            stimes=np.zeros(time_max)
            conn=np.zeros(shape=(N,N))
            for i in range(N):
                for j in np.arange(i,N,1):
                    conn[i,j]=np.exp(np.random.normal(np.log10(J),jwidth,1))
                    conn[j,i]=conn[i,j]
                    if j==i:
                        conn[i,j]=0
            if flag==1:
                conn[ninhb:N,0:ninhb]=-conn[ninhb:N,0:ninhb]/ratee
            for i in range(N):
                for j in range(N):
                    jsums[i]=jsums[i]+abs(conn[i,j])
            print (ratee, weakening, weakeninginhb,k,w)
            while time < time_max:
                np.random.seed()
                weak_dis= np.random.normal(0,0.01,N)
                # weak=weakf+weak_dis
                weak=weakf
                s=0
                n=[]
                rate=[]
                gaps=np.zeros(N)
                #step1####
                initial_pot_raise = np.min(1-potential)
                ##########
                #step2####
                tstep=int(initial_pot_raise/ext_cur)
                time+=tstep
                if time>=time_max:
                    break
                ##########
                #step3####
                potential+= initial_pot_raise
                ##########
                #step4####
                spikes = potential >= vthresh
                gaps[spikes]=delta_p[spikes]
                ##########
                if np.sum(spikes)==0:
                    ininputs[time]=0
                    exinputs[time]=0
                    singpot[time]=potential[0]
                    spikingr[time]=0
                    continue
                else:
                    stime=time
                while np.sum(spikes) > 0:
                    ftimes=np.append(ftimes,time*np.ones(sum(spikes)))
                    fneurs=np.append(fneurs,indds[spikes])
                    potential[spikes] -= delta_p[spikes]
                    inds=np.where(spikes)[0]
                    spikingr[time]=len(inds)
                    for j in range(N):
                        potential[j]+=np.sum((conn[j,inds]/(jsums[inds]+N*k))*delta_p[inds])
                    mask = (vthresh == 1.0) * spikes
                    vthresh[mask] -= (weak[mask] * delta_p[mask])
                    delta_p = (vthresh - arr_pot)
                    rate.append(inds)
                    s += np.sum(spikes)
                    n=np.append(n,list(indds[spikes]))
                    spikes=potential>=vthresh
                    ininputs[time]=np.sum((conn[0,inds]/(jsums[inds]+N*k))*delta_p[inds])
                    exinputs[time]=np.sum((inds==0))
                    singpot[time]=potential[0]
                    stress[time]=np.sum(potential)
                    time += 1
                    if time >= time_max:
                        break
                vthresh = vthresh * 0 + 1
                if time>=time_max:
                    break
                data.append(rate)
                durs[time]=time-stime
                size[time]=s
                stimes[time]=stime
                number[time]=len(np.unique(n))
            fullsizes=np.append(fullsizes,size)
            fullnumber=np.append(fullnumber,number)
            fullstimes=np.append(fullstimes,stimes+triall*time_max)
            fulldurations=np.append(fulldurations,durs)
            fullstresses=np.append(fullstresses,stress)
    return[fullsizes,fullnumber,fullstimes,fullstresses,fulldurations,ftimes,fneurs]

def Sim_for_ch7(N,Ni,weakening,weakeninginhb,trialnum,w,time_max,k,gamma):
    c=1.0/(1.0+k)
    Ne=N-Ni
    se=[]
    si=[]
    ne=[]
    ni=[]
    d=[]
    inds=np.arange(0,N,1)
    weak=np.ones(N)*weakening
    weak[:Ni]=weakeninginhb
    for trial in range(trialnum):
        np.random.seed()
        arr_pot=np.random.uniform(-w/2,w/2,N)
        vthresh=np.ones(N)
        delta_p=vthresh-arr_pot
        np.random.seed()
        rand=np.random.uniform(0,1,N)
        potential=arr_pot+rand*(delta_p)
        Sexc=np.zeros(time_max)
        Sinh=np.zeros(time_max)
        Nexc=np.zeros(time_max)
        Ninh=np.zeros(time_max)
        durs=np.zeros(time_max)
        stress=np.zeros(time_max)
        TEtimes=[]
        TEneurons=[]
        time=0
        count=0
        while time < time_max:
            vthresh = vthresh * 0 + 1
            initial_pot_raise = 1.0 - np.max(potential[Ni:])
            potential[Ni:] = potential[Ni:]+initial_pot_raise
            # initial_pot_raise = 1.0 - np.max(potential)
            # potential = potential+initial_pot_raise
            sse=0
            ssi=0
            spikes = inds[potential >= vthresh]
            sm=potential >= vthresh
            spikesi=spikes[spikes<Ni]
            spikese=spikes[spikes>=Ni]
            neuronspikes=[]
            if len(spikes)==0:
                time+=1
                count+=1
                stress[time]=len(spikes)
                continue
            else:
                starttime=time
                stress[time]=len(spikes)
            while len(spikes) > 0:
                neuronspikes=np.append(neuronspikes,spikes)
                delta_V = (vthresh - arr_pot)
                total_delta_V_exc=np.sum(delta_V[spikese])
                total_delta_V_inh=np.sum(delta_V[spikesi])
                potential[Ni:]+=(1.0/(N-1))*(total_delta_V_exc-(1.0/gamma)*total_delta_V_inh)
                potential[:Ni-1]+=(1.0/(N-1))*(total_delta_V_exc+total_delta_V_inh)
                potential[spikesi]-=delta_V[spikesi]
                potential[spikese]-=delta_V[spikese]
                potential[spikesi]-=(1.0/(N-1))*delta_V[spikesi]
                potential[spikese]-=(1.0/(N-1))*delta_V[spikese]
                sse += len(spikese)
                ssi += len(spikesi)
                mask = (vthresh == 1.0) * sm
                vthresh[mask] -= (weak[mask] * delta_V[mask])
                spikes=inds[potential >= vthresh]
                sm=potential >= vthresh
                spikesi=spikes[spikes<Ni]
                spikese=spikes[spikes>=Ni]
                tes=inds[spikes]
                TEneurons=np.append(TEneurons,tes)
                TEtimes=np.append(TEtimes,time*np.ones(len(tes)))
                time+=1
                if time>=time_max:
                    break
                stress[time]=len(spikes)
            if time>=time_max:
                break
            durs[time]=time-starttime
            Sexc[time]=sse
            Sinh[time]=ssi
            maske=neuronspikes>=Ni
            maski=neuronspikes<Ni
            Nexc[time]=len(np.unique(neuronspikes[maske]))
            Nexc[time]=len(np.unique(neuronspikes[maske]))
            Ninh[time]=len(np.unique(neuronspikes[maski]))
            ### if I add the next three lines (to de-correlate avalanches) the -1 power law
            ### that you should get from weakening does not happen.
            # np.random.seed()
            # rand=np.random.uniform(0,1,N)
            # potential=arr_pot+rand*(delta_p)
        se=np.append(se,Sexc)
        si=np.append(si,Sinh)
        ne=np.append(ne,Nexc)
        ni=np.append(ni,Ninh)
        d=np.append(d,durs)
    return[se,si,ne,ni,d,TEtimes,TEneurons,stress]

# Function 3: Multiproccesing simulation
def find_avsize(consv):
    N=consv[0]
    ninhb=consv[1]
    W=consv[2]
    K=consv[3]
    weakening=consv[4]
    weakeninginhb=consv[5]
    J=consv[6]
    jwidth=consv[7]
    ratee=consv[8]
    trialnumber=consv[9]
    time_max=consv[10]
    np.random.seed()
    dat=full_sim4([0],N,time_max,W,0.0001,K,ninhb,J,weakening,weakeninginhb,jwidth,0,ratee,1,trialnumber)
    dat=np.append(consv,dat)
    return[dat]

def find_avsize_mft(consv):
    N=consv[0]
    W=consv[1]
    con=consv[2]
    weakening=consv[3]
    trialnum=consv[4]
    time_max=consv[5]
    ext_cur=consv[6]
    distype=consv[7]
    np.random.seed()
    dat=full_sim7(N,time_max,W,con,weakening,ext_cur,trialnum,distype)
    dat=np.append(consv,dat)
    return[dat]

def find_avsize_full(consv):
    N=consv[0]
    time_max=consv[1]
    W=consv[2]
    ext_cur=consv[3]
    k=consv[4]
    ninhb=consv[5]
    J=consv[6]
    weakening=consv[7]
    weakeninginhb=consv[8]
    jwidth=consv[9]
    style=consv[10]
    ratee=consv[11]
    flag=consv[12]
    trialnumber=consv[13]
    np.random.seed()
    dat=full_sim4([],N,time_max,W,ext_cur,k,ninhb,J,weakening,weakeninginhb,jwidth,style,ratee,flag,trialnumber)
    dat=np.append(consv,dat)
    return[dat]

## example way to run the above function
# consv=[[0.9,0.006],[0.9,0.008],[0.9,0.01]]
# pool = Pool(os.cpu_count())
# dat=pool.map(find_avsize,consv)
##############

# Function 4: Generate parabolic distribution
def find_nearest(array, value):
    near = [abs(i-value) for i in array]
    indnear = near.index(min(near))
    return indnear

def parabdist(len,w):
    np.random.seed()
    x=np.linspace(-w/2.0,w/2.0,len)
    f=(3.0/(2.0*w**3))*(w**2-4*x**2)
    g=(3.0/2.0)*w**(-3.0)*(((w**(3.0)/3.0)+(w**(2.0)*x)-((4.0/3.0)*x**(3.0))))
    dis=[]
    for i in range(len):
        rand=np.random.uniform(0,1,1)
        spot=find_nearest(g,rand)
        dis.append(x[spot])
    return dis

# Function 5: Simulation with delay *** results not checked vs. other working simulation
def full_sim1(delay,N,time_max,w,weakening,weakeninginh,ext_vel,k):
    # ninhb=int(N/4)
    ninhb=0
    weak=np.ones(N)*weakening
    weak[:ninhb-1]=weakeninginh
    print (weak)
    vthresh=np.ones(N)
    jsums=np.zeros(N)
    arr_pot=-w/2.0*np.ones(N)+w*(np.random.random(N))
    delta_p=vthresh-arr_pot
    np.random.seed()
    rand=np.random.uniform(0,1,N)
    potential=arr_pot+rand*(delta_p)
    size=np.zeros(time_max)
    stress=np.zeros(time_max)
    rate=[]
    time=0
    fullpotrel=np.zeros(shape=(N,delay+1))
    conn=np.zeros(shape=(N,N))
    for i in range(N):
        for j in np.arange(i,N,1):
            conn[i,j]=np.exp(np.random.normal(0,1,1))/np.sqrt(N)
            conn[j,i]=conn[i,j]
            if j==i:
                conn[i,j]=0
    print (conn)
    conn[ninhb:N,0:ninhb]=-conn[ninhb:N,0:ninhb]/3.0
    # for i in range(N):
    #     jsums[i]=np.sum(np.abs(conn[:,i]))+N*k
    for i in range(N):
        for j in range(N):
            jsums[i]=jsums[i]+abs(conn[i,j])

    ##start sim
    avcounter=0
    strucfull=[]
    while time < time_max:
        vthresh = vthresh * 0 + 1
        potential[ninhb+1:]=potential[ninhb+1:]+k*ext_vel
        s=0
        spikes = potential >= vthresh
        a=np.sum(spikes)
        if a==0:
            stress[time]=np.sum(potential)
            time+=1
        struc=[]
        while a > 0:
            delta_pot = (vthresh - arr_pot)#*np.random.uniform(0.5,1.5)
            potential[spikes] -= delta_pot[spikes]
            for i in range (N):
                potential[i]+=fullpotrel[i,0]
            fullpotrel[:,0:delay]=fullpotrel[:,1:delay+1]
            fullpotrel[:,delay]=fullpotrel[:,delay]*0
            inds=np.where(spikes)[0]
            for i in inds:
                struc.append(i)
                for j in range(N):
                    fullpotrel[j,delay]+=(conn[j,i]/(jsums[i]+N*k))*delta_pot[i]
            s += a
            rate=np.append(rate,time*np.ones(a))
            mask = (vthresh == 1.0) * spikes
            vthresh[mask] -= (weak[mask] * delta_pot[mask])
            spikes=potential>=vthresh
            a=np.sum(spikes)
            stress[time]=np.sum(potential)
            time += 1
            if time >= time_max:
                break
        if time>=time_max:
            break
        if s>0:
            size[avcounter]=s
            strucfull.append(struc)
            avcounter+=1
        np.random.seed()
        rand=np.random.uniform(0,1,N)
        potential=arr_pot+rand*(delta_p)
    return[rate,stress]

# Function 6: MFT version of simulation (No network structure)
def full_sim7(N,time_max,w,consv,weakening,ext_cur,trialnum,distype):
    fullsizes=[]
    fullnumber=[]
    fullstimes=[]
    fulldurations=[]
    inds=np.arange(0,N,1)
    for trial in range(trialnum):
        entropy=np.zeros(time_max)
        print('triall',trial)
        np.random.seed()
        if distype=="uniform":
            print("111")
            arr_pot=np.random.uniform(-w/2,w/2,N)
        else:
            print("112")
            arr_pot=parabdist(N,w)
        vthresh=np.ones(N)
        delta_p=vthresh-arr_pot
        np.random.seed()
        rand=np.random.uniform(0,1,N)
        potential=arr_pot+rand*(delta_p)
        size=np.zeros(time_max)
        number=np.zeros(time_max)
        stress=np.zeros(time_max)
        stimes=np.zeros(time_max)
        durations=np.zeros(time_max)
        TEtimes=[]
        TEneurons=[]
        time=0
        while time < time_max:
            # if np.mod(time,time_max/10)==0:
            #     # print ('hey', time,time_max)
            vthresh = vthresh * 0 + 1
            initial_pot_raise = 1.0 - np.max(potential)
            tincrease=int(np.true_divide(initial_pot_raise,ext_cur))
            if time+tincrease>=time_max:
                break
            stress[time:time+tincrease]=np.linspace(stress[time-1],stress[time-1]+N*initial_pot_raise,tincrease)
            time+=tincrease
            # initial_pot_raise = ext_cur
            potential = potential + initial_pot_raise
            stwiddle=np.divide(potential-arr_pot,np.ones(N)-arr_pot)
            density, bins = np.histogram(stwiddle,density=True)
            delta=bins[-1]-(bins[1]-bins[0])
            ent = (-np.sum((delta*density)*np.log(density)))/len(density)
            entropy[time] = ent
            s=0
            n=[]
            spikes = potential >= vthresh
            if np.sum(spikes)==0:
                stress[time]=np.sum(potential)
                time+=1
                continue
            else:
                stime=time
                initial_stress=np.sum(potential)
                stress[time]=np.sum(potential)
            while np.sum(spikes) > 0:
                delta_pot = (vthresh - arr_pot)
                tot_pot_rel=(consv/N)*np.sum(delta_pot[spikes])
                # print ('1',consv/N*delta_pot[spikes][0])
                potential[spikes] -= delta_pot[spikes]*(1+consv/(N))
                potential += tot_pot_rel
                # s += np.sum(spikes)
                s+=tot_pot_rel*N
                n=np.append(n,list(inds[spikes]))
                mask = (vthresh == 1.0) * spikes
                vthresh[mask] -= (weakening * delta_pot[mask])
                spikes=potential>=vthresh
                tes=inds[spikes]
                TEneurons=np.append(TEneurons,tes)
                TEtimes=np.append(TEtimes,time*np.ones(len(tes)))
                time+=1
                if np.mod(time,1000)==0:
                    print(time/time_max)
                if time>=time_max:
                    break
                stress[time]=np.sum(potential)
            if time>=time_max:
                break
            # size[time]=s
            size[time]=initial_stress-stress[time]
            stimes[time]=stime
            durations[time]=time-stime
            # print('s',s)
            # print('n',n)
            # print ('np.unique',np.unique(n))
            number[time]=len(np.unique(n))
            ### if I add the next three lines (to de-correlate avalanches) the -1 power law
            ### that you should get from weakening does not happen.
            # np.random.seed()
            # rand=np.random.uniform(0,1,N)
            # potential=arr_pot+rand*(delta_p)
        fullsizes=np.append(fullsizes,size)
        fullnumber=np.append(fullnumber,number)
        fullstimes=np.append(fullstimes,stimes)
        fulldurations=np.append(fulldurations,durations)
    return[fullsizes,fullnumber,TEtimes,TEneurons,fullstimes,fulldurations,stress,entropy]

# Function 6: MFT version of simulation (No network structure)
def full_sim_dual_network(N,time_max,w,ke,ki,ninhb,J,weakening,weakeninginhb,jwidth):
    sizee=np.zeros(time_max)
    sizei=np.zeros(time_max)
    indds=np.arange(0,N,1)
    vthresh=np.ones(N)
    arr_pot=np.random.uniform(-w/2,w/2,N)
    delta_p=vthresh-arr_pot
    weak=np.ones(N)*weakening
    weak[0:ninhb-1]=weakeninginhb
    rand=np.random.uniform(0,1,N)
    potential=arr_pot+rand*(delta_p)
    datae=[]
    datai=[]
    jsums=np.zeros(N)
    time=0
    ininputs=np.zeros(time_max)
    exinputs=np.zeros(time_max)
    singpot=np.zeros(time_max)
    spikingri=np.zeros(time_max)
    spikingre=np.zeros(time_max)
    conn=np.zeros(shape=(N,N))
    Ni=ninhb
    Ne=N-Ni
    for i in range(N):
        for j in np.arange(i,N,1):
            conn[i,j]=np.exp(np.random.normal(np.log10(J),jwidth,1))
            conn[j,i]=conn[i,j]
            if j==i:
                conn[i,j]=0
    # conn[ninhb:N,0:ninhb]=-conn[ninhb:N,0:ninhb]
    for i in range(Ni):
        for j in range(Ni):
            jsums[i]=jsums[i]+abs(conn[i,j])
    for i in np.arange(ninhb,N,1):
        for j in np.arange(ninhb,N,1):
            jsums[i]=jsums[i]+abs(conn[i,j])
    while time < time_max:
        se=0
        si=0
        ratee=[]
        ratei=[]
        gaps=np.zeros(N)
        #step1####
        initial_pot_raise = np.min(1-potential)
        ##########
        #step2####
        # time+=1
        ##########
        #step3####
        potential+= initial_pot_raise
        ##########
        #step4####
        spikes = potential >= vthresh
        gaps[spikes]=delta_p[spikes]
        ##########
        if np.sum(spikes)==0:
            ininputs[time]=0
            exinputs[time]=0
            singpot[time]=potential[0]
            spikingri[time]=0
            spikingre[time]=0
            time+=1
            continue
        while np.sum(spikes) > 0:
            potential[spikes] -= delta_p[spikes]
            indsi=np.where(spikes[0:ninhb-1])[0]
            indse=np.where(spikes[ninhb:])[0]
            spikingri[time]=len(indsi)
            spikingre[time]=len(indse)
            for j in np.arange(0,ninhb-1,1):
                potential[j]+=np.sum((conn[j,indsi]/(jsums[indsi]+Ni*ki))*delta_p[indsi])
            for j in np.arange(ninhb,N,1):
                potential[j]+=np.sum((conn[j,indse]/(jsums[indse]+Ne*ke))*delta_p[indse])
            mask = (vthresh == 1.0) * spikes
            vthresh[mask] -= (weak[mask] * delta_p[mask])
            delta_p = (vthresh - arr_pot)
            ratee.append(indse)
            ratei.append(indsi)
            se += len(indse)
            si += len(indsi)
            spikes=potential>=vthresh
            time += 1
            if time >= time_max:
                break
        vthresh = vthresh * 0 + 1
        if time>=time_max:
            break
        datae.append(ratee)
        datai.append(ratei)
        sizee[time]=se
        sizei[time]=si
    return[sizee,datae,sizei,datai]
def full_sim_mft(Ne,time_max,w,ce,weakening,trialnum,ext_curr):
    inds=np.arange(0,Ne,1)
    weak=np.ones(Ne)*weakening
    for trial in range(trialnum):
        print('trial',trial)
        np.random.seed()
        arr_pot=np.random.uniform(-w/2,w/2,Ne)
        vthresh=np.ones(Ne)
        delta_p=vthresh-arr_pot
        rand=np.random.uniform(0,1,Ne)
        potential=arr_pot+rand*(delta_p)
        sizee=np.zeros(time_max)
        n=np.zeros(time_max)
        d=np.zeros(time_max)
        starts=np.zeros(time_max)
        stress=np.zeros(time_max)
        time=0
        count=0
        while time < time_max:
            vthresh = vthresh * 0 + 1
            initial_pot_raise = 1.0 - np.max(potential)
            potential = potential + initial_pot_raise
            if ext_curr==0:
                time+=1
            else:
                time+=int(initial_pot_raise/ext_curr)
            sse=0
            nn=0
            spikese = inds[potential >= vthresh]
            sm=potential >= vthresh
            if len(spikese)==0:
                time+=1
                count+=1
                print('hmm',time,count)
                continue
            else:
                stime=time
            while len(spikese) > 0:
                delta_pot = (vthresh - arr_pot)
                tot_pot_rele=(ce/Ne)*np.sum(delta_pot[spikese])
                potential[spikese] -= delta_pot[spikese]*(1+ce/Ne)
                potential += tot_pot_rele
                sse += len(spikese)
                nn=np.append(n,spikese)
                mask = (vthresh == 1.0) * sm
                vthresh[mask] -= (weak[mask] * delta_pot[mask])
                spikese=inds[potential >= vthresh]
                sm=potential >= vthresh
                time+=1
                if time>=time_max:
                    break
            if time>=time_max:
                break
            sizee[time]=sse
            d[time]=time-stime
            starts[time]=stime
            n[time]=len(np.unique(nn))
            ### if I add the next three lines (to de-correlate avalanches) the -1 power law
            ### that you should get from weakening does not happen.
            # np.random.seed()
            # rand=np.random.uniform(0,1,N)
            # potential=arr_pot+rand*(delta_p)
    return[sizee,starts,d,n,stress]
def full_sim_mft_subsampling(Ne,time_max,w,ce,weakening,trialnum,ext_curr):
    inds=np.arange(0,Ne,1)
    weak=np.ones(Ne)*weakening
    for trial in range(trialnum):
        print('trial',trial)
        np.random.seed()
        arr_pot=np.random.uniform(-w/2,w/2,Ne)
        vthresh=np.ones(Ne)
        delta_p=vthresh-arr_pot
        rand=np.random.uniform(0,1,Ne)
        potential=arr_pot+rand*(delta_p)
        rawtimes=[]
        rawfires=[]
        sizes=[]
        numbers=[]
        starts=[]
        times=[]
        fires=[]
        time=0
        count=0
        flg=0
        while time < time_max:
            vthresh = vthresh * 0 + 1
            initial_pot_raise = 1.0 - np.max(potential)
            potential = potential + initial_pot_raise
            if ext_curr==0:
                time+=1
            else:
                time+=int(initial_pot_raise/ext_curr)
                if time>=time_max:
                    break
            spikese = inds[potential >= vthresh]
            sm=potential >= vthresh
            sse=0
            if len(spikese)==0:
                time+=1
                count+=1
                print('hmm',time,count)
                continue
            else:
                tempfires=[]
                temptimes=[]
                stime=time
            while len(spikese) > 0:
                delta_pot = (vthresh - arr_pot)
                tot_pot_rele=(ce/Ne)*np.sum(delta_pot[spikese])
                potential[spikese] -= delta_pot[spikese]*(1+ce/Ne)
                potential += tot_pot_rele
                rawtimes=rawtimes+(time*np.ones(len(spikese))).tolist()
                rawfires=rawfires+spikese.tolist()
                mask = (vthresh == 1.0) * sm
                vthresh[mask] -= (weak[mask] * delta_pot[mask])
                tempfires=tempfires+spikese.tolist()
                sse+=len(spikese)
                temptimes=temptimes+(time*np.ones(len(spikese))).tolist()
                spikese=inds[potential >= vthresh]
                sm=potential >= vthresh
                time+=1
                if time>=time_max:
                    break
            if time>=time_max:
                break
            sizes.append(sse)
            starts.append(stime)
            numbers.append(len(np.unique(tempfires)))
            fires.append(tempfires)
            times.append(temptimes)
            ### if I add the next three lines (to de-correlate avalanches) the -1 power law
            ### that you should get from weakening does not happen.
            # np.random.seed()
            # rand=np.random.uniform(0,1,N)
            # potential=arr_pot+rand*(delta_p)
    return[np.array(rawtimes),np.array(rawfires),np.array(sizes),np.array(numbers),np.array(starts)]
def full_sim_dual_mft(N,Ni,time_max,w,ces,ci,weakening,trialnum):
    Ne=N-Ni
    se=[]
    si=[]
    inds=np.arange(0,N,1)
    weak=np.ones(N)*weakening
    # weak[0:Ni-1]=0
    for trial in range(trialnum):
        print('trial',trial)
        np.random.seed()
        arr_pot=np.random.uniform(-w/2,w/2,N)
        vthresh=np.ones(N)
        delta_p=vthresh-arr_pot
        rand=np.random.uniform(0,1,N)
        potential=arr_pot+rand*(delta_p)
        sizee=np.zeros(time_max)
        sizei=np.zeros(time_max)
        time=0
        count=0
        while time < time_max:
            vthresh = vthresh * 0 + 1
            initial_pot_raise = 1.0 - np.max(potential)
            potential = potential + initial_pot_raise
            sse=0
            ssi=0
            spikes = inds[potential >= vthresh]
            sm=potential >= vthresh
            spikesi=spikes[spikes<=(Ni-1)]
            spikese=spikes[spikes>=Ni]
            if len(spikes)==0:
                time+=1
                count+=1
                print('hmm',time,count)
                continue
            while len(spikes) > 0:
                # ce=ces-(len(spikesi)/Ni)
                ce=ces
                delta_pot = (vthresh - arr_pot)
                tot_pot_reli=(ci/Ni)*np.sum(delta_pot[spikesi])
                tot_pot_rele=(ce/Ne)*np.sum(delta_pot[spikese])
                potential[spikesi] -= delta_pot[spikesi]*(1+ci/Ni)
                potential[spikese] -= delta_pot[spikese]*(1+ce/Ne)
                potential[:Ni-1] += tot_pot_reli
                potential[Ni:] += tot_pot_rele
                sse += len(spikese)
                ssi += len(spikesi)
                mask = (vthresh == 1.0) * sm
                vthresh[mask] -= (weak[mask] * delta_pot[mask])
                spikes=inds[potential >= vthresh]
                sm=potential >= vthresh
                spikesi=spikes[spikes<=(Ni-1)]
                spikese=spikes[spikes>=Ni]
                time+=1
                if time>=time_max:
                    break
            if time>=time_max:
                break
            sizee[time]=sse
            sizei[time]=ssi
            ### if I add the next three lines (to de-correlate avalanches) the -1 power law
            ### that you should get from weakening does not happen.
            # np.random.seed()
            # rand=np.random.uniform(0,1,N)
            # potential=arr_pot+rand*(delta_p)
        se=np.append(se,sizee)
        si=np.append(si,sizei)
    return[se,si]
# Function 7: Pumped branching (J. Pausch et.al 2021)
# SpikeTimeTheory.py is a program for calculating the first five moments of inter-spike intervals of a pumped branching process
# created by Johannes Pausch
# copyright (2020) Johannes Pausch
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# used formulas are only exact for binary branching
# careful with how many bits longdouble precision corresponds to your machine

def creationProbability(r,gamma,s,p2,state):
    if state >1 :
        return np.longdouble(2*(r/(r+s))*(s*(state-1)/gamma+1)*((s*p2/(r+s*p2))**(state-1))*((r/(r+s*p2))**(gamma/(s*p2)))*(s*p2*(state-1)+gamma)/((state-1)*sc.beta(gamma/(s*p2),state-1)*(s*(state-1)+gamma)))
    else:
        return np.longdouble(2*(r/(r+s))*(r/(r+s*p2))**(gamma/(s*p2)))

def firstSpikeMoment(r,gamma,s,p2,precision):
    state = 1
    firstmoment = np.longdouble(0.0)
    previousSum = np.longdouble(1.0/(gamma*sc.beta(gamma/s,1)))
    powerp2 = np.longdouble(1.0)
    while state <= precision:
        powerp2 = powerp2*np.longdouble(1.0-p2)
        previousSum += np.longdouble(1.0)/(np.longdouble((state*s+gamma))*np.longdouble(sc.beta(gamma/s,state+1))*powerp2)
        # print(previousSum)
        # plt.pause(0.4)
        firstmoment += previousSum*creationProbability(r,gamma,s,p2,state)*powerp2*np.longdouble(sc.beta(gamma/s,state+1))
        state += 1

    return firstmoment

def secondSpikeMoment(r,gamma,s,p2,precision):
    state = 1
    secondmoment = np.longdouble(0.0)
    previousSum2 = np.longdouble(1.0/(gamma*sc.beta(gamma/s,1)))
    previousSum1 = np.longdouble(2.0/(gamma**2*sc.beta(gamma/s,1)))
    powerp2 = np.longdouble(1.0)
    while state <= precision:
        powerp2 = powerp2*np.longdouble(1.0-p2)
        previousSum2 += np.longdouble(1.0)/(np.longdouble((state*s+gamma))*np.longdouble(sc.beta(gamma/s,state+1))*powerp2)
        previousSum1 += np.longdouble(2.0)*previousSum2/(np.longdouble(state*s+gamma))
        secondmoment += previousSum1*creationProbability(r,gamma,s,p2,state)*powerp2*np.longdouble(sc.beta(gamma/s,state+1))
        state += 1
    return secondmoment

def thirdSpikeMoment(r,gamma,s,p2,precision):
    state = 1
    thirdmoment = np.longdouble(0.0)
    previousSum2 = np.longdouble(1.0/(gamma*sc.beta(gamma/s,1)))
    previousSum3 = np.longdouble(1.0/(sc.beta(gamma/s,1)*gamma**2))
    previousSum1 = np.longdouble(6.0/(sc.beta(gamma/s,1)*gamma**3))
    powerp2 = np.longdouble(1.0)
    while state <= precision:
        powerp2 = powerp2*np.longdouble(1.0-p2)
        previousSum2 += np.longdouble(1.0)/(np.longdouble(state*s+gamma)*np.longdouble(sc.beta(gamma/s,state+1))*powerp2)
        previousSum3 += previousSum2/np.longdouble(state*s+gamma)
        previousSum1 += np.longdouble(6.0)*previousSum3/(np.longdouble(state*s+gamma))
        thirdmoment += previousSum1*creationProbability(r,gamma,s,p2,state)*powerp2*np.longdouble(sc.beta(gamma/s,state+1))
        state += 1
    return thirdmoment

def fourthSpikeMoment(r,gamma,s,p2,precision):
    state = 1
    fourthmoment = np.longdouble(0.0)
    previousSum2 = np.longdouble(1.0/(gamma*sc.beta(gamma/s,1)))
    previousSum3 = np.longdouble(1.0/(sc.beta(gamma/s,1)*gamma**2))
    previousSum4 = np.longdouble(1.0/(sc.beta(gamma/s,1)*gamma**3))
    previousSum1 = np.longdouble(24.0/(sc.beta(gamma/s,1)*gamma**4))
    powerp2 = np.longdouble(1.0)
    while state <= precision:
        powerp2 = powerp2*np.longdouble(1.0-p2)
        previousSum2 += np.longdouble(1.0)/(np.longdouble(state*s+gamma)*np.longdouble(sc.beta(gamma/s,state+1))*powerp2)
        previousSum3 += previousSum2/np.longdouble(state*s+gamma)
        previousSum4 += previousSum3/np.longdouble(state*s+gamma)
        previousSum1 += np.longdouble(24.0)*previousSum4/(np.longdouble(state*s+gamma))
        fourthmoment += previousSum1*creationProbability(r,gamma,s,p2,state)*powerp2*np.longdouble(sc.beta(gamma/s,state+1))
        state += 1
    return fourthmoment

def fifthSpikeMoment(r,gamma,s,p2,precision):
    state = 1
    fifthmoment = np.longdouble(0.0)
    previousSum2 = np.longdouble(1.0/(gamma*sc.beta(gamma/s,1)))
    previousSum3 = np.longdouble(1.0/(sc.beta(gamma/s,1)*gamma**2))
    previousSum4 = np.longdouble(1.0/(sc.beta(gamma/s,1)*gamma**3))
    previousSum5 = np.longdouble(1.0/(sc.beta(gamma/s,1)*gamma**4))
    previousSum1 = np.longdouble(120.0/(sc.beta(gamma/s,1)*gamma**5))
    powerp2 = np.longdouble(1.0)
    while state <= precision:
        powerp2 = powerp2*np.longdouble(1.0-p2)
        previousSum2 += np.longdouble(1.0)/(np.longdouble(state*s+gamma)*np.longdouble(sc.beta(gamma/s,state+1))*powerp2)
        previousSum3 += previousSum2/np.longdouble(state*s+gamma)
        previousSum4 += previousSum3/np.longdouble(state*s+gamma)
        previousSum5 += previousSum4/np.longdouble(state*s+gamma)
        previousSum1 += np.longdouble(120.0)*previousSum5/(np.longdouble(state*s+gamma))
        fifthmoment += previousSum1*creationProbability(r,gamma,s,p2,state)*powerp2*np.longdouble(sc.beta(gamma/s,state+1))
        state += 1
    return fifthmoment

### now use these to run simulation and map r and g to X and Y
def run_sim(params,x0,y0):
    r,g=params
    s = 1.0 # rate of binary branching
    # r = 0.1 # effectice extinction rete / effective mass in field theory
    # g = 1.0 # relative spontaneous creation (g=gamma/s, where gamma is the spontaneous creation rate)
    precision = 16000 # maximum particle number that is included in calculation, requires at least 128bit double precision
    # (1-r/s)/2 = p0 = probability for a single particle to go extinct at a branching/extinction event
    firstmoment = firstSpikeMoment(r,g*s,s,(1-r/s)/2,precision)
    secondmoment = secondSpikeMoment(r,g*s,s,(1-r/s)/2,precision)
    thirdmoment = thirdSpikeMoment(r,g*s,s,(1-r/s)/2,precision)
    fourthmoment = fourthSpikeMoment(r,g*s,s,(1-r/s)/2,precision)
    fifthmoment = fifthSpikeMoment(r,g*s,s,(1-r/s)/2,precision)
    x=(thirdmoment/(firstmoment**3))-6
    y=(fourthmoment/(secondmoment**2))-6
    actual=np.array([x0,y0])
    simulated=[x,y]
    cost=np.sum((actual-simulated)**2)
    print(cost,x,y,r,g)
    # plt.subplot(2,1,1)
    # plt.plot(x,y,'o')
    # plt.subplot(2,1,2)
    # plt.plot(r,g,'o')
    # plt.pause(0.0001)
    return cost
## just for making seed map
def run_simm(r,g):
    s = 1.0 # rate of binary branching
    # r = 0.1 # effectice extinction rete / effective mass in field theory
    # g = 1.0 # relative spontaneous creation (g=gamma/s, where gamma is the spontaneous creation rate)
    precision = 16000 # maximum particle number that is included in calculation, requires at least 128bit double precision
    # (1-r/s)/2 = p0 = probability for a single particle to go extinct at a branching/extinction event
    firstmoment = firstSpikeMoment(r,g*s,s,(1-r/s)/2,precision)
    secondmoment = secondSpikeMoment(r,g*s,s,(1-r/s)/2,precision)
    thirdmoment = thirdSpikeMoment(r,g*s,s,(1-r/s)/2,precision)
    fourthmoment = fourthSpikeMoment(r,g*s,s,(1-r/s)/2,precision)
    fifthmoment = fifthSpikeMoment(r,g*s,s,(1-r/s)/2,precision)
    x=(thirdmoment/(firstmoment**3))-6
    y=(fourthmoment/(secondmoment**2))-6
    return x,y
def gradient_respecting_bounds(bounds, fun, eps=1e-8):
    """bounds: list of tuples (lower, upper)"""
    def gradient(x,x0,y0):
        fx = fun(x,x0,y0)
        grad = np.zeros(len(x))
        for k in range(len(x)):
            d = np.zeros(len(x))
            d[k] = eps if x[k] + eps <= bounds[k][1] else -eps
            grad[k] = (fun(x + d,x0,y0) - fx) / d[k]
        return grad
    return gradient
## piece of code to seed the simulation with estimated values of r an g
# ## make seed map##############################
# vals=[]
# for r in np.logspace(np.log10(0.001),np.log10(0.99),50):
#     for g in np.logspace(np.log10(0.1),np.log10(3),10):
#         x,y=run_simm(r,g)
#         plt.loglog(x,y,'o')
#         vals.append([[r,g],[x,y]])
#         np.save('vals.npy',vals)
# vals=np.load('vals.npy')[:-2]
# x0=(np.average(starts**3)/(np.average(starts)**3))-6
# y0=(np.average(starts**4)/(np.average(starts**2)**2))-6
# bounds = ((0.0005, 0.9), (0.0001,3))
# indfinder=np.zeros(len(vals))
# for kk in range(len(vals)):
#     indfinder[kk]=(x0-vals[kk][1][0])**2+(y0-vals[kk][1][1])**2
# result = minimize(run_sim, [vals[np.argmin(indfinder)][0][0], vals[np.argmin(indfinder)][0][1]],args=(x0,y0), bounds=bounds,
#                   jac=gradient_respecting_bounds(bounds, run_sim))
# rs=result.x[0]
# gs=result.x[1]
def Extract(lst,i):
    return [item[i] for item in lst]
### now extract timescale s
def Ets(s,isie,r,g):
    a=np.abs(isie-firstSpikeMoment(r*s,g*s*s,s,(1-r)/2,16000))
    print (s,a)
    return a
# isi1=np.average(starts[1:]-starts[:-1])
# s0=2
# result = scipy.optimize.minimize(Ets, s0,args=(isi1,r,g),method='nelder-mead')
# s=result.x
#######################################################

# Function 7: estimating branching ratio with MSR estimator
def mrestimate(fires,steps,isi):
    steps,rk=autocorr(fires,steps)
    res2 = sp.stats.linregress(steps, np.log(rk))
    m=np.exp(res2[0]*isi)## fix this: I don't know why isi is here

# Function 8: DFA
def dfa(signal,min,max,num):
    signal=np.cumsum(signal)
    x=np.logspace(np.log10(min),np.log10(max),num)
    y=[]
    for t in x:
        t=int(t)
        sigmafull=[]
        for j in range(len(signal)):
            if j==0:
                continue
            start=j*t-int(t/2.0)
            detrended=signal[start:start+int(t)]
            if len(detrended)>1:
                detrended=sig.detrend(detrended)
                sigmafull.append(np.std(detrended))
        y.append(np.average(sigmafull))
    slope=sp.linregress(np.log10(x),np.log10(y))
    alpha = slope[0]
    return alpha

# Function 9: Get cummulative distribution
def getcumdist(data):
    #Python sorting works with the > operator so we need to remove all nans.
    if np.isnan(data).sum()>0:
        data=np.asarray(data)
        data=data[~np.isnan(data)]
        data=data.tolist()
    #In order to remove negative values we add in a zero then sort the list. We
    #then remove the zero and all entries before it, i.e. negative numbers.
    #Similarly with infs
    data.extend([0,float('inf')])
    data.sort()

    data[data.index(float('inf')):]=[]
    data.reverse()

    data[data.index(0):]=[]

    data.reverse()

    histx=np.array(data)

    histy=(np.arange(1,0,-1.0/len(histx)))#Tyler just changed 1 to 1.0

    #Gabe found the bug that the same repeated value will return multiple
    #probabilities. This loop fixes that
    # for i in range(1,np.size(histx)):
    #     if histx[i]==histx[i-1]:
    #         histy[i]=histy[i-1]
    histx.tolist()
    histy.tolist()
    if len(histy)!= len(histx):
        histy=histy[:-1]
    return [histx,histy]

# Function 10: Kappa (see Poil 2011)
def kappa(s):
    [x,y]=getcumdist(s)
    y=1-y
    yth=1-x**(-0.7)
    logbins = np.logspace(np.log10(1),np.log10(np.max(s)),10)
    logbins=np.round(logbins)+0.5
    for ii in range(len(logbins)):
        logbins[ii]=np.int(find_nearest(x,logbins[ii]))
    logbins=logbins.astype(int)
    kappa=1+np.average(yth[logbins]-y[logbins])
    return kappa

# Function 11: Get slips
#Alan Long 6/14/16
#Last edited: Alan Long 5/23/19

#This code finds slips which have a rate of change above a certain threshold
#and determines their starting and ending times. It accepts four inputs, data
#is an list of the data to be analyzed, time an list corresponding time to the data,
#threshhold is a float that determines the slope needed to be considered a slip, and min_slip
# is a float that determines the minimum drop necessary to be considered an avalanche.
#It returns two lists, slip_sizes are the sizes of the events and slip_durations are the durations

#This code is based on the code of Jordan and Jim, but mostly Jim.

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

data=[1,5,3]
time=[0,0.1,0.2]
threshold=-1
mindrop=0
shapes=1
def get_slips(data, time, threshhold, mindrop, shapes):
    #The factor of -1 is so that we deal with positive numbers
    #for drop rates, it's not strictly neccessary but makes things nicer.
    smoothed=-1*np.array(data)
    time=np.array(time)

    #We now take a numeric derivative and get an average
    deriv=np.diff(smoothed)/np.diff(time)
    diff_avg=(smoothed[len(smoothed)-1]-smoothed[0])/(time[len(time)-1]-time[0])
    #We now set the minimum slip rate for it to be considered an avalanche
    if threshhold==-1:
        min_diff = 0.;
    else:
        min_diff = diff_avg + np.std(deriv)*threshhold;

    #Now we see if a slip is occurring, i.e. if the derivative is above the
    #minimum value. We then take the diff of the bools to get the starts and ends. +1= begin, -1=end.
    slips=np.diff([int(i>=min_diff) for i in deriv])
    index_begins=[i+1  for i in range(len(slips)) if slips[i]==1]
    index_ends=[i+1 for i in range(len(slips)) if slips[i]==-1]

    #We must consider the case where we start or end on an avalanche, this
    #checks if this is case and if so makes the start or end of the data
    #a start or end of an avalanche
    if max(index_begins)>max(index_ends):
        index_ends.append(len(time)-1)
    if min(index_begins)>min(index_ends):
        index_begins.insert(0,0)

    #Now we see if the drops were large enough to be considered an avalanche
    index_av_begins=[index_begins[i] for i in range(len(index_begins)) if mindrop<smoothed[index_ends[i]]-smoothed[index_begins[i]]]
    index_av_ends=[index_ends[i] for i in range(len(index_begins)) if mindrop<smoothed[index_ends[i]]-smoothed[index_begins[i]]]

    #Finally we use these indices to get the durations and sizes of the events
    slip_durations= time[index_av_ends]-time[index_av_begins]
    slip_sizes=-smoothed[index_av_begins]+smoothed[index_av_ends]-diff_avg*slip_durations*int(threshhold!=-1)#we subtract off the average if using a threshhold
    time_begins = time[index_av_begins]
    time_ends = time[index_av_ends]
    if shapes==1:
        velocity = []
        times = []
        time2=0.5*(time[0:len(time)-1]+time[1:len(time)])

        for k in range(len(slip_sizes)-1):
            mask=range(index_av_begins[k],index_av_ends[k]+1)
            velocity.append(deriv[mask].tolist())
            times.append((time2[mask]).tolist())
        return [slip_sizes[0:-1].tolist(),slip_durations[0:-1].tolist(),velocity,times]
    else:
        return [slip_sizes.tolist(), slip_durations.tolist(),deriv,time2,index_begins,_ends]

# Function 12: Get slips for Neurons
#Alan Long 6/14/16
#Last edited: Alan Long 5/23/19

#This code finds slips which have a rate of change above a certain threshold
#and determines their starting and ending times. It accepts four inputs, data
#is an list of the data to be analyzed, time an list corresponding time to the data,
#threshhold is a float that determines the slope needed to be considered a slip, and min_slip
# is a float that determines the minimum drop necessary to be considered an avalanche.
#It returns two lists, slip_sizes are the sizes of the events and slip_durations are the durations

#This code is based on the code of Jordan and Jim, but mostly Jim.

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def get_slipsg(data, time, threshhold, mindrop, shapes):
    #The factor of -1 is so that we deal with positive numbers
    #for drop rates, it's not strictly neccessary but makes things nicer.
    smoothed=np.array(data)
    time=np.array(time)

    #We now take a numeric derivative and get an average
    deriv=np.diff(smoothed)#/np.diff(time)
    deriv=deriv-1E-3
    diff_avg=(smoothed[len(smoothed)-1]-smoothed[0])/(time[len(time)-1]-time[0])
    #We now set the minimum slip rate for it to be considered an avalanche
    if threshhold==-1:
        min_diff = 0.;
    else:
        print('hi',np.std(deriv),diff_avg)
        min_diff = diff_avg + np.std(deriv)*threshhold;
        print('so',min_diff)
    #Now we see if a slip is occurring, i.e. if the derivative is above the
    #minimum value. We then take the diff of the bools to get the starts and ends. +1= begin, -1=end.
    slips=np.diff([int(i>=min_diff) for i in deriv])
    index_begins=[i+1  for i in range(len(slips)) if slips[i]==1]
    index_ends=[i+1 for i in range(len(slips)) if slips[i]==-1]

    #We must consider the case where we start or end on an avalanche, this
    #checks if this is case and if so makes the start or end of the data
    #a start or end of an avalanche
    if len(index_begins)==0:
        index_begins=[0]
        index_ends=[1]
    if max(index_begins)>max(index_ends):
        index_ends.append(len(time)-1)
    if min(index_begins)>min(index_ends):
        index_begins.insert(0,0)

    #Now we see if the drops were large enough to be considered an avalanche
    index_av_begins=[index_begins[i] for i in range(len(index_begins)) if mindrop<smoothed[index_ends[i]]-smoothed[index_begins[i]]]
    index_av_ends=[index_ends[i] for i in range(len(index_begins)) if mindrop<smoothed[index_ends[i]]-smoothed[index_begins[i]]]

    #Finally we use these indices to get the durations and sizes of the events
    slip_durations= time[index_av_ends]-time[index_av_begins]
    slip_sizes=-smoothed[index_av_begins]+smoothed[index_av_ends]-diff_avg*slip_durations*int(threshhold!=-1)#we subtract off the average if using a threshhold
    time_begins = time[index_av_begins]
    time_ends = time[index_av_ends]
    if shapes==1:
        velocity = []
        times = []
        time2=0.5*(time[0:len(time)-1]+time[1:len(time)])

        for k in range(len(slip_sizes)-1):
            mask=range(index_av_begins[k],index_av_ends[k]+1)
            velocity.append(deriv[mask].tolist())
            times.append((time2[mask]).tolist())
        return [slip_sizes[0:-1].tolist(),slip_durations[0:-1].tolist(),velocity,times]
    else:
        return [slip_sizes.tolist(), slip_durations.tolist(),deriv,time2,index_begins,_ends]

# Function 13: Log-binning
def log_binning(bininput,quantityinput,numbins):
    m=0
    binbounds=np.logspace(np.log10(min(bininput)),np.log10(max(bininput)),numbins+1)
    print(binbounds)
    X=np.zeros(numbins)
    Y=np.zeros(numbins)
    E = np.zeros(numbins)
    for i in range(len(binbounds)-1):
        mask = (bininput>(binbounds[i]))&(bininput<(binbounds[i+1]))
        Y[i]= np.sum(mask*quantityinput)/np.sum(mask)
        count=np.sum(mask!=0)+0.1
        if sum(mask)==0:
            E[i]=float('nan')
        else:
            E[i]=2.0*(np.std(quantityinput[mask])/count)#confidence interval
        X[i]= np.sqrt(binbounds[i]*binbounds[i+1])
    return(X,Y,E,m)
def log_binning_2(bininput,quantityinput,numbins,type):
    m=0
    binbounds=np.logspace(np.log10(min(bininput)),np.log10(max(bininput)),numbins+1)
    print(binbounds)
    X=np.zeros(numbins)
    Y=np.zeros(numbins)
    E = np.zeros(numbins)
    for i in range(len(binbounds)-1):
        mask = (bininput>(binbounds[i]))&(bininput<(binbounds[i+1]))
        Y[i]= np.sum(mask*quantityinput)/np.sum(mask)
        count=np.sum(mask!=0)+0.1
        if sum(mask)==0:
            E[i]=float('nan')
        else:
            if type=='SD':
                E[i]=np.std(quantityinput[mask])
            elif type=='SEM':
                E[i]=sp.sem(quantityinput[mask])
            else:
                E[i]=1.96*(np.std(quantityinput[mask])/np.sqrt(count))#confidence interval
        X[i]= np.sqrt(binbounds[i]*binbounds[i+1])
    return(X,Y,E,m)
def log_binning_3(bininput,quantityinput,numbins,type):
    m=0
    binbounds=np.linspace(min(bininput),max(bininput),numbins+1)
    print(binbounds)
    X=np.zeros(numbins)
    Y=np.zeros(numbins)
    E = np.zeros(numbins)
    for i in range(len(binbounds)-1):
        mask = (bininput>(binbounds[i]))&(bininput<(binbounds[i+1]))
        Y[i]= np.sum(mask*quantityinput)/np.sum(mask)
        count=np.sum(mask!=0)+0.1
        if sum(mask)==0:
            E[i]=float('nan')
        else:
            if type=='SD':
                E[i]=np.std(quantityinput[mask])
            elif type=='SEM':
                E[i]=sp.sem(quantityinput[mask])
            else:
                E[i]=1.96*(np.std(quantityinput[mask])/np.sqrt(count))#confidence interval
        X[i]= binbounds[i]+((binbounds[i+1]-binbounds[i])/2.0)
    return(X,Y,E,m,binbounds)
def log_binning_nan(bininput,quantityinput,numbins):
    binbounds=np.logspace(np.log10(1),np.log10(max(bininput)),numbins+1)
    X=np.zeros(numbins)
    Y=np.zeros(numbins)
    E = np.zeros(numbins)
    m=np.sqrt(quantityinput)
    n=np.ones(len(m))
    for i in range(len(binbounds)-1):
        mask = (bininput>(binbounds[i]))&(bininput<(binbounds[i+1]))
        count=np.sum(mask!=0)+0.1
        Y[i]= np.sum(mask*quantityinput)/np.sum(mask)
        m[mask]=m[mask]/Y[i]
        E[i]=2.0*(np.std(quantityinput[mask])/count)
        X[i]= np.sqrt(binbounds[i]*binbounds[i+1])
    return(X,Y,E,m)

def binning(bininput,quantityinput,numbins):
    binbounds=np.linspace(min(bininput),max(bininput),numbins+1)
    X=np.zeros(numbins)
    Y=np.zeros(numbins)
    E = np.zeros(numbins)
    m=np.sqrt(quantityinput)
    n=np.ones(len(m))
    for i in range(len(binbounds)-1):
        mask = (bininput>(binbounds[i]))&(bininput<(binbounds[i+1]))
        count=np.sum(mask!=0)+0.1
        Y[i]= np.sum(mask*quantityinput)/np.sum(mask)
        m[mask]=m[mask]/Y[i]
        E[i]=2.0*(np.std(quantityinput[mask])/count)
        X[i]= np.sqrt(binbounds[i]*binbounds[i+1])
    return(X,Y,E,m)

# Function 14: Bitest
def bitest(starts):
    H=[]
    for i in np.arange(2,len(starts)-2,1):
        t1=starts[i+1]-starts[i]
        t2=starts[i]-starts[i-1]
        if t1<t2:

            t=t1
            tau=starts[i+2]-starts[i+1]

        else:
            t=t2
            tau=starts[i-1]-starts[i-2]

        if (~np.isnan(t/(t+(tau/2.0))))&(~np.isinf(t/(t+(tau/2.0)))):
            H.append(t/(t+(tau/2.0)))
    return(np.array(H))

# Function 15: Create spiking rate signal from raw time stamps
def spikingrate(spikes,dt):
    spikes=np.unique(spikes)
    time = np.arange(np.min(spikes)-1,np.max(spikes)+1,dt)
    fires = np.zeros(len(time))
    binn=0
    count=0
    for _i in spikes[:-1]:
        if (_i>=time[binn])&(_i<time[binn+1]):
            fires[binn]+=1
        else:
            while _i>time[binn+1]:
                binn+=1
            fires[binn]+=1
        count+=1
    return time,fires

# Function 16: Powerspec
def powerspec(dt,data):
    n = len(data)
    spec = np.fft.rfft(data, n)
    sp = spec * np.conj(spec) / n
    freq = (1 / (dt * n)) * np.arange(n)
    L = np.arange(1, np.floor(n / 2, ), dtype='int')
    return[freq[L],sp[L]]

# Function 17: Histogram
def tyler_pdf(starts,binnum,minn,maxx,type,density):
    if type=='log':
        logbins = np.logspace(np.log10(minn),np.log10(maxx),binnum)
    else:
        logbins=np.linspace(minn,maxx,binnum)
    density=np.histogram(starts,bins=logbins,density=density)
    x=density[1][:-1]
    y=density[0]
    return[x,y]

# Function 18: MSD
def msd(shapes,sizes):
    vel = np.zeros(len(sizes))
    for i in range(len(sizes)):

        vel[i]=max(shapes[i])

    return [vel,sizes]

# Function 19: Surface plotter
def surface_plot(matrix,errmat,**kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    ax.scatter(x,y,matrix + errmat, color='black',marker='+')
    ax.scatter(x,y,matrix - errmat, color='black',marker='+')
    ax.set(facecolor='grey')
    return (fig, ax, surf)

# Function 20: Get cummulative distribution with error bars

def inc_Beta_Fit(x, a,b,target):
    return np.abs(special.betainc(a, b, x)-target)

def getcumdist2(data,steps,errorbars=True):

    if np.isnan(data).sum()>0:
        data=data[~np.isnan(data)]

    data=list(data)
    data.extend([0,float('inf')])
    data.sort()
    data[data.index(float('inf')):]=[]
    data.reverse()
    data[data.index(0):]=[]
    data.reverse()
    histx=np.array(data)
    histy=(np.arange(1,0,-1/len(histx)))

    for i in range(1,np.size(histx)):
        if histx[i]==histx[i-1]:
            histy[i]=histy[i-1]

    histx.tolist()
    histy.tolist()
    if (errorbars==False):return [histx,histy]

    loTarget=.025 #These set confidence intervals
    hiTarget=.975 #The values here give a 95% confidence interval

    lowError,highError=[],[]

    N=len(histx)
    vec=np.arange(0,N,steps)
    vec=np.append(vec,N-5)

    hist1=histx[vec]
    hist2=histy[vec]
    for i in vec:
        #S=histx[i]
        C_S=histy[i]
        k=N-i

        X=np.linspace(0,1,500000)
        y_low=inc_Beta_Fit(X,k+1,N-k+1,target=loTarget)
        y_high=inc_Beta_Fit(X,k+1,N-k+1,target=hiTarget)

        low_ind=np.argmin(y_low)
        high_ind=np.argmin(y_high)

        p_low=X[low_ind]
        p_high=X[high_ind]

        lowError=lowError+[C_S-p_low]
        highError=highError+[p_high-C_S]
    print (len(highError),len(hist2))
    return [histx,histy,np.asarray(lowError),np.asarray(highError),hist1,hist2]

'''
The first bit of this is the same as the code given to me for CCDFs, so it has the same problem in that sometimes
there is a size mismatch in X and Y, which can be fixed before or after the function call. For extra clarity, heres a simple use example

dur=ef.getcumdist(DTypes[i],True)
durX=dur[0]
durY=dur[1]
if(len(durY)>len(durX)):durY=durY[:-1]
loError=dur[2]
hiError=dur[3]

ax.errorbar(durX,durY,yerr=[loError,hiError])
'''

# Function 21: SHAPES
#Tyler Salners 5/24/19
#Last edited Alan Long 9/9/19
#This code returns the rate of change of stress over time for an avalanche.

import numpy as np
from matplotlib import pyplot as plt

#This function finds the nearest point in an array to a given value. It takes in array, an array, and value, a float.
#It outputs idx, an int, which is the index of the nearest point.
def find_nearest(array, value):
    near = [abs(i-value) for i in array]
    indnear = near.index(min(near))
    return indnear

#This function bins the shapes based on duration of size. It takes lists durs, avs, shapes, times which are the output of get_slips.
# bins is the centers of the bins to be sorted into, type is a str and either 'size' or 'duration' is what you are binning by, and
#width is the width of the bins. It outputs lists times_sorted, shapes_sorted, durs_sorted, avs_sorted which are those binned and sorted.
def shape_bins(durs,avs,shapes,times,bins,type,width):
    # first sort the arrays
    shapes=np.asarray(shapes)
    times=np.asarray(times)
    if type=='duration':
        ind=np.argsort(durs)
    else:
        ind = np.argsort(avs)
    avs = avs[ind]
    shapes = shapes[ind]
    times = times[ind]
    durs = durs[ind]
    #make return array
    shapes_sorted = []
    times_sorted = []
    durs_sorted = []
    avs_sorted = []
    save=0
    for i in range(len(bins)):
        if type=='size':
            idxx = find_nearest(avs,bins[i])#find closest event to bin center
            mask = range(idxx-width,idxx+width)#take all events within ben width
            print('begin',idxx-width)
            print ('end',idxx+width)
            if save > idxx-width:
                print ('shit')
            save=idxx+width
        elif type=='duration':
            idxx = find_nearest(durs,bins[i])#find closest event to bin center
            mask = range(idxx - width, idxx + width)#take all events within ben width
            print('begin',idxx-width)
            print ('end',idxx+width)

        shapes_sorted.append(shapes[mask])
        times_sorted.append(times[mask])
        durs_sorted.append(durs[mask])
        avs_sorted.append(avs[mask])

    return [np.array(times_sorted), np.array(shapes_sorted), np.array(durs_sorted), np.array(avs_sorted)]
#This averages over sizes. It takes lists shapes,times,avs,durs the output of get_slips. It outputs lists times_final,
#shapes_final, and err_final the time, velocity, and error vectors respectively.

def size_avg(shapes,times,avs,durs):
    shapes_final = []
    err_final = []
    times_final =[]
    for i in range(len(shapes)): # for each bin
        span = len(shapes[i])#number of shapes
        lenind = np.argmax([len(j) for j in shapes[i]])
        print ('lkjsd',lenind)
        length= len(times[i][lenind])
        avg_shape=np.zeros(length)
        avg_err =np.zeros(length)
        sort_shapes=np.zeros((np.size(shapes[i]),length))
        for k in range(np.size(shapes[i])):
            sort_shapes[k][0:np.size(shapes[i][k])] = shapes[i][k]#collect shapes, padded at end with 0 if no data
        for k in range(length):
            avg_shape[k] = sum(sort_shapes[:,k])/span#average of shapes
            avg_err[k] = np.std(sort_shapes[:,k])/np.sqrt(span)#error
        shapes_final.append(avg_shape)
        err_final.append(avg_err)
        times_final.append(times[i][lenind])
    return [np.array(times_final),np.array(shapes_final),np.array(err_final)]

#This averages over durations. It takes lists shapes,times,avs,durs the output of get_slips. It outputs lists times_final,
#shapes_final, and err_final the time, velocity, and error vectors respectively.
def duration_avg(shapes,times,avs,durs):
    shapes_final = []
    err_final = []
    times_final =[]
    for i in range(len(shapes)): # for each bin
        length = np.size(shapes[i][0])#length of time trace
        for k in range(np.size(shapes[i])):
            [times[i][k],shapes[i][k]]=resize(shapes[i][k],times[i][k],length)#conform to length
        avg_shape=np.zeros(length)
        avg_err =np.zeros(length)
        sort_shapes=np.zeros((np.size(shapes[i]),length))
        span= np.size(shapes[i])#number of shapes

        for k in range(np.size(shapes[i])):

            sort_shapes[k] = shapes[i][k]

        for k in range(length):
            avg_shape[k] = np.true_divide(np.sum(sort_shapes[:,k]),span)#average
            avg_err[k] = np.true_divide(np.std(sort_shapes[:,k]),np.sqrt(span))#error

        shapes_final.append(avg_shape)
        err_final.append(avg_err)
        times_final.append(times[i][0])
    return(times_final,shapes_final,err_final)

#This code takes a vector and resizes it to a desired length. It takes a list vector and an associate list time. length is and int and
#is the desired len of the result. It outputs two lists points, the normalized time vector, and new, the new vector.
def resize(vector,time,length):
    time=np.asarray(time)
    vector=np.asarray(vector)
    time = time-time[0]#so you allways start at 0
    time = time.astype(float)
    time = np.true_divide(time,time[-1])#normalize
    new = np.zeros(length)
    points = np.linspace(0,1,num=length)
    width2 = 1.0/length#step of points

    for i in range(length):
        # if i == 0:
        #     continue
        mask = (time>=(points[i]-(width2/2)))&(time<=(points[i]+(width2/2)))#all parts of time within the range of a point
        new[i]=np.mean(vector[mask])
        if np.isnan(new[i]):
            new[i]=new[i-1]#remove nans
    return[points,new]


def Extract(lst,i):
    return [item[i] for item in lst]

def Scaling_relation(smin,s,dmin,d):
    test_s=s[s>smin]
    resultss = powerlaw.Fit(test_s,xmin=smin)
    print('smin',resultss.xmin)
    test_d=d[d>dmin]
    resultsd = powerlaw.Fit(test_d,xmin=dmin)
    print('dmin',resultsd.xmin)
    test_s=s[(s>smin)&(d>dmin)]
    test_d=d[(s>smin)&(d>dmin)]
    [x,y,e,m]=log_binning(test_s,test_d,40)
    x=x[~np.isnan(y)]
    y=y[~np.isnan(y)]
    signuz, intercept, r_value, p_value, std_err = sp.linregress(np.log10(x), np.log10(y))
    tau=resultss.power_law.alpha
    tau_e=resultss.power_law.sigma
    alpha=resultsd.power_law.alpha
    alpha_e=resultsd.power_law.sigma
    x=1.0/signuz
    xerr=std_err
    y=(alpha-1)/(tau-1)
    yerr=(alpha/tau)*np.sqrt((alpha_e/alpha)**2+(tau_e/tau)**2)
    return[x,y,xerr,yerr,tau,alpha,signuz,tau_e,alpha_e,std_err]
import math

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

# def get_S_vs_Ninh(gcc):
#     S = []
#     N = []
#     D = []
#     tstart=[]
#     for i in range(0,len(gcc)):
#         neurons=Extract(list(dict.fromkeys(gcc[i])),0)
#         times=Extract(list(dict.fromkeys(gcc[i])),1)
#         S.append(len(neurons))
#         N.append(len(np.unique(neurons)))
#         D.append(np.subtract(np.max(times),np.min(times)))
#         tstart.append(np.min(times))
#     S = np.array(S)
#     N = np.array(N)
#     D = np.array(D)
#     tstart=np.array(tstart)
#     return S,N,D,tstart

def load_causal_webs(web,td):
    """
    Loads causal web data from specified filename, and
    returns two lists of graphs (where each element of each list
    is associated with a single avalanche embedded in the entire c-pairs
    dataset:

    gcc: directed acyclic graphs representing avalanches, where node IDs are
    (neuron, time) tuples and edges connect causally inferred triggering

    ncc: directed graphs, where node IDs are neuron indices and edges connect
    neurons associated with a triggering event, and edge weights indicate how
    many times that neuron pair is associated in the avalanche
    """
    # f = open(filename, 'rb')
    # web = pickle.load(f, encoding='bytes')
    # f.close()
    print('a')
    iweb = [((n0[0], int(n0[1])), (n1[0], int(n1[1]))) for n0, n1 in web if (int(n1[1])-int(n0[1]))<td]#n0[0]/n1[0] is neuron i/j,n0[1]/n1[1] is time i,j
    print('b')
    g = nx.DiGraph()# make empty network g
    print('c')
    g.add_edges_from(iweb)# take edges from pkl file and populate the network g
    print('d')
    del iweb
    # gccs = nx.weakly_connected_component_subgraphs(g)#DEPRECATED
    gccs = (g.subgraph(c) for c in nx.weakly_connected_components(g))#fuck if i know
    # gcc = sorted([e for e in gccs], key=lambda g: min([n[1] for n in g.nodes]))#sort based on the "temporal length" of link
    # del g
    # del gccs
    # ncc = []
    # aa = []
    # print('start')
    # for cc in gcc:
    #     nnet = nx.DiGraph()
    #     weights = {}
    #     for xn1,xn2 in cc.edges():
    #         n1, n2 = xn1[0], xn2[0]#xn1[0]/xn2[0] are neruon i/j that are causally connected
    #         if not nnet.has_edge(n1,n2):
    #             nnet.add_edge(xn1[0], xn2[0], weight=1)#add edge between i/j to directed graph
    #         else:
    #             nnet.edges[(n1,n2)]['weight'] += 1#add 1 to weight if i triggers j again
    #     # print('avalanche')
    #     # print (nnet.nodes,cc.nodes)
    #     ncc.append(nnet)
    # print ('end')
    # del nnet
    return gccs
def get_S_vs_Ninh(gcc):
    S = []
    N = []
    D = []
    firingtimes=[]
    totalspikes=[]
    totalspikepairs=[]
    tstart=[]
    print('um here?', len(gcc))
    for i in range(0,len(gcc)):
        S.append(gcc[i].number_of_nodes())
        # N.append(ncc[i].number_of_nodes())
        spikes=[]
        times=[]
        temp=list(gcc[i].edges)
        pairs=pd.unique(Extract(temp,0)+Extract(temp,1))
        spikes=np.array(Extract(pairs,0))
        times=np.array(Extract(pairs,1))
        # for j in range(len(temp)):
        #     times.append(temp[j][0][1])
        #     times.append(temp[j][1][1])
        #     spikes.append(temp[j][0])
        #     spikes.append(temp[j][1])
        D.append(np.max(times)-np.min(times))
        tstart.append(np.min(times))
        firingtimes.append(np.sort(times))
        N.append(len(pd.unique(spikes)))
        totalspikes.append([spikes])
        totalspikepairs.append(temp)
    S = np.array(S)
    N = np.array(N)
    D = np.array(D)
    tstart = np.array(tstart)
    firingtimes=np.array(firingtimes)
    return S,N,D,tstart,totalspikes,totalspikepairs,firingtimes
def get_S_vs_Ninh_2species(gcc,inspec,exspec,unspec):
    Sinh = np.zeros(len(gcc))
    Ninh = np.zeros(len(gcc))
    Sexc = np.zeros(len(gcc))
    Nexc = np.zeros(len(gcc))
    D = np.zeros(len(gcc))
    Finh=[]
    Tinh=[]
    Fexc=[]
    Texc=[]
    for i in range(0,len(gcc)):
        temp=list(gcc[i].edges)
        pairs=pd.unique(Extract(temp,0)+Extract(temp,1))
        neurons=np.array(Extract(pairs,0))
        times=np.array(Extract(pairs,1))*1E-3
        excneuron=neurons[np.isin(neurons,exspec)]
        inhneuron=neurons[np.isin(neurons,inspec)]
        exctimes=times[np.isin(neurons,exspec)]
        inhtimes=times[np.isin(neurons,inspec)]
        Sinh[i]=len(inhneuron)
        Ninh[i]=len(pd.unique(inhneuron))
        Sexc[i]=len(excneuron)
        Nexc[i]=len(pd.unique(excneuron))
        D[i]=np.max(times)-np.min(times)
        Tinh.append(inhtimes.tolist())
        Texc.append(exctimes.tolist())
        Finh.append(inhneuron.tolist())
        Fexc.append(excneuron.tolist())
    # Sinh = np.array(Sinh)
    # Ninh = np.array(Ninh)
    # Sexc = np.array(Sexc)
    # Nexc = np.array(Nexc)
    # Tinh = np.array(Tinh)
    # Texc = np.array(Texc)
    # Finh = np.array(Finh)
    # Fexc = np.array(Fexc)
    # D = np.array(D)
    return D, Sinh, Ninh, Sexc, Nexc, Finh, Fexc, Tinh, Texc
def get_S_vs_Ninh_lite(gcc):
    D = np.zeros(len(gcc))
    ft = []
    fn=[]
    S = np.zeros(len(gcc))
    N = np.zeros(len(gcc))
    for i in range(0,len(gcc)):
        temp=list(gcc[i].edges)
        pairs=pd.unique(Extract(temp,0)+Extract(temp,1))
        tempneurons=np.array(Extract(pairs,0))
        temptimes=np.array(Extract(pairs,1))

        # temptimes=(np.multiply(Extract(Extract(temp,0),1)+Extract(Extract(temp,1),1),1E-3)).tolist()
        # tempneurons=Extract(Extract(temp,0),0)+Extract(Extract(temp,1),0)
        tempneurons=np.array(tempneurons)
        temptimes=np.array(temptimes)

        tempneurons=tempneurons[np.argsort(temptimes)]
        temptimes=temptimes[np.argsort(temptimes)]
        tempneurons.tolist()
        temptimes.tolist()

        S[i]=len(tempneurons)
        fn.append(tempneurons)
        tempneurons=np.unique(tempneurons)
        D[i]=np.max(temptimes)-np.min(temptimes)#durations
        ft.append(temptimes)
        N[i]=len(tempneurons)
    D = np.array(D)
    ft=np.array(ft)
    fn=np.array(fn)
    S=np.array(S)
    N=np.array(N)
    return ft,fn,S,D,N

def T_exp_bootstrap(samplemean,df,N):
    bootstrapsample = np.random.exponential(scale=samplemean, size=(df, N))
    sample_means=np.mean(bootstrapsample, axis=0)
    mean = np.mean(sample_means)
    std = np.std(sample_means)
    return [mean,std]

def Find_x0_y0(nsamp,isif):
    ## calculating moment ratios and errors
    isi=(isif,)
    bootstrap_ci = sp.bootstrap(isi, np.average, confidence_level=0.95,random_state=1, method='percentile',n_resamples=nsamp)
    mu1=np.average(isi)
    sig1=bootstrap_ci.standard_error
    sigg1=np.std(isi)

    isi=(isif**2,)
    bootstrap_ci = sp.bootstrap(isi, np.average, confidence_level=0.95,random_state=1, method='percentile',n_resamples=nsamp)
    mu2=np.average(isi)
    sig2=bootstrap_ci.standard_error
    sigg2=np.std(isi)

    isi=(isif**3,)
    bootstrap_ci = sp.bootstrap(isi, np.average, confidence_level=0.95,random_state=1, method='percentile',n_resamples=nsamp)
    mu3=np.average(isi)
    sig3=bootstrap_ci.standard_error
    sigg3=np.std(isi)

    isi=(isif**4,)
    bootstrap_ci = sp.bootstrap(isi, np.average, confidence_level=0.95,random_state=1, method='percentile',n_resamples=nsamp)
    mu4=np.average(isi)
    sig4=bootstrap_ci.standard_error
    sigg4=np.std(isi)

    isi=isif
    mutop=mu3
    mubot=mu1**3
    sigbot=3*sig1*mubot/mu1
    sigbott=3*sigg1*mubot/mu1
    sigtop=sig3
    sigtopp=sigg3
    xerr=(mutop/mubot)*np.sqrt((sigtop/mutop)**2+(sigbot/mubot)**2)
    xer=(mutop/mubot)*np.sqrt((sigtopp/mutop)**2+(sigbott/mubot)**2)
    mutop=mu4
    mubot=mu2**2
    sigbot=2*sig2*mubot/mu2
    sigbott=2*sigg2*mubot/mu2
    sigtop=sig4
    sigtopp=sigg4
    yerr=(mutop/mubot)*np.sqrt((sigtop/mutop)**2+(sigbot/mubot)**2)
    x0=(np.average(isi**3)/(np.average(isi)**3))-6
    y0=(np.average(isi**4)/(np.average(isi**2)**2))-6
    yer=(mutop/mubot)*np.sqrt((sigtopp/mutop)**2+(sigbott/mubot)**2)

# #### optimization to find r and g
    # rmin=0.001
    # rmax=0.99
    # gmin=0.0005
    # gmax=2
    # bounds = ((rmin,rmax), (gmin,gmax))
    # os.chdir('/Users/tylersalners/Desktop/RESEARCH/gelson/sims')
    # vals=np.load('vals_3.npy')
    # indfinder=np.zeros(len(vals))
    # for kk in range(len(vals)):
    #     indfinder[kk]=(x0-vals[kk][1][0])**2+(y0-vals[kk][1][1])**2
    # result = minimize(run_sim, [vals[np.argmin(indfinder)][0][0], vals[np.argmin(indfinder)][0][1]],args=(x0,y0), bounds=bounds,jac=gradient_respecting_bounds(bounds, run_sim))
    # rs=result.x[0]
    # gs=result.x[1]
    return[x0,y0,xerr,yerr,xer,yer]

def Find_r_g(nsamp,isif):
    ## calculating moment ratios and errors
    isi=(isif,)
    bootstrap_ci = sp.bootstrap(isi, np.average, confidence_level=0.95,random_state=1, method='percentile',n_resamples=nsamp)
    print('lkj')

    mu1=np.average(isi)
    sig1=bootstrap_ci.standard_error

    isi=(isif**2,)
    bootstrap_ci = sp.bootstrap(isi, np.average, confidence_level=0.95,random_state=1, method='percentile',n_resamples=nsamp)
    mu2=np.average(isi)
    sig2=bootstrap_ci.standard_error

    isi=(isif**3,)
    bootstrap_ci = sp.bootstrap(isi, np.average, confidence_level=0.95,random_state=1, method='percentile',n_resamples=nsamp)
    mu3=np.average(isi)
    sig3=bootstrap_ci.standard_error

    isi=(isif**4,)
    bootstrap_ci = sp.bootstrap(isi, np.average, confidence_level=0.95,random_state=1, method='percentile',n_resamples=nsamp)
    mu4=np.average(isi)
    sig4=bootstrap_ci.standard_error

    isi=isif
    mutop=mu3
    mubot=mu1**3
    sigbot=3*sig1*mubot/mu1
    sigtop=sig3
    xerr=(mutop/mubot)*np.sqrt((sigtop/mutop)**2+(sigbot/mubot)**2)
    mutop=mu4
    mubot=mu2**2
    sigbot=2*sig2*mubot/mu2
    sigtop=sig4
    yerr=(mutop/mubot)*np.sqrt((sigtop/mutop)**2+(sigbot/mubot)**2)
    x0=(np.average(isi**3)/(np.average(isi)**3))-6
    y0=(np.average(isi**4)/(np.average(isi**2)**2))-6
    #### optimization to find r and g
    rmin=0.001
    rmax=0.99
    gmin=0.0005
    gmax=2
    bounds = ((rmin,rmax), (gmin,gmax))
    os.chdir('/Users/tylersalners/Desktop/RESEARCH/gelson/sims')
    vals=np.load('vals_3.npy')
    indfinder=np.zeros(len(vals))
    for kk in range(len(vals)):
        indfinder[kk]=(x0-vals[kk][1][0])**2+(y0-vals[kk][1][1])**2
    result = minimize(run_sim, [vals[np.argmin(indfinder)][0][0], vals[np.argmin(indfinder)][0][1]],args=(x0,y0), bounds=bounds,jac=gradient_respecting_bounds(bounds, run_sim))
    rs=result.x[0]
    gs=result.x[1]
    return[x0,y0,xerr,yerr]

def Find_r_g_special(nsamp,isif):
    ## calculating moment ratios and errors
    isi=(isif,)
    bootstrap_ci = sp.bootstrap(isi, np.average, confidence_level=0.95,random_state=1, method='percentile',n_resamples=nsamp)

    mu1=np.average(isi)
    sig1=bootstrap_ci.standard_error

    isi=(isif**2,)
    bootstrap_ci = sp.bootstrap(isi, np.average, confidence_level=0.95,random_state=1, method='percentile',n_resamples=nsamp)
    mu2=np.average(isi)
    sig2=bootstrap_ci.standard_error

    isi=(isif**3,)
    bootstrap_ci = sp.bootstrap(isi, np.average, confidence_level=0.95,random_state=1, method='percentile',n_resamples=nsamp)
    mu3=np.average(isi)
    sig3=bootstrap_ci.standard_error

    isi=(isif**4,)
    bootstrap_ci = sp.bootstrap(isi, np.average, confidence_level=0.95,random_state=1, method='percentile',n_resamples=nsamp)
    mu4=np.average(isi)
    sig4=bootstrap_ci.standard_error

    isi=isif
    mutop=mu3
    mubot=mu1**3
    sigbot=3*sig1*mubot/mu1
    sigtop=sig3
    xerr=(mutop/mubot)*np.sqrt((sigtop/mutop)**2+(sigbot/mubot)**2)
    mutop=mu4
    mubot=mu2**2
    sigbot=2*sig2*mubot/mu2
    sigtop=sig4
    yerr=(mutop/mubot)*np.sqrt((sigtop/mutop)**2+(sigbot/mubot)**2)
    x0=(np.average(isi**3)/(np.average(isi)**3))-6
    y0=(np.average(isi**4)/(np.average(isi**2)**2))-6
    #### optimization to find r and g
    rmin=0.001
    rmax=0.99
    gmin=0.0005
    gmax=2
    bounds = ((rmin,rmax), (gmin,gmax))
    os.chdir('/Users/tylersalners/Desktop/RESEARCH/gelson/sims')
    vals=np.load('vals_3.npy')
    indfinder=np.zeros(len(vals))
    for kk in range(len(vals)):
        indfinder[kk]=(x0-vals[kk][1][0])**2+(y0-vals[kk][1][1])**2
    result = minimize(run_sim, [vals[np.argmin(indfinder)][0][0], vals[np.argmin(indfinder)][0][1]],args=(x0,y0), bounds=bounds,jac=gradient_respecting_bounds(bounds, run_sim))
    rs=result.x[0]
    gs=result.x[1]
    return[x0,y0,xerr,yerr,rs,gs,rerr,gerr]
###### switching simulation #######
def full_sim(potential,N,time_max,w,consv,weakening,ext_cur):
    vthresh=np.ones(N)
    arr_pot=parabdist(N,w)
    delta_p=vthresh-arr_pot
    if np.sum(potential)==0:
        rand=np.random.uniform(0,1,N)
        potential=arr_pot+rand*(delta_p)
    size=np.zeros(time_max)
    unique=np.zeros(time_max)
    stress = np.zeros(time_max)
    entropy=np.zeros(time_max)
    # charT=np.average(vthresh-arr_pot)/ext_cur
    charT=1.0
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ent_switch=0
    trf=[]
    tfr=[]
    switch=[]
    last=0
    for time in range(time_max):
        vthresh = vthresh * 0 + 1
        ############alans way
        initial_pot_raise = 1.0 - np.max(potential)
        potential = potential + initial_pot_raise
        ###################
        # ######my/karins? way
        # potential=potential+ext_cur
        if np.mod(time,100000)==0:
            print ('time',100*time/time_max,len(tfr))
        s=0
        u=np.zeros(N)
        spikes = potential >= vthresh
        while np.sum(spikes) > 0:
            delta_pot = (vthresh - arr_pot)#*np.random.uniform(0.5,1.5)
            tot_pot_rel=(consv/(N-1))*np.sum(delta_pot[spikes])
            potential[spikes] -= delta_pot[spikes]
            potential += tot_pot_rel
            # time += 1
            s += np.sum(spikes)
            mask = (vthresh == 1.0) * spikes
            vthresh[mask] -= (weakening * delta_pot[mask])
            u[spikes] = 1
            spikes=potential>=vthresh
            ax[0].plot(spikes)
            ax[0].set_ylim([0,4])
            ax[1].plot(tot_pot_rel)
            plt.pause(0.1)
            ax[0].cla()
        stwiddle=np.divide(potential-arr_pot,np.ones(N)-arr_pot)
        density, bins = np.histogram(stwiddle,density=True)
        delta=bins[-1]-(bins[1]-bins[0])
        ent = (-np.sum((delta*density)*np.log(density)))/len(density)
        entropy[time] = ent
        if (ent_switch==0)&(ent<np.log(1-weakening)):
            tfr.append(time/charT-last)
            ent_switch=1
            last=time/charT
            print ('Switched to Runaway %s'%(len(tfr)))
            if len(trf)>1:
                [x, y] = getcumdist(trf)
                plt.plot(x, 1 - y, color='b', label=r'$r\rightarrow f$')
                [x, y] = getcumdist(tfr)
                # plt.plot(x, 1 - y, color='r', label=r'$f \rightarrow r$')
                # plt.ylabel(r'Prob T$_{sw}<$T')
                # plt.xlabel('T')
                # plt.legend()
                # plt.pause(0.5)
                # plt.cla()
                c=int(consv*100)
                we=int(weakening*100)
                np.save('%s_%spotential.npy'%(c,we),potential)
                np.save('%s_%strf.npy'%(c,we),trf)
                np.save('%s_%stfr.npy'%(c,we),tfr)
        elif (ent_switch==1)&(np.sum(entropy[time-100:time]>np.log(1-weakening))==100):
            trf.append((time-100)/charT-last)
            ent_switch=0
            last=(time-100)/charT
            switch.append(time-100)
            print ('Switched to GR %s')
        if s==0:
            size[time]=float('NaN')
            entropy[time]=float('NaN')
        else:
            size[time]=s
        unique[time] = np.sum(u)
        stress[time]=np.sum(potential)
        # if np.mod(time,1)==0:
            # ax[3].hist(stwiddle,density=True)
            # ax[3].plot(bins[:-1],density,'r')
            # ax[0].plot(np.ones(len(potential))-np.sort(potential),'.')
            # ax[2].plot(entropy[time-50:time],'r.')
            # ax[0].set_ylim([0,1])
            # ax[0].set_title('Failed cells')
            # ax[0].set_xlabel('Time')
            # ax[2].set_title('Entropy')
            # ax[2].plot([np.log(1-weakening),np.log(1-weakening)],'-r')
            # ax[2].set_xlabel('Time')
            # ax[1].plot(np.sort(potential),'b')
            # ax[3].set_xlim([0,1])
            # ax[3].set_ylim([0,4])
            # ax[2].set_ylim([-1,0.25])
            # ax[1].set_ylim([-1,10])
            # # ax[1].matshow(np.reshape(potential, (-1, 10)),cmap='Reds',vmin=-0.0,vmax=1.0)
            # plt.pause(.00001)
            # ax[0].cla()
            # ax[1].cla()
            # ax[2].cla()
            # ax[3].cla()
    # unique=[]
    # stress=[]
    # size=[]
    return[unique,stress,size,entropy,potential,tfr,trf,switch]

def full_sim_multi(potential,N,time_max,w,consv,weakening,ext_cur):
    vthresh=np.ones(N)
    np.random.seed()
    arr_pot=parabdist(N,w)
    delta_p=vthresh-arr_pot
    if np.sum(potential)==0:
        rand=np.random.uniform(0,1,N)
        potential=arr_pot+rand*(delta_p)
    size=np.zeros(time_max)
    unique=np.zeros(time_max)
    stress = np.zeros(time_max)
    entropy=np.zeros(time_max)
    # charT=np.average(vthresh-arr_pot)/ext_cur
    charT=1.0
    # fig, ax = plt.subplots(nrows=1, ncols=4)
    ent_switch=0
    trf=[]
    tfr=[]
    switch=[]
    last=0
    rate=0.01
    for time in range(time_max):
        vthresh = vthresh * 0 + 1
        ############alans way
        initial_pot_raise = 1.0 - np.max(potential)
        potential = potential + initial_pot_raise
        time+=int(initial_pot_raise/rate)
        ##################
        # ######my/karins? way
        # potential=potential+ext_cur
        if np.mod(time,100000)==0:
            print ('time',100*time/time_max,len(tfr))
        s=0
        u=np.zeros(N)
        spikes = potential >= vthresh
        while np.sum(spikes) > 0:
            delta_pot = (vthresh - arr_pot)#*np.random.uniform(0.5,1.5)
            tot_pot_rel=(consv/(N-1))*np.sum(delta_pot[spikes])
            potential[spikes] -= delta_pot[spikes]
            potential += tot_pot_rel
            if time>=time_max:
                break
            time += 1
            s += np.sum(spikes)
            mask = (vthresh == 1.0) * spikes
            vthresh[mask] -= (weakening * delta_pot[mask])
            u[spikes] = 1
            spikes=potential>=vthresh
            # plt.plot(spikes)
            # plt.ylim([0,4])
            # plt.pause(0.1)
            # plt.cla()
        stwiddle=np.divide(potential-arr_pot,np.ones(N)-arr_pot)
        density, bins = np.histogram(stwiddle,density=True)
        delta=bins[-1]-(bins[1]-bins[0])
        ent = (-np.sum((delta*density)*np.log(density)))/len(density)
        if time>=time_max:
            break
        entropy[time] = ent
        if (ent_switch==0)&(ent<np.log(1-weakening)):
            tfr.append(time/charT-last)
            ent_switch=1
            last=time/charT
            print ('Switched to Runaway %s'%(len(tfr)))
            if len(trf)>1:
                # [x, y] = getcumdist(trf)
                # plt.plot(x, 1 - y, color='b', label=r'$r\rightarrow f$')
                # [x, y] = getcumdist(tfr)
                # plt.plot(x, 1 - y, color='r', label=r'$f \rightarrow r$')
                # plt.ylabel(r'Prob T$_{sw}<$T')
                # plt.xlabel('T')
                # plt.legend()
                # plt.pause(1.0)
                # plt.cla()
                c=int(consv*100)
                we=int(weakening*100)
                np.save('%s_%spotential.npy'%(c,we),potential)
                np.save('%s_%strf.npy'%(c,we),trf)
                np.save('%s_%stfr.npy'%(c,we),tfr)
        elif (ent_switch==1)&(np.sum(entropy[time-100:time]>np.log(1-weakening))==100):
            trf.append((time-100)/charT-last)
            ent_switch=0
            last=(time-100)/charT
            switch.append(time-100)
            print ('Switched to GR %s')
        if s==0:
            size[time]=float('NaN')
            entropy[time]=float('NaN')
        else:
            size[time]=s
        unique[time] = np.sum(u)
        stress[time]=np.sum(potential)
        # if np.mod(time,1)==0:
        #     ax[3].hist(stwiddle,density=True)
        #     ax[3].plot(bins[:-1],density,'r')
        #     ax[0].plot(unique[time-20:time],'.')
        #     ax[2].plot(entropy[time-20:time],'r.')
        #     ax[0].set_ylim([0,N])
        #     ax[0].set_title('Failed cells')
        #     ax[0].set_xlabel('Time')
        #     ax[2].set_title('Entropy')
        #     ax[2].plot([np.log(1-weakening),np.log(1-weakening)],'-r')
        #     ax[2].set_xlabel('Time')
        #     ax[1].plot(stwiddle,'r')
        #     ax[1].plot(potential,'b')
        #     ax[3].set_xlim([0,1])
        #     ax[3].set_ylim([0,4])
        #     # ax[1].matshow(np.reshape(potential, (-1, 10)),cmap='Reds',vmin=-0.0,vmax=1.0)
        #     plt.pause(.00001)
        #     ax[0].cla()
        #     ax[1].cla()
        #     ax[2].cla()
        #     ax[3].cla()
    # unique=[]
    # stress=[]
    # size=[]
    return[trf,tfr]

def full_sim_test(potential,N,time_max,w,consv,weakening,ext_cur):
    vthresh=np.ones(N)
    arr_pot=parabdist(N,w)
    delta_p=vthresh-arr_pot
    if np.sum(potential)==0:
        rand=np.random.uniform(0,1,N)
        potential=arr_pot+rand*(delta_p)
    size=np.zeros(time_max)
    # charT=np.average(vthresh-arr_pot)/ext_cur
    charT=1.0
    fig, ax = plt.subplots(nrows=1, ncols=4)
    ent_switch=0
    last=0
    pbar=np.average(1.0/(1-np.asarray(arr_pot)))
    for time in range(time_max):
        vthresh = vthresh * 0 + 1
        ############alans way
        initial_pot_raise = 1.0 - np.max(potential)
        potential = potential + initial_pot_raise
        # ###################
        # ######my/karins? way
        # potential=potential+ext_cur
        if np.mod(time,100000)==0:
            print ('time')
        s=0
        u=np.zeros(N)
        spikes = potential >= vthresh
        while np.sum(spikes) > 0:
            print('avvss')
            delta_pot = (vthresh - arr_pot)#*np.random.uniform(0.5,1.5)
            tot_pot_rel=(consv/(N-1))*np.sum(delta_pot[spikes])
            potential[spikes] -= delta_pot[spikes]
            potential += tot_pot_rel
            Xn=1-np.flip(np.sort(potential))
            dn=Xn[:-1]-Xn[1:]
            # time += 1
            s += np.sum(spikes)
            mask = (vthresh == 1.0) * spikes
            vthresh[mask] -= (weakening * delta_pot[mask])
            u[spikes] = 1
            [x,y]=tyler_pdf(dn,20,np.min(dn),np.max(dn),'lo',False)
            ax[0].semilogy(x,10*np.exp(-N*x*pbar),'-k')
            ax[0].semilogy(x,y)
            ax[0].set_xlim([0,0.05])
            ax[0].set_ylim([1E-1,15])
            ax[1].plot(Xn)
            ax[1].set_xlim([0,10])
            ax[1].set_ylim([0.9,1.1])
            ax[2].plot(Xn-tot_pot_rel)
            ax[2].set_ylim([-.1,.1])
            ax[2].plot(0*Xn,'-k')
            # ax[3].plot(potential)
            plt.pause(1)
            ax[0].cla()
            ax[1].cla()
            ax[2].cla()
            spikes=potential>=vthresh
        print('end')
        stwiddle=np.divide(potential-arr_pot,np.ones(N)-arr_pot)
    return[size,potential]

def prune_beggs(experiment,td):
    os.chdir('/Users/tylersalners/Desktop/beggs/data/causal_web_pkl')
    filename='c-pairs_2013-01-%s-000_d-lt-20_rt0.5.pkl'%(experiment)
    f = open(filename, 'rb')
    web = pickle.load(f, encoding='bytes')
    # web=web[:10000]
    f.close()
    # iweb = [((n0[0], int(n0[1])), (n1[0], int(n1[1]))) for n0, n1 in web if int(n1[1])-int(n0[1])<td]#n0[0]/n1[0] is neuron i/j,n0[1]/n1[1] is time i,j
    gcc = load_causal_webs(web,td)
    [size,number,duration,tstart,total_spikes,total_spike_pairs]=get_S_vs_Ninh(list(gcc))
    return total_spikes

def splitter(t,sig,n):
    l=np.linspace(np.min(t),np.max(t),n)
    full_sig=[]
    full_t=[]
    for i in range(len(l)-1):
        full_sig.append(sig[(t>=l[i])&(t<l[i+1])])
        full_t.append(t[(t>=l[i])&(t<l[i+1])])
    return [full_t,full_sig,l]

def tyler_specgram(t,sig,n):
    f=[]
    s=[]
    [split_t,split_sig,t_centers]=splitter(t,sig,n)
    for i in range(len(split_t)):
        [freq,spec]=powerspec(t[1]-t[0],split_sig[i])
        f.append(freq)
        s.append(spec)
    return[f,s,t_centers]

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def surface_plot(matrix,**kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.scatter(x, y, matrix, **kwargs)
    ax.set(facecolor='grey')
    return (fig, ax, surf)