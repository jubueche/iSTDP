import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
import matplotlib.pyplot as plt
from numba import njit


def generate_sk(dt, tau_s, duration):
    # - Generate epsilon per group
    T = int(duration/dt)
    sk = np.zeros(T)
    for t in range(1,T):
        epsilon = np.random.uniform()-0.5 # Draw from uniform [-0.5,0.5]
        sk[t] = epsilon - (epsilon - sk[t-1])*np.exp(-dt/tau_s)
    # - Rectify
    sk[sk < 0] = 0
    # - Scale max value to 500 Hz * dt
    sk = 500*dt*sk/np.max(sk)
    return sk

@njit
def sparsify_input_signal(sk, dt, duration):
    toggle = 0
    tslt = 0 # - Time since last toggle in s
    for t in range(int(duration/dt)-1):
        if(sk[t] == 0 and toggle == 0 and ((t*dt - tslt) > 0.001)):
            toggle = 1
            tslt = t*dt
        
        if(toggle==1):
            sk[t] = 0
            # - If we have been cancelling the signal for more than 5ms and there is an end to the bump, stop cancelling 
            if(sk[t+1] == 0 and ((t*dt - tslt) > 0.0005)):
                toggle = 0
    return sk

def generate_signal(dt, tau_s, duration):
    sk_tmp = generate_sk(dt=dt, tau_s=tau_s, duration=duration)
    sk = sparsify_input_signal(sk_tmp, dt=dt,duration=duration)
    # - Add 5Hz * dt to the signal
    sk += 5 * dt
    return sk

def generate_spike_train(sk, duration, dt, refr):
    tsls = 0 # - Time since last spike in s
    spike_times = []
    for t in range(int(duration/dt)):
        if(tsls < 0): # - Neuron can spike. Not in refr. period anymore
            if(np.random.uniform() < sk[t]):
                spike_times.append(t)
                tsls = refr
        else:
            tsls -= dt
    return spike_times

def generate_spike_trains(nn, sk, duration, dt, refr):
    spike_trains = []
    spike_indices = []
    for i in range(nn):
        spike_times = generate_spike_train(sk=sk, duration=duration,dt=dt,refr=refr)
        spike_trains.append(spike_times)
        spike_indices.append(i*np.ones(len(spike_times)))
    # - Flatten the lists
    spike_times = [t for sublist in spike_trains for t in sublist]
    spike_channels = [t for sublist in spike_indices for t in sublist]
    return spike_times, spike_channels

@njit
def generate_spike_trains_njit(nn, sk, duration, dt, refr):
    spike_trains = np.zeros((nn,int(duration/dt)))

    for i in range(nn):
        tsls = 0.0
        for t in range(int(duration/dt)):
            if(tsls < 0): # - Neuron can spike. Not in refr. period anymore
                if(np.random.uniform(a=0.0,b=1.0) < sk[t]):
                    spike_trains[i,t] = 1.0
                    tsls = refr
            else:
                tsls -= dt
    return spike_trains

######### Input #########

# - Small plot of input
sk = generate_signal(dt=0.0001,tau_s=0.05,duration=5.0)

# - Generate nn many spike trains
spike_trains = generate_spike_trains_njit(nn=100, sk=sk, duration=5.0, dt=0.0001, refr=0.005)

# - Average firing rate in Hz
avetrain = np.sum(spike_trains) / (100*5.0)
print(f"Average firing rate is {avetrain} Hz.")

# # Uncomment to view synaptic strengths
# synapse_realistic_units = np.copy(synapse)
# synapse_realistic_units[:int(n_sigs*ex_groupsize)] *= 140
# synapse_realistic_units[int(n_sigs*ex_groupsize):] *= 350
# min_ex_syn = np.min(synapse_realistic_units[:int(n_sigs*ex_groupsize)])
# max_ex_syn = np.max(synapse_realistic_units[:int(n_sigs*ex_groupsize)])
# mean_ex_syn = np.mean(synapse_realistic_units[:int(n_sigs*ex_groupsize)])
# print(f"Min. exc. synaptic weight is {min_ex_syn} pS, max is {max_ex_syn} pS average is {mean_ex_syn}")
# plt.plot(synapse_realistic_units); plt.xlabel("pS"); plt.show()

######### Experiment #########

class iSTDP:

    def __init__(self,
                eta = 0.001,
                tauPlasticity = 0.02, # 20 ms
                tau_s = 0.05, # 50ms
                dt = 0.0001, # 0.1ms
                duration = 1.5, # s
                refr = 0.005, # 5ms refractory period
                n_cells = 1000, # Number of total input neurons to the single neuron. Including excitatory and inhibitory
                ex_frac = 0.8, # Percentage of excitatory input neurons
                in_frac = 0.2,
                gBarEx = 0.014,
                gBarIn = 0.035,
                VRest = -60,
                Vth = -50,
                tau_mem = 0.002,
                EAMPA = 0,
                EGABA = -80.0,
                tauEx = 0.005,
                tauIn = 0.01,
                noise_tau = 0.05,
                n_sigs = 8): # 8 different signals with different spiking frequencies

        self.eta = eta
        self.alpha = 0.25 * self.eta
        self.tauPlasticity = tauPlasticity
        self.tau_s = tau_s # - Signal time constant
        self.dt = dt # - Simulation time step. 0.0001 s (0.1 ms)
        self.duration = duration # - Signal duration, 1.5 s
        self.refr = refr # - Refractory period, 0.005 s (5 ms)
        self.n_cells = n_cells # - Total number of cells (1000)
        self.ex_frac = ex_frac # - Fraction of excitatory neurons
        self.in_frac = in_frac # - Fraction of inhibitory neurons
        self.num_exc = int(np.round(n_cells * self.ex_frac))
        self.gBarEx = gBarEx # - Scaling factor to scale ex. syn. conductance up from pico to 10 nano Siemens, 0.014
        self.gBarIn = gBarIn # - Scaling factor to scale inh. syn. conductance up from pico to 10 nano Siemens, 0.035
        self.VRest = VRest # - Resting potential of the neuron, -60 mV
        self.Vth = Vth # - Threshold potential of the neuron, -50 mV
        self.tau_mem = tau_mem # - Membrane time constant, 0.002 s (2 ms)
        self.EAMPA = EAMPA # - Exc. reversal potential, 0 mV
        self.EGABA = EGABA # - Inh. reversal potential, -80 mV
        self.tauEx = tauEx # - Exc. syn. TC, 0.005 s (5 ms)
        self.tauIn = tauIn # - Inh. syn. TC, 0.01 s (10 ms)
        self.noise_tau = noise_tau # - Filter time for for the input in s, 0.05 s (50 ms) 
        self.n_sigs = n_sigs # - Number of distinct signals, 8
        self.bg_rate = 5*self.dt # - Constant background firing rate in Hz, 5 Hz
        self.max_rate = 500*self.dt # - Maximum firing rate in Hz, 500 Hz
        self.norm_factor = 0.03 # - Normalize input trace to 1 so that we can then rescale to get ~500 Hz peak value (0.05 w.r.t. 0.1 ms dt)

        # - Precompute exponential kernels
        self.expGEx=np.exp(-self.dt/self.tauEx)
        self.expGIn=np.exp(-self.dt/self.tauIn)
        self.expPlasticity=np.exp(-self.dt/self.tauPlasticity)
        self.expnoise=np.exp(-self.dt/self.noise_tau)

        # - Vector for input signal
        self.input = np.zeros(self.n_sigs)

        # - Vector for time
        self.time_vector = np.arange(0.0, self.duration, self.dt)

        # - Track exc. and inh. currents
        self.Exkeep = np.zeros(len(self.time_vector))
        self.Inkeep = np.zeros(len(self.time_vector))

        # - White noise used for input generation
        self.filtered_white_noise = np.zeros(self.n_sigs) 

        # - Excitatory and inhibitory group sizes
        self.ex_groupsize = int(np.round((n_cells*ex_frac) / n_sigs))
        self.in_groupsize = int(np.round((n_cells*in_frac) / n_sigs))

        # - Assign group index to each exc. and inh. neuron. Inh. neurons have a negative sign (but also range from 1..n_sigs) e.g. 1..8
        self.input_group = self.assign_input_group()

        # - Keep track of refractory periods of input neurons
        self.input_spike_refr = np.zeros(self.n_cells)

        # - Initialize synaptic weights
        self.synapse = self.assign_synapse_weights(n_exc=int(self.n_sigs*self.ex_groupsize), n_inh=int(self.n_sigs*self.in_groupsize))

        # - Vectors to store group-wise synaptic conductances the cell experiences
        self.sgEx = np.zeros(self.n_sigs)
        self.sgIn = np.zeros(self.n_sigs)

        # - Vectors to store group-wise synaptic currents the cell receives
        self.AveExCurr = np.zeros(n_sigs)
        self.AveInCurr = np.zeros(n_sigs)

        # - Save pre-syn. learning trace
        self.pre = np.zeros(n_cells)

        # - Time of last output spike to keep track of spike times for implementing refractory period
        self.tolos = 0.0

        # - Initialize the output voltage to rest potential
        self.V = np.zeros(len(self.time_vector))
        self.V[0] = self.VRest # - Start with V being at rest potential

        # - Overall exc. and inh. synaptic conductance
        self.gEx = 0.0
        self.gIn = 0.0
        # - Postsynaptic learning trace
        self.post = 0.0
        # - Leak conductance (Everything else is normalized in respect to the leak.)
        self.gLeak = 1.0
        # - Counters for rates and averaging
        self.InputSpikeCount = 0
        self.OutputSpikeCount = 0
        self.AveCurrCounter = 0

    def reset_all(self):
        """
        @brief Reset states necessary for simulation, not learning
        """
        self.input *= 0
        self.Exkeep *= 0
        self.Inkeep *= 0
        self.filtered_white_noise *= 0 
        self.input_spike_refr *= 0
        self.sgEx *= 0
        self.sgIn *= 0
        self.AveExCurr *= 0
        self.AveInCurr *= 0
        self.pre *= 0
        self.tolos = 0.0
        self.V *= 0
        self.gEx *= 0
        self.gIn *= 0
        self.post *= 0
        self.InputSpikeCount *= 0
        self.OutputSpikeCount *= 0
        self.AveCurrCounter *= 0

    def evolve(self):
        """
        @brief Evolve over signal that is generated ad-hoc for self.duration seconds
        """

        for idx,t in enumerate(self.time_vector):
            if(idx == 0): # - Skip first time step
                continue

            # - Decay the group and general synaptic conductances
            self.gEx *= self.expGEx
            self.gIn *= self.expGIn
            self.sgEx *= self.expGEx
            self.sgIn *= self.expGIn

            # - Exponential decay of the traces used for learning
            self.pre[:self.num_exc] *= self.expPlasticity
            self.post *= self.expPlasticity

            self.generate_input()
            self.generate_input_spikes_and_update_conductances_and_weights()

            # - Update membrane potentials and generate output spike
            # - Clamp voltages to resting potential if neuron still in refr. period
            if((t - self.tolos) < self.refr):
                self.V[idx] = self.VRest
            else: # - Neuron can emit a spike
                # - Total membrane conductance
                gTot = self.gLeak + self.gEx + self.gIn
                # - Effective time constant
                tauEff = self.tau_mem / gTot
                # - Membrane potential V strives towards
                VInf =  ((self.gLeak*self.VRest + self.gEx * self.EAMPA + self.gIn*self.EGABA) / gTot)
                # - Update the membrane potential
                self.V[idx] = VInf + (self.V[idx-1] - VInf)*np.exp(-self.dt / tauEff)

                # - Keep track of group-wise input currents for plotting
                for i in range(self.n_sigs):
                    self.AveExCurr += self.sgEx[i]*(self.V[idx]-self.EAMPA)
                    self.AveInCurr += self.sgIn[i]*(self.V[idx]-self.EGABA) + (self.gLeak*(self.V[idx]-self.VRest))/self.n_sigs
                self.AveCurrCounter += 1

                if(self.V[idx] > self.Vth):
                    # - Output neuron spiked

                    # - Set the refr. period
                    self.tolos = t
                    self.V[idx-1] = 0.0 # - For plotting
                    self.V[idx] = self.VRest
                    self.OutputSpikeCount += 1
                    # - Update post-syn. learning trace
                    self.post += self.eta
                    # - Other part of the rule: Update inh. syn. on post-syn. spike
                    for j in range(self.num_exc,self.n_cells):
                        self.synapse[j] += self.pre[j]
                
                # - Keep track of exc. and inh. input currents
                self.Exkeep[idx] = self.gEx*(self.V[idx]-self.EAMPA)
                self.Inkeep[idx] = self.gIn*(self.V[idx]-self.EGABA)

        # - End time loop
        print(f"Spike count is {self.OutputSpikeCount}")

    def run(self, nruns = 100):
        fig = plt.figure(figsize=(6,5))
        for num_run in range(nruns):
            self.reset_all()
            self.evolve()

            plt.clf()
            ax1 = fig.add_subplot(311)
            ax1.plot(self.time_vector, self.Exkeep / 100.0, color='r', label='Exc. current')
            ax1.plot(self.time_vector, self.Inkeep / 100.0, color='g', label='Inh. current')
            ax1.plot(self.time_vector, (self.Exkeep+self.Inkeep) / 100.0, color='k', label='Total current')
            ax1.legend()
            ax1.set_ylabel("I(t) [nA]")
            ax2 = fig.add_subplot(312)
            ax2.plot(self.time_vector, self.V)
            ax2.set_ylabel("V(t) [mV]")
            ax3 = fig.add_subplot(313)
            ax3.plot(self.sgEx, marker="s", label="Exc. current")
            ax3.plot(self.sgIn, marker="D",linestyle='--',label="Inh. current")
            ax3.legend()
            plt.draw()
            plt.pause(0.001)

    def generate_input_spikes_and_update_conductances_and_weights(self):
        for i in range(self.n_cells):
            if(np.random.uniform() < self.input[np.abs(self.input_group[i])-1] and (self.input_spike_refr[i] <= 0.0)):
                # - i-th input neuron fired a spike
                if(self.input_group[i] >= 0): # - Input neuron is excitatory
                    # - Update general exc. syn. conductance
                    self.gEx += self.gBarEx*self.synapse[i]
                    # - Keep track of group wise conductance
                    self.sgEx[self.input_group[i]-1] += self.gBarEx*self.synapse[i]
                else: # - Inhibitory
                    # - Update general inh. syn. conductance
                    self.gIn += self.gBarIn*self.synapse[i]
                    # - Update group-wise inh. conductances
                    self.sgIn[np.abs(self.input_group[i])-1] += self.gBarIn*self.synapse[i]
                    # - Update pre-syn. trace
                    self.pre[i] += self.eta
                    # - Update synaptic strength according to rule
                    self.synapse[i] += self.post - self.alpha
                    # - Clamp to 0
                    if(self.synapse[i] < 0):
                        self.synapse[i] = 0.0

                self.input_spike_refr[i] = self.refr
                self.InputSpikeCount += 1

            # - No input spike emitted
            else:
                self.input_spike_refr[i] -= self.dt
        # - End for      
        return


    def generate_input(self):
        """
        @brief Fill self.input with new input, dependent of self.filtered_white_noise
        """
        for i in range(self.n_sigs):
            re = np.random.uniform()-0.5
            self.filtered_white_noise[i] = re - (re - self.filtered_white_noise[i])*self.expnoise
            self.input[i] = self.bg_rate + np.maximum(0, self.max_rate* self.filtered_white_noise[i])/self.norm_factor
        return


    def assign_input_group(self):
        input_group = np.empty(self.n_cells)
        temptype = 0
        for i in range(self.n_cells):
            if(i < self.n_cells*self.ex_frac):
                if(np.mod(i,self.ex_groupsize) == 0):
                    temptype += 1
                input_group[i] = temptype
            else:
                input_group[i] = -temptype
                if(np.mod(i+1,self.in_groupsize)==0):
                    temptype -= 1
        return np.asarray(input_group, dtype=int)

    def assign_synapse_weights(self, n_exc, n_inh):
        synapse = np.empty(n_exc+n_inh)
        for i in range(n_exc+n_inh):
            if(i < n_exc):
                synapse[i] = 0.3 + (1.1/(1+(self.input_group[i]-5)**4)) + np.random.uniform()*0.1
            else:
                synapse[i] = 0.1
        return synapse


np.random.seed(42)

istdp_experiment = iSTDP(dt=0.001)
istdp_experiment.run()