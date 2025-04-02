# Internal Clocks

This folder includes a set of demos that relate to a model using internal clocks or metronomes to determine the temporal dynamics of actions. These were constructed for the paper <a href="https://www.sciencedirect.com/science/article/pii/S0149763424004536"> 'Inferring When To Move'</a> (Parr, Oswal, and Manohar).

### Background
The simulations developed in the DEMO_Metronomes.m routine are designed to show how we might make use of internal clocks (or metronomes) to synchronise our behaviour to events in the world. This is of particular relevance for conditions such as Parkinson's disease or Lewy Body Dementia in which everything from movement to thought can be slowed. 

The theoretical background for this work depends upon ideas about chunking of continuous trajectories into sequences of discrete events. Previously this problem has been addressed either by assuming fixed durations for each discrete event (e.g., <a href="https://direct.mit.edu/netn/article/1/4/381/5401/The-graphical-brain-Belief-propagation-and-active">The Graphical Brain</a>, <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC6115199/">The Discrete and Continuous Brain</a>), by forming priors based upon the spectral envelopes of certain kinds of data (e.g., <a href="https://pubmed.ncbi.nlm.nih.gov/32732017/">Active Listening</a>), or by tying events to peaks in Lotka-Volterra like dynamical systems (<a href="https://pubmed.ncbi.nlm.nih.gov/27391681/">Active Inference and Learning in the Cerebellum</a>). We approach this problem through use of a set of internal clocks with different speeds, such that our brains must identify the clock speed(s) that best explain temporally structured sensory data and use the same schedule to predict the proprioceptive data necessary to generate movements.

In what follows, we illustrate the 'healthy' simulation under several different conditions (changing only the input stimulus but not the model). The task here, inspired by clinical assessment of Parkinson's disease, is to time a regular alternating movement to a periodic stimulus. When the stimulus is omitted, participants must maintain the same rhythm. Following this, we consider what happens when we 'lesion' the model, by changing prior beliefs. We then consider the relationship between aspects of the model and neurobiology (including both the relationship between computational architecture and anatomy and the role of electrophysiological observations - including time-frequency and spectral analyses of these data - in supporting healthy motor cognition), and finish by evaluating the potential to recover parameters from this model through fitting to behavioural data. 

### Simulation under default settings
The simulation presented here shows baseline 'healthy' performance of the task. A clock speed is quickly inferred and movements are well-timed relative to the stimulus.
<img src="./Graphics/Animation Default.gif"/>

### Simulation with faster stimulus presentation
The simulation presented here shows that when a stimulus is presented with faster frequency, a faster clock speed is inferred, and consequently, movements alternate with faster frequences.
<img src="./Graphics/Animation Fast.gif"/>

### Simulation with occlusions
The simulation presented here shows a return to the baseline frequency but now introduces an omission period in which the rhythm that was initially externally entrained now must be internally generated.
<img src="./Graphics/Animation Default  Occlusion.gif"/>

### Simulation with attenuated policy precision
The simulation presented here shows that when one has reduced confidence in the selection of the target (here, a downweighting of a softmax parameter), there is a decline in the amplitude of the alternating movements that is independent of whether the timings are internally or externally entrained.
<img src="./Graphics/Animation Impaired Policy Precision.gif"/>

### Simulation with impaired segregation of preparation and execution phases
The simulation presented here shows that when the boundaries between action preparation and execution phases (i.e., the boundaries from one discrete time-step to the next) are blurred, both beliefs and behaviour rapidly deteriorate as uncertainty accumulates.
<img src="./Graphics/Animation Impaired chunking precision.gif"/>

### Simulation with belief in persistence of occluder status
The simulation presented here shows the situation in which the ability to redirect attention to externally or internally generated stimuli is disrupted. Here we see a specific loss of rhythm and amplitude, with a paradoxical hastening or festination emerging as faster clocks are unmasked by the loss of confidence in slower clocks.
<img src="./Graphics/Animation Persistent Occluders.gif"/>


