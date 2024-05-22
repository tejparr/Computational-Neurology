# Internal Clocks

This folder includes a set of demos that relate to a model using internal clocks or metronomes to determine the temporal dynamics of actions. 

### Background
The simulations developed in the DEMO_Metronomes.m routine are designed to show how we might make use of internal clocks (or metronomes) to synchronise our behaviour to events in the world. This is of particular relevance for conditions such as Parkinson's disease or Lewy Body Dementia in which everything from movement to thought can be slowed. 

The theoretical background for this work depends upon ideas about chunking of continuous events into ...

In what follows, we illustrate the 'healthy' simulation under several different conditions (changing only the input stimulus but not the model). Following this, we consider what happens when we 'lesion' the model, by changing prior beliefs. We then consider the relationship between aspects of the model and neurobiology (including both the relationship between computational architecture and anatomy and the role of electrophysiological observations - including time-frequency and spectral analyses of these data - in supporting healthy motor cognition), and finish by evaluating the potential to recover parameters from this model through fitting to behavioural data. 

### Simulation under default settings
The simulation presented here shows...
<img src="./Graphics/Animation Default.gif"/>

### Simulation with faster stimulus presentation
The simulation presented here shows...
<img src="./Graphics/Animation Fast.gif"/>

### Simulation with occlusions
The simulation presented here shows...
<img src="./Graphics/Animation Default  Occlusion.gif"/>

### Simulation with attenuated policy precision
The simulation presented here shows...
<img src="./Graphics/Animation Impaired Policy Precision.gif"/>

### Simulation with impaired segregation of preparation and execution phases
The simulation presented here shows...
<img src="./Graphics/Animation Impaired chunking precision.gif"/>

### Simulation with belief in persistence of occluder status
The simulation presented here shows...
<img src="./Graphics/Animation Persistent Occluders.gif"/>


