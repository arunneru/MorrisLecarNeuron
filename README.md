@ArunNeru

# Interactive visualization of morris lecar model

To run (on a linux machine, via commandline): 
```console
foo@bar:~ python morrisInteractive.py
```

There are three subplots and four ways to interact with the figure generated once you run the script
- LEFT subplot: 
	- Nullcline
	- Phase curve starting from an initial condition
- RIGHT subplot: calcium and potassium conductance parameter space
	- different colors indicate different combination of the signs of eigenvalues of the local linearization (Jacobian matrix) of the system at the single equilibrium point 
- BOTTOM subplot: Voltage trace as a function of time. The total duration of the simulation is for 400 ms

## Here are the HANDLES that are available
1. Radio buttons for different external currents (Iext)
	- Upon chosing any one of them (ranges from 100 to 500 micoampere/centimetersquared), 
		1. The phase curve and nullcline will be updated on the left plot,
 		2. The parameter space will be replaced with corrosponding matrix of eigenvalue combinations
		3. The voltage trace will be updated in the lower plot
	It might take a moment to load. 
2. Sliders for calcium (gCa) and potassium conductance (gK)
	- Once slider position is picked, changing the value of calcium or potassium conductance,
	The phase curve (and voltage trace) and the nullcline will be updated
3. Initial conditions on the phase space
	- The phase space in the left subplot can be used to plot phase curve with any intial condition by clicking inside the subplot.
4. Chosing conductance pair from parameter space
	- Clicking anywhere on the parameter space changes the value of calcium and potassium conductances accordingly.
And the phase curve, nullcline and voltage trace will be updated.


