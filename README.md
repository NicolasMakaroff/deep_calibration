[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# Deep Calibration

## Features

- [x] nn module
- [x] tensorboard
- [ ] pricing speed comparison
- [ ] gp notebook
- [ ] prediction result comparison
- [ ] 

## Getting Started

### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name deep_cal python=3.6
	source activate deep_cal
	```
	- __Windows__: 
	```bash
	conda create --name deep_cal python=3.6 
	activate deep_cal
	```

3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/NicolasMakaroff/deep_calibration.git
cd deep_calibration/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `deep_cal` environment.  
```bash
python -m ipykernel install --user --name deep_cal --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `deep_cal` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

### Instructions

Follow the instructions in the `notebook` folder to get started!  