## SEIDR
----------------------

A revamped repo for Seidr - an instrument to perform photonic lantern fed kernel nulling at the VLTI.

**very early developmental stage


### Development installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/nathanske42/seidr2.0.git
cd seidr2.0
pip install -e .
```

### Structure

```
seidr
│   README.md
│   requirements.txt    
│
└───src
│   │
│   └───seidr
│       │   __init__.py
│       │   lanternfiber.py
│       │   SeidrSim.py
│       │   star_planet.py
│
│
└───test
│   │   test_kernel_nulling.py
│   │   test_star_planet.py
│
│
```


### TO DO

* Continue sifting through Seidr1.0 to extract useful code
* Map extrasolar planet (++) to kernel nulling chip performance


### Citation

If you found this code useful, please cite the following paper:

```bibtex
@inproceedings{taras2024kernel,
  title={Kernel nulling at VLTI with photonic lanterns for optimal fibre injection},
  author={Taras, Adam K and Norris, Barnaby and Chhabra, Sorabh and Cvetojevic, Nick and Foriel, Vincent and Ireland, Michael and Kraus, Stefan and Leon-Saval, Sergio and Martinache, Frantz and Paul, Jyotirmay and Spaldin, Eckhart and Sweeney, David and Tuthill, Peter},
  booktitle={Optical and Infrared Interferometry and Imaging IX},
  volume={13095},
  pages={242--250},
  year={2024},
  organization={SPIE}
}
```

