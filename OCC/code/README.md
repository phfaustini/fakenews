## Installing MLPack

```
sudo apt install cmake libboost-all-dev git
git clone https://github.com/mlpack/mlpack 
cmake ../
make -j4 # Number of cores
sudo make install
export LD_LIBRARY_PATH="/usr/local/lib/:$LD_LIBRARY_PATH" # add this also in ~/.bashrc
```

## Compiling this code

```
make ARGS=-O3 # Full optimisation
make ARGS=-g # debug mode
```

## Running experiments

```
sh run.sh
```

### Libraries

Experiments done with MLPack 3.1.1
Experiments done with Armadillo 8.4
