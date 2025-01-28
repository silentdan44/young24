# Young24
Young24 is an automatic tool for calculating Young modulus of polymers. It works with [LAMMPS](https://github.com/lammps/lammps) code.
## Usage
In order to use this code for simulation you have to create a variable with path to you LAMMPS executable:
```bash
export LMP_EXEC=path/to/your/lammps/executable/lmp_mpi 
```
The simplest pipeline should look like this:
```bash
python main.py eq start.data
python main.py run-nemd --maxdef 0.1
python main.py analyse
```