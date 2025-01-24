import os
import typer
# from rich import print
from typing_extensions import Annotated
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
# from scipy.constants import R
import subprocess

app = typer.Typer()
lmp = os.getenv('LMP_EXEC')
lmp_template = \
"""
units real
atom_style     full
pair_style      lj/cut/coul/long 10
bond_style      harmonic
angle_style     harmonic
dihedral_style  harmonic
improper_style umbrella
kspace_style    pppm 1.0e-4
special_bonds   dreiding 
pair_modify     tail yes mix arithmetic

read_data {name_of_data_file}

variable tmp equal "l{direction}"
variable L0 equal ${tmp}

timestep 1
fix             1 all npt temp 300 300 100 {first_dir} 1 1 1000 {second_dir} 1 1 1000
fix             2 all deform 1 {direction} erate 0.0000001 units box remap x

variable strain equal "(l{direction} - v_L0)/v_L0"
variable p1 equal "v_strain"
variable p2 equal "-pxx/10000*1.01325"
variable p3 equal "-pyy/10000*1.01325"
variable p4 equal "-pzz/10000*1.01325"
fix def1 all print 100 "${p1} ${p2} ${p3} ${p4}" file {number}{direction}.txt screen no

thermo 10000
thermo_style       custom step v_strain temp v_p2 v_p3 v_p4 ke pe press

run {number_of_steps}
"""
eq_template = \
"""
units real
dimension      3
boundary       p p p
atom_style     full
pair_style      lj/cut/coul/long 10
bond_style      harmonic
angle_style     harmonic
dihedral_style  harmonic
improper_style umbrella
kspace_style    pppm 1.0e-4
special_bonds   dreiding  
pair_modify     tail yes mix arithmetic
read_data {input_file}
minimize 1.0e-10 1.0e-10 8000 8000
timestep 1
velocity     all create 300 {seed} mom yes rot yes dist gaussian
fix 1 all npt temp 300 300 100 iso 1 1 1000 
thermo 1000
thermo_style custom step lx ly lz  pe temp
run 1000000
write_data eq{number}.data
"""
slurm_template_eq = \
"""#!/bin/bash
#SBATCH -N 1                  
#SBATCH -n 32 
#SBATCH --output=eq{number}.out
module load mpi/latest

mpirun -np 32 {lmp} -in eq{number}.in
"""
slurm_template_nemd = \
"""#!/bin/bash
#SBATCH -N 1                  
#SBATCH -n 8 
#SBATCH --output={number}{direction}.out
module load mpi/latest

mpirun -np 8 {lmp} -in {number}{direction}.in
"""

@app.command()
def run_nemd(
        inp:Annotated[str, typer.Argument(help="List of .data files")] = 'eq1.data,eq2.data,eq3.data,eq4.data,eq5.data',
        maxdef:Annotated[float, typer.Argument(help="Max strain applied")] = 0.4):
    num_str_steps = int(maxdef / 0.0000001)
    for index, file in enumerate(inp.split(',')):
        for var in ['x', 'y', 'z']:
            if var == 'x':
                first_dir = 'y'
                second_dir = 'z'
            elif var == 'y':
                first_dir = 'x'
                second_dir = 'z'
            else:
                first_dir = 'x'
                second_dir = 'y'
            lmpinp = lmp_template.replace('{name_of_data_file}', file).replace('{number}', f'{index+1}').replace('{number_of_steps}', f'{num_str_steps}').replace('{direction}', f'{var}').replace('{first_dir}', first_dir).replace('{second_dir}', second_dir)
            slm = slurm_template_nemd.replace('{number}', str(index+1)).replace('{lmp}', lmp).replace('{direction}', str(var))
            with open(f'{index+1}{var}.in', 'w') as f:
                f.write(lmpinp)
            with open(f'{index+1}{var}.sh', 'w') as f:
                f.write(slm)
            subprocess.run(f'sbatch {index+1}{var}.sh', shell=True)
            # subprocess.run(f'mpirun -np 32 {lmp} -i {index+1}{var}.in', shell=True)



@app.command()
def eq(inp:Annotated[str, typer.Argument(help="Name of alpha remd output .data file")],
       numrep:Annotated[int, typer.Argument(help="Number of replicas")] = 5):
    for i in range(numrep):
        seed = str(i) * 4
        eq_content = eq_template.replace('{number}', str(i)).replace('{input_file}', inp).replace('{seed}', seed)
        slm = slurm_template.replace('{number}', str(i)).replace('{lmp}', lmp)
        with open(f'eq{i}.in', 'w') as f:
            f.write(eq_content)
        with open(f'eq{i}.sh', 'w') as f:
            f.write(slm)
        subprocess.run(f'sbatch eq{i}.sh', shell=True)
        # subprocess.run(f'mpirun -np 32 {lmp} -i eq{i}.in', shell=True)



@app.command()
def analyse(inp:Annotated[str, typer.Argument(help="List of .txt files for analysis")] = '1x.txt,1y.txt,1z.txt,2x.txt,2y.txt,2z.txt,3x.txt,3y.txt,3z.txt,4x.txt,4y.txt,4z.txt,5x.txt,5y.txt,5z.txt',
            output:Annotated[str, typer.Argument(help="Name of the plot")] = 'plot.png'):
    df = pd.DataFrame()

    # Чтение файлов и создание общего dataframe
    for index, file in enumerate(inp.split(',')):
        file = pd.read_csv(f'{file}', skiprows=1, header=None, sep='\s+')
        file.columns = ['strain', 'px', 'py', 'pz']
        pxs = file.px[1000:7000].sum()
        pys = file.py[1000:7000].sum()
        pzs = file.pz[1000:7000].sum()
        if pxs > pys and pxs > pzs:
            stress = file.px
        elif pys > pxs and pys > pzs:
            stress = file.py
        else:
            stress = file.pz
        df[index] = stress
        df['strain'] = file.strain

    # Усреднение значений
    df['Mean'] = df.mean(axis=1)
    numrep = len(df.columns) - 2
    std_0 = df[0][:5000].std().round(3)
    std_mean = df['Mean'][:5000].std().round(3)

    # Линейная регрессия
    deformation_range = (0, 0.02)
    x = df[df['strain'] <= max(deformation_range)]['strain']
    y = df[df['strain'] <= max(deformation_range)]['Mean']

    def calc_y(params, x):
        return params[0] * x + params[1]

    def fun(params):
        return calc_y(params, x) - y

    params = [1, 2]
    result = least_squares(fun, params)

    young = result.x[0]

    # Построение графиков
    fig, axs = plt.subplots(2)
    fig.set_size_inches(10, 8)

    # Stress-strain curve усредненная
    axs[0].scatter(df.strain, df['Mean'], alpha=0.1)
    axs[0].plot(x, result.x[0] * x + result.x[1], linewidth=5, color='b')
    axs[0].set_xlabel('Strain')
    axs[0].set_ylabel('Stress, Gpa')
    axs[0].text(2, 3, f'Young modulus={young} Gpa')

    # Распределение стресса до усреднения и после
    axs[1].hist(df[0][:5000], bins=50, label=f'w/o replicas, std={std_0} Gpa')
    axs[1].hist(df['Mean'][:5000], bins=50, label=f'mean value of {numrep} replicas, std={std_mean} Gpa')
    axs[1].legend()
    axs[1].set_xlabel('Stress')

    fig.savefig(f'{output}')


if __name__ == "__main__":
    """
    
    """
    app()