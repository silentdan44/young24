import os
import typer
from typing_extensions import Annotated
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
from scipy.stats import linregress



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
        slm = slurm_template_eq.replace('{number}', str(i)).replace('{lmp}', lmp)
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
        pxs = file.px[1000:7000].sum() # Чтобы понять по какому из направлений происходит деформация
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


    # Линейная регрессия
    deformation_range = (0, 0.02) # Границы линейного участка
    df_def = df[df['strain'] <= max(deformation_range)]
    x = df_def.strain
    errors = []
    youngs = []

    for i in range(len(df.columns)): # Цикл считает, как изменется модуль Юнга и std напряжения от усреднения
        y = df_def.iloc[:,:i+1].mean(axis=1)
        res = linregress(x, y)
        error = res.stderr
        young = res.slope
        errors.append(error)
        youngs.append(young)

    y = df_def.mean(axis=1)
    reg_of_all_reps = linregress(x, y)
    # Усреднение stress-strain curve
    df['Mean'] = df.mean(axis=1)

    # Построение графиков
    fig, axs = plt.subplots(3)
    fig.set_size_inches(10, 8)

    # Stress-strain curve усредненная
    axs[0].scatter(df.strain, df['Mean'], alpha=0.1)
    axs[0].plot(x, reg_of_all_reps.slope * x + reg_of_all_reps.x[1], linewidth=5, color='b')
    axs[0].set_xlabel('Strain')
    axs[0].set_ylabel('Stress, Gpa')
    axs[0].text(2, 3, f'Young modulus={reg_of_all_reps.slope} Gpa')

    # Модуль Юнга от числа реплик
    axs[1].plot(youngs)
    axs[1].set_xlabel('Number of replicas')
    axs[1].set_ylabel('Young modulus, Gpa')

    # Ошибки модуля Юнга
    axs[2].plot(errors)
    axs[2].set_xlabel('Number of replicas')
    axs[2].set_ylabel('Estimated std of Young modulus, Gpa')

    fig.savefig(f'{output}')



if __name__ == "__main__":
    """
    
    """
    app()