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

read_data name_of_data_fle

variable tmp equal "l_direction"
variable L0 equal ${tmp}

timestep 1
fix             1 all npt temp 300 300 100 y 1 1 1000 z 1 1 1000
fix             2 all deform 1 direction erate 0.0000001 units box remap x

variable strain equal "(l_direction - v_L0)/v_L0"
variable p1 equal "v_strain"
variable p2 equal "-pxx/10000*1.01325"
variable p3 equal "-pyy/10000*1.01325"
variable p4 equal "-pzz/10000*1.01325"
fix def1 all print 100 "${p1} ${p2} ${p3} ${p4}" file number_direction.txt screen no

thermo 10000
thermo_style       custom step v_strain temp v_p2 v_p3 v_p4 ke pe press

run number_of_steps
"""

@app.command()
def run_nemd(
        inp:Annotated[str, typer.Argument(help="List of .data files")] = 'eq1.data,eq2.data,eq3.data,eq4.data,eq5.data',
        maxdef:Annotated[float, typer.Argument(help="Max strain applied")] = 0.4):
    for index, file in enumerate(inp.split(',')):
        lmp_template.replace('name_of_data_file', f'{file}')
        lmp_template.replace('number', f'{index+1}')
        lmp_template.replace('number_of_steps', f'{maxdef/0.0000001}')
        for var in ['x', 'y', 'z']:
            lmp_template.replace('direction', f'{var}')
            with open(f'{index+1}{var}.in', 'w') as f:
                f.write(lmp_template)
            subprocess.run(f'mpirun -np 32 ./{lmp} -i {index+1}{var}.in')



@app.command()
def eq(inp:Annotated[str, typer.Argument(help="Name of alpha remd output .data file")],
       numrep:Annotated[int, typer.Argument(help="Number of replicas")]):
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
    read_data name_of_input
    minimize 1.0e-10 1.0e-10 8000 8000
    timestep 1
    fix 1 all npt temp 300 300 100 iso 1 1 1000 
    thermo 1000
    thermo_style custom step lx ly lz  pe temp
    run 1000000
    write_data eqnumber.data
    """
    eq_template.replace('name_of_input', inp)
    for i in range(numrep):
        eq_template.replace('number', str(i))
        with open(f'eq{i}.inp', 'w') as f:
            f.write(eq_template)
        subprocess.run(f'mpirun -np 32 ./{lmp} -i eq{i}.in')



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