# -*- coding: utf-8 -*-
r"""
Plot all demos

"""

import glob
import locale
import os
import shutil
import subprocess

import matplotlib.pyplot as plt
import microstructpy as msp

locale.setlocale(locale.LC_NUMERIC, "C")

example_dir = '../../../src/microstructpy/examples'

welcome_fnames = ['intro_2_quality/trimesh.png',
                  'minimal/polymesh.png',
                  'basalt_circle/trimesh.png',
                  'foam/trimesh.png',
                  'two_phase_3D/trimesh.png',
                  'colormap/trimesh.png']

def main():
    # Copy Supporting Files
    supporting_files = ['aphanitic_cdf.csv', 'olivine_cdf.csv']
    for fname in supporting_files:
        filename = os.path.join(example_dir, fname)
        shutil.copy(filename, '.')

    # Run XML files
    xml_pattern = os.path.join(example_dir, '*.xml')
    for filename in glob.glob(xml_pattern):
        msp.cli.run_file(filename)

    # Run Python Scripts
    py_pattern = os.path.join(example_dir, '*.py')
    for filename in glob.glob(py_pattern):
        subprocess.call(['python', filename])

    # Remove Supporting Files
    for fname in supporting_files:
        os.remove(fname)

    # Create welcome figure
    create_welcome()

    # Create example subfigures
    seed_poly_tri('.')


def create_welcome():
    fig, axes = plt.subplots(2, 3, figsize=(21, 15))
    plt.subplots_adjust(wspace=0.05, hspace=0)
    for i, fname in enumerate(welcome_fnames):
        filename = os.path.join(example_dir, fname)
        im = plt.imread(filename)
        
        row_num = int(i / 3)
        col_num = i % 3
        ax = axes[row_num, col_num]
        ax.imshow(im)

        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    sub_fname = 'welcome_examples.png'
    sub_filename = os.path.join(example_dir, sub_fname)
    plt.savefig(sub_filename, pad_inches=0, bbox_inches='tight', dpi=200)
    plt.close('all')


def seed_poly_tri(filepath):
    basenames = ['seeds.png', 'polymesh.png', 'trimesh.png']
    ex_path = os.path.join(example_dir, filepath)
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.05, hspace=0)
    for i, fname in enumerate(basenames):
        filename = os.path.join(ex_path, fname)
        im = plt.imread(filename)

        ax = axes[i]
        ax.imshow(im)

        ax.set_axis_off()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    sub_fname = 'joined.png'
    sub_filename = os.path.join(ex_path, sub_fname)
    plt.savefig(sub_filename, pad_inches=0, bbox_inches='tight', dpi=300)
    plt.close('all')


if __name__ == '__main__':
    main()
    


