'''Utility functions to generate plots'''
import os
import itertools

from typing import List, Dict, Tuple

import matplotlib as mpl
# mpl.use('PS')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from matplotlib import tight_layout
from dl4nlp_pos_tagging.config import Config
import seaborn as sns
import pandas as pd
import numpy as np


def set_styles(mpl_style: str = Config.mpl_style, sns_palette: str = Config.sns_palette):
    mpl.style.use(mpl_style)
    sns.set_palette(sns.color_palette(sns_palette))

def save_figure(dirname, filename):
    plt.savefig(os.path.join(dirname, filename + '.png'), bbox_inches='tight', dpi=500)
    # plt.savefig(os.path.join(dirname, filename + '.svg'), bbox_inches='tight')

def new_figure(**kwargs):
    fig, ax = plt.subplots()
    return fig, ax

def annotate(fig, ax, **kwargs) -> None:

    xlabel = kwargs.get("xlabel")
    if xlabel:
        ax.set_xlabel(xlabel)

    ylabel = kwargs.get("ylabel")
    if ylabel:
        ax.set_ylabel(ylabel)

    title = kwargs.get("title")
    subtitle = kwargs.get("subtitle")

    if title and subtitle:
        fig.suptitle(title, fontsize=plt.rcParams['axes.titlesize'], y=1)
        plt.title(subtitle, fontsize=plt.rcParams['axes.titlesize'] * 0.75)

    elif title:
        plt.title(title)

    legend_location = kwargs.get("legend_location", "upper left")

    plt.tight_layout()
    sns.despine(ax=ax)
    ax.legend(loc=legend_location)

def multi_univariate_kde(frame, data_col, label_col, **kwargs):
    formats = [('.','-'), ('.','--'), ('.',':'), ('.','-.'), ('*','-'), ('*','--'), ('*',':'), ('*','-.')]
    labels = frame[label_col].unique()
    assert len(labels) <= len(formats), f"Can only plot a maximum of {len(formats)} on one figure"
    for i, l in enumerate(labels):
        marker, style = formats[i]
        sns.kdeplot(frame[frame[label_col] == l][data_col], label=l, linestyle=style, **kwargs)

def grouped_boxplot(frame, x, y, hue, **kwargs):
    sns.boxplot(
        x=x,
        y=y,
        hue=hue,
        data=frame,
        **kwargs
    )

def heatmap(frame, ax, **kwargs):
    sns.heatmap(
        frame,
        ax=ax,
        **kwargs
    )

def regplot(frame, x, y, **kwargs):
    sns.regplot(
        x=x,
        y=y,
        data=frame,
        **kwargs
    )

def displot(frame, x, y, **kwargs):
    sns.displot(
        frame,
        x=x,
        y=y,
        **kwargs
    )

def kdeplot(frame, x, y, **kwargs):
    sns.kdeplot(
        data=frame,
        x=x,
        y=y,
        **kwargs
    )

def jointplot(frame, x, y, **kwargs):
    sns.jointplot(
      x=x,
      y=y,
      data=frame,
      **kwargs
    )
