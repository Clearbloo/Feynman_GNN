import math
import os
import os.path as osp

import constants as const
import numpy as np
import pandas as pd
from feynman_graphs import (
    S_Channel,
    T_Channel,
    # U_Channel,
    build_tree_diagrams,
    diagram_builder_gluon
)
from particles import Particle, ParticleRegistry


def QED_dataset_builder():
    """
    Main function for building the dataset.

    This function builds the dataset for various physics processes, including QED and QCD interactions.
    It generates dataframes for different combinations of particles and channels, and saves them as CSV files.
    It also performs sampling and assigns weights to the data based on the inverse of the values in the 'y' column.

    Returns:
        None
    """
    E_Minus: Particle = ParticleRegistry.get_particle_class("e_minus")
    E_Plus: Particle = ParticleRegistry.get_particle_class("e_plus")
    Mu_Minus: Particle = ParticleRegistry.get_particle_class("mu_minus")
    Mu_Plus: Particle = ParticleRegistry.get_particle_class("mu_plus")
    q_e = const.q_e
    m_e = const.m_e

    CURRENT_DIR = osp.dirname(osp.abspath(__file__))
    RAW_DIR = osp.join(osp.dirname(CURRENT_DIR), "data/raw")

    # write the matrix element as a function
    def Mfi_squared(p, theta):
        return (
            (q_e**2 * (1 + np.cos(theta))) ** 2 + (q_e**2 * (1 - np.cos(theta))) ** 2
        ) / 2

    graphs: list[Particle] = build_tree_diagrams(
        E_Minus(), E_Plus(), Mu_Minus(), Mu_Plus(), S_Channel(), True
    )
    graph = graphs[0]
    df_e_annih = graph.dataframe_builder(
        0,
        ang_res=100,
        p_res=100,
        p_min=10**3,
        p_max=10**6,
        Mfi_squared=Mfi_squared,
        graph=graph,
    )

    # save the file
    df_e_annih_mu = df_e_annih
    os.makedirs(RAW_DIR, exist_ok=True)
    df_e_annih_mu.to_csv(path_or_buf=f"{RAW_DIR}/QED_data_e_annih_mu.csv")

    df_e_annih_mu.plot("theta", "y", kind="scatter")

    # ## **Extending the dataset**
    #
    # Repeating for
    # $$e^-e^+\to e^-e^+$$
    #
    # This requires the inclusion of the t-channel diagrams

    def Mfi_squared(p, theta):
        """
        s = p_1^2 + p_2^2 + 2p_1p_2
        t = p_1^2 + p_3^2 - 2p_1p_3
        """
        s = 4 * (m_e**2 + p**2)
        t = 2 * m_e**2 - 2 * p**2 * (1 - math.cos(theta))
        u = 2 * m_e**2 - 2 * p**2 * (1 + math.cos(theta))
        return (
            2
            * q_e**4
            * ((u**2 + t**2) / s**2 + (u**2 + s**2) / t**2 + 2 * u**2 / (s * t))
        )  # 8*(q_e**4)*(s**4+t**4+u**4)/(4*s**2*t**2)

    graph_t = build_tree_diagrams(
        E_Minus(), E_Plus(), E_Minus(), E_Plus(), T_Channel(), True
    )
    graph_s = build_tree_diagrams(
        E_Minus(), E_Plus(), E_Minus(), E_Plus(), S_Channel(), True
    )
    graph = graph_s + graph_t
    df = graph.dataframe_builder(
        0.5,
        ang_res=400,
        p_res=100,
        p_min=10**3,
        p_max=10**6,
        Mfi_squared=Mfi_squared,
        graph=graph,
    )
    df_e_annih_e = df
    df_e_annih = pd.concat([df_e_annih, df], ignore_index=True)
    df_merge = df_e_annih

    # save the file
    os.makedirs(RAW_DIR, exist_ok=True)
    df_e_annih_e.to_csv(path_or_buf=RAW_DIR + "/QED_data_e_annih_e.csv")
    df_e_annih.to_csv(path_or_buf=RAW_DIR + "/QED_data_e_annih.csv")

    """Other combinations"""

    def Mfi_squared(p, theta):
        return (
            (q_e**2 * (1 + np.cos(theta))) ** 2 + (q_e**2 * (1 - np.cos(theta))) ** 2
        ) / 2

    graph = build_tree_diagrams(
        Mu_Minus(), Mu_Plus(), E_Minus(), E_Plus(), S_Channel(), True
    )
    df = graph.df(
        0,
        ang_res=100,
        p_res=100,
        p_min=0,
        p_max=10**6,
        Mfi_squared=Mfi_squared,
        graph=graph,
    )
    df_annih = df_e_annih
    df_annih = pd.concat([df_annih, df], ignore_index=True)

    graph = build_tree_diagrams(
        Mu_Minus(), Mu_Plus(), Mu_Minus(), Mu_Plus(), S_Channel(), True
    )
    df = graph[0].build_df(
        0,
        ang_res=100,
        p_res=100,
        p_min=10**3,
        p_max=10**6,
        Mfi_squared=Mfi_squared,
        graph=graph,
    )
    df_annih = pd.concat([df_annih, df], ignore_index=True)

    # save the file
    os.makedirs(RAW_DIR, exist_ok=True)
    df_annih.to_csv(path_or_buf=RAW_DIR + "/QED_data_annih.csv")

    df_annih.plot("theta", "y_norm", kind="scatter")
    return df_merge, df_e_annih_mu, df_annih, df_e_annih_e, df_e_annih

def QCD_dataset_builder():
    # ## QCD
    m_top = const.m_top
    alpha_S = const.alpha_S
    Up_r: Particle = ParticleRegistry.get_particle_class("Up_r")
    AntiUp_b: Particle = ParticleRegistry.get_particle_class("AntiUp_b")
    Top_r: Particle = ParticleRegistry.get_particle_class("Top_r")
    AntiTop_b: Particle = ParticleRegistry.get_particle_class("AntiTop_b")

    CURRENT_DIR = osp.dirname(osp.abspath(__file__))
    RAW_DIR = osp.join(osp.dirname(CURRENT_DIR), "data/raw")

    def Mfi_squared(p, theta):
        """
        s = p_1^2 + p_2^2 + 2p_1p_2
        t = p_1^2 + p_3^2 - 2p_1p_3
        """
        if p <= m_top:
            return 0
        s = 4 * p**2
        t = m_top**2 - 2 * p**2 + 2 * p * math.cos(theta) * math.sqrt(p**2 - m_top**2)
        u = m_top**2 - 2 * p**2 - 2 * p * math.cos(theta) * math.sqrt(p**2 - m_top**2)
        return (
            16 * alpha_S**4 * (t**4 + u**4 + 2 * m_top**2 * (2 * s - m_top**2)) / (s**2)
        )

    graph = diagram_builder_gluon(
        Up_r(), AntiUp_b(), Top_r(), AntiTop_b(), S_Channel(), True
    )
    df = graph[0].build_df(
        0,
        ang_res=1000,
        p_res=10,
        p_min=m_top * 0.9,
        p_max=m_top * 2,
        Mfi_squared=Mfi_squared,
        graph=graph,
    )
    df_QCD = df

    # save the file
    os.makedirs(RAW_DIR, exist_ok=True)
    df_QCD.to_csv(path_or_buf=RAW_DIR + "/QCD_data.csv")

    df_merge, df_e_annih_mu, df_annih, df_e_annih_e, df_e_annih = QED_dataset_builder()

    df_all = pd.concat([df_merge, df_QCD], ignore_index=True)
    df_all.to_csv(path_or_buf=RAW_DIR + "/all_data.csv")

    df_all.plot("theta", "y_norm", kind="scatter")

    """#**Testing Sampling**"""

    w = df_e_annih_mu["y"].astype(float).round(4)
    w_dict = 1 / w.value_counts()

    def weight_lookup(x):
        x = np.round(x, 4)
        return w_dict[x]

    df_e_annih_mu["sample_weight"] = df_e_annih_mu["y"].apply(
        lambda x: weight_lookup(x)
    )

    rep_df = df_e_annih_mu.sample(n=1000, weights=df_e_annih_mu["sample_weight"])
    # print(rep_df['y'])
    rep_df["y"].hist(bins=10)
