from ctypes.macholib import dyld
import json
import os
from tkinter import END
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import show

PLOT_LIB = "plotly"

DIAGONALS = {4, 5, 6, 7}

DIRECTION_TO_VECTOR = {
    0: (0, 1),  # Up
    1: (0, -1),  # Bas
    2: (-1, 0),  # Left
    3: (1, 0),  # Right
    4: (-1, 1),  # Left Up
    5: (1, 1),  # Right Up
    6: (-1, -1),  # Left Down
    7: (1, -1),  # Right Down
    8: (0, 0),  # Free
}


class BeatSaberMapAnalyzer:

    def __init__(self, map_folder):
        self.map_folder = map_folder
        self.stats = {
            "map_name": os.path.basename(map_folder),
            "total_notes": 0,
            "diagonal_notes": 0,
            "duration": {
                "seconds": 0,
                "minutes": 0,
            },
            "density": 0.0,
        }

    def load_data(self):
        # Ouverture du fichier info.dat et gestion des erreurs
        info_path = os.path.join(self.map_folder, "info.dat")
        if not os.path.exists(info_path):
            print(f"info.dat not found in {self.map_folder}")
        with open(info_path, "r") as info_file:
            try:
                self.info_data = json.load(info_file)
                print("info file opened")
            except json.JSONDecodeError:
                print(f"Error decoding JSON in {info_path}")

        # Ouverture du fichier expertplus.dat et gestion des erreurs
        expert_path = os.path.join(self.map_folder, "ExpertPlusStandard.dat")
        if not os.path.exists(expert_path):
            print(f"ExpertPlusStandard.dat not found in {self.map_folder}")
        with open(expert_path, "r") as expert_file:
            try:
                self.expert_data = json.load(expert_file)
                print("expert file opened")
            except json.JSONDecodeError:
                print(f"Error decoding JSON in {info_path}")
        self.bpm = self.info_data.get("_beatsPerMinute", 120)
        self.notes = self.expert_data.get("_notes", [])
        # Analyse des donn�es de base
        self.stats["duration"]["seconds"] = self.info_data.get("_songLength", 0)
        # Approximation de la dur�e si elle est manquante
        if self.stats["duration"]["seconds"] <= 0:
            print("Approximating duration from last note")
            last_beat = max([note["_time"] for note in self.notes], default=0)
            # conversion de beats en secondes
            self.stats["duration"]["seconds"] = round(last_beat * 60 / self.bpm, 2)

    def analyze_maps(self):
        # Analyse a map
        # info_data: dict, expert_data should be loaded before calling this method

        if self.info_data is None or self.expert_data is None:
            print("Data should be loaded before analysing map")
            return self.stats
        print(f"Analyzing map: {self.stats['map_name']}")

        # Analyse des notes
        self.stats["total_notes"] = len(self.notes)
        for note in self.notes:
            if note.get("_cutDirection") in DIAGONALS:
                self.stats["diagonal_notes"] += 1

            self.stats["duration"]["minutes"] = round(
                self.stats["duration"]["seconds"] / 60, 2
            )
            # Calcul de la densit�
            if self.stats["duration"]["seconds"] > 0:
                self.stats["density"] = round(
                    self.stats["total_notes"] / self.stats["duration"]["seconds"], 2
                )

            return self.stats

    def visualize_map(self):
        # Visualisation d'une map en 3D
        # les data doivent �tre charg�es avant d'appeler cette m�thode

        print(f"Visualizing map: {self.stats['map_name']}")
        if not self.notes:
            print("No notes to visualize")
            return

        # Initialisation des listes pour les coordonn�es
        Xr, Yr, Zr = [], [], []
        Xb, Yb, Zb = [], [], []
        U_r, V_r, W_r = [], [], []  # Pour les vecteurs de direction rouges
        U_b, V_b, W_b = [], [], []  # Pour les vecteurs de direction bleus
        vector_length = 0.6
        windows_size = 4.0

        # Plotly 3D scatter plot
        for note in self.notes:
            # Extraction des coordonn�es et des couleurs
            color = ["red" if note["_type"] == 0 else "blue"]
            cut = note["_cutDirection"]
            dx, dz = DIRECTION_TO_VECTOR[cut]
            dx *= vector_length
            dy = 0
            dz *= vector_length
            dx, dy, dz = normalize(dx, dy, dz)  # Normalisation des vecteurs

            if color == ["red"]:
                Xr.append(note["_lineIndex"])
                Yr.append(note["_time"] * 60 / self.bpm)
                Zr.append(note["_lineLayer"])
                U_r.append(dx)
                V_r.append(dy)
                W_r.append(dz)

            elif color == ["blue"]:
                Xb.append(note["_lineIndex"])
                Yb.append(note["_time"] * 60 / self.bpm)
                Zb.append(note["_lineLayer"])
                U_b.append(dx)
                V_b.append(dy)
                W_b.append(dz)

        Xr = np.asarray(Xr)
        Yr = np.asarray(Yr)
        Zr = np.asarray(Zr)
        Xb = np.asarray(Xb)
        Yb = np.asarray(Yb)
        Zb = np.asarray(Zb)
        U_r = np.asarray(U_r)
        V_r = np.asarray(V_r)
        W_r = np.asarray(W_r)
        U_b = np.asarray(U_b)
        V_b = np.asarray(V_b)
        W_b = np.asarray(W_b)

        # Calcul des bornes pour les fen�tres de temps
        tmin = float(
            min(Yr.min() if Yr.size else np.inf, Yb.min() if Yb.size else np.inf)
        )
        tmax = float(
            max(Yr.max() if Yr.size else -np.inf, Yb.max() if Yb.size else -np.inf)
        )

        bins = np.arange(tmin, tmax + windows_size, windows_size)

        frames = []
        for start in bins[:-1]:
            end = start + windows_size
            # Filtrer les notes dans la fen�tre de temps
            # mr et mb sont des masques booleens pour les notes rouges et bleues respectivement qui sont dans la fen�tre de temps [start, end)]
            mr = (Yr >= start) & (Yr < end)
            mb = (Yb >= start) & (Yb < end)
            frames.append(
                go.Frame(
                    name=f"{start:.2f}-{end:.2f}",
                    data=[
                        go.Scatter3d(
                            x=Xr[
                                mr
                            ],  # on garde que les items de Xr, Yr, Zr qui sont dans la fen�tre de temps gr�ce au masque mr
                            y=Yr[mr],
                            z=Zr[mr],
                            mode="lines+markers",
                            marker=dict(size=3, color="red"),
                            line=dict(color="red", width=3),
                            showlegend=False,
                        ),
                        go.Scatter3d(
                            x=Xb[mb],
                            y=Yb[mb],
                            z=Zb[mb],
                            mode="lines+markers",
                            marker=dict(size=3, color="blue"),
                            line=dict(color="blue", width=3),
                            showlegend=False,
                        ),
                        go.Cone(
                            x=Xr[mr],
                            y=Yr[mr],
                            z=Zr[mr],
                            u=U_r[mr],
                            v=V_r[mr],
                            w=W_r[mr],
                            sizemode="raw",
                            sizeref=0.6,
                            anchor="tail",
                            colorscale=[[0, "red"], [1, "red"]],
                            showscale=False,
                            name="Red Notes",
                            showlegend=False,
                        ),
                        go.Cone(
                            x=Xb[mb],
                            y=Yb[mb],
                            z=Zb[mb],
                            u=U_b[mb],
                            v=V_b[mb],
                            w=W_b[mb],
                            sizemode="raw",
                            sizeref=0.6,
                            anchor="tail",
                            colorscale=[[0, "blue"], [1, "blue"]],
                            showscale=False,
                            name="Blue Notes",
                            showlegend=False,
                        ),
                    ],
                )
            )

        init = frames[0].data if frames else []

        sliders = [
            {
                "steps": [
                    {
                        "method": "animate",
                        "args": [
                            [frame.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": frame.name,
                    }
                    for frame in frames
                ],
                "transition": {"duration": 0},
                "x": 0,
                "y": -0.05,
                "len": 1.0,
            }
        ]

        updatemenus = [
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 400, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {"frame": {"duration": 0, "redraw": True}},
                        ],
                    },
                ],
                "x": 0,
                "y": -0.12,
                "direction": "left",
                "showactive": False,
            }
        ]

        fig = go.Figure(
            data=init,
            layout=go.Layout(
                title=f"Map: {self.stats['map_name']}",
                scene=dict(
                    xaxis_title="Colonne (0-3)",
                    yaxis_title="Temps (s)",
                    zaxis_title="Etage (0-2)",
                    aspectmode="cube",
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                sliders=sliders,
                updatemenus=updatemenus,
            ),
            frames=frames,
        )

        fig.show()


@staticmethod
def normalize(u, v, w):
    # Normalise les vecteurs u, v, w
    norm = np.sqrt(u**2 + v**2 + w**2)
    if norm == 0:
        return 0, 0, 0
    return u / norm, v / norm, w / norm
